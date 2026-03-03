from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _ws(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _strip_answer(ans: str) -> str:
    """
    Heuristic cleanup to improve comparability across prompting styles.
    Official MQuAKE-Remastered eval uses strict string equality against answer/aliases;
    we keep this conservative: trim whitespace and common leading tokens.
    """
    s = str(ans or "").strip()
    # remove common prefixes like "Answer:" / "The answer is"
    s = re.sub(r"^(answer\\s*:\\s*)", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^(the\\s+answer\\s+is\\s+)", "", s, flags=re.IGNORECASE).strip()
    # strip trailing punctuation
    s = s.rstrip(" .\n\t\r")
    return s.strip()


def _check_answer_new(rec: Dict[str, Any], ans: str) -> bool:
    """
    Mirrors the official check logic from MQuAKE-Remastered (case-insensitive exact match
    to the new answer or any new aliases). We apply minimal stripping first.
    """
    ans = _strip_answer(ans)
    if not ans:
        return False
    ans_u = ans.upper()
    tgt = _ws(rec.get("answer_new", "")).upper()
    if ans_u == tgt:
        return True
    aliases = rec.get("answer_new_alias") or []
    for a in aliases:
        a_u = str(a).strip().upper()
        if a_u and ans_u == a_u:
            return True
    return False


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_vllm_generate(model_path: str, prompts: List[str], *, tp_size: int, max_tokens: int):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    mp = str(Path(model_path).expanduser())
    if Path(mp).exists():
        mp = str(Path(mp).resolve())
    llm = LLM(model=mp, tensor_parallel_size=tp_size, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        stop=["\n", ".", tokenizer.eos_token],
    )
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]


@dataclass
class Agg:
    n: int = 0
    ok: int = 0

    def add(self, ok: bool) -> None:
        self.n += 1
        self.ok += int(bool(ok))

    def to_dict(self) -> Dict[str, float]:
        return {"n": self.n, "acc": self.ok / max(1, self.n)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=24)
    parser.add_argument("--max_records", type=int, default=0, help="0 = all")
    parser.add_argument("--q_idxs", type=str, default="0,1,2", help="Question indices to query per record.")
    parser.add_argument("--cuda", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    q_idxs = [int(x.strip()) for x in str(args.q_idxs).split(",") if x.strip()]
    rows = load_json(args.data_path)
    if int(args.max_records) > 0:
        rows = rows[: int(args.max_records)]

    prompts: List[str] = []
    meta: List[Tuple[int, int]] = []
    for ridx, rec in enumerate(rows):
        qs = rec.get("questions") or []
        if not isinstance(qs, list) or not qs:
            continue
        for qi in q_idxs:
            if 0 <= qi < len(qs):
                q = _ws(qs[qi])
                if q:
                    prompts.append(q)
                    meta.append((ridx, qi))

    preds = run_vllm_generate(args.model_path, prompts, tp_size=int(args.tp_size), max_tokens=int(args.max_tokens))

    # Per-case aggregation: case is correct if ANY answer matches the (new) answer/alias.
    per_case_answers: Dict[int, List[str]] = {}
    pred_by_key: Dict[Tuple[int, int], str] = {}
    for (ridx, _qi), pred in zip(meta, preds):
        per_case_answers.setdefault(ridx, []).append(pred)
    for (ridx, qi), pred in zip(meta, preds):
        pred_by_key[(ridx, qi)] = pred

    case_any = Agg()
    case_q0 = Agg()
    by_q: Dict[int, Agg] = {qi: Agg() for qi in q_idxs}

    for ridx, rec in enumerate(rows):
        answers = per_case_answers.get(ridx, [])
        if not answers:
            continue
        ok_any = any(_check_answer_new(rec, a) for a in answers)
        case_any.add(ok_any)

        # Per-q accuracies
        qs = rec.get("questions") or []
        if isinstance(qs, list):
            for qi in q_idxs:
                if qi < len(qs):
                    pred = pred_by_key.get((ridx, qi), "")
                    by_q[qi].add(_check_answer_new(rec, pred))

        # Canonical-only (q0 if requested)
        if 0 in q_idxs:
            pred0 = pred_by_key.get((ridx, 0), "")
            case_q0.add(_check_answer_new(rec, pred0))

    report = {
        "data_path": args.data_path,
        "model_path": args.model_path,
        "max_records": int(args.max_records),
        "q_idxs": q_idxs,
        "case_acc_any": case_any.to_dict(),
        "case_acc_q0": case_q0.to_dict(),
        "by_q": {str(qi): by_q[qi].to_dict() for qi in q_idxs},
    }

    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
