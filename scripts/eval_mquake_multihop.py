from __future__ import annotations

import argparse
import json
import os
import string
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ws(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _norm(text: str) -> str:
    s = _ws(text).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    for art in (" a ", " an ", " the "):
        s = s.replace(art, " ")
    return " ".join(s.split())


def _matches_any(pred: str, answers: Iterable[str]) -> bool:
    p = _norm(pred)
    if not p:
        return False
    p_toks = p.split()
    for ans in answers:
        a = _norm(ans)
        if not a:
            continue
        a_toks = a.split()
        if len(a_toks) > len(p_toks):
            continue
        if p_toks == a_toks:
            return True
        for i in range(len(p_toks) - len(a_toks) + 1):
            if p_toks[i : i + len(a_toks)] == a_toks:
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
    sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens, stop=["\n", ".", tokenizer.eos_token])
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]


@dataclass
class Agg:
    n: int = 0
    ok: int = 0
    old_contains: int = 0

    def add(self, success: bool, old_contains: bool) -> None:
        self.n += 1
        self.ok += int(bool(success))
        self.old_contains += int(bool(old_contains))

    def to_dict(self) -> Dict[str, float]:
        return {
            "n": self.n,
            "success": self.ok / max(1, self.n),
            "old_contains": self.old_contains / max(1, self.n),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str, help="Converted MQuAKE-Remastered JSON.")
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=24)
    parser.add_argument("--max_records", type=int, default=0, help="0 = all")
    parser.add_argument("--eval_q_idx", type=str, default="", help="Optional comma-separated question indices to eval.")
    parser.add_argument("--cuda", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    rows = load_json(args.data_path)
    if int(args.max_records) > 0:
        rows = rows[: int(args.max_records)]

    eval_q_idxs = None
    if str(args.eval_q_idx).strip():
        eval_q_idxs = [int(x.strip()) for x in str(args.eval_q_idx).split(",") if x.strip()]

    # Build prompt list: single-hop edit prompt + multi-hop questions.
    prompt_rows: List[Tuple[str, str, List[str], str, List[str]]] = []
    # tuple: (kind, prompt, new_answers, old_answer, old_alias)
    for rec in rows:
        # single-hop edit check
        edit_prompt = _ws(rec.get("prompt", ""))
        tgt_new = _ws(rec.get("target_new", ""))
        tgt_old = _ws(rec.get("target_old", ""))
        if edit_prompt and tgt_new:
            prompt_rows.append(("single", edit_prompt, [tgt_new], tgt_old, []))

        qs = rec.get("questions") or []
        if not isinstance(qs, list):
            continue
        for qi, q in enumerate(qs):
            if eval_q_idxs is not None and qi not in eval_q_idxs:
                continue
            q = _ws(q)
            if not q:
                continue
            new_ans = _ws(rec.get("answer_new", ""))
            new_alias = rec.get("answer_new_alias") or []
            old_ans = _ws(rec.get("answer_old", ""))
            old_alias = rec.get("answer_old_alias") or []
            prompt_rows.append((f"mh_q{qi}", q, [new_ans, *map(str, new_alias)], old_ans, list(map(str, old_alias))))

    prompts = [p for _, p, *_ in prompt_rows]
    preds = run_vllm_generate(args.model_path, prompts, tp_size=int(args.tp_size), max_tokens=int(args.max_tokens))

    overall = Agg()
    by_kind: Dict[str, Agg] = defaultdict(Agg)
    for (kind, _prompt, new_answers, old_ans, old_alias), pred in zip(prompt_rows, preds):
        ok = _matches_any(pred, new_answers)
        old_in = _matches_any(pred, [old_ans, *old_alias]) if old_ans else False
        overall.add(ok, old_in)
        by_kind[kind].add(ok, old_in)

    report = {
        "data_path": args.data_path,
        "model_path": args.model_path,
        "max_records": int(args.max_records),
        "eval_q_idx": eval_q_idxs,
        "overall": overall.to_dict(),
        "by_kind": {k: by_kind[k].to_dict() for k in sorted(by_kind.keys())},
    }
    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()

