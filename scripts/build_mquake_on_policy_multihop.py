from __future__ import annotations

import argparse
import json
import os
import string
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str, help="Converted MQuAKE-Remastered JSON.")
    parser.add_argument("--output_path", required=True, type=str, help="Output on-policy JSONL.")
    parser.add_argument("--train_q_idx", type=str, default="0", help="Comma-separated indices in record['questions'] to include.")
    parser.add_argument("--rollout_model_path", type=str, default="", help="If set, generate rejected=y_hat and conflict_score.")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument("--cuda", type=str, default="")
    args = parser.parse_args()

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    idxs = [int(x.strip()) for x in str(args.train_q_idx).split(",") if x.strip()]
    rows = load_json(args.data_path)

    prompts: List[str] = []
    metas: List[Tuple[Dict[str, Any], int, str]] = []
    for rec in rows:
        qs = rec.get("questions") or []
        if not isinstance(qs, list):
            continue
        for qi in idxs:
            if 0 <= qi < len(qs):
                q = _ws(qs[qi])
                if not q:
                    continue
                prompts.append(q)
                metas.append((rec, qi, q))

    preds: List[str] = []
    if args.rollout_model_path:
        preds = run_vllm_generate(args.rollout_model_path, prompts, tp_size=int(args.tp_size), max_tokens=int(args.max_tokens))
    else:
        preds = ["" for _ in prompts]

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as handle:
        for (rec, qi, q), y_hat in zip(metas, preds):
            chosen = _ws(rec.get("answer_new", ""))
            chosen_alias = rec.get("answer_new_alias") or []
            old = _ws(rec.get("answer_old", ""))
            old_alias = rec.get("answer_old_alias") or []

            # Conflict = mismatch on (new answer or aliases)
            is_match = _matches_any(y_hat, [chosen, *map(str, chosen_alias)])
            old_in_pred = _matches_any(y_hat, [old, *map(str, old_alias)]) if old else False
            mismatch = float(not is_match)
            # Optional: penalize old-answer inclusion (helps old-resurgence analysis)
            score = float(max(mismatch, float(old_in_pred)))

            row = {
                "edit_id": rec.get("case_id"),
                "prompt": q,
                "chosen": chosen,
                "rejected": _ws(y_hat),
                "conflict_score": score,
                "mismatch_rate": mismatch,
                "old_contains": float(old_in_pred),
                "meta": {"q_idx": int(qi)},
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"data_path": args.data_path, "output_path": str(out_path), "n_rows": len(metas)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

