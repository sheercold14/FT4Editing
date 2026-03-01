from __future__ import annotations

import argparse
import json
import os
import re
import string
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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


def contains_token_seq(pred: str, target: str) -> bool:
    p = _norm(pred).split()
    t = _norm(target).split()
    if not t:
        return False
    if len(t) > len(p):
        return False
    if p == t:
        return True
    for i in range(len(p) - len(t) + 1):
        if p[i : i + len(t)] == t:
            return True
    return False


def load_json(path: str) -> List[Dict]:
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


def build_family_prompts(rec: Dict, *, per_family_k: int, ctx_offset: int = 0) -> List[Tuple[str, str]]:
    """
    Returns (family, prompt) pairs.
    CHED expects context prefixes in families like SBJ / OBJ_OLD / OBJ_NEW and their hop variants.
    """
    base = _ws(rec.get("prompt", ""))
    out: List[Tuple[str, str]] = []
    if base:
        out.append(("prompt", base))

    # Rephrases are part of benchmark generalization; keep as a separate family.
    for rp in (rec.get("rephrased_prompt") or [])[:2]:
        rp = _ws(rp)
        if rp:
            out.append(("rephrase", rp))

    def add_ctx(family: str, ctx_list_key: str) -> None:
        ctxs = rec.get(ctx_list_key) or []
        added = 0
        start = max(0, int(ctx_offset))
        for s in ctxs[start:]:
            if added >= int(per_family_k):
                break
            s = _ws(s)
            if not (s and base):
                continue
            out.append((family, f"{s}\n\n{base}"))
            added += 1

    add_ctx("sbj", "sbj_sentence")
    add_ctx("obj_old", "obj_old_sentence")
    add_ctx("obj_new", "obj_new_sentence")
    add_ctx("sbj_hop", "sbj_hop_sentence")
    add_ctx("obj_old_hop", "obj_old_hop_sentence")
    add_ctx("obj_new_hop", "obj_new_hop_sentence")
    return out


@dataclass
class Agg:
    n: int = 0
    ok_sum: float = 0.0  # contains new && not contains old
    new_sum: float = 0.0
    old_sum: float = 0.0

    def add(self, *, ok: bool, new_contains: bool, old_contains: bool) -> None:
        self.n += 1
        self.ok_sum += float(ok)
        self.new_sum += float(new_contains)
        self.old_sum += float(old_contains)

    def to_dict(self) -> Dict:
        return {
            "n": self.n,
            "success": self.ok_sum / max(1, self.n),
            "new_contains": self.new_sum / max(1, self.n),
            "old_contains": self.old_sum / max(1, self.n),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str, help="Converted CHED json (FT4Editing-style).")
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument("--per_family_k", type=int, default=2)
    parser.add_argument(
        "--ctx_offset",
        type=int,
        default=0,
        help="Use context sentences starting from this offset within each family list (for holdout eval).",
    )
    parser.add_argument("--max_records", type=int, default=0, help="0 = all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    rows = load_json(args.data_path)
    if int(args.max_records) > 0:
        import random

        rng = random.Random(int(args.seed))
        idxs = list(range(len(rows)))
        rng.shuffle(idxs)
        rows = [rows[i] for i in idxs[: int(args.max_records)]]

    family_pairs: List[Tuple[str, int, str, str, str]] = []
    for ridx, rec in enumerate(rows):
        target_new = _ws(rec.get("target_new", ""))
        target_old = _ws(rec.get("target_old", ""))
        if not (target_new and rec.get("prompt")):
            continue
        for family, prompt in build_family_prompts(
            rec,
            per_family_k=int(args.per_family_k),
            ctx_offset=int(args.ctx_offset),
        ):
            family_pairs.append((family, ridx, prompt, target_new, target_old))

    prompts = [p[2] for p in family_pairs]
    preds = run_vllm_generate(args.model_path, prompts, tp_size=int(args.tp_size), max_tokens=int(args.max_tokens))

    overall = Agg()
    by_family: Dict[str, Agg] = defaultdict(Agg)
    examples: Dict[str, Dict] = {}
    for (family, ridx, prompt, new, old), pred in zip(family_pairs, preds):
        new_contains = contains_token_seq(pred, new)
        old_contains = contains_token_seq(pred, old) if old else False
        ok = bool(new_contains and (not old_contains))
        overall.add(ok=ok, new_contains=new_contains, old_contains=old_contains)
        by_family[family].add(ok=ok, new_contains=new_contains, old_contains=old_contains)

        # Save one failure example per family for debugging.
        if not ok and family not in examples:
            examples[family] = {
                "record_index": ridx,
                "prompt": prompt[:500],
                "pred": pred[:200],
                "target_new": new,
                "target_old": old,
                "new_contains": new_contains,
                "old_contains": old_contains,
            }

    report = {
        "data_path": args.data_path,
        "model_path": args.model_path,
        "per_family_k": int(args.per_family_k),
        "ctx_offset": int(args.ctx_offset),
        "max_records": int(args.max_records),
        "overall": overall.to_dict(),
        "by_family": {k: by_family[k].to_dict() for k in sorted(by_family.keys())},
        "failure_examples": examples,
    }

    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
