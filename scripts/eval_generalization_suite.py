from __future__ import annotations

import argparse
import json
import os
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


def _norm_em(text: str) -> str:
    s = _ws(text).lower()
    # remove punctuation
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    # remove articles
    for art in (" a ", " an ", " the "):
        s = s.replace(art, " ")
    return " ".join(s.split())


def exact_match(pred: str, target: str) -> float:
    return float(_norm_em(pred) == _norm_em(target))


def contains_match(pred: str, target: str) -> float:
    p = _norm_em(pred)
    t = _norm_em(target)
    if not t:
        return 0.0
    return float(t == p or t in p)


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_vllm_generate(model_path: str, prompts: List[str], *, tp_size: int, max_tokens: int):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    llm = LLM(model=model_path, tensor_parallel_size=tp_size, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        stop=[".", "\n", tokenizer.eos_token],
    )
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]


@dataclass
class Agg:
    n: int = 0
    em_sum: float = 0.0
    contains_sum: float = 0.0

    def add(self, em: float, contains: float) -> None:
        self.n += 1
        self.em_sum += float(em)
        self.contains_sum += float(contains)

    def to_dict(self) -> Dict:
        return {
            "n": self.n,
            "em": self.em_sum / max(1, self.n),
            "contains": self.contains_sum / max(1, self.n),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_path", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--cuda", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    rows = load_jsonl(args.suite_path)
    prompts = [_ws(r.get("prompt", "")) for r in rows]
    targets = [_ws(r.get("target_new", "")) for r in rows]
    transforms = [str(r.get("transform", "unknown")) for r in rows]

    preds = run_vllm_generate(args.model_path, prompts, tp_size=int(args.tp_size), max_tokens=int(args.max_tokens))

    overall = Agg()
    by_transform: Dict[str, Agg] = defaultdict(Agg)
    for pred, tgt, tr in zip(preds, targets, transforms):
        em = exact_match(pred, tgt)
        cont = contains_match(pred, tgt)
        overall.add(em, cont)
        by_transform[tr].add(em, cont)

    report = {
        "suite_path": args.suite_path,
        "model_path": args.model_path,
        "overall": overall.to_dict(),
        "by_transform": {k: by_transform[k].to_dict() for k in sorted(by_transform.keys())},
    }

    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
