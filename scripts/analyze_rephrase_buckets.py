from __future__ import annotations

import argparse
import json
import random
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _ws(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _norm_for_match(text: str) -> str:
    text = _ws(text).lower()
    return text


def _norm_for_em(text: str) -> str:
    def remove_articles(t: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def remove_punc(t: str) -> str:
        return "".join(ch for ch in t if ch not in set(string.punctuation))

    return " ".join(remove_articles(remove_punc(_ws(text).lower())).split())


def exact_match(pred: str, target: str) -> float:
    return float(_norm_for_em(pred) == _norm_for_em(target))


def contains_match(pred: str, target: str) -> float:
    pred_n = _norm_for_em(pred)
    tgt_n = _norm_for_em(target)
    if not tgt_n:
        return 0.0
    return float(tgt_n == pred_n or tgt_n in pred_n)


def _tokenize(text: str) -> List[str]:
    return [t for t in _norm_for_match(text).split(" ") if t]


def suffix_token_overlap(prompt: str, rephrase: str, *, min_k: int = 2) -> int:
    pt = _tokenize(prompt)
    rt = _tokenize(rephrase)
    if not pt or not rt:
        return 0
    max_k = min(len(pt), len(rt))
    for k in range(max_k, min_k - 1, -1):
        if pt[-k:] == rt[-k:]:
            return k
    return 0


def seq_ratio(a: str, b: str) -> float:
    a_n = _norm_for_match(a)
    b_n = _norm_for_match(b)
    if not a_n or not b_n:
        return 0.0
    return float(SequenceMatcher(a=a_n, b=b_n).ratio())


def _has_prefix_delimiter(text: str) -> bool:
    # Rough signal for "prefix noise" style prompts: multiple clauses/lines before the actual question.
    t = str(text or "")
    return ("\n" in t) or ("." in t) or (":" in t)


def _bigram_in(prompt: str, rephrase: str) -> bool:
    pt = _tokenize(prompt)
    rt = _tokenize(rephrase)
    if len(pt) >= 2:
        a, b = pt[0], pt[1]
        for i in range(len(rt) - 1):
            if rt[i] == a and rt[i + 1] == b:
                return True
        return False
    if len(pt) == 1:
        return pt[0] in rt
    return False


@dataclass(frozen=True)
class BucketResult:
    bucket: str
    ratio: float
    suffix_k: int
    prompt_len: int
    rephrase_len: int


def bucket_rephrase(prompt: str, rephrase: str) -> BucketResult:
    prompt = _ws(prompt)
    rephrase = _ws(rephrase)
    ratio = seq_ratio(prompt, rephrase)
    suffix_k = suffix_token_overlap(prompt, rephrase)

    pt = _tokenize(prompt)
    rt = _tokenize(rephrase)
    prompt_len = len(pt)
    rephrase_len = len(rt)

    prompt_n = _norm_for_match(prompt)
    rephrase_n = _norm_for_match(rephrase)

    # Heuristics:
    # - CounterFact-style: unrelated prefix then truncated prompt tail.
    # - WikiBigEdit/ZsRE-style: semantic paraphrase (high ratio, low suffix overlap).
    if not prompt_n or not rephrase_n:
        bucket = "missing"
    elif rephrase_n == prompt_n:
        bucket = "identical"
    elif ratio >= 0.78:
        bucket = "semantic_paraphrase_high"
    elif ratio >= 0.60:
        bucket = "semantic_paraphrase_mid"
    elif prompt_n in rephrase_n:
        # rephrase contains full prompt text (usually prefix-noise).
        bucket = "distractor_prefix_full"
    elif _has_prefix_delimiter(rephrase) and _bigram_in(prompt, rephrase):
        # Prefix-noise + semantic restatement, but not a clean paraphrase string match.
        # This is common in CounterFact rephrase prompts.
        bucket = "prefix_noise_semantic"
    elif _has_prefix_delimiter(rephrase) and suffix_k >= 2 and suffix_k < max(2, prompt_len) and ratio < 0.75:
        bucket = "distractor_prefix_trunc"
    else:
        bucket = "other"

    return BucketResult(
        bucket=bucket,
        ratio=ratio,
        suffix_k=suffix_k,
        prompt_len=prompt_len,
        rephrase_len=rephrase_len,
    )


def load_records(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def pick(record: Dict, keys: Tuple[str, ...], default: str = "") -> str:
    for k in keys:
        val = record.get(k)
        if val is None:
            continue
        val = str(val).strip()
        if val:
            return val
    return default


def run_generation_vllm(
    model_path: str,
    prompts: List[str],
    *,
    tp_size: int,
    max_tokens: int,
):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=0, help="0=all")
    parser.add_argument("--examples_per_bucket", type=int, default=3)
    parser.add_argument("--model_path", type=str, default="", help="Optional: compute per-bucket EM using vLLM")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=32)
    args = parser.parse_args()

    random.seed(args.seed)
    records = load_records(args.data_path)
    if args.num_samples and len(records) > args.num_samples:
        records = random.sample(records, args.num_samples)

    bucket_counts = Counter()
    bucket_examples: Dict[str, List[Dict]] = defaultdict(list)
    bucket_stats: Dict[str, List[BucketResult]] = defaultdict(list)

    parsed = []
    for rec in records:
        prompt = pick(rec, ("prompt", "src"))
        rephrase = pick(rec, ("rephrase_prompt", "rephrase"))
        target = pick(rec, ("target_new", "alt"))
        res = bucket_rephrase(prompt, rephrase)
        bucket_counts[res.bucket] += 1
        bucket_stats[res.bucket].append(res)
        if len(bucket_examples[res.bucket]) < args.examples_per_bucket:
            bucket_examples[res.bucket].append(
                {
                    "prompt": prompt,
                    "rephrase_prompt": rephrase,
                    "target_new": target,
                    "ratio": round(res.ratio, 4),
                    "suffix_k": res.suffix_k,
                }
            )
        parsed.append((prompt, rephrase, target, res.bucket))

    report: Dict[str, object] = {
        "data_path": args.data_path,
        "num_records": len(records),
        "bucket_counts": dict(bucket_counts),
        "bucket_examples": dict(bucket_examples),
    }

    # Basic aggregate stats per bucket.
    agg = {}
    for b, rows in bucket_stats.items():
        if not rows:
            continue
        agg[b] = {
            "avg_ratio": sum(r.ratio for r in rows) / len(rows),
            "avg_suffix_k": sum(r.suffix_k for r in rows) / len(rows),
            "avg_prompt_len": sum(r.prompt_len for r in rows) / len(rows),
            "avg_rephrase_len": sum(r.rephrase_len for r in rows) / len(rows),
        }
    report["bucket_agg"] = agg

    if args.model_path:
        src_prompts = [p for p, _, _, _ in parsed]
        re_prompts = [r for _, r, _, _ in parsed]
        targets = [t for _, _, t, _ in parsed]
        buckets = [b for _, _, _, b in parsed]

        src_preds = run_generation_vllm(args.model_path, src_prompts, tp_size=args.tp_size, max_tokens=args.max_tokens)
        re_preds = run_generation_vllm(args.model_path, re_prompts, tp_size=args.tp_size, max_tokens=args.max_tokens)

        overall = {
            "reliability_src_em": sum(exact_match(p, t) for p, t in zip(src_preds, targets)) / max(1, len(targets)),
            "generalization_rephrase_em": sum(exact_match(p, t) for p, t in zip(re_preds, targets)) / max(1, len(targets)),
        }

        by_bucket = {}
        for b in bucket_counts.keys():
            idxs = [i for i, bb in enumerate(buckets) if bb == b]
            if not idxs:
                continue
            re_em = sum(exact_match(re_preds[i], targets[i]) for i in idxs) / len(idxs)
            re_contains = sum(contains_match(re_preds[i], targets[i]) for i in idxs) / len(idxs)
            by_bucket[b] = {"n": len(idxs), "rephrase_em": re_em, "rephrase_contains": re_contains}

        report["model_eval"] = {"model_path": args.model_path, "overall": overall, "by_bucket": by_bucket}

    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
