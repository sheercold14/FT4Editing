from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _ws(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _sim(a: str, b: str) -> float:
    a = _ws(a).lower()
    b = _ws(b).lower()
    if not a or not b:
        return 0.0
    return SequenceMatcher(a=a, b=b).ratio()


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@dataclass
class Hit:
    sim: float
    edit_id: str
    prompt: str
    rephrase: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ched_path", required=True, type=str, help="Converted CHED json (FT4Editing-style).")
    parser.add_argument("--on_policy_path", required=True, type=str, help="On-policy JSONL produced by stage-2 rebuild.")
    parser.add_argument("--near_thresholds", type=str, default="0.92,0.95")
    parser.add_argument("--global_sample_prompts", type=int, default=300)
    parser.add_argument("--global_sample_rephrases", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report_path", type=str, default="")
    args = parser.parse_args()

    thresholds = [float(x.strip()) for x in str(args.near_thresholds).split(",") if x.strip()]
    thresholds = sorted(set(thresholds))

    ched = load_json(args.ched_path)
    on_rows = load_jsonl(args.on_policy_path)

    by_id: Dict[str, Dict[str, Any]] = {str(r.get("case_id")): r for r in ched}
    all_rephrases: List[str] = []
    for r in ched:
        rps = r.get("rephrased_prompt") or []
        if isinstance(rps, list):
            for rp in rps:
                rp = _ws(rp)
                if rp:
                    all_rephrases.append(rp)
    reph_set = {rp.lower() for rp in all_rephrases}

    exact_global = 0
    exact_same_edit = 0
    near_counts_same_edit = {t: 0 for t in thresholds}
    max_hit = Hit(sim=0.0, edit_id="", prompt="", rephrase="")
    missing_edit_id = 0

    # Per-edit leakage check: compare on-policy prompts against the same edit's rephrases.
    for row in on_rows:
        p = _ws(row.get("prompt", ""))
        if not p:
            continue
        if p.lower() in reph_set:
            exact_global += 1

        edit_id = str(row.get("edit_id", ""))
        rec = by_id.get(edit_id)
        if rec is None:
            missing_edit_id += 1
            continue
        rps = rec.get("rephrased_prompt") or []
        if not isinstance(rps, list):
            continue
        rps = [_ws(x) for x in rps if _ws(x)]
        if any(p.lower() == rp.lower() for rp in rps):
            exact_same_edit += 1
        best = max((( _sim(p, rp), rp) for rp in rps), default=(0.0, ""))
        best_sim, best_rp = best
        if best_sim > max_hit.sim:
            max_hit = Hit(sim=float(best_sim), edit_id=edit_id, prompt=p, rephrase=best_rp)
        for t in thresholds:
            if best_sim >= t:
                near_counts_same_edit[t] += 1

    # Conservative global approximate check: random subset of prompts x random subset of rephrases.
    rng = random.Random(int(args.seed))
    sample_prompts = rng.sample(on_rows, k=min(int(args.global_sample_prompts), len(on_rows)))
    sample_rephrases = rng.sample(all_rephrases, k=min(int(args.global_sample_rephrases), len(all_rephrases)))
    global_max_sim = 0.0
    global_near_counts = {t: 0 for t in thresholds}
    for row in sample_prompts:
        p = _ws(row.get("prompt", ""))
        if not p:
            continue
        ms = max(_sim(p, rp) for rp in sample_rephrases) if sample_rephrases else 0.0
        global_max_sim = max(global_max_sim, ms)
        for t in thresholds:
            if ms >= t:
                global_near_counts[t] += 1

    report = {
        "ched_path": args.ched_path,
        "on_policy_path": args.on_policy_path,
        "n_ched_records": len(ched),
        "n_on_policy_rows": len(on_rows),
        "n_rephrased_prompts": len(all_rephrases),
        "exact_match_global_rows": exact_global,
        "exact_match_same_edit_rows": exact_same_edit,
        "near_same_edit_rows": {str(t): near_counts_same_edit[t] for t in thresholds},
        "max_similarity_same_edit": {
            "sim": round(max_hit.sim, 4),
            "edit_id": max_hit.edit_id,
            "prompt": max_hit.prompt,
            "rephrase": max_hit.rephrase,
        },
        "missing_edit_id_rows": missing_edit_id,
        "global_nearcheck": {
            "sample_prompts": len(sample_prompts),
            "sample_rephrases": len(sample_rephrases),
            "near_hits_in_prompt_sample": {str(t): global_near_counts[t] for t in thresholds},
            "max_similarity_in_sample": round(global_max_sim, 4),
        },
    }

    if args.report_path:
        Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()

