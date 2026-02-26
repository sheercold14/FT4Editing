from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from data.datasets import load_records


def _norm(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def trigger_error_rate(records: List[Dict]) -> float:
    if not records:
        return 0.0
    mismatches = 0
    total = 0
    for record in records:
        chosen = record.get("chosen") or record.get("target_new", "")
        rejected = record.get("rejected", "")
        if chosen and _norm(chosen) != _norm(rejected):
            mismatches += 1
        total += 1
    return mismatches / max(total, 1)


def conflict_mass(records: List[Dict]) -> float:
    if not records:
        return 0.0
    values = [max(0.0, 1.0 - float(record.get("margin", 0.0))) for record in records]
    return sum(values) / len(values)


def eval_on_policy(path: str) -> Dict[str, float]:
    rows = load_records(path)
    metrics = {
        "num_samples": len(rows),
        "trigger_error_rate": trigger_error_rate(rows),
        "conflict_mass": conflict_mass(rows),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--on_policy_path", required=True, type=str)
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()

    metrics = eval_on_policy(args.on_policy_path)
    print(json.dumps(metrics, indent=2))
    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
