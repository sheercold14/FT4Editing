from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


def _guess_repo_root() -> Path:
    """
    Works both when executed inside a worktree (e.g., .../.worktrees/ched/scripts)
    and when copied elsewhere.
    """
    here = Path(__file__).resolve()
    parts = set(here.parts)
    if ".worktrees" in parts:
        # .../FT4Editing/.worktrees/<name>/scripts/...
        p = here
        while p.name != ".worktrees" and p.parent != p:
            p = p.parent
        if p.name == ".worktrees":
            return p.parent
    return here.parents[1]


def _fmt_prompt(prompt_tmpl: str, subject: str) -> str:
    p = str(prompt_tmpl or "").strip()
    s = str(subject or "").strip()
    if not p:
        return ""
    if "{}" in p:
        try:
            return p.format(s).strip()
        except Exception:
            # Fall back to a simple replacement if format() fails for any reason.
            return p.replace("{}", s).strip()
    # Some prompts may omit {} but still need a subject prefix.
    if s and s.lower() not in p.lower():
        return f"{p} {s}".strip()
    return p


def _pick_first(items: Any) -> str:
    if isinstance(items, list) and items:
        return str(items[0]).strip()
    if isinstance(items, str):
        return items.strip()
    return ""


def convert_record(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    subject = str(rec.get("subject") or "").strip()
    prompt = _fmt_prompt(rec.get("prompt") or "", subject)
    target_new = str(rec.get("edited_knowledge") or "").strip()
    target_old = str(rec.get("fact_knowledge") or "").strip()
    if not (prompt and target_new):
        return None

    locality_prompts = rec.get("locality_prompts") or []
    locality_ground_truth = rec.get("locality_ground_truth") or []

    out: Dict[str, Any] = {
        "case_id": rec.get("case_id"),
        "relation_id": rec.get("relation_id"),
        "counterfact_id": rec.get("counterfact_id"),
        "subject": subject,
        "prompt": prompt,
        "target_new": target_new,
        "target_old": target_old,
        # Keep the full lists for CHED-style evaluation / trigger generation.
        "rephrased_prompt": rec.get("rephrased_prompt") or [],
        "sbj_sentence": rec.get("sbj_sentence") or [],
        "obj_old_sentence": rec.get("obj_old_sentence") or [],
        "obj_new_sentence": rec.get("obj_new_sentence") or [],
        "sbj_hop_sentence": rec.get("sbj_hop_sentence") or [],
        "obj_old_hop_sentence": rec.get("obj_old_hop_sentence") or [],
        "obj_new_hop_sentence": rec.get("obj_new_hop_sentence") or [],
        "locality_prompts": locality_prompts,
        "locality_ground_truth": locality_ground_truth,
        # Provide a single locality prompt/gt for compatibility with existing FT4Editing codepaths.
        "locality_prompt": _pick_first(locality_prompts),
        "locality_ground_truth": _pick_first(locality_ground_truth),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="external/CoRE/data/CHED.json",
        help="Path to CHED.json from CoRE repo.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/ched/ched_21k.json",
        help="Output JSON path (FT4Editing-style records).",
    )
    parser.add_argument("--max_records", type=int, default=0, help="0 = all")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = _guess_repo_root()
    input_path = (repo_root / args.input_path).resolve() if not Path(args.input_path).is_absolute() else Path(args.input_path)
    output_path = (repo_root / args.output_path).resolve() if not Path(args.output_path).is_absolute() else Path(args.output_path)

    with open(input_path, "r", encoding="utf-8") as handle:
        raw: List[Dict[str, Any]] = json.load(handle)

    rng = random.Random(int(args.seed))
    idxs = list(range(len(raw)))
    rng.shuffle(idxs)
    if int(args.max_records) > 0:
        idxs = idxs[: int(args.max_records)]

    converted: List[Dict[str, Any]] = []
    for i in idxs:
        out = convert_record(raw[i])
        if out is not None:
            converted.append(out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(converted, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"input": str(input_path), "output": str(output_path), "n": len(converted)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

