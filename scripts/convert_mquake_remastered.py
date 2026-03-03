from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


def _guess_repo_root() -> Path:
    here = Path(__file__).resolve()
    if ".worktrees" in set(here.parts):
        p = here
        while p.name != ".worktrees" and p.parent != p:
            p = p.parent
        if p.name == ".worktrees":
            return p.parent
    return here.parents[1]


def _ws(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _first(lst: Any) -> Any:
    if isinstance(lst, list) and lst:
        return lst[0]
    return None


def _find_edit_cloze(
    row: Dict[str, Any],
    *,
    edit_question: str,
    subject: str,
    prompt_tmpl: str,
) -> str:
    """
    Choose the cloze string corresponding to the factual edit.

    MQuAKE-Remastered rows contain multiple single-hop facts (a reasoning chain). The edit
    corresponds to `requested_rewrite[0]` (subject + relation + new object). We try to align
    the edit to a hop by exact question match, then by template match, and finally fall back
    to the first hop.
    """

    def norm(s: Any) -> str:
        return _ws(str(s or ""))

    hops_new = row.get("new_single_hops") if isinstance(row.get("new_single_hops"), list) else []
    hops_old = row.get("single_hops") if isinstance(row.get("single_hops"), list) else []
    hops = [h for h in (hops_new or hops_old) if isinstance(h, dict)]

    # 1) Exact question match (most reliable).
    if edit_question:
        for h in hops:
            if norm(h.get("question")) == edit_question:
                return norm(h.get("cloze"))

    # 2) Template match: replace subject with {} and compare against requested rewrite template.
    if subject and prompt_tmpl and "{}" in prompt_tmpl:
        for h in hops:
            cloze = norm(h.get("cloze"))
            if not cloze or subject not in cloze:
                continue
            tmpl = cloze.replace(subject, "{}", 1)
            if tmpl == prompt_tmpl:
                return cloze

    # 3) Fallback: first hop cloze if present.
    for h in hops:
        cloze = norm(h.get("cloze"))
        if cloze:
            return cloze

    return ""


def convert_row(row: Dict[str, Any], *, prompt_mode: str) -> Optional[Dict[str, Any]]:
    rr0 = _first(row.get("requested_rewrite"))
    if not isinstance(rr0, dict):
        return None

    subject = _ws(rr0.get("subject", ""))
    prompt_tmpl = _ws(rr0.get("prompt", ""))
    edit_question = _ws(rr0.get("question", ""))
    target_old = _ws(rr0.get("target_true_str", ""))
    target_new = _ws(rr0.get("target_new_str", ""))

    # Canonical single-hop prompts (aligned to the edit)
    cloze = _find_edit_cloze(row, edit_question=edit_question, subject=subject, prompt_tmpl=prompt_tmpl)

    if prompt_mode == "question":
        prompt = edit_question
    elif prompt_mode == "cloze":
        prompt = cloze
    elif prompt_mode == "template":
        if subject and "{}" in prompt_tmpl:
            try:
                prompt = prompt_tmpl.format(subject)
            except Exception:
                prompt = prompt_tmpl.replace("{}", subject)
        else:
            prompt = prompt_tmpl
    else:
        raise ValueError(f"Unknown prompt_mode={prompt_mode!r}")

    prompt = _ws(prompt)
    if not (prompt and target_new):
        return None

    out: Dict[str, Any] = {
        "case_id": row.get("case_id"),
        "subject": subject,
        # Editing supervision (single-hop)
        "prompt": prompt,
        "edit_prompt_template": prompt_tmpl,
        "edit_question": edit_question,
        "edit_cloze": cloze,
        "target_old": target_old,
        "target_new": target_new,
        "relation_id": rr0.get("relation_id"),
        # Multi-hop evaluation prompts/answers
        "questions": [_ws(q) for q in (row.get("questions") or []) if _ws(q)],
        "answer_old": _ws(row.get("answer", "")),
        "answer_new": _ws(row.get("new_answer", "")),
        "answer_old_alias": row.get("answer_alias") or [],
        "answer_new_alias": row.get("new_answer_alias") or [],
        # Keep single-hop variants (optional eval/trigger use later)
        "single_hops": row.get("single_hops") or [],
        "new_single_hops": row.get("new_single_hops") or [],
        # Keep triples for analysis
        "orig_triples_labeled": row.get("orig_triples_labeled") or [],
        "new_triples_labeled": row.get("new_triples_labeled") or [],
        "edit_triples": row.get("edit_triples") or [],
        # Dataset metadata (the dataset also stores a dict field named 'split')
        "mquake_split_meta": row.get("split"),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", type=str, default="henryzhongsc/MQuAKE-Remastered")
    parser.add_argument("--hf_split", type=str, default="CF3k", help="CF3k | CF9k | CF6334 | T")
    parser.add_argument("--prompt_mode", type=str, default="question", help="question | cloze | template")
    parser.add_argument("--output_path", type=str, default="data/mquake_remastered/mquake_remastered_cf3k.json")
    parser.add_argument("--max_records", type=int, default=0, help="0 = all")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = _guess_repo_root()
    out_path = (repo_root / args.output_path).resolve() if not Path(args.output_path).is_absolute() else Path(args.output_path)

    from datasets import load_dataset

    ds = load_dataset(str(args.hf_repo), split=str(args.hf_split))
    idxs = list(range(len(ds)))
    rng = random.Random(int(args.seed))
    rng.shuffle(idxs)
    if int(args.max_records) > 0:
        idxs = idxs[: int(args.max_records)]

    rows: List[Dict[str, Any]] = []
    for i in idxs:
        out = convert_row(ds[int(i)], prompt_mode=str(args.prompt_mode))
        if out is not None:
            rows.append(out)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"hf_repo": args.hf_repo, "hf_split": args.hf_split, "n": len(rows), "output": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
