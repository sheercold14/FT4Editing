from __future__ import annotations

from typing import Dict, List


def _pick_prompt(edit_record: Dict) -> str:
    return (
        edit_record.get("prompt")
        or edit_record.get("src")
        or edit_record.get("rephrase_prompt")
        or edit_record.get("rephrase")
        or ""
    ).strip()

def generate_suffix_completions(prompt: str, *, max_variants: int = 2) -> List[str]:
    """
    Generate lightweight suffix-extended prompt variants to better match CounterFact-style rephrase prompts,
    which often restate the relation more explicitly (e.g., "speaks the language", "is located in the continent").

    NOTE: Heuristic and dataset-dependent; should be used only as synthetic triggers (on-policy), not for evaluation.
    """
    p = (prompt or "").strip()
    if not p:
        return []
    p_l = p.lower()

    suffixes: List[str] = []
    def add(s: str) -> None:
        s = s.strip()
        if s and s.lower() != p_l:
            suffixes.append(s)

    # Common CounterFact prompt patterns (fragments).
    if p_l.endswith(" speaks") or p_l.endswith(" speaks the"):
        add(p + " language")
        add(p + " the language")
    if p_l.endswith(" died in") or p_l.endswith(" died at"):
        add(p + " the city of")
        add(p + " the country of")
    if p_l.endswith(" was born in") or p_l.endswith(" born in"):
        add(p + " the city of")
        add(p + " the country of")
    if p_l.endswith(" is in") or p_l.endswith(" is within") or p_l.endswith(" is located in"):
        add(p + " the continent")
        add(p + " the country")
    if p_l.endswith(" originated in") or p_l.endswith(" was founded in") or p_l.endswith(" was created in"):
        add(p + " the city of")
        add(p + " the country of")
    if p_l.endswith(" is written in") or p_l.endswith(" written in"):
        add(p + " language")
        add(p + " the language")
    if p_l.endswith(" plays") or p_l.endswith(" play") or p_l.endswith(" performs on the"):
        add(p + " sport")
        add(p + " instrument")

    # Dedup + cap.
    out: List[str] = []
    seen = set()
    for s in suffixes:
        k = s.lower()
        if k and k not in seen:
            out.append(s)
            seen.add(k)
        if len(out) >= int(max_variants):
            break
    return out


def generate_triggers(
    edit_record: Dict,
    k: int,
    mode: str = "template",
    *,
    include_rephrase: bool = False,
) -> List[str]:
    prompt = _pick_prompt(edit_record)
    subject = edit_record.get("subject", "").strip()
    if not prompt:
        return []
    if mode == "model":
        mode = "template"

    variants = [prompt]
    # NOTE: Dataset-provided rephrases are typically used only for *evaluation*.
    # Including them in training/on-policy triggers can leak eval prompts and inflate "generalization".
    if include_rephrase:
        if edit_record.get("rephrase_prompt"):
            variants.append(edit_record["rephrase_prompt"].strip())
        if edit_record.get("rephrase"):
            variants.append(edit_record["rephrase"].strip())
    if subject:
        variants.extend(
            [
                f"What is the correct answer for: {prompt}",
                f"Answer briefly: {prompt}",
                f"{subject} - {prompt}",
                f"Please update your answer: {prompt}",
            ]
        )
    else:
        variants.extend(
            [
                f"Answer briefly: {prompt}",
                f"Please correct this: {prompt}",
            ]
        )

    deduped = []
    seen = set()
    for item in variants:
        key = item.lower()
        if key and key not in seen:
            deduped.append(item)
            seen.add(key)
        if len(deduped) >= k:
            break
    return deduped
