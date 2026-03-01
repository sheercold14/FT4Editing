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


def _chat_wraps(prompt: str) -> List[str]:
    p = " ".join((prompt or "").strip().split())
    if not p:
        return []
    return [f"User: {p}\nAssistant:", f"### Question:\n{p}\n### Answer:"]


def _ched_context_prompts(edit_record: Dict, base_prompt: str, *, per_family: int = 2) -> List[str]:
    """
    CHED-style context prefixes: SBJ / OBJ_OLD / OBJ_NEW and hop variants.
    We include both supportive and adversarial contexts (e.g., OBJ_OLD) to train context-robust edits.
    """
    keys = [
        "sbj_sentence",
        "obj_old_sentence",
        "obj_new_sentence",
        "sbj_hop_sentence",
        "obj_old_hop_sentence",
        "obj_new_hop_sentence",
    ]
    pools: List[List[str]] = []
    for k in keys:
        ctxs = edit_record.get(k) or []
        if not isinstance(ctxs, list):
            continue
        ctxs = [str(s).strip() for s in ctxs if str(s).strip()]
        if ctxs:
            pools.append([f"{s}\n\n{base_prompt}" for s in ctxs[: int(per_family)]])

    out: List[str] = []
    # Round-robin across pools for coverage.
    idxs = [0] * len(pools)
    turns = 0
    while turns < 256:
        progressed = False
        for pi, pool in enumerate(pools):
            if idxs[pi] < len(pool):
                out.append(pool[idxs[pi]])
                idxs[pi] += 1
                progressed = True
        if not progressed:
            break
        turns += 1
    return out


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

    # CHED prompts sometimes keep "{}" placeholders; make this function robust if conversion was skipped.
    if subject and "{}" in prompt:
        try:
            prompt = prompt.format(subject)
        except Exception:
            prompt = prompt.replace("{}", subject)

    variants = [prompt]
    # NOTE: Dataset-provided rephrases are typically used only for *evaluation*.
    # Including them in training/on-policy triggers can leak eval prompts and inflate "generalization".
    if include_rephrase:
        if edit_record.get("rephrase_prompt"):
            variants.append(edit_record["rephrase_prompt"].strip())
        if edit_record.get("rephrase"):
            variants.append(edit_record["rephrase"].strip())
        if isinstance(edit_record.get("rephrased_prompt"), list):
            for rp in edit_record.get("rephrased_prompt", [])[:2]:
                rp = str(rp).strip()
                if rp:
                    variants.append(rp)

    if mode == "ched":
        variants.extend(_chat_wraps(prompt))
        variants.extend(_ched_context_prompts(edit_record, prompt, per_family=2))
    elif subject:
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
