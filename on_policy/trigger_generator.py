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


def generate_triggers(edit_record: Dict, k: int, mode: str = "template") -> List[str]:
    prompt = _pick_prompt(edit_record)
    subject = edit_record.get("subject", "").strip()
    if not prompt:
        return []
    if mode == "model":
        mode = "template"

    variants = [prompt]
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

