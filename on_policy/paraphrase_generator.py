from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional

from on_policy.rollout import rollout


_WS = re.compile(r"\\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", (text or "").strip().lower())


def _similar(a: str, b: str) -> float:
    a_n = _norm(a)
    b_n = _norm(b)
    if not a_n or not b_n:
        return 0.0
    return float(SequenceMatcher(a=a_n, b=b_n).ratio())


def _clean_paraphrase(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    # Keep the first non-empty line; many models emit extra commentary.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    text = lines[0] if lines else text
    # Remove common prefixes
    for prefix in ("paraphrase:", "rewrite:", "rephrased:", "output:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :].strip()
    text = text.strip(" \t\"'")
    return text


def generate_paraphrase_triggers(
    model,
    tokenizer,
    prompts: Iterable[str],
    *,
    per_prompt: int,
    gen_cfg: Optional[Dict] = None,
    seed: int = 0,
    similarity_threshold: float = 0.92,
    filter_against: Optional[Dict[str, List[str]]] = None,
    forbid_substrings: Optional[List[str]] = None,
) -> List[List[str]]:
    """
    Generate paraphrased prompt variants without using dataset-provided rephrases.

    filter_against: map original_prompt -> list of "protected" strings (e.g., eval rephrase prompts).
    similarity_threshold: drop paraphrases too similar to any protected string.
    forbid_substrings: drop paraphrases containing any forbidden substring (e.g., target answer).
    """
    prompt_list = [p.strip() for p in prompts]
    if per_prompt <= 0 or not prompt_list:
        return [[] for _ in prompt_list]

    forbid = [s for s in (forbid_substrings or []) if s]
    filter_against = filter_against or {}

    inst = (
        "Paraphrase the following question. Keep the meaning identical. "
        "Do NOT answer the question. Output ONLY the rewritten question.\n"
        "Question: {q}\nParaphrase:"
    )

    # Expand prompts to per_prompt replicas for sampling diversity.
    expanded = []
    backref = []
    for i, p in enumerate(prompt_list):
        for j in range(per_prompt):
            expanded.append(inst.format(q=p))
            backref.append(i)

    raw = rollout(model=model, tokenizer=tokenizer, prompts=expanded, gen_cfg=gen_cfg or {}, seed=seed)
    grouped: List[List[str]] = [[] for _ in prompt_list]
    for out, src_idx in zip(raw, backref):
        candidate = _clean_paraphrase(out)
        if not candidate:
            continue
        # Heuristic filters
        if _similar(candidate, prompt_list[src_idx]) >= similarity_threshold:
            continue
        protected = filter_against.get(prompt_list[src_idx], [])
        if any(_similar(candidate, p) >= similarity_threshold for p in protected if p):
            continue
        if any(sub.lower() in candidate.lower() for sub in forbid):
            continue
        grouped[src_idx].append(candidate)

    # Dedup within each group (case-insensitive)
    final: List[List[str]] = []
    for i, group in enumerate(grouped):
        seen = set()
        out = []
        for cand in group:
            key = _norm(cand)
            if key and key not in seen:
                out.append(cand)
                seen.add(key)
        final.append(out)
    return final

