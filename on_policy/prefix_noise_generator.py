from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional

from on_policy.rollout import rollout


_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", (text or "").strip().lower())


def _similar(a: str, b: str) -> float:
    a_n = _norm(a)
    b_n = _norm(b)
    if not a_n or not b_n:
        return 0.0
    return float(SequenceMatcher(a=a_n, b=b_n).ratio())


def _clean(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    # Keep first non-empty line.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    text = lines[0] if lines else text
    text = text.strip(" \t\"'")
    # Remove common prefixes.
    for prefix in ("output:", "rewrite:", "rephrase:", "paraphrase:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :].strip()
    return text


def _tail(prompt: str, k: int) -> str:
    toks = [t for t in (prompt or "").strip().split() if t]
    if not toks:
        return ""
    k = max(1, min(int(k), len(toks)))
    return " ".join(toks[-k:])


def generate_prefix_noise_tail_anchored_triggers(
    model,
    tokenizer,
    prompts: Iterable[str],
    *,
    per_prompt: int,
    tail_words: int = 2,
    gen_cfg: Optional[Dict] = None,
    seed: int = 0,
    similarity_threshold: float = 0.92,
    filter_against: Optional[Dict[str, List[str]]] = None,
) -> List[List[str]]:
    """
    Generate CounterFact-style rephrases: an unrelated prefix sentence + a restated prompt fragment,
    with a constraint that the output must END WITH the last `tail_words` from the original prompt.

    This directly targets the `distractor_prefix_trunc` family without using dataset-provided `rephrase_prompt`.
    """
    prompt_list = [p.strip() for p in prompts]
    if per_prompt <= 0 or not prompt_list:
        return [[] for _ in prompt_list]

    filter_against = filter_against or {}
    tail_map = {p: _tail(p, int(tail_words)) for p in prompt_list}

    inst = (
        "You will rewrite a short factual prompt fragment.\n"
        "1) Add ONE unrelated sentence as distracting context.\n"
        "2) Then rewrite the fragment to keep the same meaning.\n"
        "3) Do NOT answer.\n"
        "4) Your final text MUST end with: \"{tail}\" (exactly).\n"
        "Output ONLY the rewritten text.\n"
        "Fragment: {q}\n"
        "Output:"
    )

    expanded: List[str] = []
    backref: List[int] = []
    for i, p in enumerate(prompt_list):
        tail = tail_map.get(p, "")
        if not tail:
            continue
        for _ in range(int(per_prompt)):
            expanded.append(inst.format(q=p, tail=tail))
            backref.append(i)

    raw = rollout(model=model, tokenizer=tokenizer, prompts=expanded, gen_cfg=gen_cfg or {}, seed=seed)
    grouped: List[List[str]] = [[] for _ in prompt_list]
    for out, src_idx in zip(raw, backref):
        base = prompt_list[src_idx]
        tail = tail_map.get(base, "")
        cand = _clean(out)
        if not cand or not tail:
            continue
        if not _norm(cand).endswith(_norm(tail)):
            continue
        if _similar(cand, base) >= float(similarity_threshold):
            continue
        protected = filter_against.get(base, [])
        if any(_similar(cand, p) >= float(similarity_threshold) for p in protected if p):
            continue
        grouped[src_idx].append(cand)

    # Dedup within each prompt.
    final: List[List[str]] = []
    for group in grouped:
        seen = set()
        out = []
        for cand in group:
            key = _norm(cand)
            if key and key not in seen:
                out.append(cand)
                seen.add(key)
        final.append(out)
    return final

