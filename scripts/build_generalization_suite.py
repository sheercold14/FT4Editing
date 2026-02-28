from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.datasets import RecordMapper, load_records


def _ws(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _dedup_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for it in items:
        it = _ws(it)
        if not it:
            continue
        k = it.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


@dataclass(frozen=True)
class SuiteItem:
    edit_id: str
    transform: str
    prompt: str
    target_new: str
    base_prompt: str


def _pick_edit_id(rec: Dict, idx: int) -> str:
    return str(rec.get("case_id", rec.get("id", idx)))


def make_instruction_wraps(prompt: str) -> List[str]:
    p = _ws(prompt)
    return [
        f"Answer briefly: {p}",
        f"Just the answer: {p}",
        f"Please answer: {p}",
        f"{p}\nAnswer:",
    ]


def make_chat_wraps(prompt: str) -> List[str]:
    p = _ws(prompt)
    return [
        f"User: {p}\nAssistant:",
        f"### Question:\n{p}\n### Answer:",
    ]


def make_trunc_variants(prompt: str, *, drop_last_words: int = 1) -> List[str]:
    toks = [t for t in _ws(prompt).split(" ") if t]
    if len(toks) <= max(1, drop_last_words):
        return []
    short = " ".join(toks[: -int(drop_last_words)])
    return [short]


def make_prefix_noise(
    prompt: str,
    *,
    distractors: List[str],
    rng: random.Random,
    with_trunc: bool,
) -> List[str]:
    if not distractors:
        return []
    p = _ws(prompt)
    if not p:
        return []
    prefix = _ws(rng.choice(distractors))
    if not prefix:
        return []
    out = []
    out.append(f"{prefix}. {p}")
    if with_trunc:
        trunc = make_trunc_variants(p, drop_last_words=1)
        if trunc:
            out.append(f"{prefix}. {trunc[0]}")
    return out


def _apply_replacements(text: str, replacements: List[Tuple[str, str]]) -> str:
    out = _ws(text)
    for a, b in replacements:
        if a in out:
            out = out.replace(a, b)
    return _ws(out)


def make_rule_paraphrases(prompt: str, *, subject: str = "") -> List[str]:
    p = _ws(prompt)
    subj = _ws(subject)
    if not p:
        return []

    # Completion-style prompts (CounterFact-like).
    if not p.endswith("?"):
        variants: List[str] = []
        if subj and p.lower().startswith(subj.lower()):
            tail = _ws(p[len(subj) :])
            for repls in (
                [(" is in", " is located in")],
                [(" is in", " can be found in")],
                [(" is in", " is situated in")],
            ):
                cand = _apply_replacements(f"{subj}{tail}", repls)
                if cand and cand.lower() != p.lower():
                    variants.append(cand)
        else:
            variants.append(_apply_replacements(p, [(" is in", " is located in")]))
            variants.append(_apply_replacements(p, [(" is in", " can be found in")]))
        return _dedup_keep_order([v for v in variants if v and v.lower() != p.lower()])

    # Question-style prompts (WikiBigEdit-like).
    variants: List[str] = []
    variants.append(
        _apply_replacements(
            p,
            [
                ("In which traditional geographical division", "In which traditional region"),
                ("traditional geographical division", "traditional region"),
                ("historic county", "historical county"),
                (" is located?", " can be found?"),
                (" is located", " can be found"),
            ],
        )
    )
    variants.append(_apply_replacements(p, [("In which", "Which"), (" is located", " can be found")]))

    # Common question-to-completion shim (keeps meaning but shifts format).
    lower = p.lower()
    if subj and lower.startswith("in which") and lower.endswith("?") and " is " in lower:
        variants.append(f"{subj} is located in")

    return _dedup_keep_order([v for v in variants if v and v.lower() != p.lower()])


def make_context_prefix(
    prompt: str,
    *,
    ctx_prompt: str,
    ctx_answer: str,
    style: str,
) -> str:
    p = _ws(prompt)
    cqp = _ws(ctx_prompt)
    cqa = _ws(ctx_answer)
    if not p or not cqp or not cqa:
        return ""
    if style == "qa":
        return _ws(f"Q: {cqp} A: {cqa}. Q: {p} A:")
    return _ws(f"Context: {cqp} Answer: {cqa}. {p}")


def build_suite(
    records: List[Dict],
    *,
    seed: int,
    suite_version: str,
    per_edit_max: int,
    include_dataset_rephrase: bool,
    add_instruction_wraps: bool,
    add_chat_wraps: bool,
    add_prefix_noise: bool,
    add_prefix_noise_trunc: bool,
    add_trunc: bool,
    add_rule_paraphrase: bool,
    add_context_irrelevant: bool,
    add_context_confusable: bool,
    distractor_pool_size: int,
) -> List[SuiteItem]:
    mapper = RecordMapper()
    rng = random.Random(seed)
    rows: List[SuiteItem] = []

    canon = [mapper.canonicalize(r) for r in records]

    # Build distractor pool from locality_prompt (preferred) or other prompts.
    distractors: List[str] = []
    seen = set()
    for rec in canon:
        d = _ws(rec.get("locality_prompt") or rec.get("prompt") or rec.get("src") or "")
        if not d:
            continue
        k = d.lower()
        if k in seen:
            continue
        seen.add(k)
        distractors.append(d)
        if len(distractors) >= int(distractor_pool_size):
            break

    # Context pools.
    irrelevant_ctx: List[Tuple[str, str]] = []
    confusable_ctx: List[Tuple[str, str]] = []
    for rec in canon:
        lp = _ws(rec.get("locality_prompt", ""))
        lg = _ws(rec.get("locality_ground_truth", ""))
        if lp and lg:
            irrelevant_ctx.append((lp, lg))
        cp = _ws(rec.get("prompt", ""))
        ct = _ws(rec.get("target_new", ""))
        if cp and ct:
            confusable_ctx.append((cp, ct))

    def _sample_other(pool: List[Tuple[str, str]], *, avoid_prompt: str) -> Tuple[str, str]:
        if not pool:
            return ("", "")
        for _ in range(8):
            qp, qa = pool[rng.randrange(0, len(pool))]
            if _ws(qp).lower() != _ws(avoid_prompt).lower():
                return (qp, qa)
        return pool[rng.randrange(0, len(pool))]

    for idx, rec in enumerate(canon):
        base_prompt = _ws(rec.get("prompt", ""))
        target = _ws(rec.get("target_new", ""))
        if not base_prompt or not target:
            continue

        edit_id = _pick_edit_id(rec, idx)
        subject = _ws(rec.get("subject", ""))

        candidates: List[Tuple[str, str]] = [("src", base_prompt)]

        if include_dataset_rephrase:
            rp = _ws(rec.get("rephrase_prompt", ""))
            if rp:
                candidates.append(("rephrase_dataset", rp))

        if add_instruction_wraps:
            for p in make_instruction_wraps(base_prompt):
                candidates.append(("instr_wrap", p))

        if add_chat_wraps:
            for p in make_chat_wraps(base_prompt):
                candidates.append(("chat_wrap", p))

        if add_trunc:
            for p in make_trunc_variants(base_prompt, drop_last_words=1):
                candidates.append(("trunc_drop1", p))

        if add_prefix_noise:
            for p in make_prefix_noise(base_prompt, distractors=distractors, rng=rng, with_trunc=False):
                candidates.append(("prefix_noise", p))

        if add_prefix_noise_trunc:
            for p in make_prefix_noise(base_prompt, distractors=distractors, rng=rng, with_trunc=True):
                candidates.append(("prefix_noise_trunc", p))

        if add_rule_paraphrase:
            for p in make_rule_paraphrases(base_prompt, subject=subject):
                candidates.append(("rule_paraphrase", p))

        if add_context_irrelevant:
            qp, qa = _sample_other(irrelevant_ctx, avoid_prompt=base_prompt)
            pref = make_context_prefix(base_prompt, ctx_prompt=qp, ctx_answer=qa, style="ctx")
            if pref:
                candidates.append(("ctx_irrelevant", pref))

        if add_context_confusable:
            qp, qa = _sample_other(confusable_ctx, avoid_prompt=base_prompt)
            pref = make_context_prefix(base_prompt, ctx_prompt=qp, ctx_answer=qa, style="qa")
            if pref:
                candidates.append(("ctx_confusable", pref))

        # Dedup + cap.
        out_prompts = _dedup_keep_order([p for _, p in candidates])
        out_lc = {q.lower() for q in out_prompts}
        # Preserve transform labels by re-deriving in order.
        emitted = 0
        for name, p in candidates:
            p = _ws(p)
            if not p or p.lower() not in out_lc:
                continue
            rows.append(
                SuiteItem(
                    edit_id=edit_id,
                    transform=name,
                    prompt=p,
                    target_new=target,
                    base_prompt=base_prompt,
                )
            )
            emitted += 1
            if per_edit_max > 0 and emitted >= int(per_edit_max):
                break

    return rows


def dump_jsonl(path: str, rows: List[SuiteItem]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "edit_id": row.edit_id,
                        "transform": row.transform,
                        "prompt": row.prompt,
                        "target_new": row.target_new,
                        "base_prompt": row.base_prompt,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def dump_meta(path: str, meta: Dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--suite_version", type=str, default="v0", choices=("v0", "v1"))
    parser.add_argument("--num_edits", type=int, default=0, help="0=all")
    parser.add_argument("--per_edit_max", type=int, default=0, help="0=no cap")
    parser.add_argument("--include_dataset_rephrase", action="store_true")
    parser.add_argument("--no_instruction_wraps", action="store_true")
    parser.add_argument("--chat_wraps", action="store_true")
    parser.add_argument("--prefix_noise", action="store_true")
    parser.add_argument("--prefix_noise_trunc", action="store_true")
    parser.add_argument("--trunc", action="store_true")
    parser.add_argument("--rule_paraphrase", action="store_true")
    parser.add_argument("--ctx_irrelevant", action="store_true")
    parser.add_argument("--ctx_confusable", action="store_true")
    parser.add_argument("--distractor_pool_size", type=int, default=2048)
    parser.add_argument("--write_meta", action="store_true")
    args = parser.parse_args()

    records = load_records(args.data_path)
    if args.num_edits and len(records) > args.num_edits:
        random.seed(args.seed)
        records = random.sample(records, args.num_edits)

    suite_version = str(args.suite_version)
    chat_wraps = bool(args.chat_wraps)
    rule_paraphrase = bool(args.rule_paraphrase)
    ctx_irrelevant = bool(args.ctx_irrelevant)
    ctx_confusable = bool(args.ctx_confusable)
    if suite_version == "v1":
        chat_wraps = True
        rule_paraphrase = True
        ctx_irrelevant = True
        ctx_confusable = True

    rows = build_suite(
        records,
        seed=int(args.seed),
        suite_version=suite_version,
        per_edit_max=int(args.per_edit_max),
        include_dataset_rephrase=bool(args.include_dataset_rephrase),
        add_instruction_wraps=not bool(args.no_instruction_wraps),
        add_chat_wraps=chat_wraps,
        add_prefix_noise=bool(args.prefix_noise),
        add_prefix_noise_trunc=bool(args.prefix_noise_trunc),
        add_trunc=bool(args.trunc),
        add_rule_paraphrase=rule_paraphrase,
        add_context_irrelevant=ctx_irrelevant,
        add_context_confusable=ctx_confusable,
        distractor_pool_size=int(args.distractor_pool_size),
    )
    dump_jsonl(args.output_path, rows)

    by_transform: Dict[str, int] = {}
    for r in rows:
        by_transform[r.transform] = by_transform.get(r.transform, 0) + 1
    meta = {
        "suite_version": suite_version,
        "data_path": args.data_path,
        "output_path": args.output_path,
        "seed": int(args.seed),
        "num_rows": len(rows),
        "by_transform": {k: by_transform[k] for k in sorted(by_transform.keys())},
        "argv": sys.argv,
        "generated_at_unix": int(time.time()),
    }
    if args.write_meta or suite_version != "v0":
        dump_meta(str(args.output_path) + ".meta.json", meta)
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
