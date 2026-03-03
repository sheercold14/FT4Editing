from __future__ import annotations

import argparse
import json
import os
import re
import string
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _ws(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _strip_answer(ans: str) -> str:
    s = str(ans or "").strip()
    s = re.sub(r"^(answer\\s*:\\s*)", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^(the\\s+answer\\s+is\\s+)", "", s, flags=re.IGNORECASE).strip()
    s = s.splitlines()[0].strip() if s else ""
    s = s.rstrip(" .\n\t\r")
    return s.strip()


def _strip_json(text: str) -> str:
    """
    vLLM outputs sometimes wrap JSON in code-fences or prepend extra tokens.
    Extract the first JSON object substring heuristically.
    """
    s = str(text or "").strip()
    if "```" in s:
        s = s.replace("```json", "```").replace("```JSON", "```")
        parts = [p.strip() for p in s.split("```") if p.strip()]
        # Prefer a fenced payload if present
        for p in parts:
            if p.startswith("{") and p.endswith("}"):
                return p
        s = parts[0] if parts else s
    m = re.search(r"\\{.*\\}", s, flags=re.DOTALL)
    return m.group(0).strip() if m else s


def _norm_rel(text: str) -> str:
    s = _ws(text).lower()
    # keep {} placeholder, strip other punctuation
    keep = set(["{", "}", "_"])
    s2 = []
    for ch in s:
        if ch in keep:
            s2.append(ch)
        elif ch in string.punctuation:
            s2.append(" ")
        else:
            s2.append(ch)
    s = "".join(s2)
    s = " ".join(s.split())
    return s


def _check_answer_new(rec: Dict[str, Any], ans: str) -> bool:
    ans = _strip_answer(ans)
    if not ans:
        return False
    ans_u = ans.upper()
    tgt = _ws(rec.get("answer_new", "")).upper()
    if ans_u == tgt:
        return True
    for a in rec.get("answer_new_alias") or []:
        a_u = str(a).strip().upper()
        if a_u and ans_u == a_u:
            return True
    return False


def _build_edit_bank(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], str]:
    bank: Dict[Tuple[str, str], str] = {}
    for rec in rows:
        subj = _ws(rec.get("subject", ""))
        rel = _ws(rec.get("edit_prompt_template", ""))
        obj = _ws(rec.get("target_new", ""))
        if subj and rel and obj and "{}" in rel:
            bank[(subj.lower(), _norm_rel(rel))] = obj
    return bank


def _best_rel_key(rel_pred: str, rel_keys: List[str], *, thr: float) -> Optional[str]:
    rp = _norm_rel(rel_pred)
    if not (rp and rel_keys):
        return None
    if rp in rel_keys:
        return rp
    best_k = None
    best_s = 0.0
    for k in rel_keys:
        s = SequenceMatcher(a=rp, b=k).ratio()
        if s > best_s:
            best_s, best_k = s, k
    if best_k is not None and best_s >= float(thr):
        return best_k
    return None


def _format_rel_query_prompt(q: str, *, prev_answer: str, subject_hint: str) -> str:
    # Keep prompt short; we want stable JSON output.
    hint = f"Subject hint: {subject_hint}\n" if subject_hint else ""
    prev = f"Previous hop answer: {prev_answer}\n" if prev_answer else ""
    return (
        "You are decomposing a multi-hop question into a single-hop cloze query.\n"
        "Return STRICT JSON with keys: subject, relation_template, done.\n"
        "- subject: an entity string for the next hop (no punctuation at end)\n"
        "- relation_template: a cloze template containing EXACTLY ONE '{}' placeholder.\n"
        "- done: true only if no more hops are needed and you can directly answer.\n"
        f"{hint}{prev}"
        f"Question: {q}\n"
        "JSON:"
    )


def _format_fill_prompt(subject: str, rel_template: str) -> str:
    # Keep it single-line completion-friendly.
    st = rel_template
    if "{}" in st:
        try:
            st = st.format(subject)
        except Exception:
            st = st.replace("{}", subject)
    return f"Complete with the missing object. Reply with only the object.\n{st} ____\nAnswer:"


def _load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _make_vllm(model_path: str, *, tp_size: int):
    from vllm import LLM

    mp = str(Path(model_path).expanduser())
    if Path(mp).exists():
        mp = str(Path(mp).resolve())
    return LLM(
        model=mp,
        tensor_parallel_size=int(tp_size),
        trust_remote_code=True,
        enforce_eager=bool(os.environ.get("GWALK_ENFORCE_EAGER", "")),
    )


def _vllm_generate(llm, prompts: List[str], *, max_tokens: int, stop: List[str]):
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0, max_tokens=int(max_tokens), stop=stop)
    outs = llm.generate(prompts, sp)
    return [o.outputs[0].text for o in outs]


@dataclass
class Agg:
    n: int = 0
    ok: int = 0

    def add(self, ok: bool) -> None:
        self.n += 1
        self.ok += int(bool(ok))

    def to_dict(self) -> Dict[str, float]:
        return {"n": self.n, "acc": self.ok / max(1, self.n)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str, help="LLM used as M in GWalk.")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--max_records", type=int, default=0, help="0 = all")
    parser.add_argument("--q_idxs", type=str, default="0,1,2")
    parser.add_argument("--cuda", type=str, default="")
    parser.add_argument("--max_hops", type=int, default=4)
    parser.add_argument("--use_memory", action="store_true", help="Enable edited-fact bank override.")
    parser.add_argument("--use_subject_hint", action="store_true", help="Provide dataset subject to M as a hint.")
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Disable torch.compile/cudagraph in vLLM for stability on shared GPUs.",
    )
    parser.add_argument(
        "--legacy_vllm",
        action="store_true",
        help="Use vLLM V0 engine (sets VLLM_USE_V1=0) to avoid V1 memory-profiling flakiness.",
    )
    parser.add_argument("--rel_match_threshold", type=float, default=0.88)
    parser.add_argument("--rel_query_tokens", type=int, default=64)
    parser.add_argument("--fill_tokens", type=int, default=8)
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    q_idxs = [int(x.strip()) for x in str(args.q_idxs).split(",") if x.strip()]
    rows = _load_json(args.data_path)
    if int(args.max_records) > 0:
        rows = rows[: int(args.max_records)]

    # Build a memory bank mapping (subject, rel_template) -> edited object.
    edit_bank = _build_edit_bank(rows) if bool(args.use_memory) else {}
    rel_keys_by_subject: Dict[str, List[str]] = {}
    if edit_bank:
        for (s, rel_k), _o in edit_bank.items():
            rel_keys_by_subject.setdefault(s, []).append(rel_k)

    if bool(args.legacy_vllm):
        os.environ["VLLM_USE_V1"] = "0"
    if bool(args.enforce_eager):
        # Plumb via env to avoid importing vllm arg names across versions.
        os.environ["GWALK_ENFORCE_EAGER"] = "1"
    llm = _make_vllm(args.model_path, tp_size=int(args.tp_size))
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    stop = ["\n", tok.eos_token] if tok.eos_token else ["\n"]

    # Evaluate each question variant; aggregate per-case any-correct (official-style).
    case_any = Agg()
    by_q: Dict[int, Agg] = {qi: Agg() for qi in q_idxs}
    per_case_preds: Dict[int, List[str]] = {ridx: [] for ridx in range(len(rows))}

    for qi in q_idxs:
        prompts_q: List[str] = []
        meta_q: List[int] = []
        for ridx, rec in enumerate(rows):
            qs = rec.get("questions") or []
            if isinstance(qs, list) and 0 <= qi < len(qs):
                q = _ws(qs[qi])
                if q:
                    prompts_q.append(q)
                    meta_q.append(ridx)

        # Per-question run: iterative walk.
        pred_final: Dict[int, str] = {ridx: "" for ridx in meta_q}
        prev_answer: Dict[int, str] = {ridx: "" for ridx in meta_q}

        for hop in range(int(args.max_hops)):
            # Step 1: ask M for next hop relation template and subject.
            rel_prompts: List[str] = []
            rel_meta: List[int] = []
            for ridx, q in zip(meta_q, prompts_q):
                # Skip if already marked done (store as pred_final non-empty with a sentinel).
                if pred_final.get(ridx, "") == "__DONE__":
                    continue
                subj_hint = _ws(rows[ridx].get("subject", "")) if bool(args.use_subject_hint) else ""
                rel_prompts.append(_format_rel_query_prompt(q, prev_answer=prev_answer.get(ridx, ""), subject_hint=subj_hint))
                rel_meta.append(ridx)

            rel_out = _vllm_generate(llm, rel_prompts, max_tokens=int(args.rel_query_tokens), stop=stop) if rel_prompts else []

            hop_subject: Dict[int, str] = {}
            hop_rel: Dict[int, str] = {}
            hop_done: Dict[int, bool] = {}
            for ridx, raw in zip(rel_meta, rel_out):
                s = _strip_json(raw)
                try:
                    obj = json.loads(s)
                except Exception:
                    obj = {}
                subj = _ws(obj.get("subject", ""))
                rel = _ws(obj.get("relation_template", ""))
                done = bool(obj.get("done", False))
                hop_done[ridx] = done
                hop_subject[ridx] = subj
                hop_rel[ridx] = rel

            # Step 2: apply memory override or fill with LLM.
            fill_prompts: List[str] = []
            fill_meta: List[int] = []
            for ridx in rel_meta:
                if hop_done.get(ridx, False):
                    # Mark done; final answer is the previous hop answer (or empty).
                    pred_final[ridx] = prev_answer.get(ridx, "") or ""
                    pred_final[ridx] = pred_final[ridx] or "__DONE__"
                    continue

                subj = hop_subject.get(ridx, "")
                rel = hop_rel.get(ridx, "")
                if not (subj and rel and "{}" in rel):
                    # If invalid, stop and use previous answer (best-effort).
                    pred_final[ridx] = prev_answer.get(ridx, "") or ""
                    pred_final[ridx] = pred_final[ridx] or "__DONE__"
                    continue

                subj_key = subj.lower()
                rel_key = None
                if edit_bank:
                    rel_key = _best_rel_key(rel, rel_keys_by_subject.get(subj_key, []), thr=float(args.rel_match_threshold))
                if rel_key and (subj_key, rel_key) in edit_bank:
                    ans = edit_bank[(subj_key, rel_key)]
                    prev_answer[ridx] = ans
                    continue

                fill_prompts.append(_format_fill_prompt(subj, rel))
                fill_meta.append(ridx)

            fill_out = _vllm_generate(llm, fill_prompts, max_tokens=int(args.fill_tokens), stop=stop) if fill_prompts else []
            for ridx, raw in zip(fill_meta, fill_out):
                ans = _strip_answer(raw)
                if ans:
                    prev_answer[ridx] = ans

        # After max hops, final answer = last prev_answer.
        preds_for_q: Dict[int, str] = {}
        for ridx in meta_q:
            final = pred_final.get(ridx, "")
            if final == "__DONE__":
                final = prev_answer.get(ridx, "")
            if not final:
                final = prev_answer.get(ridx, "")
            preds_for_q[ridx] = _strip_answer(final)

        # Score per-q and accumulate per-case lists.
        for ridx, rec in enumerate(rows):
            if ridx not in preds_for_q:
                continue
            ok = _check_answer_new(rec, preds_for_q[ridx])
            by_q[qi].add(ok)

        # Store per-case for aggregation later.
        for ridx, pred in preds_for_q.items():
            per_case_preds[ridx].append(pred)

    # Official-style: case correct if ANY asked question yields a match.
    for ridx, rec in enumerate(rows):
        preds = per_case_preds.get(ridx, [])
        if not preds:
            continue
        case_any.add(any(_check_answer_new(rec, p) for p in preds))

    report = {
        "data_path": args.data_path,
        "model_path": args.model_path,
        "max_records": int(args.max_records),
        "q_idxs": q_idxs,
        "max_hops": int(args.max_hops),
        "use_memory": bool(args.use_memory),
        "use_subject_hint": bool(args.use_subject_hint),
        "rel_match_threshold": float(args.rel_match_threshold),
        "case_acc_any": case_any.to_dict(),
        "by_q": {str(qi): by_q[qi].to_dict() for qi in q_idxs},
    }

    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
