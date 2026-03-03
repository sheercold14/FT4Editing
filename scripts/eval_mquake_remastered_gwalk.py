from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from dataclasses import dataclass
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


def _first_line(ans: str) -> str:
    s = str(ans or "").strip()
    return s.splitlines()[0].strip() if s else ""


def _norm_key(text: str) -> str:
    return _ws(text).lower()


def _rel_template_from_cloze(cloze: str, *, subject_hint: str) -> str:
    c = _ws(cloze)
    if not c:
        return ""
    if "{}" in c:
        return c
    if subject_hint and subject_hint in c:
        return _ws(c.replace(subject_hint, "{}", 1))
    return c


def _make_vllm(model_path: str, *, tp_size: int):
    from vllm import LLM

    mp = str(Path(model_path).expanduser())
    if Path(mp).exists():
        mp = str(Path(mp).resolve())
    return LLM(model=mp, tensor_parallel_size=int(tp_size), trust_remote_code=True)


def _vllm_generate(llm, prompts: List[str], *, max_tokens: int, stop: List[str]):
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0, max_tokens=int(max_tokens), stop=stop)
    outs = llm.generate(prompts, sp)
    return [o.outputs[0].text for o in outs]


def _load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "external" / "MQuAKE-Remastered" / "data_utils.py").exists():
            return p
    return start


def _load_mquake_utils(repo_root: Path) -> Any:
    """
    Load external/MQuAKE-Remastered/data_utils.py via importlib (repo path contains a hyphen).
    """
    utils_path = repo_root / "external" / "MQuAKE-Remastered" / "data_utils.py"
    if not utils_path.exists():
        raise FileNotFoundError(f"Missing {utils_path} (did you git clone external/MQuAKE-Remastered?)")
    spec = importlib.util.spec_from_file_location("mquake_utils", utils_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec for {utils_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


@dataclass(frozen=True)
class EditFact:
    subject: str
    prompt: str
    target: str
    triple_ids: Tuple[str, str, str]  # (subj_id, rel_id, obj_id)


def _iter_edit_facts(rec: Dict[str, Any]) -> Iterable[EditFact]:
    reqs = rec.get("requested_rewrite") or []
    triples = (rec.get("orig") or {}).get("edit_triples") or []
    for req, triple in zip(reqs, triples):
        subj = _ws(req.get("subject", ""))
        prompt = _ws(req.get("prompt", ""))
        tgt = _ws(((req.get("target_new") or {}) if isinstance(req.get("target_new"), dict) else {}).get("str", ""))
        if not tgt:
            tgt = _ws(req.get("target_new", ""))
        if not (subj and prompt and tgt and "{}" in prompt):
            continue
        if not (isinstance(triple, list) and len(triple) == 3):
            continue
        s_id, r_id, o_id = (str(triple[0]), str(triple[1]), str(triple[2]))
        yield EditFact(subject=subj, prompt=prompt, target=tgt, triple_ids=(s_id, r_id, o_id))


def _build_edit_bank(
    rows: List[Dict[str, Any]],
    *,
    edited_case_ids: Optional[set[int]],
) -> Dict[Tuple[str, str], EditFact]:
    bank: Dict[Tuple[str, str], EditFact] = {}
    for rec in rows:
        cid = int(rec.get("case_id", -1))
        if edited_case_ids is not None and cid not in edited_case_ids:
            continue
        for fact in _iter_edit_facts(rec):
            bank[(_norm_key(fact.subject), _norm_key(fact.prompt))] = fact
    return bank


def _correct_path(
    rec: Dict[str, Any],
    *,
    edited: bool,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Returns (triples_ids, triples_labeled) for the correct path for this case.
    """
    orig = rec.get("orig") or {}
    if edited:
        ids = orig.get("new_triples") or []
        labeled = orig.get("new_triples_labeled") or []
    else:
        ids = orig.get("triples") or []
        labeled = orig.get("triples_labeled") or []
    if not (isinstance(ids, list) and isinstance(labeled, list)):
        return [], []
    return ids, labeled


def _hop_templates(
    rec: Dict[str, Any],
    *,
    edited: bool,
    labeled_path: List[List[str]],
) -> List[str]:
    hops = (rec.get("new_single_hops") if edited else rec.get("single_hops")) or []
    if not isinstance(hops, list):
        return []
    rels: List[str] = []
    for i, hop in enumerate(hops):
        if not isinstance(hop, dict):
            continue
        cloze = _ws(hop.get("cloze", ""))
        hint = ""
        if i < len(labeled_path) and isinstance(labeled_path[i], list) and labeled_path[i]:
            hint = _ws(labeled_path[i][0])
        rels.append(_rel_template_from_cloze(cloze, subject_hint=hint))
    return [r for r in rels if r and "{}" in r]


def _format_fill_prompt(subject: str, rel_template: str) -> str:
    st = rel_template
    if "{}" in st:
        try:
            st = st.format(subject)
        except Exception:
            st = st.replace("{}", subject)
    return (
        "Complete the statement with the missing object. Respond with only the object.\n"
        f"{st} ____\n"
        "Answer:"
    )


def _should_mask_override(
    fact: EditFact,
    *,
    correct_obj_by_sr: Dict[Tuple[str, str], str],
) -> bool:
    s_id, r_id, o_id = fact.triple_ids
    ok_obj = correct_obj_by_sr.get((s_id, r_id))
    return ok_obj is not None and ok_obj != o_id


def _make_correct_obj_map(triples_ids: List[List[str]]) -> Dict[Tuple[str, str], str]:
    out: Dict[Tuple[str, str], str] = {}
    for t in triples_ids:
        if not (isinstance(t, list) and len(t) == 3):
            continue
        s_id, r_id, o_id = str(t[0]), str(t[1]), str(t[2])
        out[(s_id, r_id)] = o_id
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str, help="LLM used as M in GWalk.")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--cuda", type=str, default="")
    parser.add_argument("--max_tokens", type=int, default=8)
    parser.add_argument(
        "--stop_mode",
        type=str,
        default="strict",
        choices=["strict", "paper"],
        help=(
            "strict: stop on newline/dot/eos (can inflate accuracy by preventing trailing punctuation). "
            "paper: stop on newline/eos (closer to typical generation)."
        ),
    )
    parser.add_argument(
        "--report_raw",
        action="store_true",
        help="Also compute accuracy using unnormalized raw answers (uppercases only, no stripping).",
    )
    parser.add_argument(
        "--hop_update",
        type=str,
        default="strip",
        choices=["strip", "first_line_raw"],
        help=(
            "How to update the intermediate hop entity from model output. "
            "strip: aggressive normalization (may inflate accuracy). "
            "first_line_raw: only take first line, keep punctuation."
        ),
    )
    parser.add_argument(
        "--init_subject_mode",
        type=str,
        default="gold_path",
        choices=["gold_path", "llm_question"],
        help=(
            "gold_path: initialize subject from gold triples_labeled (oracle). "
            "llm_question: extract subject from the question text using the model."
        ),
    )
    parser.add_argument(
        "--use_all_questions",
        action="store_true",
        help=(
            "Run one walk per question paraphrase and aggregate answers per case (matches cal_accuracy any-of-answers). "
            "Useful with --init_subject_mode llm_question."
        ),
    )
    parser.add_argument("--max_records", type=int, default=0, help="0 = all")
    parser.add_argument("--use_memory", action="store_true", help="Override hop answer if (s, prompt) in edit bank.")
    parser.add_argument("--use_6334_split", action="store_true", help="Use MQuAKE-Remastered-CF-6334 split protocol.")
    parser.add_argument("--edit_num", type=int, default=6334, help="Edit budget for 6334 split (e.g. 100|1000|3000|6334).")
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    rows = _load_json(args.data_path)
    if int(args.max_records) > 0:
        rows = rows[: int(args.max_records)]

    repo_root = _find_repo_root(Path(__file__).resolve().parent)
    mquake_utils = _load_mquake_utils(repo_root)

    # Decide eval set + which case_ids are considered edited (i.e., present in the edit bank).
    eval_rows: List[Dict[str, Any]] = rows
    edited_case_ids: Optional[set[int]] = None

    if bool(args.use_6334_split):
        # Uses the official protocol for the CF-6334 variant:
        # - train_set_edited_caseid = edits present in the edit bank
        # - evaluate on the provided test_set
        train_set, test_set, train_ids, _test_ids, *_ = mquake_utils.process_mquake_remastered_cf_6334(
            rows, edit_num=int(args.edit_num)
        )
        edited_case_ids = set(int(x) for x in train_ids)
        eval_rows = list(test_set)
        print(f"[split] edit_num={int(args.edit_num)} edit_bank_cases={len(edited_case_ids)} eval_cases={len(eval_rows)}")
    else:
        # Default: treat ALL cases as edited (CF-3k / CF-9k style).
        edited_case_ids = set(int(r.get("case_id", -1)) for r in rows if int(r.get("case_id", -1)) >= 0)

    edit_bank = _build_edit_bank(rows, edited_case_ids=edited_case_ids) if bool(args.use_memory) else {}

    llm = _make_vllm(args.model_path, tp_size=int(args.tp_size))
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    eos = [tok.eos_token] if tok.eos_token else []
    if str(args.stop_mode) == "paper":
        stop = ["\n"] + eos
    else:
        stop = ["\n", "."] + eos

    # Build per-question instances (optionally), then walk hop-by-hop.
    inst_case_ids: List[int] = []
    inst_edited: List[bool] = []
    inst_questions: List[str] = []
    inst_hop_rels: List[List[str]] = []
    inst_correct_obj_maps: List[Dict[Tuple[str, str], str]] = []
    inst_cur: List[str] = []
    inst_cur_raw: List[str] = []

    for rec in eval_rows:
        cid = int(rec.get("case_id", -1))
        edited = bool(edited_case_ids is not None and cid in edited_case_ids)
        triples_ids, triples_labeled = _correct_path(rec, edited=edited)
        rels = _hop_templates(rec, edited=edited, labeled_path=triples_labeled)
        correct_map = _make_correct_obj_map(triples_ids)

        qs = rec.get("questions") or []
        if not isinstance(qs, list) or not qs:
            qs = [""]
        if not bool(args.use_all_questions):
            qs = [qs[0]]

        for q in qs:
            inst_case_ids.append(cid)
            inst_edited.append(edited)
            inst_questions.append(_ws(q))
            inst_hop_rels.append(rels)
            inst_correct_obj_maps.append(correct_map)
            inst_cur.append("")  # filled below
            inst_cur_raw.append("")

            if str(args.init_subject_mode) == "gold_path":
                init = ""
                if triples_labeled and isinstance(triples_labeled[0], list) and triples_labeled[0]:
                    init = _ws(triples_labeled[0][0])
                if not init:
                    reqs = rec.get("requested_rewrite") or []
                    if reqs and isinstance(reqs, list) and isinstance(reqs[0], dict):
                        init = _ws(reqs[0].get("subject", ""))
                inst_cur[-1] = init
                inst_cur_raw[-1] = init

    if str(args.init_subject_mode) == "llm_question":
        subj_prompts: List[str] = []
        subj_meta: List[int] = []
        for i, q in enumerate(inst_questions):
            if inst_cur[i]:
                continue
            if not q:
                continue
            subj_prompts.append(
                "Extract the starting subject entity from the question. "
                "Return ONLY the entity string (no quotes, no punctuation).\n"
                f"Question: {q}\n"
                "Subject:"
            )
            subj_meta.append(i)
        if subj_prompts:
            subj_out = _vllm_generate(llm, subj_prompts, max_tokens=16, stop=["\n"] + eos)
            for i, pred in zip(subj_meta, subj_out):
                inst_cur_raw[i] = str(pred or "")
                inst_cur[i] = _strip_answer(pred)

    max_hops = max((len(r) for r in inst_hop_rels), default=0)

    for hop_idx in range(max_hops):
        prompts: List[str] = []
        meta: List[int] = []

        for i in range(len(inst_case_ids)):
            rels = inst_hop_rels[i]
            if hop_idx >= len(rels):
                continue
            subj = _ws(inst_cur[i])
            rel_t = _ws(rels[hop_idx])
            if not (subj and rel_t and "{}" in rel_t):
                continue

            if edit_bank:
                key = (_norm_key(subj), _norm_key(rel_t))
                fact = edit_bank.get(key)
                if fact is not None and not _should_mask_override(fact, correct_obj_by_sr=inst_correct_obj_maps[i]):
                    inst_cur[i] = fact.target
                    inst_cur_raw[i] = fact.target
                    continue

            prompts.append(_format_fill_prompt(subj, rel_t))
            meta.append(i)

        if prompts:
            preds = _vllm_generate(llm, prompts, max_tokens=int(args.max_tokens), stop=stop)
            for i, pred in zip(meta, preds):
                inst_cur_raw[i] = str(pred or "")
                if str(args.hop_update) == "first_line_raw":
                    ans = _first_line(pred)
                else:
                    ans = _strip_answer(pred)
                if ans:
                    inst_cur[i] = ans

    raw_answer_dict_norm: Dict[str, Dict[str, Any]] = {}
    raw_answer_dict_raw: Dict[str, Dict[str, Any]] = {}
    for cid, edited, pred, pred_raw in zip(inst_case_ids, inst_edited, inst_cur, inst_cur_raw):
        if cid < 0:
            continue
        entry = raw_answer_dict_norm.setdefault(str(cid), {"answers": [], "edited": bool(edited)})
        entry["answers"].append(_strip_answer(pred))
        if bool(args.report_raw):
            entry2 = raw_answer_dict_raw.setdefault(str(cid), {"answers": [], "edited": bool(edited)})
            entry2["answers"].append(_first_line(pred_raw))

    use_6334 = bool(args.use_6334_split) and any("6334_split" in (r or {}) for r in eval_rows)
    metrics_norm, correct_norm, total_norm = mquake_utils.cal_accuracy(
        eval_rows, raw_answer_dict_norm, int(args.edit_num), use_6334=use_6334
    )
    metrics_raw = None
    if bool(args.report_raw):
        metrics_raw, _correct_raw, _total_raw = mquake_utils.cal_accuracy(
            eval_rows, raw_answer_dict_raw, int(args.edit_num), use_6334=use_6334
        )

    report = {
        "data_path": args.data_path,
        "model_path": args.model_path,
        "use_memory": bool(args.use_memory),
        "use_6334_split": bool(args.use_6334_split),
        "edit_num": int(args.edit_num),
        "max_records": int(args.max_records),
        "max_tokens": int(args.max_tokens),
        "stop_mode": str(args.stop_mode),
        "stop": stop,
        "hop_update": str(args.hop_update),
        "init_subject_mode": str(args.init_subject_mode),
        "use_all_questions": bool(args.use_all_questions),
        "metrics_norm": metrics_norm,
        "metrics_raw": metrics_raw,
        "n_eval": len(eval_rows),
        "n_edit_bank": len(edit_bank),
        "n_instances": len(inst_case_ids),
        "n_edited_cases": len(edited_case_ids) if edited_case_ids is not None else None,
    }
    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        raw_path = Path(args.output_path).with_suffix(".raw_answers.norm.json")
        raw_path.write_text(json.dumps(raw_answer_dict_norm, ensure_ascii=False, indent=2), encoding="utf-8")
        if bool(args.report_raw):
            raw_path2 = Path(args.output_path).with_suffix(".raw_answers.raw.json")
            raw_path2.write_text(json.dumps(raw_answer_dict_raw, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
