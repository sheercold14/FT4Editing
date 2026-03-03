from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


def _ws(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _strip_answer(ans: str) -> str:
    s = str(ans or "").strip()
    s = re.sub(r"^(answer\s*:\s*)", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^(the\s+answer\s+is\s+)", "", s, flags=re.IGNORECASE).strip()
    s = s.splitlines()[0].strip() if s else ""
    s = s.rstrip(" .\n\t\r")
    return s.strip()


def _norm(text: str) -> str:
    return _ws(text).lower()


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "external" / "MQuAKE-Remastered" / "data_utils.py").exists():
            return p
    return start


def _load_py_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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


@dataclass(frozen=True)
class EditFact:
    subj: str
    rel: str
    prompt: str
    obj: str
    triple_ids: Tuple[str, str, str]  # (subj_id, rel_id, obj_id)


def _build_relid_to_label(rows: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for rec in rows:
        orig = rec.get("orig") or {}
        for key_ids, key_lab in [
            ("triples", "triples_labeled"),
            ("new_triples", "new_triples_labeled"),
        ]:
            ids = orig.get(key_ids) or []
            lab = orig.get(key_lab) or []
            if not (isinstance(ids, list) and isinstance(lab, list) and len(ids) == len(lab)):
                continue
            for t_ids, t_lab in zip(ids, lab):
                if not (isinstance(t_ids, list) and len(t_ids) == 3 and isinstance(t_lab, list) and len(t_lab) >= 2):
                    continue
                r_id = str(t_ids[1])
                r_lab = _ws(t_lab[1])
                if r_id and r_lab and r_id not in out:
                    out[r_id] = r_lab
    return out


def _iter_edit_facts(rec: Dict[str, Any], *, relid_to_label: Dict[str, str]) -> Iterable[EditFact]:
    reqs = rec.get("requested_rewrite") or []
    triples = (rec.get("orig") or {}).get("edit_triples") or []
    for req, triple in zip(reqs, triples):
        if not (isinstance(triple, list) and len(triple) == 3):
            continue
        s_id, r_id, o_id = (str(triple[0]), str(triple[1]), str(triple[2]))

        subj = _ws(req.get("subject", ""))
        prompt = _ws(req.get("prompt", ""))
        obj = _ws(((req.get("target_new") or {}) if isinstance(req.get("target_new"), dict) else {}).get("str", ""))
        if not obj:
            obj = _ws(req.get("target_new", ""))

        rel = _ws(relid_to_label.get(r_id, ""))
        if not rel:
            rel = prompt.replace("{}", "").strip()

        if subj and prompt and rel and obj and "{}" in prompt:
            yield EditFact(subj=subj, rel=rel, prompt=prompt, obj=obj, triple_ids=(s_id, r_id, o_id))


def _correct_path_ids(rec: Dict[str, Any], *, edited: bool) -> List[List[str]]:
    orig = rec.get("orig") or {}
    ids = orig.get("new_triples" if edited else "triples") or []
    return ids if isinstance(ids, list) else []


def _make_correct_obj_map(triples_ids: List[List[str]]) -> Dict[Tuple[str, str], str]:
    out: Dict[Tuple[str, str], str] = {}
    for t in triples_ids:
        if not (isinstance(t, list) and len(t) == 3):
            continue
        s_id, r_id, o_id = str(t[0]), str(t[1]), str(t[2])
        out[(s_id, r_id)] = o_id
    return out


def _should_mask(fact: EditFact, *, correct_obj_by_sr: Dict[Tuple[str, str], str]) -> bool:
    s_id, r_id, o_id = fact.triple_ids
    ok_obj = correct_obj_by_sr.get((s_id, r_id))
    return ok_obj is not None and ok_obj != o_id


def _parse_rel_list(text: str, *, max_hops: int) -> List[str]:
    s = str(text or "").strip()
    if "```" in s:
        s = s.replace("```json", "```").replace("```JSON", "```")
        parts = [p.strip() for p in s.split("```") if p.strip()]
        s = parts[0] if parts else s
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            rels = [_ws(x) for x in obj if _ws(x)]
            return rels[:max_hops]
    except Exception:
        pass
    m = re.search(r"\[(.*)\]", s, flags=re.DOTALL)
    if m:
        inner = m.group(1)
        rels = re.split(r"\s*[,\n]\s*", inner)
        rels = [_ws(r.strip("'\" ")) for r in rels if _ws(r.strip("'\" "))]
        return rels[:max_hops]
    rels = re.split(r"[\n;，,]+", s)
    rels = [_ws(r) for r in rels if _ws(r)]
    return rels[:max_hops]


class ContrieverIndex:
    def __init__(self, model_name: str, *, device: str, batch_size: int = 64):
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        self._cache: Dict[str, torch.Tensor] = {}

    @torch.inference_mode()
    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        outs: List[torch.Tensor] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tok(batch, padding=True, truncation=True, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            last = self.model(**enc).last_hidden_state  # [b, t, d]
            mask = enc["attention_mask"].unsqueeze(-1)  # [b, t, 1]
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            outs.append(pooled.detach().cpu())
        return torch.cat(outs, dim=0) if outs else torch.empty((0, 0), dtype=torch.float32)

    def embed_one(self, text: str) -> torch.Tensor:
        key = _ws(text)
        if key in self._cache:
            return self._cache[key]
        v = self.embed_texts([key])[0]
        self._cache[key] = v
        return v


def _best_match(
    query: str,
    cands: List[str],
    cand_embs: torch.Tensor,
    idx: ContrieverIndex,
    *,
    thr: float,
) -> Optional[int]:
    if not (query and cands):
        return None
    q = idx.embed_one(query)  # [d] on CPU
    sims = torch.mv(cand_embs, q)  # [n]
    best = int(torch.argmax(sims).item())
    return best if float(sims[best].item()) >= float(thr) else None


def _infer_dataset_tag(data_path: str) -> Optional[str]:
    p = str(data_path)
    if "CF-3k" in p or "CF-3K" in p or "cf3k" in p:
        return "3k"
    if "CF-9k" in p or "CF-9K" in p or "cf9k" in p:
        return "9k"
    if "CF-3151" in p:
        return "3151"
    if p.endswith("MQuAKE-Remastered-T.json"):
        return "T"
    return None


def _pick_rand_list(edit_cases_mod: Any, tag: str, *, edit_num: int, dataset_size: int) -> List[int]:
    if edit_num <= 0 or edit_num >= dataset_size:
        cand = [f"rand_list_{tag}_all"]
    else:
        cand = [f"rand_list_{tag}_{edit_num}", f"rand_list_{tag}_all"]
    for name in cand:
        if hasattr(edit_cases_mod, name):
            v = getattr(edit_cases_mod, name)
            if isinstance(v, list):
                return [int(x) for x in v]
    raise ValueError(f"Missing rand list for tag={tag} edit_num={edit_num} in external/MQuAKE-Remastered/edit_cases.py")


def _build_rel_template_from_cloze(cloze: str, *, subject: str) -> Optional[str]:
    c = _ws(cloze)
    s = _ws(subject)
    if not (c and s):
        return None
    if "{}" in c and c.count("{}") == 1:
        return c
    if s in c:
        t = c.replace(s, "{}", 1)
        return t if t.count("{}") == 1 else None
    # Case-insensitive fallback (replace first match span).
    m = re.search(re.escape(s), c, flags=re.IGNORECASE)
    if m:
        t = c[: m.start()] + "{}" + c[m.end() :]
        return t if t.count("{}") == 1 else None
    return None


def _build_dataset_rel_templates(rows: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """
    Build a mapping rel_label_norm -> most common cloze template (with exactly one '{}')
    using dataset-provided single_hops/new_single_hops cloze strings.
    """
    from collections import Counter, defaultdict

    ctrs: Dict[str, Counter] = defaultdict(Counter)
    for rec in rows:
        orig = rec.get("orig") or {}
        for triples_key, labeled_key, hops_key in [
            ("triples", "triples_labeled", "single_hops"),
            ("new_triples", "new_triples_labeled", "new_single_hops"),
        ]:
            labeled = orig.get(labeled_key) or []
            hops = rec.get(hops_key) or []
            if not (isinstance(labeled, list) and isinstance(hops, list) and len(labeled) == len(hops)):
                continue
            for t_lab, hop in zip(labeled, hops):
                if not (isinstance(t_lab, list) and len(t_lab) >= 2 and isinstance(hop, dict)):
                    continue
                subj = _ws(t_lab[0])
                rel = _norm(t_lab[1])
                cloze = _ws(hop.get("cloze", ""))
                tmpl = _build_rel_template_from_cloze(cloze, subject=subj)
                if tmpl and tmpl.count("{}") == 1:
                    ctrs[rel][tmpl] += 1

    out: Dict[str, str] = {}
    for rel, c in ctrs.items():
        if c:
            out[rel] = c.most_common(1)[0][0]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str, help="LLM used as M in GWalk.")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--cuda", type=str, default="")
    parser.add_argument("--max_records", type=int, default=0, help="0 = all")
    parser.add_argument("--eval_start", type=int, default=0, help="Start index into eval set (after split).")
    parser.add_argument("--eval_end", type=int, default=0, help="End index (exclusive); 0 = end.")
    parser.add_argument("--edit_num", type=int, default=3000)
    parser.add_argument("--use_6334_split", action="store_true")
    parser.add_argument("--decompose_mode", type=str, default="rel_labels", choices=["rel_labels", "cloze_templates"])
    parser.add_argument("--max_hops", type=int, default=4)
    parser.add_argument("--max_tokens_subject", type=int, default=16)
    parser.add_argument("--max_tokens_rels", type=int, default=64)
    parser.add_argument("--max_tokens_hop", type=int, default=8)
    parser.add_argument("--max_tokens_template", type=int, default=32)
    parser.add_argument("--entity_thr", type=float, default=0.45)
    parser.add_argument("--rel_thr", type=float, default=0.45)
    parser.add_argument("--embed_model", type=str, default="facebook/contriever-msmarco")
    parser.add_argument("--embed_device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--embed_batch_size", type=int, default=64)
    parser.add_argument("--template_source", type=str, default="hybrid", choices=["llm", "dataset", "hybrid"])
    parser.add_argument("--use_chat_template", action="store_true", help="Format prompts with tokenizer.chat_template (Instruct-style).")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.", help="Used when --use_chat_template is set.")
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    repo_root = _find_repo_root(Path(__file__).resolve().parent)
    mquake_utils = _load_py_module("mquake_utils", repo_root / "external" / "MQuAKE-Remastered" / "data_utils.py")
    edit_cases_mod = _load_py_module("edit_cases", repo_root / "external" / "MQuAKE-Remastered" / "edit_cases.py")

    rows_all = _load_json(args.data_path)
    relid_to_label = _build_relid_to_label(rows_all)
    ds_rel_templates = _build_dataset_rel_templates(rows_all)

    eval_rows: List[Dict[str, Any]] = list(rows_all)
    edited_case_ids: set[int]

    if bool(args.use_6334_split):
        train_set, test_set, train_ids, _test_ids, *_ = mquake_utils.process_mquake_remastered_cf_6334(
            eval_rows, edit_num=int(args.edit_num)
        )
        edited_case_ids = set(int(x) for x in train_ids)
        eval_rows = list(test_set)
        print(f"[split] CF6334 edit_num={int(args.edit_num)} edit_bank_cases={len(edited_case_ids)} eval_cases={len(eval_rows)}")
    else:
        tag = _infer_dataset_tag(args.data_path)
        if tag is None:
            raise ValueError(f"Cannot infer dataset tag from --data_path={args.data_path}; use --use_6334_split or rename file.")
        rand_list = _pick_rand_list(edit_cases_mod, tag, edit_num=int(args.edit_num), dataset_size=len(rows_all))
        edited_case_ids = set(int(x) for x in rand_list)
        print(f"[split] tag={tag} edit_num={int(args.edit_num)} edited_cases={len(edited_case_ids)} eval_cases={len(eval_rows)}")

    # Limit evaluation set AFTER we decide the edit bank.
    start = max(0, int(args.eval_start))
    end = int(args.eval_end)
    if end <= 0:
        end = len(eval_rows)
    if start or end != len(eval_rows):
        eval_rows = eval_rows[start:end]
    if int(args.max_records) > 0:
        eval_rows = eval_rows[: int(args.max_records)]

    facts: List[EditFact] = []
    for rec in rows_all:
        cid = int(rec.get("case_id", -1))
        if cid not in edited_case_ids:
            continue
        facts.extend(list(_iter_edit_facts(rec, relid_to_label=relid_to_label)))

    facts_by_subj: Dict[str, List[EditFact]] = {}
    for f in facts:
        facts_by_subj.setdefault(_norm(f.subj), []).append(f)

    llm = _make_vllm(args.model_path, tp_size=int(args.tp_size))
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    eos_stop = [tok.eos_token] if tok.eos_token else []
    subj_stop = ["\n"] + eos_stop
    hop_stop = ["\n"] + eos_stop
    rel_stop = eos_stop
    tmpl_stop = ["\n"] + eos_stop

    def fmt(user_text: str) -> str:
        if not bool(args.use_chat_template):
            return user_text
        if not getattr(tok, "chat_template", None):
            return user_text
        msgs = [
            {"role": "system", "content": str(args.system_prompt)},
            {"role": "user", "content": user_text},
        ]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return user_text

    # Embedding index (init AFTER vLLM forks workers to avoid tokenizers-before-fork deadlocks).
    subj_texts = sorted({f.subj for f in facts})
    idx = ContrieverIndex(str(args.embed_model), device=str(args.embed_device), batch_size=int(args.embed_batch_size))
    subj_embs = idx.embed_texts(subj_texts) if subj_texts else torch.empty((0, 0), dtype=torch.float32)
    rel_cache: Dict[str, Tuple[List[str], torch.Tensor]] = {}

    # Flatten question instances.
    inst_case_ids: List[int] = []
    inst_q: List[str] = []
    inst_edited: List[bool] = []
    inst_correct_map: List[Dict[Tuple[str, str], str]] = []

    for rec in eval_rows:
        cid = int(rec.get("case_id", -1))
        edited = bool(cid in edited_case_ids)
        correct_map = _make_correct_obj_map(_correct_path_ids(rec, edited=edited))
        qs = rec.get("questions") or []
        if not isinstance(qs, list):
            continue
        for q in qs:
            q = _ws(q)
            if not q:
                continue
            inst_case_ids.append(cid)
            inst_q.append(q)
            inst_edited.append(edited)
            inst_correct_map.append(correct_map)

    # Step 1: extract initial subject.
    subj_prompts = [
        fmt(
            "Extract the starting subject entity from the question. "
            "Return ONLY the entity string (no quotes, no punctuation).\n"
            f"Question: {q}\n"
            "Subject:"
        )
        for q in inst_q
    ]
    subj_out = _vllm_generate(llm, subj_prompts, max_tokens=int(args.max_tokens_subject), stop=subj_stop)
    cur_s = [_strip_answer(x) for x in subj_out]

    # Step 2: decompose into relations.
    if str(args.decompose_mode) == "cloze_templates":
        rel_prompts = [
            fmt(
                "Decompose the multi-hop question into an ordered list of SINGLE-HOP cloze templates.\n"
                "IMPORTANT: list hops from INNERMOST to OUTERMOST (apply to the subject first, then to the result).\n"
                "Return a JSON array of 1 to 4 templates.\n"
                "Each template MUST contain EXACTLY ONE '{}' placeholder, and should be a short English statement missing the object.\n"
                "Examples:\n"
                '- Q: "Who is the head of state in the country of citizenship of Ellie Kemper?"\n'
                '  A: ["{} is a citizen of", "{} is the head of state of"]\n'
                '- Q: "What is the capital of the country where X is a citizen?"\n'
                '  A: ["{} is a citizen of", "The capital of {} is"]\n'
                f"Question: {q}\n"
                "Templates:"
            )
            for q in inst_q
        ]
    else:
        rel_prompts = [
            fmt(
                "Decompose the multi-hop question into an ordered list of relation labels.\n"
                "IMPORTANT: list relations from INNERMOST to OUTERMOST (apply to the subject first, then to the result).\n"
                "Return a JSON array of 1 to 4 short relation phrases.\n"
                "Do NOT include modifiers like 'current', 'present', 'name of', etc. Use the core relation only.\n"
                "Examples:\n"
                '- Q: "Who is the head of state in the country of citizenship of Ellie Kemper?"\n'
                '  A: ["country of citizenship", "head of state"]\n'
                '- Q: "What is the capital of the country where X is a citizen?"\n'
                '  A: ["country of citizenship", "capital"]\n'
                f"Question: {q}\n"
                "Relations:"
            )
            for q in inst_q
        ]
    rel_out = _vllm_generate(llm, rel_prompts, max_tokens=int(args.max_tokens_rels), stop=rel_stop)
    rels_by_inst = [_parse_rel_list(x, max_hops=int(args.max_hops)) for x in rel_out]
    rel_stats = {
        "n_inst": len(rels_by_inst),
        "empty_rels": sum(1 for rs in rels_by_inst if not rs),
        "avg_hops": (sum(len(rs) for rs in rels_by_inst) / max(1, len(rels_by_inst))),
    }

    rel_template_cache: Dict[str, str] = {}

    # Walk.
    for hop_idx in range(int(args.max_hops)):
        hop_prompts: List[str] = []
        hop_meta: List[int] = []
        hop_rel_norm: List[str] = []
        hop_subject: List[str] = []
        hop_tmpl_raw: List[str] = []

        for i in range(len(inst_q)):
            rels = rels_by_inst[i]
            if hop_idx >= len(rels):
                continue

            s = _ws(cur_s[i])
            r = _ws(rels[hop_idx])
            if not (s and r):
                continue

            subj_key = _norm(s) if _norm(s) in facts_by_subj else None
            if subj_key is None and subj_texts:
                j = _best_match(s, subj_texts, subj_embs, idx, thr=float(args.entity_thr))
                if j is not None:
                    subj_key = _norm(subj_texts[j])

            fact: Optional[EditFact] = None
            if subj_key is not None:
                outs = facts_by_subj.get(subj_key) or []
                if outs:
                    r_norm = _norm(r)
                    if str(args.decompose_mode) == "cloze_templates":
                        exact = next((f for f in outs if _norm(f.prompt) == r_norm), None)
                    else:
                        exact = next((f for f in outs if _norm(f.rel) == r_norm), None)
                    if exact is not None:
                        fact = exact
                    else:
                        if subj_key not in rel_cache:
                            rel_texts = [f.prompt if str(args.decompose_mode) == "cloze_templates" else f.rel for f in outs]
                            rel_cache[subj_key] = (rel_texts, idx.embed_texts(rel_texts))
                        rel_texts, rel_embs = rel_cache[subj_key]
                        k = _best_match(r, rel_texts, rel_embs, idx, thr=float(args.rel_thr))
                        if k is not None:
                            fact = outs[k]

            if fact is not None and not _should_mask(fact, correct_obj_by_sr=inst_correct_map[i]):
                cur_s[i] = fact.obj
                continue

            hop_meta.append(i)
            hop_rel_norm.append(_norm(r))
            hop_subject.append(s)
            hop_tmpl_raw.append(r)

        # (1) Build cloze templates for unseen relation labels (cached).
        # Skipped if the decomposition already returns cloze templates.
        if str(args.decompose_mode) != "cloze_templates":
            need_rels = sorted({rn for rn in hop_rel_norm if rn and rn not in rel_template_cache})
            if need_rels:
                # First try dataset-derived templates (if enabled).
                if str(args.template_source) in ("dataset", "hybrid"):
                    for rel in need_rels:
                        t = ds_rel_templates.get(rel)
                        if t:
                            rel_template_cache[rel] = t

                remaining = [rel for rel in need_rels if rel not in rel_template_cache]
                if remaining and str(args.template_source) in ("llm", "hybrid"):
                    tmpl_prompts = [
                        fmt(
                            "Convert the relation label into a cloze statement template with EXACTLY ONE '{}' placeholder.\n"
                            "The template should be a short English statement missing the object.\n"
                            'Example: country of citizenship -> "{} is a citizen of"\n'
                            f"Relation label: {rel}\n"
                            "Template:"
                        )
                        for rel in remaining
                    ]
                    tmpl_out = _vllm_generate(llm, tmpl_prompts, max_tokens=int(args.max_tokens_template), stop=tmpl_stop)
                    for rel, raw in zip(remaining, tmpl_out):
                        t = _ws(raw)
                        if t.count("{}") != 1:
                            t = "{} " + _ws(rel)
                        rel_template_cache[rel] = t

        # (2) Answer hops via fill-in-the-blank using the template.
        if hop_meta:
            hop_prompts = []
            for rn, subj, raw_t in zip(hop_rel_norm, hop_subject, hop_tmpl_raw):
                if str(args.decompose_mode) == "cloze_templates":
                    tmpl = _ws(raw_t)
                    if tmpl.count("{}") != 1:
                        tmpl = "{} " + rn
                else:
                    tmpl = rel_template_cache.get(rn, "{} " + rn)
                try:
                    stmt = tmpl.format(subj)
                except Exception:
                    stmt = tmpl.replace("{}", subj)
                hop_prompts.append(
                    fmt(
                    "Complete the statement with the missing object. Respond with only the object.\n"
                    f"{stmt} ____\n"
                    "Answer:"
                    )
                )
            hop_out = _vllm_generate(llm, hop_prompts, max_tokens=int(args.max_tokens_hop), stop=hop_stop)
            for inst_i, raw in zip(hop_meta, hop_out):
                ans = _strip_answer(raw)
                if ans:
                    cur_s[inst_i] = ans

    # Aggregate answers per case (any-of-questions).
    raw_answer_dict: Dict[str, Dict[str, Any]] = {}
    for cid, edited, pred in zip(inst_case_ids, inst_edited, cur_s):
        if cid < 0:
            continue
        entry = raw_answer_dict.setdefault(str(cid), {"answers": [], "edited": bool(edited)})
        entry["answers"].append(_strip_answer(pred))

    use_6334 = bool(args.use_6334_split) and any("6334_split" in (r or {}) for r in eval_rows)
    metrics, correct, total = mquake_utils.cal_accuracy(
        eval_rows, raw_answer_dict, int(args.edit_num), use_6334=use_6334
    )

    report = {
        "data_path": args.data_path,
        "model_path": args.model_path,
        "embed_model": str(args.embed_model),
        "embed_device": str(args.embed_device),
        "entity_thr": float(args.entity_thr),
        "rel_thr": float(args.rel_thr),
        "edit_num": int(args.edit_num),
        "use_6334_split": bool(args.use_6334_split),
        "max_records": int(args.max_records),
        "max_hops": int(args.max_hops),
        "metrics": metrics,
        "n_eval": len(eval_rows),
        "n_edited_cases": len(edited_case_ids),
        "n_facts": len(facts),
        "decompose_stats": rel_stats,
        "subject_empty": sum(1 for s in cur_s if not _ws(s)),
    }

    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        raw_path = Path(args.output_path).with_suffix(".raw_answers.json")
        raw_path.write_text(json.dumps(raw_answer_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
