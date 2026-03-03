from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _ws(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _strip_answer(ans: str) -> str:
    s = str(ans or "").strip()
    s = re.sub(r"^(answer\\s*:\\s*)", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^(the\\s+answer\\s+is\\s+)", "", s, flags=re.IGNORECASE).strip()
    s = s.splitlines()[0].strip() if s else ""
    s = s.rstrip(" .\n\t\r")
    return s.strip()


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


def _rel_template_from_cloze(cloze: str, *, subject_hint: str) -> str:
    c = _ws(cloze)
    if not c:
        return ""
    if "{}" in c:
        return c
    if subject_hint and subject_hint in c:
        return _ws(c.replace(subject_hint, "{}", 1))
    # Fallback: if cloze looks like "<ent> <relation words>", template by taking first token span
    # up to the first verb/preposition is brittle; keep conservative.
    return c


def _initial_subject(rec: Dict[str, Any]) -> str:
    triples = rec.get("new_triples_labeled") or rec.get("orig_triples_labeled") or []
    if isinstance(triples, list) and triples and isinstance(triples[0], list) and triples[0]:
        return _ws(triples[0][0])
    # Fallback: try to extract from first hop cloze by removing trailing relation words.
    hops = rec.get("new_single_hops") or rec.get("single_hops") or []
    if isinstance(hops, list) and hops and isinstance(hops[0], dict):
        cloze = _ws(hops[0].get("cloze", ""))
        if cloze:
            # naive: subject is first two tokens if looks like name; better than empty.
            toks = cloze.split()
            return " ".join(toks[: min(4, len(toks))])
    return ""


def _hop_subject_hint(rec: Dict[str, Any], hop_idx: int) -> str:
    triples = rec.get("new_triples_labeled") or []
    if isinstance(triples, list) and hop_idx < len(triples):
        t = triples[hop_idx]
        if isinstance(t, list) and t:
            return _ws(t[0])
    hops = rec.get("new_single_hops") or rec.get("single_hops") or []
    if isinstance(hops, list) and hop_idx < len(hops):
        h = hops[hop_idx]
        if isinstance(h, dict):
            cloze = _ws(h.get("cloze", ""))
            # best-effort: if cloze begins with an entity-like span before ' is/ was/ are'
            for cue in (" is ", " was ", " are ", " were ", " has ", " have "):
                if cue in f" {cloze} ":
                    return _ws(cloze.split(cue.strip())[0])
    return ""


def _hop_templates(rec: Dict[str, Any]) -> List[str]:
    hops = rec.get("new_single_hops") or rec.get("single_hops") or []
    if not isinstance(hops, list):
        return []
    rels: List[str] = []
    for i, h in enumerate(hops):
        if not isinstance(h, dict):
            continue
        cloze = _ws(h.get("cloze", ""))
        hint = _hop_subject_hint(rec, i)
        rels.append(_rel_template_from_cloze(cloze, subject_hint=hint))
    return [r for r in rels if r]


def _build_edit_bank(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], str]:
    bank: Dict[Tuple[str, str], str] = {}
    for rec in rows:
        s = _ws(rec.get("subject", ""))
        r = _ws(rec.get("edit_prompt_template", ""))
        o = _ws(rec.get("target_new", ""))
        if s and r and o and "{}" in r:
            bank[(s, r)] = o
    return bank


def _format_fill_prompt(subject: str, rel_template: str) -> str:
    st = rel_template
    if "{}" in st:
        try:
            st = st.format(subject)
        except Exception:
            st = st.replace("{}", subject)
    # Avoid adding punctuation that may shift exact-string outputs.
    return (
        "Complete the statement with the missing object. Respond with only the object.\n"
        f"{st} ____\n"
        "Answer:"
    )


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
    parser.add_argument("--max_tokens", type=int, default=8)
    parser.add_argument("--max_records", type=int, default=0, help="0 = all")
    parser.add_argument("--cuda", type=str, default="")
    parser.add_argument("--use_memory", action="store_true", help="Override hop answer if (s, r) in edit bank.")
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    rows = _load_json(args.data_path)
    if int(args.max_records) > 0:
        rows = rows[: int(args.max_records)]

    edit_bank = _build_edit_bank(rows) if bool(args.use_memory) else {}

    llm = _make_vllm(args.model_path, tp_size=int(args.tp_size))
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    stop = ["\n", ".", tok.eos_token] if tok.eos_token else ["\n", "."]

    # Precompute hop templates per record.
    hop_rels: List[List[str]] = []
    for rec in rows:
        hop_rels.append(_hop_templates(rec))
    max_hops = max((len(r) for r in hop_rels), default=0)

    # Current subject/entity per record.
    cur: List[str] = [_initial_subject(rec) for rec in rows]

    # Walk hop-by-hop; batch all records at the same hop.
    for hop_idx in range(max_hops):
        prompts: List[str] = []
        meta: List[int] = []
        # Apply memory hits first; queue misses for LLM.
        for i, rec in enumerate(rows):
            rels = hop_rels[i]
            if hop_idx >= len(rels):
                continue
            s = _ws(cur[i])
            r = _ws(rels[hop_idx])
            if not (s and r and "{}" in r):
                continue
            if edit_bank and (s, r) in edit_bank:
                cur[i] = edit_bank[(s, r)]
                continue
            prompts.append(_format_fill_prompt(s, r))
            meta.append(i)

        if prompts:
            preds = _vllm_generate(llm, prompts, max_tokens=int(args.max_tokens), stop=stop)
            for i, pred in zip(meta, preds):
                ans = _strip_answer(pred)
                if ans:
                    cur[i] = ans

    # Final predicted answer is the entity after the last hop.
    agg = Agg()
    for rec, pred in zip(rows, cur):
        agg.add(_check_answer_new(rec, pred))

    report = {
        "data_path": args.data_path,
        "model_path": args.model_path,
        "max_records": int(args.max_records),
        "use_memory": bool(args.use_memory),
        "max_tokens": int(args.max_tokens),
        "acc": agg.to_dict(),
    }
    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()

