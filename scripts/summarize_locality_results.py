from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _first_metric(d: Dict[str, Any], preferred_prefixes: Iterable[str]) -> Optional[Tuple[str, float]]:
    for prefix in preferred_prefixes:
        for key, val in d.items():
            if key.endswith("_stderr,none") or key.endswith("_stderr"):
                continue
            if key.startswith(prefix) and isinstance(val, (int, float)):
                return key, float(val)
    # fallback: any numeric metric (excluding stderr)
    for key, val in d.items():
        if "stderr" in key:
            continue
        if isinstance(val, (int, float)):
            return key, float(val)
    return None


def _extract_task_metric(result_json: Dict[str, Any], task: str) -> Optional[float]:
    results = result_json.get("results") or {}
    if task not in results:
        return None
    metric = _first_metric(results[task], preferred_prefixes=("acc", "exact_match", "bleu", "chrf"))
    if metric is None:
        return None
    return metric[1]


def _extract_mmlu_mean(result_json: Dict[str, Any]) -> Optional[float]:
    results = result_json.get("results") or {}
    group_subtasks = result_json.get("group_subtasks") or {}
    subtasks = group_subtasks.get("mmlu")
    if isinstance(subtasks, list) and subtasks:
        vals: List[float] = []
        for sub in subtasks:
            if sub not in results:
                continue
            metric = _first_metric(results[sub], preferred_prefixes=("acc",))
            if metric is not None:
                vals.append(metric[1])
        if vals:
            return sum(vals) / len(vals)
    # If lm-eval exposes an aggregated "mmlu" entry, use it.
    return _extract_task_metric(result_json, "mmlu")


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    return f"{x:.4f}"


def _summarize_pair(edited_json_path: Path, base_json_path: Optional[Path]) -> Dict[str, Any]:
    edited = _load_json(edited_json_path)
    base = _load_json(base_json_path) if base_json_path else None

    def get_all(d: Dict[str, Any]) -> Dict[str, Optional[float]]:
        return {
            "mmlu": _extract_mmlu_mean(d),
            "gsm8k": _extract_task_metric(d, "gsm8k"),
            "sst2": _extract_task_metric(d, "sst2"),
            "wmt16-de-en": _extract_task_metric(d, "wmt16-de-en"),
        }

    edited_m = get_all(edited)
    base_m = get_all(base) if base else {k: None for k in edited_m}

    out: Dict[str, Any] = {"edited": edited_m, "base": base_m}
    out["paths"] = {"edited": str(edited_json_path), "base": str(base_json_path) if base_json_path else ""}
    return out


def _find_meta_files(meta_dir: Path, tag_substr: str) -> List[Path]:
    metas = sorted(meta_dir.glob("meta_*.json"))
    if not tag_substr:
        return metas
    return [p for p in metas if tag_substr in p.name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_dir", type=str, default="runs/locality_general_tasks")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--out_md", type=str, default="")
    args = parser.parse_args()

    meta_dir = Path(args.meta_dir)
    metas = _find_meta_files(meta_dir, args.tag)
    if not metas:
        raise SystemExit(f"No meta files found under {meta_dir} (tag filter='{args.tag}').")

    rows: List[str] = []
    rows.append("| run | mmlu | gsm8k | sst2 | wmt16-de-en |")
    rows.append("|---|---:|---:|---:|---:|")

    for meta_path in metas:
        meta = _load_json(meta_path)
        edited_info = (meta.get("edited") or {})
        base_info = (meta.get("base") or {})
        edited_json = edited_info.get("result_json_path") or ""
        base_json = base_info.get("result_json_path") or ""
        if not edited_json:
            # Best-effort fallback: find newest JSON with prefix "{model_name}_{tag}".
            prefix = meta_path.name.replace("meta_", "").replace(".json", "")
            candidates = sorted(meta_dir.glob(f"{prefix}_*.json"), key=lambda p: p.stat().st_mtime)
            if candidates:
                edited_json = str(candidates[-1])
        if edited_json and (not Path(edited_json).exists()):
            raise SystemExit(f"Missing edited result json: {edited_json} (meta={meta_path})")
        if base_json and (not Path(base_json).exists()):
            base_json = ""

        summary = _summarize_pair(Path(edited_json), Path(base_json) if base_json else None)
        run_name = meta_path.name.replace("meta_", "").replace(".json", "")
        edited_m = summary["edited"]
        base_m = summary["base"]

        # Emit "edited (Δ)" where Δ is vs base when available.
        def cell(k: str) -> str:
            e = edited_m.get(k)
            b = base_m.get(k)
            if e is None:
                return "NA"
            if b is None:
                return _fmt(e)
            return f"{_fmt(e)} ({e - b:+.4f})"

        rows.append(f"| {run_name} | {cell('mmlu')} | {cell('gsm8k')} | {cell('sst2')} | {cell('wmt16-de-en')} |")

    md = "\n".join(rows) + "\n"
    if args.out_md:
        out_path = Path(args.out_md)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
    else:
        print(md)


if __name__ == "__main__":
    main()

