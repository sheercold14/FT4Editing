from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import os
import tempfile

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.datasets import OffPolicySFTDataset, RecordMapper, dump_jsonl
from on_policy.conflict_score import compute_conflict_score
from on_policy.editor_interface import RuleBasedEditor
from on_policy.paraphrase_generator import generate_paraphrase_triggers
from on_policy.rollout import rollout
from on_policy.trigger_generator import generate_triggers


def _count_jsonl_lines(path: str, max_lines: int = 1_000_000) -> int:
    try:
        lines = 0
        with open(path, "r", encoding="utf-8") as handle:
            for _ in handle:
                lines += 1
                if lines >= max_lines:
                    break
        return lines
    except FileNotFoundError:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--off_data_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--trigger_mode", type=str, default="template", choices=["template", "template_paraphrase"])
    parser.add_argument("--template_triggers", type=int, default=2)
    parser.add_argument("--paraphrase_per_prompt", type=int, default=0)
    parser.add_argument("--paraphrase_similarity_threshold", type=float, default=0.92)
    parser.add_argument(
        "--include_rephrase",
        action="store_true",
        help="Include dataset-provided rephrase prompts as triggers (can leak eval prompts).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--force", action="store_true", help="Regenerate even if output seems complete.")
    parser.add_argument(
        "--score_mode",
        type=str,
        default="mismatch",
        choices=["mismatch", "full"],
        help="Conflict score mode: mismatch-only (fast) or full (includes logprob margins; slow).",
    )
    parser.add_argument("--progress_every", type=int, default=50)
    args = parser.parse_args()

    off_ds = OffPolicySFTDataset(args.off_data_path, mapper=RecordMapper())
    records = off_ds.records[: args.max_samples] if args.max_samples > 0 else off_ds.records
    expected_rows = len(records) * args.k
    existing_rows = _count_jsonl_lines(args.output_path)
    if (not args.force) and existing_rows >= expected_rows and expected_rows > 0:
        print(
            json.dumps(
                {
                    "output_path": args.output_path,
                    "num_rows": existing_rows,
                    "status": "already_complete",
                    "expected_rows": expected_rows,
                },
                ensure_ascii=False,
            )
        )
        return

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "left"
    torch_dtype = torch.float16 if device.type == "cuda" else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch_dtype
    ).to(device)
    model.eval()

    editor = RuleBasedEditor()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    rows_written = 0

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(output_path.parent),
        prefix=output_path.name + ".",
        suffix=".tmp",
    ) as handle:
        tmp_path = Path(handle.name)
        for idx, rec in enumerate(records):
            if args.progress_every > 0 and idx > 0 and idx % args.progress_every == 0:
                elapsed = max(1e-6, time.time() - start_time)
                rate = rows_written / elapsed
                remaining = (expected_rows - rows_written) / max(1e-6, rate)
                print(
                    json.dumps(
                        {
                            "status": "progress",
                            "records_done": idx,
                            "records_total": len(records),
                            "rows_written": rows_written,
                            "rows_expected": expected_rows,
                            "rows_per_sec": round(rate, 3),
                            "eta_sec": int(remaining),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

            k_template = int(args.k)
            if args.trigger_mode == "template_paraphrase":
                k_template = max(1, min(int(args.template_triggers), int(args.k)))
            prompts = generate_triggers(rec, k_template, include_rephrase=args.include_rephrase)
            if args.trigger_mode == "template_paraphrase" and args.paraphrase_per_prompt > 0 and prompts:
                base_prompts = list(prompts)
                protected = {}
                if not args.include_rephrase:
                    rp = (rec.get("rephrase_prompt") or rec.get("rephrase") or "").strip()
                    if rp:
                        for p in base_prompts:
                            protected.setdefault(p, []).append(rp)
                paraphrases = generate_paraphrase_triggers(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=base_prompts,
                    per_prompt=int(args.paraphrase_per_prompt),
                    gen_cfg={"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 64, "batch_size": 16},
                    seed=int(args.seed) + 1337 + idx,
                    similarity_threshold=float(args.paraphrase_similarity_threshold),
                    filter_against=protected,
                )
                for base, group in zip(base_prompts, paraphrases):
                    for cand in group:
                        if len(prompts) >= int(args.k):
                            break
                        if cand.strip():
                            prompts.append(cand.strip())
            if not prompts:
                continue
            preds = rollout(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                gen_cfg={"temperature": 0.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 32, "batch_size": 16},
                seed=args.seed + idx,
            )
            y_pos = rec.get("target_new", "")
            edits = [editor.edit(prompt, pred, rec) for prompt, pred in zip(prompts, preds)]
            if args.score_mode == "full":
                score = compute_conflict_score(prompts, preds, y_pos, model=model, tokenizer=tokenizer)
            else:
                score = compute_conflict_score(prompts, preds, y_pos)
            for prompt, pred, chosen in zip(prompts, preds, edits):
                row = {
                    "edit_id": rec.get("case_id", idx),
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": pred,
                    "target_new": y_pos,
                    "conflict_score": score["score"],
                    "mismatch_rate": score["mismatch_rate"],
                    "margin": score["margin"],
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_written += 1

        handle.flush()
        os.fsync(handle.fileno())

    tmp_path.replace(output_path)
    print(json.dumps({"output_path": str(output_path), "num_rows": rows_written}, ensure_ascii=False))


if __name__ == "__main__":
    main()
