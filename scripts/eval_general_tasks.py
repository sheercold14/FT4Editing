from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def _now_tag() -> str:
    return datetime.now().strftime("%m%d_%H%M%S")


def _require_lm_eval() -> None:
    if shutil.which("lm_eval") is not None:
        return
    # The CLI may not be on PATH if installed as a module, but -m will still work.
    try:
        subprocess.run(["python", "-c", "import lm_eval"], check=True, capture_output=True, text=True)
        return
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "lm-evaluation-harness is not installed. Install with: pip install lm-eval"
        ) from exc


def run_lm_eval(
    model_path: str,
    tasks: List[str],
    backend: str,
    tp_size: int,
    batch_size: int,
    limit: Optional[int],
    output_path: str,
    extra_model_args: str = "",
) -> Dict:
    _require_lm_eval()

    model_args_parts = [f"pretrained={model_path}"]
    if backend == "vllm":
        model_args_parts.append(f"tensor_parallel_size={tp_size}")
        model_args_parts.append(f"gpu_memory_utilization=0.90")
        model_args_parts.append("trust_remote_code=True")
    elif backend == "hf":
        model_args_parts.append("trust_remote_code=True")
    else:
        raise ValueError(f"Unknown backend={backend} (expected vllm|hf)")

    if extra_model_args:
        model_args_parts.append(extra_model_args.strip().lstrip(","))

    cmd = [
        "python",
        "-m",
        "lm_eval",
        "--model",
        backend,
        "--model_args",
        ",".join(model_args_parts),
        "--tasks",
        ",".join(tasks),
        "--output_path",
        output_path,
        "--write_out",
    ]
    if batch_size > 0:
        cmd += ["--batch_size", str(batch_size)]
    if limit is not None:
        cmd += ["--limit", str(limit)]

    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    result = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    if proc.returncode != 0:
        raise RuntimeError(f"lm_eval failed (rc={proc.returncode}). Stderr tail:\n{result['stderr_tail']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, default="")
    # Paper-aligned defaults: GSM8K, MMLU, WMT16 (use one representative direction by default).
    parser.add_argument("--tasks", type=str, default="mmlu,gsm8k,wmt16-en-de")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "hf"])
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cuda", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="runs/general_tasks")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--extra_model_args", type=str, default="")
    args = parser.parse_args()

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag or _now_tag()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    payload = {"tasks": tasks, "backend": args.backend, "tp_size": args.tp_size, "batch_size": args.batch_size, "limit": args.limit}

    model_out = str(out_dir / f"{Path(args.model_path).name}_{tag}.json")
    payload["edited"] = run_lm_eval(
        model_path=args.model_path,
        tasks=tasks,
        backend=args.backend,
        tp_size=args.tp_size,
        batch_size=args.batch_size,
        limit=args.limit,
        output_path=model_out,
        extra_model_args=args.extra_model_args,
    )

    if args.base_model_path:
        base_out = str(out_dir / f"{Path(args.base_model_path).name}_BASE_{tag}.json")
        payload["base"] = run_lm_eval(
            model_path=args.base_model_path,
            tasks=tasks,
            backend=args.backend,
            tp_size=args.tp_size,
            batch_size=args.batch_size,
            limit=args.limit,
            output_path=base_out,
            extra_model_args=args.extra_model_args,
        )

    meta_path = out_dir / f"meta_{Path(args.model_path).name}_{tag}.json"
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(str(meta_path))


if __name__ == "__main__":
    main()
