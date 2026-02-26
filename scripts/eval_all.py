from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run_cmd(command: list[str]) -> dict:
    proc = subprocess.run(command, check=False, text=True, capture_output=True)
    return {"code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "cmd": command}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--edit_eval_data", required=True, type=str)
    parser.add_argument("--on_policy_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=300)
    args = parser.parse_args()

    edit_cmd = [
        "python",
        "-m",
        "eval.off_policy_eval",
        "--model_path",
        args.model_path,
        "--data_path",
        args.edit_eval_data,
        "--num_samples",
        str(args.num_samples),
    ]
    on_cmd = ["python", "-m", "eval.on_policy_eval", "--on_policy_path", args.on_policy_path]

    edit_result = run_cmd(edit_cmd)
    on_result = run_cmd(on_cmd)
    report = {"edit_eval": edit_result, "on_policy_eval": on_result}

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"saved": args.output_path, "edit_code": edit_result["code"], "on_code": on_result["code"]}, indent=2))


if __name__ == "__main__":
    main()
