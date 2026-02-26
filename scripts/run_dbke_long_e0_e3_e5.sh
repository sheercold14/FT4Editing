#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs reports data/generated

ts="$(date +%Y%m%d_%H%M%S)"
log="logs/run_dbke_long_${ts}.log"
report="reports/dbke_long_${ts}.json"

echo "[START] $(date -Iseconds)" | tee -a "$log"
echo "[INFO] pwd=$(pwd)" | tee -a "$log"
echo "[INFO] log=$log" | tee -a "$log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate ftedit
python -c "import torch; print('[INFO] torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())" | tee -a "$log"

lock=".dbke_long.lock"
if [[ -f "$lock" ]]; then
  old_pid="$(cat "$lock" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "[ERROR] Another long run active (pid=$old_pid). Remove $lock to override." | tee -a "$log"
    exit 2
  fi
fi
echo "$$" > "$lock"
trap 'rm -f "$lock"' EXIT

echo "[STEP] build base on-policy dataset (long)" | tee -a "$log"
python scripts/build_on_policy_dataset.py \
  --model_path /data/shichao/data/Qwen3-1.7B \
  --off_data_path ./data/zsre/zsre_3k.json \
  --output_path ./data/generated/on_policy_zsre_long.jsonl \
  --k 4 --max_samples 600 --device cuda:0 --score_mode mismatch --progress_every 50 | tee -a "$log"

if [[ -d ./saves/long_e0_off_sft ]]; then
  echo "[STEP] train long E0 (skip; exists)" | tee -a "$log"
else
  echo "[STEP] train long E0" | tee -a "$log"
  python scripts/train.py --config_path ./configs/long_e0.yaml | tee -a "$log"
fi

if [[ -d ./saves/long_e3_mix_dpo ]]; then
  echo "[STEP] train long E3 (skip; exists)" | tee -a "$log"
else
  echo "[STEP] train long E3" | tee -a "$log"
  python scripts/train.py --config_path ./configs/long_e3.yaml | tee -a "$log"
fi

if [[ -d ./saves/long_e5_dynamic_gate ]]; then
  echo "[STEP] train long E5 (skip; exists)" | tee -a "$log"
else
  echo "[STEP] train long E5" | tee -a "$log"
  python scripts/train.py --config_path ./configs/long_e5.yaml | tee -a "$log"
fi

models=(long_e0_off_sft long_e3_mix_dpo long_e5_dynamic_gate)
for model in "${models[@]}"; do
  echo "[STEP] on-policy dataset for ${model}" | tee -a "$log"
  python scripts/build_on_policy_dataset.py \
    --model_path "./saves/${model}" \
    --off_data_path ./data/zsre/zsre_3k.json \
    --output_path "./data/generated/${model}_on.jsonl" \
    --k 4 --max_samples 400 --device cuda:0 --score_mode mismatch --force | tee -a "$log"

  echo "[STEP] on-policy eval for ${model}" | tee -a "$log"
  python -m eval.on_policy_eval \
    --on_policy_path "./data/generated/${model}_on.jsonl" \
    --output_path "./saves/${model}/on_policy_metrics.json" | tee -a "$log"

  echo "[STEP] off-policy eval for ${model}" | tee -a "$log"
  python -m eval.off_policy_eval \
    --model_path "./saves/${model}" \
    --data_path ./data/zsre/zsre_3k.json \
    --num_samples 500 \
    --output_path "./saves/${model}/off_policy_metrics.json" | tee -a "$log"
done

echo "[STEP] summarize metrics -> ${report}" | tee -a "$log"
REPORT_PATH="$report" python - <<'PY' | tee -a "$log"
import json, os
models=["long_e0_off_sft","long_e3_mix_dpo","long_e5_dynamic_gate"]
out={}
for m in models:
    out[m]={}
    for kind in ["on_policy","off_policy"]:
        path=f"./saves/{m}/{kind}_metrics.json"
        with open(path,"r",encoding="utf-8") as f:
            out[m][kind]=json.load(f)
report=os.environ["REPORT_PATH"]
with open(report,"w",encoding="utf-8") as f:
    json.dump(out,f,indent=2,ensure_ascii=False)
keys=["edit_success","edit_success_contains","locality","locality_contains","trigger_error_rate","conflict_mass"]
print("model\t" + "\t".join(keys))
for m in models:
    off=out[m]["off_policy"]
    on=out[m]["on_policy"]
    row={
        "edit_success": off.get("edit_success","NA"),
        "edit_success_contains": off.get("edit_success_contains","NA"),
        "locality": off.get("locality","NA"),
        "locality_contains": off.get("locality_contains","NA"),
        "trigger_error_rate": on.get("trigger_error_rate","NA"),
        "conflict_mass": on.get("conflict_mass","NA"),
    }
    def fmt(v):
        return f"{v:.6f}" if isinstance(v,(int,float)) else str(v)
    print(m + "\t" + "\t".join(fmt(row[k]) for k in keys))
print(json.dumps({"report": report}, ensure_ascii=False))
PY

echo "[END] $(date -Iseconds)" | tee -a "$log"
echo "[INFO] log=$log report=$report"
