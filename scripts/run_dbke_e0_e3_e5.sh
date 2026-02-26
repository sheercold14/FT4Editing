#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs reports data/generated

ts="$(date +%Y%m%d_%H%M%S)"
log="logs/run_dbke_${ts}.log"
report="reports/dbke_${ts}.json"

echo "[START] $(date -Iseconds)" | tee -a "$log"
echo "[INFO] pwd=$(pwd)" | tee -a "$log"
echo "[INFO] log=$log" | tee -a "$log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate ftedit
python -c "import torch; print('[INFO] torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())" | tee -a "$log"

echo "[STEP] build base on-policy dataset (skip if complete)" | tee -a "$log"
python scripts/build_on_policy_dataset.py \
  --model_path /data/shichao/data/Qwen3-1.7B \
  --off_data_path ./data/zsre/zsre_3k.json \
  --output_path ./data/generated/on_policy_zsre.jsonl \
  --k 4 --max_samples 200 --device cuda:0 | tee -a "$log"

echo "[STEP] train E0" | tee -a "$log"
python scripts/train.py --config_path ./configs/e0.yaml | tee -a "$log"

echo "[STEP] train E3" | tee -a "$log"
python scripts/train.py --config_path ./configs/e3.yaml | tee -a "$log"

echo "[STEP] train E5" | tee -a "$log"
python scripts/train.py --config_path ./configs/e5.yaml | tee -a "$log"

models=(e0_off_sft e3_mix_dpo e5_dynamic_gate)
for model in "${models[@]}"; do
  echo "[STEP] on-policy dataset for ${model}" | tee -a "$log"
  python scripts/build_on_policy_dataset.py \
    --model_path "./saves/${model}" \
    --off_data_path ./data/zsre/zsre_3k.json \
    --output_path "./data/generated/${model}_on.jsonl" \
    --k 4 --max_samples 100 --device cuda:0 --force | tee -a "$log"

  echo "[STEP] on-policy eval for ${model}" | tee -a "$log"
  python -m eval.on_policy_eval \
    --on_policy_path "./data/generated/${model}_on.jsonl" \
    --output_path "./saves/${model}/on_policy_metrics.json" | tee -a "$log"

  echo "[STEP] off-policy eval for ${model}" | tee -a "$log"
  python -m eval.off_policy_eval \
    --model_path "./saves/${model}" \
    --data_path ./data/zsre/zsre_3k.json \
    --num_samples 200 \
    --output_path "./saves/${model}/off_policy_metrics.json" | tee -a "$log"
done

echo "[STEP] summarize metrics -> ${report}" | tee -a "$log"
python - <<'PY' | tee -a "$log"
import json, os, time
models=["e0_off_sft","e3_mix_dpo","e5_dynamic_gate"]
out={}
for m in models:
    out[m]={}
    for kind in ["on_policy","off_policy"]:
        path=f"./saves/{m}/{kind}_metrics.json"
        with open(path,"r",encoding="utf-8") as f:
            out[m][kind]=json.load(f)
report_path=os.environ.get("REPORT_PATH")
if report_path:
    with open(report_path,"w",encoding="utf-8") as f:
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
PY

REPORT_PATH="$report" python - <<'PY' | tee -a "$log"
import json, os
models=["e0_off_sft","e3_mix_dpo","e5_dynamic_gate"]
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
print(json.dumps({"report": report}, ensure_ascii=False))
PY

echo "[END] $(date -Iseconds)" | tee -a "$log"
echo "[INFO] log=$log report=$report"
