# DBKE (Dynamic Batchwise Knowledge Editing) — Run Report

## Research objective (restated)
Build a pure fine-tuning, batchwise editing system that **dynamically chooses on-policy vs off-policy updates** per edit to reduce **on-policy conflicts** while keeping **off-policy edit success** and **locality**.

This branch implements an end-to-end runnable prototype and runs E0/E3/E5 on ZsRE-style edits (repo’s `data/zsre/zsre_3k.json`).

## What we implemented
- **On-policy pipeline**
  - Trigger generation: `on_policy/trigger_generator.py`
  - Model rollout: `on_policy/rollout.py` (batched greedy rollout)
  - Conflict scoring: `on_policy/conflict_score.py`
  - Dataset build script: `scripts/build_on_policy_dataset.py` (streaming JSONL, progress, fast mismatch scoring by default)
- **Training loop (single loop, switchable objective + dynamic gate)**
  - `train/train_patch.py`
  - Supports CE (SFT), DPO, fixed mix, and threshold gate (`tau_low`, `tau_high`) with micro-batching.
  - DPO optimized to compute chosen/rejected logps in one pass per model (`losses/objectives.py`).
- **Evaluation**
  - Off-policy: `eval/off_policy_eval.py` (reports both strict EM and “contains” success)
  - On-policy: `eval/on_policy_eval.py` (trigger error + conflict mass; conflict mass falls back to mismatch proxy when margins aren’t computed)
- **Runnable experiment drivers**
  - Short run: `scripts/run_dbke_e0_e3_e5.sh`
  - Longer run: `scripts/run_dbke_long_e0_e3_e5.sh` (lock file `.dbke_long.lock`)

## Experiments we ran (GPU)
Long run script:
- `scripts/run_dbke_long_e0_e3_e5.sh`
- Output log: `logs/run_dbke_long_20260226_123740.log`
- Output JSON: `reports/dbke_long_20260226_123740.json`

Config summary (long run):
- Base on-policy dataset: 600 edits × 4 triggers = 2400 preference rows (mismatch-only conflict scoring).
- E0: off-policy SFT only (`configs/long_e0.yaml`)
- E3: fixed mix, on-policy DPO + off-policy CE (`configs/long_e3.yaml`)
- E5: dynamic gate (`configs/long_e5.yaml`)
- Per-model on-policy eval set: 400 edits × 4 triggers = 1600 rows
- Off-policy eval set: 500 samples

## Results (ZsRE)
### First long run (before loss masking fix)
This run produced near-zero strict EM and ~6% “contains” success. The primary root cause was a **loss masking bug** in `losses/objectives.py` when the tokenizer used right-padding (target tokens were masked incorrectly). This made learning much weaker than intended.

### Rerun (after loss masking fix)
After fixing masking to respect `tokenizer.padding_side` and setting training/eval tokenizers to left padding, strict EM increased substantially.

Run artifacts:
- Log: `logs/run_dbke_long_rerun_20260226_141551.log`
- JSON: `reports/dbke_long_rerun_20260226_141551.json`

Metrics printed by the runner (off-policy eval `num_samples=500`, on-policy eval `400×4=1600`):

| model | edit_success (EM) | edit_success_contains | locality (EM) | locality_contains | trigger_error_rate | conflict_mass |
|---|---:|---:|---:|---:|---:|---:|
| long_e0_off_sft | 0.0880 | 0.1380 | 0.0000 | 0.0000 | 0.9125 | 0.9125 |
| long_e3_mix_dpo | 0.0540 | 0.0800 | 0.0000 | 0.0000 | 0.9338 | 0.9338 |
| long_e5_dynamic_gate | 0.0520 | 0.0780 | 0.0000 | 0.0000 | 0.9288 | 0.9288 |

## Baseline reproduction + full-scale (3k) DBKE results
We aligned our “Edit Success” reporting with the **repo’s official evaluation script** `eval_edit_metric.py` (vLLM greedy EM with normalization), which reports:
- **Reliability (Src) EM** on `src` prompts
- **Generalization (Rephrase) EM** on `rephrase` prompts

### Repo baseline (paper script)
Fine-tune command (repo):
- `python fine-tune.py --config_path ./hparams/qwen3-1.7b_zsre3k_baseline.yaml`

Eval command (repo):
- `CUDA_VISIBLE_DEVICES=0 python eval_edit_metric.py --data_path ./data/zsre/zsre_3k.json --model_path ./saves/baseline_qwen3_zsre3k --tp_size 1 --num_samples 3000 --seed 0 --quiet --output_path ./runs/baseline_qwen3_zsre3k_eval3000_seed0.json`

Result (3000 samples):
- Reliability (Src) EM: **0.9743**
- Generalization (Rephrase) EM: **0.5073**

### Full-scale DBKE runs (this branch)
Key changes vs the earlier “long run” prototype:
- Training targets now add **leading space + EOS** (same stop-encouragement trick as the repo baseline).
- Mixed sampling now uses an **infinite shuffled stream** for off-policy indices (no replacement within a cycle), which is important when `total_steps ≈ dataset_size`.
- On-policy dataset rebuild is now **batched** (single rollout over ~12k triggers) and prints progress.

Configs / artifacts:
- E0 (off-policy SFT): `configs/zsre3k_full_e0.yaml` → `saves/zsre3k_full_e0_off_sft` → `runs/zsre3k_full_e0_eval3000_seed0.json`
- E3 (fixed mix, DPO): `configs/zsre3k_full_e3.yaml` → `saves/zsre3k_full_e3_mix_dpo` → `runs/zsre3k_full_e3_eval3000_seed0.json`
- E5 (dynamic gate, DPO): `configs/zsre3k_full_e5.yaml` → `saves/zsre3k_full_e5_dynamic_gate` → `runs/zsre3k_full_e5_eval3000_seed0.json`
- E5 (stage-2, dynamic gate + **on-policy SFT**, starts from E0): `configs/zsre3k_stage2_e5_sft.yaml` → `saves/zsre3k_stage2_e5_sft_gate` → `runs/zsre3k_stage2_e5_sft_gate_eval3000_seed0.json`

Results (3000 samples):

| method | model | Reliability (Src) EM | Generalization (Rephrase) EM |
|---|---|---:|---:|
| Repo baseline | `./saves/baseline_qwen3_zsre3k` | 0.9743 | 0.5073 |
| DBKE E0 | `./saves/zsre3k_full_e0_off_sft` | **0.9950** | 0.4943 |
| DBKE E3 | `./saves/zsre3k_full_e3_mix_dpo` | 0.9153 | 0.5253 |
| DBKE E5 (DPO gate) | `./saves/zsre3k_full_e5_dynamic_gate` | 0.9133 | 0.5227 |
| DBKE E5 (stage-2, SFT gate) | `./saves/zsre3k_stage2_e5_sft_gate` | **0.9960** | **0.6690** |

### Important note: rephrase prompt leakage
The ZsRE records include a dataset-provided `rephrase` prompt that is used by `eval_edit_metric.py` to measure "generalization".
If we include these rephrase prompts as **training triggers**, we contaminate the eval set and inflate Rephrase EM.

We fixed this by defaulting triggers to **exclude dataset-provided rephrases** (`include_rephrase_triggers: false`).

Rerun (no leakage) for stage-2 on ZsRE-3k:
- `./saves/zsre3k_stage2_e5_sft_gate_norephrase`: Reliability **0.9960**, Rephrase **0.5197** (`runs/zsre3k_stage2_e5_sft_gate_norephrase_eval3000_seed0.json`)

### Takeaway vs research objective
- Pure off-policy SFT (E0) matches/exceeds baseline reliability, but **does not improve rephrase generalization**.
- Fixed-mix / gated **on-policy DPO** improves generalization slightly, but **hurts reliability** (trade-off).
- A **dynamic gate** that uses a **milder on-policy objective (SFT)** in a stage-2 run improves Rephrase EM **without** using dataset-provided rephrase prompts (small but real gain on ZsRE-3k).

## Interpretation vs research objective
1) **The system is runnable end-to-end**, and after fixing loss masking, **off-policy edit success is measurable**.
   - E0 reaches **8.8% strict EM** and **13.8% contains** on 500 samples.
   - On-policy trigger error remains high (~91–93%), so the main remaining problem is on-policy conflict suppression.

2) **Dynamic gate didn’t produce a large separation** vs fixed mix in this run.
   - A key reason: the conflict score used in this run is **mismatch-only**, so most samples are “high conflict”.
   - That makes the gate behave close to a constant-weight mix (i.e., little real “dynamic” behavior).

3) **Likely bottlenecks**
   - **Model capacity / parameter budget**: we intentionally restrict trainable parameters to a narrow module slice for locality; it may be too small for thousands of edits.
   - **Hyperparameters**: LR/steps may still be insufficient for noticeable behavior shift.
   - **Scoring signal**: mismatch-only score is too coarse; need a softer confidence/margin signal (but full margins were expensive).
   - **Evaluation strictness**: EM is brittle; “contains” is more informative here and should be the primary early metric.

## Next runs to make results more analyzable
If we want E5 to show its intended advantage (reduce on-policy conflicts without sacrificing off-policy success), prioritize:
1) **Use a softer conflict score** (e.g., mismatch + positive-target NLL) to avoid score saturation and make the gate actually switch regimes.
2) **Increase effective learning signal**:
   - More steps or slightly higher LR, and/or unfreeze a bit more capacity (multiple layers or wider module).
3) **Add a small general/ability set** for regression monitoring (optional but aligns with “locality” story).
