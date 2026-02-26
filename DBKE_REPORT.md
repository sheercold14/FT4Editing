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
Metrics printed by the runner:

| model | edit_success (EM) | edit_success_contains | locality (EM) | locality_contains | trigger_error_rate | conflict_mass |
|---|---:|---:|---:|---:|---:|---:|
| long_e0_off_sft | 0.0000 | 0.0600 | 0.0000 | 0.0000 | 0.9500 | 0.9500 |
| long_e3_mix_dpo | 0.0020 | 0.0640 | 0.0000 | 0.0000 | 0.9513 | 0.9513 |
| long_e5_dynamic_gate | 0.0000 | 0.0660 | 0.0000 | 0.0000 | 0.9494 | 0.9494 |

## Interpretation vs research objective
1) **The system is runnable end-to-end**, but **editing effectiveness is still low** on this setup.
   - Strict EM is ~0–0.2% and “contains” success is ~6–6.6% on 500 off-policy samples.
   - On-policy trigger error remains ~95% (model outputs don’t contain the target answer on most triggers).

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

