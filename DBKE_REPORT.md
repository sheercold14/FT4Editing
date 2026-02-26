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
