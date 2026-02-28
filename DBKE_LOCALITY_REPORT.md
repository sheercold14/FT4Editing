# DBKE — Capability / Locality (Catastrophic Forgetting) Eval

This repo’s baseline paper setup uses **lm-evaluation-harness** to measure capability regression (“locality” in the broad sense). In this workspace we run the same *task set* but (for iteration speed) with `--limit` (so these are **smoke** results, not publishable final numbers).

## 1) Quick-run results (limit=10)

Tasks: `mmlu,gsm8k,sst2,wmt16-de-en` (vLLM backend).

Numbers are **Edited score** with **(Δ vs Base)** in parentheses.

| run | mmlu | gsm8k | sst2 | wmt16-de-en |
|---|---:|---:|---:|---:|
| counterfact3k_stage2_suitev1_from_baseline_sft_gate_loc_cf_l10 | 0.5410 (-0.0018) | 0.4000 (+0.0000) | 0.7000 (+0.1000) | 25.6394 (+2.5343) |
| wikibigedit3k_stage2_suitev1_from_baseline_sft_gate_loc_wbe_l10_bs16 | 0.5096 (-0.0019) | 0.5000 (+0.1000) | 0.6000 (+0.1000) | 6.2233 (+1.2559) |
| zsre3k_stage2_e5_sft_gate_norephrase_loc_zsre_l10_bs16 | 0.5297 (-0.0113) | 0.3000 (+0.0000) | 1.0000 (+0.0000) | 28.0638 (-2.8300) |

**Interpretation (for limit=10 only)**: deltas are small and mixed; no obvious catastrophic forgetting signal under this quick slice. For MMLU, the observed drops are within the expected noise floor at this tiny limit.

## 2) How to reproduce (commands)

Env:
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ftedit`

Run one dataset (edited vs base, same GPU) — CounterFact example:
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/eval_general_tasks.py \
  --model_path saves/counterfact3k_stage2_suitev1_from_baseline_sft_gate \
  --base_model_path saves/baseline_qwen3_counterfact3k \
  --tasks mmlu,gsm8k,sst2,wmt16-de-en \
  --backend vllm --tp_size 1 --batch_size 16 --limit 10 \
  --tag loc_cf_l10_bs16 --out_dir runs/locality_general_tasks
```

Summarize all meta files under a tag substring:
```bash
python scripts/summarize_locality_results.py \
  --meta_dir runs/locality_general_tasks \
  --tag l10 \
  --out_md runs/locality_general_tasks/summary_l10.md
```

## 3) Notes / gotchas

- **OOM**: `batch_size=30` + `gpu_memory_utilization=0.90` can OOM on vLLM loglikelihood (MMLU is worst). Use `batch_size=16` and keep `gpu_memory_utilization=0.80` (script default) for stability on 24GB GPUs.
- **lm-eval version skew**: `lm-eval==0.4.3` breaks with `datasets==4.x` due to `trust_remote_code` API changes. We upgraded to `lm-eval==0.4.11` in the `ftedit` env.
- **Publishable runs**: remove `--limit` and follow paper’s exact sampling/few-shot; also record decoding/stopping and exact checkpoints.

