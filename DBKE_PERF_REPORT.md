# DBKE — 3-Dataset Performance Summary (ZsRE / CounterFact / WikiBigEdit)

This report consolidates the **baseline vs DBKE** results on the three in-repo editing datasets:
- `data/zsre/zsre_3k.json`
- `data/counterfact/counterfact_3k.json`
- `data/wikibigedit/wikibigedit_3k.json`

We report two complementary views:
1) **Repo main metric** (`eval_edit_metric.py`): `Reliability (Src) EM` + `Generalization (Rephrase) EM`.
2) **Generalization Suite v1** (deterministic): multi-axis robustness (`chat_wrap`, `ctx_irrelevant`, `prefix_noise`, etc.).

Notes:
- For ZsRE, we use the **no-leak** stage-2 checkpoint (`*_norephrase`) as the primary result.
- For CounterFact/WikiBigEdit, “stage-2 suite_v1” means on-policy rebuild used `trigger_mode: suite_v1` (suite-aligned triggers).

---

## A) Repo main metric (`eval_edit_metric.py`, 3000 samples, seed=0)

| dataset | model | checkpoint | Src EM | Rephrase EM | Δ Rephrase |
|---|---|---|---:|---:|---:|
| ZsRE | baseline | `/data/shichao/FT4Editing/saves/baseline_qwen3_zsre3k` | 0.9743 | 0.5073 | — |
| ZsRE | DBKE stage-2 (no-leak) | `saves/zsre3k_stage2_e5_sft_gate_norephrase` | 0.9960 | 0.5197 | +0.0123 |
| CounterFact | baseline | `saves/baseline_qwen3_counterfact3k` | 0.9977 | 0.1947 | — |
| CounterFact | DBKE stage-2 (suite_v1, gate) | `saves/counterfact3k_stage2_suitev1_from_baseline_sft_gate` | 0.9977 | 0.2063 | +0.0117 |
| CounterFact | DBKE stage-2 (suite_v1, fixedmix) | `saves/counterfact3k_stage2_suitev1_fixedmix_from_baseline_sft` | 0.9977 | 0.2050 | +0.0103 |
| WikiBigEdit | baseline | `saves/baseline_qwen3_wikibigedit3k` | 0.9953 | 0.7583 | — |
| WikiBigEdit | DBKE stage-2 (suite_v1, gate) | `saves/wikibigedit3k_stage2_suitev1_from_baseline_sft_gate` | 0.9957 | 0.7643 | +0.0060 |
| WikiBigEdit | DBKE stage-2 (suite_v1, fixedmix) | `saves/wikibigedit3k_stage2_suitev1_fixedmix_from_baseline_sft` | 0.9953 | 0.7647 | +0.0063 |

Run files (JSON):
- ZsRE baseline: `/data/shichao/FT4Editing/runs/baseline_qwen3_zsre3k_eval3000_seed0.json`
- ZsRE stage-2 no-leak: `runs/zsre3k_stage2_e5_sft_gate_norephrase_eval3000_seed0.json`
- CounterFact baseline: `runs/baseline_qwen3_counterfact3k_eval3000_seed0.json`
- CounterFact suite_v1 stage-2 (gate): `runs/counterfact3k_stage2_suitev1_from_baseline_eval3000_seed0.json`
- CounterFact suite_v1 stage-2 (fixedmix): `runs/counterfact3k_stage2_suitev1_fixedmix_from_baseline_eval3000_seed0.json`
- WikiBigEdit baseline: `runs/baseline_qwen3_wikibigedit3k_eval3000_seed0.json`
- WikiBigEdit suite_v1 stage-2 (gate): `runs/wikibigedit3k_stage2_suitev1_from_baseline_eval3000_seed0.json`
- WikiBigEdit suite_v1 stage-2 (fixedmix): `runs/wikibigedit3k_stage2_suitev1_fixedmix_from_baseline_eval3000_seed0.json`

---

## B) Generalization Suite v1 (EM; baseline vs best DBKE variant)

Suite generation:
- `scripts/build_generalization_suite.py --suite_version v1 ...` → `data/eval_generated/*_suite_v1.jsonl`

Suite evaluation:
- `scripts/eval_generalization_suite.py` → `runs/suite_*_v1_*.json`

### ZsRE (baseline vs stage-2 no-leak)

| metric (EM) | baseline | stage-2 | Δ |
|---|---:|---:|---:|
| overall | 0.5026 | 0.5478 | +0.0452 |
| src | 0.9743 | 0.9957 | +0.0213 |
| instr_wrap | 0.7635 | 0.8451 | +0.0816 |
| chat_wrap | 0.4750 | 0.4348 | -0.0402 |
| ctx_irrelevant | 0.3344 | 0.4192 | +0.0848 |
| prefix_noise | 0.3173 | 0.4067 | +0.0893 |
| prefix_noise_trunc | 0.1623 | 0.2062 | +0.0438 |

Run files:
- `runs/suite_zsre3k_v1_baseline.json`
- `runs/suite_zsre3k_v1_stage2_noleak.json`

### CounterFact (baseline vs stage-2 suite_v1 from baseline, gate)

| metric (EM) | baseline | stage-2 | Δ |
|---|---:|---:|---:|
| overall | 0.3943 | 0.4538 | +0.0595 |
| src | 0.9980 | 0.9977 | -0.0003 |
| instr_wrap | 0.5758 | 0.6183 | +0.0425 |
| chat_wrap | 0.0638 | 0.2747 | +0.2108 |
| ctx_irrelevant | 0.2969 | 0.3372 | +0.0403 |
| prefix_noise | 0.2563 | 0.3040 | +0.0477 |
| prefix_noise_trunc | 0.1440 | 0.1732 | +0.0292 |

Run files:
- `runs/suite_counterfact3k_v1_baseline.json`
- `runs/suite_counterfact3k_v1_suitev1_from_baseline_stage2.json`

### WikiBigEdit (baseline vs stage-2 suite_v1 from baseline, gate)

| metric (EM) | baseline | stage-2 | Δ |
|---|---:|---:|---:|
| overall | 0.5914 | 0.6810 | +0.0895 |
| src | 0.9947 | 0.9947 | +0.0000 |
| instr_wrap | 0.9273 | 0.9396 | +0.0122 |
| chat_wrap | 0.1668 | 0.5653 | +0.3985 |
| ctx_irrelevant | 0.5991 | 0.7096 | +0.1105 |
| prefix_noise | 0.6067 | 0.6827 | +0.0760 |
| prefix_noise_trunc | 0.3292 | 0.3660 | +0.0368 |

Run files:
- `runs/suite_wikibigedit3k_v1_baseline.json`
- `runs/suite_wikibigedit3k_v1_suitev1_from_baseline_stage2.json`

---

## Takeaway (story-ready)
Across three datasets, the best-performing runs follow a consistent pattern:
- **`Src EM` stays stable**, so edits remain reliable.
- The biggest improvements appear when **on-policy triggers are aligned with the target robustness axes** (suite v1), supporting the DBKE thesis that many “edits” are distributional: we can often “pull” the model’s trigger distribution toward correct behavior without heavy overwrite.

