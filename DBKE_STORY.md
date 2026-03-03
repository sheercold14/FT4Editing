# DBKE / FT4Editing — Experiments, Code Map, and Paper Story (Working Draft)

This document consolidates:
1) **What we implemented** (DBKE + evaluation extensions),
2) **Where the code lives** (worktrees/branches + file paths),
3) **What we ran** (datasets, checkpoints, commands, result artifacts),
4) A **reviewer-facing story** with threat-model + rigor notes.

Repo root: `/data/shichao/FT4Editing`

---

## TL;DR (current evidence-based story)

**Observation.** On editing benchmarks, “edit success” decomposes into (at least) three separable requirements:
1) **Write** the new answer on canonical prompts (Reliability),
2) **Trigger** the new answer under realistic formatting/context/prefix distributions (Robustness),
3) Avoid **side effects** (locality / general capability regression; and “old-answer resurgence” where defined).

**Method.** DBKE is a *pure fine-tuning* recipe that treats many failures as (2): we **pull the trigger distribution** toward the edited behavior using rollout-based, on-policy training; and only “hard-write” (off-policy) when conflict is high (dynamic gate).

**Empirical signals we already have.**
- Across 3 in-repo datasets (ZsRE/CounterFact/WikiBigEdit), stage-2 improves **suite-defined robustness axes** (esp. `chat_wrap`, `prefix_noise`, `ctx_irrelevant`) while keeping canonical `Src EM` stable. See `.worktrees/dbke/DBKE_PERF_REPORT.md`.
- Capability regression (lm-eval suite) stays small/mixed at current budgets. See `.worktrees/dbke/DBKE_LOCALITY_REPORT.md`.
- On CHED (context robustness benchmark), CHED-family triggers raise success from `0.40 → 0.64` (eval200), and **holdout-context** still improves (`0.41 → 0.61`), reducing old-answer inclusion. See `.worktrees/ched/runs/*`.

**Conservative claim (hard to refute).** “Editing robustness is distributional; aligning the on-policy trigger distribution yields large gains on the axes you cover, and CHED provides a direct context-robustness validation (including holdout contexts).”

## 0) Repo layout + branches (ground truth)

We use `git worktree` so multiple branches/experiments can coexist under one repo root.

- Root working dir (docs + shared data): `/data/shichao/FT4Editing`
  - Branch: `docs/skill-updates` (contains `SKILL.md` updates and this doc)
- DBKE main implementation + 3-dataset experiments:
  - Worktree: `/data/shichao/FT4Editing/.worktrees/dbke`
  - Branch: `feat/dbke-dynamic-gate`
  - Reports:
    - `.worktrees/dbke/DBKE_PERF_REPORT.md`
    - `.worktrees/dbke/DBKE_LOCALITY_REPORT.md`
- CHED adapter + CHED-specific triggers/eval (external benchmark alignment):
  - Worktree: `/data/shichao/FT4Editing/.worktrees/ched`
  - Branch: `feat/ched-adapter`
  - Scripts + configs added for CHED:
    - `.worktrees/ched/scripts/convert_ched_to_ft4e.py`
    - `.worktrees/ched/scripts/eval_ched.py`
    - `.worktrees/ched/scripts/check_ched_rephrase_leak.py`
    - `.worktrees/ched/configs/ched3k_full_e0.yaml`
    - `.worktrees/ched/configs/ched3k_stage2_e5_gate_sft_chedtrig.yaml`

---

## 1) Method in one page (what we claim, operationally)

### Problem framing
Many “knowledge editing failures” are **not** “the new fact is missing everywhere”; instead, the edit is correct on the canonical prompt, but **fails under the model’s real trigger distribution** (format shifts, prefix noise, irrelevant context, chat wrapper, etc.). These are *distributional* failures.

### DBKE thesis (minimal but testable)
Use a **simple, batchwise fine-tuning** pipeline that dynamically chooses between:
- **off-policy hard-write** (canonical edit prompts + target) when conflict is high, and
- **on-policy alignment** (triggers sampled from rollout distribution + teacher correction) when conflict is low/medium,
to reduce on-policy conflicts while keeping off-policy edit success and preserving locality.

### Implementation components (where)
- Training loop + gate: `.worktrees/dbke/train/train_patch.py`
- On-policy rebuild: `.worktrees/dbke/train/train_patch.py` → `rebuild_on_policy_dataset`
- Conflict score: `.worktrees/dbke/on_policy/conflict_score.py` (mismatch + optional logprob margin)
- Trigger generation (baseline templates + suite-aligned modes): `.worktrees/dbke/on_policy/trigger_generator.py`
- Preference/SFT objectives: `.worktrees/dbke/losses/objectives.py`

---

## 2) Evaluation philosophy (what we measure and why)

We report three views; each answers a different reviewer question.

### 2.1 Repo main metric (fast + widely used in this repo)
Script: `eval_edit_metric.py`  
Outputs: `Reliability (Src) EM` and `Generalization (Rephrase) EM`.

### 2.2 Multi-axis generalization suite (diagnostic, not just one “rephrase”)
Goal: split “generalization” into **controlled failure modes** and measure each.

Where:
- Generate suite: `.worktrees/dbke/scripts/build_generalization_suite.py`
- Eval suite: `.worktrees/dbke/scripts/eval_generalization_suite.py`
- Summary report: `.worktrees/dbke/DBKE_PERF_REPORT.md`

### 2.3 Capability regression / catastrophic forgetting (paper-aligned)
We align to the baseline paper’s setup using **lm-eval-harness**.

Where:
- Eval runner: `.worktrees/dbke/scripts/eval_general_tasks.py`
- Summarizer: `.worktrees/dbke/scripts/summarize_locality_results.py`
- Report: `.worktrees/dbke/DBKE_LOCALITY_REPORT.md`

---

## 3) 3 in-repo datasets: what we ran + where results are

All 3-dataset consolidated results (baseline vs DBKE stage-2) live in:
- `.worktrees/dbke/DBKE_PERF_REPORT.md`

### 3.1 ZsRE-3k (`data/zsre/zsre_3k.json`)
Key point: we use **no-leak** stage-2 checkpoint as primary.

Main metric (from `.worktrees/dbke/DBKE_PERF_REPORT.md`):
- Baseline: `saves/baseline_qwen3_zsre3k` → Rephrase EM `0.5073`
- Stage-2 (no-leak): `saves/zsre3k_stage2_e5_sft_gate_norephrase` → Rephrase EM `0.5197` (+0.0123)

Suite v1 (diagnostic): overall EM `0.5026` → `0.5478` (+0.0452), with large gains on `ctx_irrelevant/prefix_noise/instr_wrap`.

### 3.2 CounterFact-3k (`data/counterfact/counterfact_3k.json`)
Main metric:
- Baseline Rephrase EM `0.1947`
- Stage-2 (suite-aligned triggers): `0.2063` (+0.0117)

Suite v1: overall EM `0.3943` → `0.4538` (+0.0595); biggest gain on `chat_wrap` (+0.2108).

### 3.3 WikiBigEdit-3k (`data/wikibigedit/wikibigedit_3k.json`)
Main metric:
- Baseline Rephrase EM `0.7583`
- Stage-2 (suite-aligned triggers): `0.7643` (+0.0060)

Suite v1: overall EM `0.5914` → `0.6810` (+0.0895); biggest gain on `chat_wrap` (+0.3985).

---

## 4) Locality / forgetting (lm-eval-harness; edited vs baseline)

Full “no limit + NQ” table is in:
- `.worktrees/dbke/DBKE_LOCALITY_REPORT.md`

Takeaway: deltas are small and mixed on the general-task suite, suggesting no obvious catastrophic forgetting at the current edit budgets/configs (but this should be repeated with full publishable settings).

---

## 5) External benchmark alignment: CHED (Context-Robust Knowledge Editing)

### 5.1 Why CHED matters for our paper positioning
If we claim “on-policy triggers reduce conflicts under real usage contexts”, CHED is a **direct stress test**: same edit must succeed under multiple *context prefix families* (SBJ/OBJ_OLD/OBJ_NEW + hop variants), and evaluation requires:
- answer **contains new knowledge** AND
- answer **does not contain old knowledge**.

### 5.2 CHED adapter (where)
Branch: `feat/ched-adapter` in `.worktrees/ched`.

- CoRE repo (downloaded locally): `external/CoRE`
- Convert CHED → FT4Editing schema:
  - Script: `.worktrees/ched/scripts/convert_ched_to_ft4e.py`
  - Output used in experiments: `/data/shichao/FT4Editing/data/ched/ched_3k.json`
- Training triggers:
  - `trigger_mode: ched` wired in:
    - `.worktrees/ched/on_policy/trigger_generator.py`
    - `.worktrees/ched/train/train_patch.py`
- CHED evaluation (bucketed by family):
  - Script: `.worktrees/ched/scripts/eval_ched.py`
  - Leakage self-check:
    - `.worktrees/ched/scripts/check_ched_rephrase_leak.py`

### 5.3 Results (CHED-3k; eval200 smoke)
Artifacts:
- E0: `.worktrees/ched/runs/ched3k_e0_eval200.json`
- Stage-2 (CHED triggers): `.worktrees/ched/runs/ched3k_stage2_sftgate_eval200.json`

Overall success (ctx_offset=0):
- E0: `0.4011` (old_contains `0.1172`)
- Stage-2: `0.6350` (old_contains `0.0650`)

**Holdout-context evaluation** (to address “context leakage” concerns):
`eval_ched.py` supports `--ctx_offset` to evaluate on *different* context sentences than those likely used in training triggers.

Artifacts:
- E0 holdout: `.worktrees/ched/runs/ched3k_e0_eval200_ctxoff2.json`
- Stage-2 holdout: `.worktrees/ched/runs/ched3k_stage2_sftgate_eval200_ctxoff2.json`

Overall success (ctx_offset=2):
- E0: `0.4070` (old_contains `0.1203`)
- Stage-2: `0.6071` (old_contains `0.0697`)

Interpretation: CHED-family triggers mainly improve **context robustness families** (SBJ/OBJ_OLD/OBJ_NEW + hop), while `rephrase` improves only modestly—consistent with the “distribution coverage” hypothesis.

---

## 5.4) Multi-hop knowledge editing: MQuAKE-Remastered (CF3k smoke)

### Why this matters
CHED tests robustness to *context prefixes*, but the answer is still a **single fact string**. Multi-hop editing asks a stronger question:
> after editing one fact, does the model’s *downstream multi-hop answer* update accordingly?

This directly supports the “generalization boundary” story: canonical write success can be high while multi-hop behavior remains broken.

### Adapter + scripts (where)
Worktree: `/data/shichao/FT4Editing/.worktrees/mquake` (branch `feat/mquake-multihop`)

- Convert/download (HF → JSON):
  - `.worktrees/mquake/scripts/convert_mquake_remastered.py`
  - Output used: `/data/shichao/FT4Editing/data/mquake_remastered/mquake_remastered_cf3k.json`
- Build on-policy multi-hop dataset (q0 only; with rollouts as rejected):
  - `.worktrees/mquake/scripts/build_mquake_on_policy_multihop.py`
  - Output: `.worktrees/mquake/data/generated/mquake_cf3k_multihop_on.jsonl`
- Eval (single-hop edit prompt + multi-hop questions):
  - `.worktrees/mquake/scripts/eval_mquake_multihop.py`

### Experiment definition (minimal, story-relevant)
- **Stage-1 (E0)** trains only the *edited single-hop* supervision (`requested_rewrite`) and evaluates multi-hop questions.
- **Stage-2** adds *on-policy-like* supervision on **one** multi-hop question per record (q0), then evaluates on **held-out** paraphrases q1/q2.

### Results (CF3k; eval200 smoke)
Artifacts:
- E0: `.worktrees/mquake/runs/mquake_cf3k_e0_eval200.json`
- Stage-2: `.worktrees/mquake/runs/mquake_cf3k_stage2_eval200.json`

Scores are “match new answer (or aliases)”:
- Single-hop (edited prompt): `0.975 → 0.985` (stable)
- Multi-hop:
  - `mh_q0`: `0.070 → 0.255` (trained index)
  - `mh_q1`: `0.030 → 0.205` (held-out index)
  - `mh_q2`: `0.045 → 0.180` (held-out index)

Interpretation: **writing the edited fact does not propagate to multi-hop answers** (stage-1), but adding multi-hop triggers in stage-2 yields large gains even on held-out question variants—evidence that multi-hop failures are largely a *trigger distribution* mismatch (plus potentially reasoning brittleness), and that “pulling the trigger distribution” is necessary to cross this boundary.

---

## 6) Rigor / threats-to-validity checklist (what reviewers will ask)

### 6.1 Prompt leakage (rephrase)
We explicitly avoid training on dataset-provided rephrases by default (`include_rephrase_triggers: false`) and run self-checks:
- Script: `.worktrees/ched/scripts/check_ched_rephrase_leak.py`
- Current CHED stage-2 on-policy dataset:
  - exact match with any rephrase: 0 / 24000
  - exact match with same-edit rephrase: 0 / 24000

### 6.2 Context-prefix overlap (CHED)
Even if rephrase is clean, CHED has fixed context sentence lists. We therefore report holdout-context results:
- `eval_ched.py --ctx_offset 2` (evaluate on later sentences in each family list).

For the final paper, the rigorous version should do a **disjoint split** of context sentences per record (train vs eval) rather than offset-only.

### 6.3 “Isn’t this just data augmentation?”
We must include ablations that separate:
- trigger coverage effect (suite/CHED triggers),
- dynamic gate effect (fixed mix vs dynamic),
- on-policy objective effect (SFT vs DPO vs OPA-style),
and report the locality tradeoff.

---

## 7) Reviewer-facing story (polished positioning draft)

### Main claim (narrow, defensible)
**Knowledge editing is often distributional.** A successful “write” on canonical prompts does not ensure correct behavior under a model’s trigger distribution. We show that a lightweight, pure fine-tuning stage that **aligns on-policy triggers** can significantly reduce such trigger failures, while maintaining edit reliability and not inducing large capability regression.

### Contribution map
1) **DBKE algorithm**: a simple gate that allocates update budget between off-policy hard-write and on-policy alignment based on a conflict score from rollouts.
2) **Evaluation clarity**: replace “one rephrase metric” with a multi-axis suite (and align to CHED for context robustness).
3) **Empirical insight**: improvements concentrate on the axes we explicitly cover (e.g., chat wrappers, prefix noise, irrelevant context), supporting the distribution-coverage hypothesis.

### Positioning vs “just data augmentation” (how we should phrase it)
It’s valid (and often *correct*) to interpret a large portion of “editing generalization” as **targeted data augmentation over trigger distributions**. The novelty is *not* “we discovered augmentation”, but:
- we tie augmentation to a **measurable conflict signal** (rollout mismatch / margin),
- we show a **budgeted, editing-style fine-tuning** can achieve robustness without large global retraining,
- we separate evaluation into axes, so “generalization” is no longer a single ambiguous scalar.

The paper should explicitly acknowledge: *for some axes, the right solution is better trigger coverage*, and the scientific contribution is to (i) define axes, (ii) show which axes respond to editing-style updates, and (iii) show where the boundary is (axes that remain hard).

### What we do *not* claim (to avoid overreach)
- We do not claim “one-shot editing yields unlimited semantic generalization”.
- We do not claim synthetic triggers are identical to real user prompts; we treat them as a controlled approximation and validate via holdout-context tests (CHED) and multi-axis suites.

---

## 8) Reproduction index (commands)

### Env
`source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ftedit`

### CHED (3k subset)
Convert:
```bash
python .worktrees/ched/scripts/convert_ched_to_ft4e.py \
  --input_path external/CoRE/data/CHED.json \
  --output_path data/ched/ched_3k.json \
  --max_records 3000 --seed 0
```

Train E0:
```bash
CUDA_VISIBLE_DEVICES=0 python -m train.train_patch \
  --config_path .worktrees/ched/configs/ched3k_full_e0.yaml
```

Train stage-2 (CHED triggers):
```bash
CUDA_VISIBLE_DEVICES=0 python -m train.train_patch \
  --config_path .worktrees/ched/configs/ched3k_stage2_e5_gate_sft_chedtrig.yaml
```

Eval (in-context):
```bash
CUDA_VISIBLE_DEVICES=0 python .worktrees/ched/scripts/eval_ched.py \
  --data_path data/ched/ched_3k.json \
  --model_path .worktrees/ched/saves/ched3k_stage2_e5_gate_sft_chedtrig \
  --max_records 200 --per_family_k 1 --max_tokens 8
```

Eval (holdout-context):
```bash
CUDA_VISIBLE_DEVICES=0 python .worktrees/ched/scripts/eval_ched.py \
  --data_path data/ched/ched_3k.json \
  --model_path .worktrees/ched/saves/ched3k_stage2_e5_gate_sft_chedtrig \
  --max_records 200 --per_family_k 1 --ctx_offset 2 --max_tokens 8
```

Leak check:
```bash
python .worktrees/ched/scripts/check_ched_rephrase_leak.py \
  --ched_path data/ched/ched_3k.json \
  --on_policy_path .worktrees/ched/data/generated/ched3k_stage2_on_chedtrig.jsonl
```

---

## 9) Immediate next steps (to maximize “top-tier” acceptance odds)

1) **CHED full 21k**: scale from 3k smoke to full dataset, and run holdout-context split properly.
2) **Ablations to kill “augmentation only” critique**:
   - template triggers vs suite triggers vs CHED triggers,
   - fixed mix vs dynamic gate,
   - on-policy SFT vs on-policy DPO vs OPA-style (if enabled).
3) **Report locality with paper-aligned, publishable settings** (no `--limit`, fixed few-shot, recorded decode/stops).
4) **Generalization boundary**: define which transforms are “editing-appropriate” vs “requires data-driven retraining”, and show failure modes explicitly (suite + CHED).
