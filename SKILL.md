# FT4Editing / DBKE 实验技能卡（Skill）

适用目录：`/data/shichao/FT4Editing`（以及该 repo 的各个 `git worktree`）。

## 1) 现有 DBKE 实验目标（dbke worktree 已实现）
目标：做一个**纯 fine-tuning、batchwise** 的知识编辑原型，按编辑项的冲突程度在 **on-policy** 与 **off-policy** 更新之间做动态选择（Dynamic Gate），以降低 on-policy 冲突，同时保持 off-policy 的编辑成功率与 locality。

对应实现与跑法（见 worktree：`/data/shichao/FT4Editing/.worktrees/dbke`）：
- Runner（长实验）：`scripts/run_dbke_long_e0_e3_e5.sh`
- 数据：ZsRE 风格编辑 `data/zsre/zsre_3k.json`
- 模型：`/data/shichao/data/Qwen3-1.7B`
- E0：off-policy SFT（CE）
- E3：固定混合（off-policy CE + on-policy DPO）
- E5：Dynamic Gate（阈值 `tau_low/tau_high` 控制权重切换）
- 训练参数“局部化”：默认只解冻一个层的 MLP（例如 `train_layer: 6`，`rewrite_module: model.layers.{}.mlp`）
- 评测：
  - Off-policy：`python -m eval.off_policy_eval ...`（edit_success + locality）
  - On-policy：`python -m eval.on_policy_eval ...`（trigger_error_rate + conflict_mass）
  - 论文/仓库主指标：`eval_edit_metric.py`（vLLM greedy，输出 Reliability(Src) EM / Generalization(Rephrase) EM）

## 1.1) 关键成功经验（ZsRE-3k 已验证有效）
结论先行：把训练/评测对齐到仓库论文设置后，我们的 DBKE 能把 Reliability/Src EM 做到和 baseline 同量级；但 **Generalization/Rephrase EM 必须做“去泄漏（no-leak）检查”**，否则很容易被训练数据污染而“虚高”（尤其当 on-policy triggers 直接包含数据集自带 `rephrase_prompt`/`rephrase` 时）。

**经验 1：必须对齐仓库的生成/停止方式（否则 EM 会“假低”）**
- `target_new` 建议做两步处理：
  1) 若不以空格开头，补一个 leading space（tokenization 细节）
  2) 末尾追加 `eos_token`（鼓励停止，避免生成多余 token 导致 EM 失败）
- 我们在 `.worktrees/dbke/losses/objectives.py` 内部对 CE/DPO targets 统一做了上述处理（chosen 用 EOS；rejected 不加 EOS）。

**经验 2：混合采样不要“带放回抽样”覆盖不全**
- 当 `steps_per_epoch * batch_size ≈ dataset_size` 时，`randrange` 带放回会让不少 edits 永远看不到。
- 我们在 `.worktrees/dbke/data/mixer.py` 改成了“无限 shuffled stream”，确保一个周期内无放回覆盖（对 3k 数据很关键）。

**经验 3：Dynamic Gate 的 on-policy 目标要“够温和”**
- 直接用 on-policy DPO 往往会牺牲 Reliability，提升 Generalization 有但不稳定。
- 有效做法：**先跑 off-policy SFT（E0）把 edits 写进去**，再做 **stage-2 的 dynamic gate + on-policy SFT**（用触发 prompt 拉分布，抑制冲突/提升泛化）。
- 这个 stage-2 方法在 ZsRE-3k 上显著提升 Rephrase EM，并保持/提升 Src EM。

**经验 0（重要）：避免“rephrase prompt 泄漏”导致 Generalization 虚高**
- 很多数据集（ZsRE/CounterFact/WikiBigEdit）自带 `rephrase_prompt` 字段，用于评测“泛化到改写问法”。
- 如果训练/on-policy trigger 里直接包含这条 `rephrase_prompt`，就会出现 **train-test contamination**：Generalization (Rephrase EM) 会被高估。
- 建议默认：训练/构建 triggers **只用 src/prompt + 模板改写**，把 dataset 自带 rephrase 只留给 `eval_edit_metric.py`（作为真正 hold-out 的 generalization）。

**泄漏自检（必须做，写进报告）**
- 目标：确认 `on_policy.jsonl` 的 `prompt` **没有覆盖**评测用的 `rephrase` prompts（覆盖率接近 1.0 基本等同于“generalization contamination”）。
- 一键统计（在对应 worktree 下跑）：
  - `python - <<'PY'\nimport json\nfrom pathlib import Path\n\noff=json.load(open('data/zsre/zsre_3k.json','r',encoding='utf-8'))\nrephrases=set([(d.get('rephrase_prompt') or d.get('rephrase') or '').strip() for d in off])\nrephrases.discard('')\n\non_path=Path('data/generated/zsre3k_stage2_on.jsonl')\non=set()\nwith on_path.open('r',encoding='utf-8') as f:\n    for line in f:\n        if line.strip():\n            on.add(json.loads(line)['prompt'].strip())\n\nhit=len(rephrases & on)\nprint('unique_rephrase',len(rephrases))\nprint('unique_on_prompts',len(on))\nprint('rephrase_in_on_count',hit)\nprint('rephrase_in_on_frac',hit/max(1,len(rephrases)))\nPY`

**不改代码的规避方案（推荐）**
- 训练前，把 on-policy 数据里“等于 rephrase prompt 的行”过滤掉（只保留 src + 模板触发）：
  - `python - <<'PY'\nimport json\nfrom pathlib import Path\n\noff=json.load(open('data/zsre/zsre_3k.json','r',encoding='utf-8'))\nban=set([(d.get('rephrase_prompt') or d.get('rephrase') or '').strip() for d in off])\nban.discard('')\n\ninp=Path('data/generated/zsre3k_stage2_on.jsonl')\nout=Path('data/generated/zsre3k_stage2_on_noleak.jsonl')\nout.parent.mkdir(parents=True, exist_ok=True)\nkept=0\nwith inp.open('r',encoding='utf-8') as fi, out.open('w',encoding='utf-8') as fo:\n    for line in fi:\n        if not line.strip():\n            continue\n        row=json.loads(line)\n        if row.get('prompt','').strip() in ban:\n            continue\n        fo.write(json.dumps(row,ensure_ascii=False)+'\\n')\n        kept+=1\nprint('kept_rows',kept,'->',out)\nPY`
- 报告里把指标命名区分清楚：`generalization_rephrase_em`（no-leak） vs `generalization_rephrase_em_contaminated`（如果你确认有泄漏）。

**ZsRE-3k 复现路径（已跑通）**
- Repo baseline（论文脚本）：`./saves/baseline_qwen3_zsre3k`
- DBKE full-scale configs（见 worktree）：`.worktrees/dbke/configs/zsre3k_full_e0.yaml`、`.worktrees/dbke/configs/zsre3k_full_e5.yaml`
- DBKE 最佳 stage-2：`.worktrees/dbke/configs/zsre3k_stage2_e5_sft.yaml` → `./saves/zsre3k_stage2_e5_sft_gate`

## 2) 快速命令索引
查看 worktree：
- `git worktree list`

构建 on-policy 偏好数据（JSONL，含 chosen/rejected + conflict_score）：
- `python scripts/build_on_policy_dataset.py --model_path ... --off_data_path ... --output_path ... --k 4 --max_samples 600 --device cuda:0 --score_mode mismatch`
- `--score_mode full`：会额外算 logprob margin（更慢，但 gate 更不容易饱和）

训练（读取 YAML 配置）：
- `python scripts/train.py --config_path configs/long_e5.yaml`

并行/隔离跑（推荐；按 tag 分离输出目录）：
- worktree：`/data/shichao/FT4Editing/.worktrees/dbke-2026`
- runner：`bash scripts/run_dbke_tagged_e0_e3_e5.sh --tag <TAG> --gpu <1|2|3> --preset fast --score_mode full`

评测：
- `python -m eval.on_policy_eval --on_policy_path data/generated/xxx.jsonl --output_path saves/xxx/on_policy_metrics.json`
- `python -m eval.off_policy_eval --model_path saves/xxx --data_path data/zsre/zsre_3k.json --num_samples 500 --output_path saves/xxx/off_policy_metrics.json`

## 3) 多卡并行跑实验（CUDA 1/2/3）
原则：
- 所有脚本内部常用 `--device cuda:0`，所以请用 `CUDA_VISIBLE_DEVICES` 映射到你希望用的物理卡。
  - 例如：`CUDA_VISIBLE_DEVICES=2` 时，脚本里的 `cuda:0` 实际就是物理 GPU2。
- 并行跑多个实验时，要确保**输出目录不冲突**（`saves/`、`data/generated/`、`logs/`、`reports/`）。

推荐方式（最简单、隔离最好）：
- 每个实验一个 worktree + 分支（避免路径冲突，互不影响）。
  - 新建：`git worktree add -b feat/dbke-run-gpu2 .worktrees/dbke-run-gpu2 feat/dbke-2026`
  - 在该 worktree 内跑：`CUDA_VISIBLE_DEVICES=2 bash scripts/run_dbke_long_e0_e3_e5.sh`

## 4) 读日志 / 查进程 / 停止实验
查 GPU 占用：
- `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv`

查进程 cwd（定位是哪一个 worktree 起的）：
- `readlink -f /proc/<PID>/cwd`

停止：
- `kill -TERM <PID>`（必要时 `kill -KILL <PID>`）

## 5) 经验性排障（当前观察到的瓶颈）
如果出现 “Dynamic Gate 不分化 / gate 类似常数混合”：
- 优先把 on-policy 构建的 `--score_mode` 从 `mismatch` 切到 `full`（用 margin 让 conflict_score 不那么饱和）。

如果 edit_success 长期接近 0：
- 放宽可训练参数预算（多解冻几层 / 更宽模块），或增加 steps / LR（注意 locality 回归）。
- **先检查 CE/DPO 的 label mask 是否把 padding token 计入 loss**（典型症状：loss 很低但 edit 指标不涨）。对齐修复见：`.worktrees/dbke/losses/objectives.py` 与 `.worktrees/dbke/train/train_patch.py`。

如果 Generalization（Rephrase EM）很低但 Src EM 很高：
- 多半是 “写入成功，但触发/重述分布没被拉过去”，优先尝试：
  - stage-2：dynamic gate + on-policy SFT（从 E0 checkpoint 继续训）
  - 提高 `k_triggers`（但注意 on-policy 数据构建耗时）
  - 用“未泄漏”的 paraphrase/trigger（不要直接用数据集自带 `rephrase`；或者用它但必须做 no-leak eval 并在报告里标注）

## 6) 2026 继续迭代建议（从论文动向抽象成可落地 ablation）
（这里先记“方向”，具体落地以 runner + configs 为准）
- 更可靠的冲突信号：mismatch + margin / NLL（减少阈值饱和）
- 加 locality 约束：对 ref 模型 KL / RPO 类正则（轻量、可控）
- 参数范数控制（Norm Anchoring）：对可训练参数加 `norm_anchor_lambda`，约束参数 L2 norm 不漂移（适合长序列编辑稳定性）
- 动态策略更细：按样本 conflict 做 reweight / curriculum（而不是 batch 平均）
