from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.datasets import OffPolicySFTDataset, OnPolicySFTDataset, RecordMapper, dump_jsonl
from data.mixer import MixedBatchSampler
from losses.objectives import ce_loss, dpo_loss
from on_policy.conflict_score import compute_conflict_score
from on_policy.editor_interface import RuleBasedEditor
from on_policy.paraphrase_generator import generate_paraphrase_triggers
from on_policy.prefix_noise_generator import generate_prefix_noise_tail_anchored_triggers
from on_policy.rollout import rollout
from on_policy.trigger_generator import generate_suffix_completions, generate_triggers


@dataclass
class TrainConfig:
    model_name_or_path: str
    off_data_path: str
    on_data_path: str
    save_model_dir: str
    objective: str = "online_dpo"
    use_dynamic_gate: bool = False
    tau_low: float = 0.2
    tau_high: float = 0.7
    lambda_on: float = 0.7
    lambda_off: float = 0.3
    lambda_schedule: str = "fixed"
    on_ratio: float = 0.5
    beta: float = 0.1
    lr: float = 5e-5
    weight_decay: float = 0.0
    optimizer: str = "adamw"  # "adamw" or "adam"
    torch_dtype: str = "bf16"  # "bf16", "fp16", "fp32"
    batch_size: int = 4
    num_epochs: int = 1
    steps_per_epoch: int = 200
    refresh_interval: int = 1
    k_triggers: int = 4
    gen_cfg: Dict = None
    device: int = 0
    seed: int = 42
    max_new_tokens: int = 32
    min_conflict_for_on: float = 0.0
    max_refresh_records: int = 200
    conflict_score_mode: str = "mismatch"  # "mismatch" or "full" (logprob margin)
    include_rephrase_triggers: bool = False
    trigger_mode: str = "template"  # "template" | "template_paraphrase" | "template_distractor" | "bucket_mix"
    template_triggers: int = 2
    paraphrase_per_prompt: int = 0
    paraphrase_gen_cfg: Dict | None = None
    paraphrase_similarity_threshold: float = 0.92
    paraphrase_filter_against_rephrase: bool = True
    suffix_triggers_per_record: int = 0
    distractor_pool_size: int = 2048
    distractor_prefixes_per_record: int = 1
    distractor_max_chars: int = 120
    filter_against_eval_rephrase_threshold: float = 0.92
    prefix_tail_per_prompt: int = 0
    prefix_tail_words: int = 2
    prefix_tail_gen_cfg: Dict | None = None
    on_objective: str = "dpo"  # for dynamic_gate: "dpo", "sft", or "auto"
    micro_batch_size: int = 0
    early_stop_loss: float | None = None
    skip_step_loss: float | None = None
    train_layer: int | None = None
    rewrite_module: str = ""
    # OPA-style schedule
    opa_align_epochs: int = 0
    opa_refresh_after_align: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        data = data or {}
        if "gen_cfg" not in data:
            data["gen_cfg"] = {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": data.get("max_new_tokens", 32)}
        if "paraphrase_gen_cfg" not in data:
            data["paraphrase_gen_cfg"] = {
                "temperature": 0.7,
                "top_p": 0.95,
                "max_new_tokens": 64,
                "batch_size": 16,
            }
        if "prefix_tail_gen_cfg" not in data:
            data["prefix_tail_gen_cfg"] = {
                "temperature": 0.9,
                "top_p": 0.95,
                "max_new_tokens": 96,
                "batch_size": 16,
            }
        return cls(**data)


def _chunk(items: List[Dict], size: int) -> List[List[Dict]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _gate_lambda(config: TrainConfig, conflict: float) -> Tuple[float, float]:
    if not config.use_dynamic_gate:
        return config.lambda_on, config.lambda_off
    if conflict <= config.tau_low:
        return 1.0, 0.0
    if conflict >= config.tau_high:
        return 0.5, 1.0
    return 0.7, 0.7


def rebuild_on_policy_dataset(
    config: TrainConfig,
    model,
    tokenizer,
    off_records: List[Dict],
) -> List[Dict]:
    editor = RuleBasedEditor()
    refreshed: List[Dict] = []
    sampled_records = off_records[: config.max_refresh_records] if config.max_refresh_records > 0 else off_records
    print(f"[ON] rebuild dataset: records={len(sampled_records)} k={config.k_triggers} mode={config.conflict_score_mode}")

    prompt_rows: List[Tuple[int, str]] = []
    # For counterfact-style rephrase prompts, adding an unrelated prefix often matches evaluation distribution better.
    distractor_pool: List[str] = []
    if config.trigger_mode in {"template_distractor", "bucket_mix"}:
        seen = set()
        for rec in sampled_records:
            cand = (rec.get("locality_prompt") or rec.get("prompt") or rec.get("src") or "").strip()
            if not cand:
                continue
            cand = cand.replace("\n", " ").strip()
            if len(cand) > int(config.distractor_max_chars):
                cand = cand[: int(config.distractor_max_chars)].rsplit(" ", 1)[0].strip() or cand[: int(config.distractor_max_chars)]
            key = cand.lower()
            if key and key not in seen:
                distractor_pool.append(cand)
                seen.add(key)
            if len(distractor_pool) >= int(config.distractor_pool_size):
                break
        if distractor_pool:
            print(f"[ON] distractor_pool: size={len(distractor_pool)}")
    base_prompts_by_rec: Dict[int, List[str]] = {}
    k_template = int(config.k_triggers)
    if config.trigger_mode == "template_paraphrase":
        k_template = max(1, min(int(config.template_triggers), int(config.k_triggers)))
    for index, rec in enumerate(sampled_records):
        prompts = generate_triggers(
            rec,
            k_template,
            mode="template",
            include_rephrase=config.include_rephrase_triggers,
        )
        base_prompts_by_rec[index] = prompts
        for prompt in prompts:
            prompt_rows.append((index, prompt))

    def _too_similar_to_eval(prompt: str, rec: Dict) -> bool:
        rp = (rec.get("rephrase_prompt") or rec.get("rephrase") or "").strip()
        if not rp:
            return False
        from difflib import SequenceMatcher

        a = " ".join(prompt.lower().split())
        b = " ".join(rp.lower().split())
        if not a or not b:
            return False
        return SequenceMatcher(a=a, b=b).ratio() >= float(config.filter_against_eval_rephrase_threshold)

    if config.trigger_mode in {"template_distractor", "bucket_mix"} and distractor_pool:
        import random

        rng = random.Random(int(config.seed) + 2026)
        new_rows: List[Tuple[int, str]] = []
        for rec_index, base_prompts in base_prompts_by_rec.items():
            rec = sampled_records[rec_index]
            prompt = (rec.get("prompt") or rec.get("src") or "").strip()
            if not prompt:
                continue
            prompt_short = " ".join(prompt.split(" ")[:-1]).strip() or prompt
            keep = list(base_prompts)
            for _ in range(int(config.distractor_prefixes_per_record)):
                prefix = rng.choice(distractor_pool)
                if config.trigger_mode == "bucket_mix":
                    # Reserve slots for other synthetic triggers (e.g., tail-anchored rephrases).
                    candidates = [f"{prefix}. {prompt_short}"]
                else:
                    candidates = [
                        f"{prefix}. {prompt}",
                        f"{prefix}. {prompt_short}",
                    ]
                for cand in candidates:
                    if len(keep) >= int(config.k_triggers):
                        break
                    if (not cand) or _too_similar_to_eval(cand, rec):
                        continue
                    if cand.lower() not in {k.lower() for k in keep}:
                        keep.append(cand)
            for p in keep[: int(config.k_triggers)]:
                new_rows.append((rec_index, p))
        prompt_rows = new_rows

    if config.trigger_mode in {"bucket_mix"} and int(config.suffix_triggers_per_record) > 0:
        new_rows: List[Tuple[int, str]] = []
        for rec_index, base_prompts in base_prompts_by_rec.items():
            rec = sampled_records[rec_index]
            prompt = (rec.get("prompt") or rec.get("src") or "").strip()
            if not prompt:
                continue
            keep = list({p.lower(): p for p in base_prompts}.values())
            suffixes = generate_suffix_completions(prompt, max_variants=int(config.suffix_triggers_per_record))
            for cand in suffixes:
                if len(keep) >= int(config.k_triggers):
                    break
                if (not cand) or _too_similar_to_eval(cand, rec):
                    continue
                if cand.lower() not in {k.lower() for k in keep}:
                    keep.append(cand)
            for p in keep[: int(config.k_triggers)]:
                new_rows.append((rec_index, p))
        prompt_rows = new_rows

    if config.trigger_mode in {"bucket_mix"} and int(config.prefix_tail_per_prompt) > 0:
        # Generate prefix-noise + tail-anchored variants (CounterFact-style).
        # We apply this at the record level using the canonical prompt (not templates), then cap to k_triggers.
        rec_prompts: List[Tuple[int, str]] = []
        for rec_index, _ in base_prompts_by_rec.items():
            rec = sampled_records[rec_index]
            prompt = (rec.get("prompt") or rec.get("src") or "").strip()
            if prompt:
                rec_prompts.append((rec_index, prompt))
        flat_prompts = [p for _, p in rec_prompts]
        protected = {}
        if config.paraphrase_filter_against_rephrase:
            for rec_index, p in rec_prompts:
                rec = sampled_records[rec_index]
                rp = (rec.get("rephrase_prompt") or rec.get("rephrase") or "").strip()
                if rp:
                    protected.setdefault(p, []).append(rp)
        print(f"[ON] prefix_tail: prompts={len(flat_prompts)} per_prompt={int(config.prefix_tail_per_prompt)} tail_words={int(config.prefix_tail_words)}")
        grouped = generate_prefix_noise_tail_anchored_triggers(
            model=model,
            tokenizer=tokenizer,
            prompts=flat_prompts,
            per_prompt=int(config.prefix_tail_per_prompt),
            tail_words=int(config.prefix_tail_words),
            gen_cfg=config.prefix_tail_gen_cfg,
            seed=int(config.seed) + 4242,
            similarity_threshold=float(config.paraphrase_similarity_threshold),
            filter_against=protected,
        )
        kept = sum(len(g) for g in grouped)
        print(f"[ON] prefix_tail: kept={kept}")
        new_rows: List[Tuple[int, str]] = []
        for (rec_index, base_prompt), outs in zip(rec_prompts, grouped):
            rec = sampled_records[rec_index]
            # Start from existing rows for this record.
            existing = [p for ri, p in prompt_rows if ri == rec_index]
            keep = list(existing) if existing else list(base_prompts_by_rec.get(rec_index, []))
            for cand in outs:
                if len(keep) >= int(config.k_triggers):
                    break
                if (not cand) or _too_similar_to_eval(cand, rec):
                    continue
                if cand.lower() not in {k.lower() for k in keep}:
                    keep.append(cand)
            for p in keep[: int(config.k_triggers)]:
                new_rows.append((rec_index, p))
        if new_rows:
            prompt_rows = new_rows

    if config.trigger_mode in {"template_paraphrase", "bucket_mix"} and config.paraphrase_per_prompt > 0:
        flat_base = [p for _, p in prompt_rows]
        print(f"[ON] paraphrase: base_prompts={len(flat_base)} per_prompt={int(config.paraphrase_per_prompt)}")
        # Build a map of protected strings we want to avoid matching too closely (to prevent leakage).
        protected_by_prompt: Dict[str, List[str]] = {}
        if config.paraphrase_filter_against_rephrase:
            # For each base prompt, protect against its record's eval rephrase (if present).
            for rec_index, prompts in base_prompts_by_rec.items():
                rec = sampled_records[rec_index]
                rp = (rec.get("rephrase_prompt") or rec.get("rephrase") or "").strip()
                if not rp:
                    continue
                for p in prompts:
                    protected_by_prompt.setdefault(p, []).append(rp)

        # Generate paraphrases per base prompt; keep only a few to avoid dataset bloat.
        paraphrases_grouped = generate_paraphrase_triggers(
            model=model,
            tokenizer=tokenizer,
            prompts=flat_base,
            per_prompt=int(config.paraphrase_per_prompt),
            gen_cfg=config.paraphrase_gen_cfg,
            seed=int(config.seed) + 1337,
            similarity_threshold=float(config.paraphrase_similarity_threshold),
            filter_against=protected_by_prompt,
            forbid_substrings=[],
        )
        kept = sum(len(g) for g in paraphrases_grouped)
        print(f"[ON] paraphrase: kept={kept}")
        # Append paraphrases, capped by k_triggers per record.
        new_rows: List[Tuple[int, str]] = []
        cursor = 0
        for rec_index, base_prompts in base_prompts_by_rec.items():
            keep = list(base_prompts)
            for _ in base_prompts:
                for cand in paraphrases_grouped[cursor]:
                    if len(keep) >= int(config.k_triggers):
                        break
                    keep.append(cand)
                cursor += 1
            for p in keep:
                new_rows.append((rec_index, p))
        prompt_rows = new_rows

    if not prompt_rows:
        dump_jsonl(config.on_data_path, refreshed)
        return refreshed

    flat_prompts = [p for _, p in prompt_rows]
    print(f"[ON] rollout: prompts={len(flat_prompts)} batch_size={int((config.gen_cfg or {}).get('batch_size', 16))}")
    flat_preds = rollout(
        model=model,
        tokenizer=tokenizer,
        prompts=flat_prompts,
        gen_cfg=config.gen_cfg,
        seed=config.seed,
    )
    print("[ON] rollout: done")

    # Group prompts/preds by record index for per-edit conflict scoring.
    grouped: Dict[int, List[Tuple[str, str]]] = {}
    for (rec_index, prompt), pred in zip(prompt_rows, flat_preds):
        grouped.setdefault(rec_index, []).append((prompt, pred))

    score_model = model if config.conflict_score_mode == "full" else None
    score_tokenizer = tokenizer if config.conflict_score_mode == "full" else None

    for rec_index, rec in enumerate(sampled_records):
        if rec_index > 0 and rec_index % 200 == 0:
            print(f"[ON] score/proc: {rec_index}/{len(sampled_records)} rows={len(refreshed)}")
        target = rec.get("target_new") or rec.get("alt") or ""
        if not target:
            continue
        pairs = grouped.get(rec_index, [])
        if not pairs:
            continue
        prompts = [p for p, _ in pairs]
        preds = [y for _, y in pairs]
        y_edit = [editor.edit(prompt, pred, rec) for prompt, pred in pairs]
        score_dict = compute_conflict_score(prompts, preds, target, model=score_model, tokenizer=score_tokenizer)
        for prompt, pred, chosen in zip(prompts, preds, y_edit):
            refreshed.append(
                {
                    "edit_id": rec.get("case_id", rec_index),
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": pred,
                    "target_new": target,
                    "conflict_score": float(score_dict["score"]),
                    "mismatch_rate": float(score_dict["mismatch_rate"]),
                    "margin": float(score_dict["margin"]),
                }
            )
    dump_jsonl(config.on_data_path, refreshed)
    print(f"[ON] saved: {config.on_data_path} rows={len(refreshed)}")
    return refreshed


def train(config: TrainConfig) -> None:
    torch.manual_seed(config.seed)
    device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "left"
    if device.type != "cuda":
        torch_dtype = None
    else:
        dt = (config.torch_dtype or "bf16").lower()
        if dt == "bf16":
            torch_dtype = torch.bfloat16
        elif dt == "fp16":
            torch_dtype = torch.float16
        elif dt == "fp32":
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unknown torch_dtype={config.torch_dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path, trust_remote_code=True, torch_dtype=torch_dtype
    ).to(device)
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    mapper = RecordMapper()
    off_dataset = OffPolicySFTDataset(config.off_data_path, mapper=mapper)
    off_records = [item["record"] for item in off_dataset]

    use_on_policy = config.on_ratio > 0.0 and config.objective != "off_sft"
    on_dataset = None
    if use_on_policy:
        on_path = Path(config.on_data_path)
        if not on_path.exists():
            rebuild_on_policy_dataset(config, model, tokenizer, off_records)
        on_dataset = OnPolicySFTDataset(config.on_data_path, min_conflict=config.min_conflict_for_on)

    ref_model = None
    needs_ref = config.objective in {"online_dpo", "mix_dpo", "opa_align_then_dpo"} or (
        config.objective == "dynamic_gate" and config.on_objective in {"dpo", "auto"}
    )
    if use_on_policy and needs_ref:
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path, trust_remote_code=True, torch_dtype=torch_dtype
        ).to(device)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

    # Freeze all weights by default; optionally unfreeze a targeted module (editing-style FT)
    for param in model.parameters():
        param.requires_grad = False

    trainable_params = []
    if config.rewrite_module and config.train_layer is not None:
        needle = config.rewrite_module.format(config.train_layer)
        for name, param in model.named_parameters():
            if needle in name:
                param.requires_grad = True
                trainable_params.append(param)
        if not trainable_params:
            raise ValueError(f"No parameters matched rewrite_module={config.rewrite_module} at layer={config.train_layer}")
    else:
        # Fallback: train all params (not recommended for large runs)
        for param in model.parameters():
            param.requires_grad = True
        trainable_params = list(model.parameters())

    opt_name = (config.optimizer or "adamw").lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=float(config.weight_decay))
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=config.lr, weight_decay=float(config.weight_decay))
    else:
        raise ValueError(f"Unknown optimizer={config.optimizer}")
    model.train()
    use_amp = False
    scaler = None

    for epoch in range(config.num_epochs):
        is_align_phase = config.objective == "opa_align_then_dpo" and epoch < int(config.opa_align_epochs)
        if use_on_policy and epoch % config.refresh_interval == 0 and epoch > 0:
            rebuild_on_policy_dataset(config, model, tokenizer, off_records)
            on_dataset = OnPolicySFTDataset(config.on_data_path, min_conflict=config.min_conflict_for_on)
        if (
            config.objective == "opa_align_then_dpo"
            and int(config.opa_align_epochs) > 0
            and int(epoch) == int(config.opa_align_epochs)
            and bool(config.opa_refresh_after_align)
        ):
            # Refresh rejected rollouts after alignment so DPO sees current-policy negatives.
            rebuild_on_policy_dataset(config, model, tokenizer, off_records)
            on_dataset = OnPolicySFTDataset(config.on_data_path, min_conflict=config.min_conflict_for_on)

        if on_dataset is not None and len(on_dataset):
            conflict_scores = [float(r.get("conflict_score", 0.0)) for r in on_dataset.records]
        else:
            conflict_scores = []
        ratio = config.on_ratio if use_on_policy else 0.0
        if is_align_phase:
            ratio = 1.0
        sampler = MixedBatchSampler(
            off_size=len(off_dataset),
            on_size=len(on_dataset) if on_dataset is not None else 0,
            ratio=ratio,
            total_steps=config.steps_per_epoch * config.batch_size,
            seed=config.seed + epoch,
            conflict_scores=conflict_scores if conflict_scores else None,
        )
        step_losses = []
        samples = list(sampler)
        micro = config.micro_batch_size or config.batch_size
        for sampled_batch in _chunk(samples, config.batch_size):
            if not sampled_batch:
                continue
            optimizer.zero_grad(set_to_none=True)

            micro_batches = _chunk(sampled_batch, micro)
            total_items = sum(len(mb) for mb in micro_batches)
            for micro_batch in micro_batches:
                off_items = []
                on_items = []
                for source, idx in micro_batch:
                    if source == "on" and on_dataset is not None and len(on_dataset) > 0:
                        on_items.append(on_dataset[idx % len(on_dataset)])
                    else:
                        off_items.append(off_dataset[idx % len(off_dataset)])

                conflict = None
                if config.use_dynamic_gate and on_items:
                    conflict = sum(float(item["conflict_score"]) for item in on_items) / len(on_items)

                losses = []
                if off_items:
                    off_prompts = [item["prompt"] for item in off_items]
                    off_targets = [item["target"] for item in off_items]
                    losses.append(("off", ce_loss(model, tokenizer, off_prompts, off_targets, device)))

                if on_items:
                    on_prompts = [item["prompt"] for item in on_items]
                    on_chosen = [item["target"] for item in on_items]
                    on_rejected = [item["rejected"] for item in on_items]
                    use_dpo_on = config.objective in {"online_dpo", "mix_dpo", "opa_align_then_dpo"}
                    if is_align_phase:
                        use_dpo_on = False
                    elif config.objective == "dynamic_gate":
                        if config.on_objective == "dpo":
                            use_dpo_on = True
                        elif config.on_objective == "sft":
                            use_dpo_on = False
                        elif config.on_objective == "auto":
                            use_dpo_on = (conflict is not None) and (float(conflict) >= float(config.tau_high))
                        else:
                            raise ValueError(f"Unknown on_objective={config.on_objective}")

                    if use_dpo_on:
                        if ref_model is None:
                            raise RuntimeError("ref_model is required for DPO objectives but was not initialized")
                        on_loss = dpo_loss(
                            policy_model=model,
                            ref_model=ref_model,
                            tokenizer=tokenizer,
                            prompts=on_prompts,
                            chosen=on_chosen,
                            rejected=on_rejected,
                            beta=config.beta,
                            device=device,
                        )
                    else:
                        on_loss = ce_loss(model, tokenizer, on_prompts, on_chosen, device)
                    losses.append(("on", on_loss))

                if not losses:
                    continue

                if config.use_dynamic_gate and (not is_align_phase) and on_items and conflict is not None:
                    lambda_on, lambda_off = _gate_lambda(config, float(conflict))
                else:
                    lambda_on, lambda_off = config.lambda_on, config.lambda_off

                if config.objective == "off_sft":
                    lambda_on, lambda_off = 0.0, 1.0
                elif (config.objective in {"online_sft", "online_dpo", "grpo"} and not config.use_dynamic_gate) or is_align_phase:
                    if config.objective == "online_sft":
                        lambda_on, lambda_off = 1.0, 0.0
                    elif config.objective == "online_dpo":
                        lambda_on, lambda_off = config.lambda_on, config.lambda_off
                    elif is_align_phase:
                        # OPA alignment: default to on-policy SFT only (can override by setting lambda_off in config).
                        lambda_on, lambda_off = 1.0, 0.0

                total_loss = torch.zeros((), device=device)
                for name, value in losses:
                    if name == "on":
                        total_loss = total_loss + lambda_on * value
                    else:
                        total_loss = total_loss + lambda_off * value

                if config.skip_step_loss is not None and float(total_loss.detach().item()) < float(config.skip_step_loss):
                    continue

                (total_loss * (len(micro_batch) / total_items)).backward()
                step_losses.append(float(total_loss.detach().item()))

            optimizer.step()

        avg_loss = sum(step_losses) / max(1, len(step_losses))
        on_size = len(on_dataset) if on_dataset is not None else 0
        print(f"[Epoch {epoch + 1}/{config.num_epochs}] avg_loss={avg_loss:.4f} on_size={on_size}")
        if config.early_stop_loss is not None and avg_loss < config.early_stop_loss:
            print(f"[Epoch {epoch + 1}] early stop: avg_loss {avg_loss:.4f} < {config.early_stop_loss:.4f}")
            break

    Path(config.save_model_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.save_model_dir)
    tokenizer.save_pretrained(config.save_model_dir)
    with open(Path(config.save_model_dir) / "train_config_dump.json", "w", encoding="utf-8") as handle:
        json.dump(config.__dict__, handle, indent=2, ensure_ascii=False)
    print(f"Saved model to {config.save_model_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, type=str)
    args = parser.parse_args()
    config = TrainConfig.from_yaml(args.config_path)
    train(config)


if __name__ == "__main__":
    main()
