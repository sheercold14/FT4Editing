from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F


def _prepare_targets(
    tokenizer,
    targets: List[str],
    *,
    add_leading_space: bool,
    add_eos: bool,
) -> List[str]:
    eos = getattr(tokenizer, "eos_token", None) if add_eos else None
    prepared: List[str] = []
    for target in targets:
        text = "" if target is None else str(target)
        text = text.strip()
        if add_leading_space and text and not text.startswith(" "):
            text = " " + text
        if eos and text and not text.endswith(eos):
            text = text + eos
        prepared.append(text)
    return prepared


def _tokenize_prompt_target(tokenizer, prompts: List[str], targets: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
    prompt_batch = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    full_texts = [p + t for p, t in zip(prompts, targets)]
    full_batch = tokenizer(full_texts, return_tensors="pt", padding=True).to(device)
    prompt_lens = (prompt_batch["input_ids"] != tokenizer.pad_token_id).sum(dim=1)
    full_lens = (full_batch["input_ids"] != tokenizer.pad_token_id).sum(dim=1)

    labels = full_batch["input_ids"].clone()
    labels[:] = -100
    for idx in range(labels.shape[0]):
        pad_len = labels.shape[1] - int(full_lens[idx])
        if getattr(tokenizer, "padding_side", "right") == "left":
            start = max(0, pad_len + int(prompt_lens[idx]))
        else:
            start = max(0, int(prompt_lens[idx]))
        labels[idx, start:] = full_batch["input_ids"][idx, start:]
    return {"input_ids": full_batch["input_ids"], "attention_mask": full_batch["attention_mask"], "labels": labels}


def ce_loss(
    model,
    tokenizer,
    prompts: List[str],
    targets: List[str],
    device: torch.device,
    *,
    add_leading_space: bool = True,
    add_eos: bool = True,
) -> torch.Tensor:
    prepared = _prepare_targets(tokenizer, targets, add_leading_space=add_leading_space, add_eos=add_eos)
    batch = _tokenize_prompt_target(tokenizer, prompts, prepared, device)
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    # Per-sample length-normalized CE (matches fine-tune.py behavior more closely than HF's token-mean loss).
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch["labels"][:, 1:].contiguous()
    loss_flat = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape[0], -1)
    valid = shift_labels != -100
    denom = valid.sum(dim=1).clamp(min=1)
    per_sample = (loss_flat * valid).sum(dim=1) / denom
    return per_sample.mean()


def continuation_logp(
    model,
    tokenizer,
    prompts: List[str],
    targets: List[str],
    device: torch.device,
    *,
    add_leading_space: bool = True,
    add_eos: bool = False,
) -> torch.Tensor:
    prepared = _prepare_targets(tokenizer, targets, add_leading_space=add_leading_space, add_eos=add_eos)
    batch = _tokenize_prompt_target(tokenizer, prompts, prepared, device)
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    valid_mask = labels != -100
    safe_labels = labels.masked_fill(~valid_mask, 0)
    token_logp = F.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * valid_mask
    return token_logp.sum(dim=1)


def dpo_loss(
    policy_model,
    ref_model,
    tokenizer,
    prompts: List[str],
    chosen: List[str],
    rejected: List[str],
    beta: float,
    device: torch.device,
) -> torch.Tensor:
    # Compute chosen/rejected logps in one pass each for policy/ref to cut tokenization + forward overhead.
    combined_prompts = prompts + prompts
    chosen_prepared = _prepare_targets(tokenizer, chosen, add_leading_space=True, add_eos=True)
    rejected_prepared = _prepare_targets(tokenizer, rejected, add_leading_space=True, add_eos=False)
    combined_targets = chosen_prepared + rejected_prepared
    pi_all = continuation_logp(policy_model, tokenizer, combined_prompts, combined_targets, device, add_leading_space=False, add_eos=False)
    pi_chosen = pi_all[: len(prompts)]
    pi_rejected = pi_all[len(prompts) :]
    with torch.no_grad():
        ref_all = continuation_logp(ref_model, tokenizer, combined_prompts, combined_targets, device, add_leading_space=False, add_eos=False)
        ref_chosen = ref_all[: len(prompts)]
        ref_rejected = ref_all[len(prompts) :]
    logits = beta * ((pi_chosen - pi_rejected) - (ref_chosen - ref_rejected))
    return -F.logsigmoid(logits).mean()


def grpo_loss(
    policy_logps: torch.Tensor,
    rewards: torch.Tensor,
    kl_penalty: torch.Tensor | None = None,
) -> torch.Tensor:
    advantages = rewards - rewards.mean(dim=1, keepdim=True)
    loss = -(advantages.detach() * policy_logps).mean()
    if kl_penalty is not None:
        loss = loss + kl_penalty.mean()
    return loss
