from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F


def _tokenize_prompt_target(tokenizer, prompts: List[str], targets: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
    prompt_batch = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    full_texts = [p + t for p, t in zip(prompts, targets)]
    full_batch = tokenizer(full_texts, return_tensors="pt", padding=True).to(device)
    prompt_lens = (prompt_batch["input_ids"] != tokenizer.pad_token_id).sum(dim=1)
    full_lens = (full_batch["input_ids"] != tokenizer.pad_token_id).sum(dim=1)

    labels = full_batch["input_ids"].clone()
    labels[:] = -100
    for idx in range(labels.shape[0]):
        start = max(0, labels.shape[1] - int(full_lens[idx]) + int(prompt_lens[idx]))
        labels[idx, start:] = full_batch["input_ids"][idx, start:]
    return {"input_ids": full_batch["input_ids"], "attention_mask": full_batch["attention_mask"], "labels": labels}


def ce_loss(model, tokenizer, prompts: List[str], targets: List[str], device: torch.device) -> torch.Tensor:
    batch = _tokenize_prompt_target(tokenizer, prompts, targets, device)
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
    return outputs.loss


def continuation_logp(model, tokenizer, prompts: List[str], targets: List[str], device: torch.device) -> torch.Tensor:
    batch = _tokenize_prompt_target(tokenizer, prompts, targets, device)
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
    pi_chosen = continuation_logp(policy_model, tokenizer, prompts, chosen, device)
    pi_rejected = continuation_logp(policy_model, tokenizer, prompts, rejected, device)
    with torch.no_grad():
        ref_chosen = continuation_logp(ref_model, tokenizer, prompts, chosen, device)
        ref_rejected = continuation_logp(ref_model, tokenizer, prompts, rejected, device)
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
