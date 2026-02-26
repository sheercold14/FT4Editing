from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch


def rollout(
    model,
    tokenizer,
    prompts: Iterable[str],
    gen_cfg: Optional[Dict] = None,
    seed: int = 0,
    device: Optional[torch.device] = None,
    cache: Optional[Dict[str, str]] = None,
) -> List[str]:
    gen_cfg = gen_cfg or {}
    prompt_list = list(prompts)
    outputs: List[str] = []
    if device is None:
        device = next(model.parameters()).device
    if cache is None:
        cache = {}

    max_new_tokens = int(gen_cfg.get("max_new_tokens", 32))
    temperature = float(gen_cfg.get("temperature", 0.0))
    top_p = float(gen_cfg.get("top_p", 1.0))
    batch_size = int(gen_cfg.get("batch_size", 16))
    do_sample = temperature > 0.0

    # Fast path: greedy decoding can be batched efficiently and is deterministic given model weights.
    if not do_sample:
        if tokenizer.padding_side != "left":
            tokenizer.padding_side = "left"
        pending = []
        for idx, prompt in enumerate(prompt_list):
            cache_key = f"{prompt}||{seed}||{max_new_tokens}||{temperature}||{top_p}"
            if cache_key in cache:
                outputs.append(cache[cache_key])
            else:
                outputs.append("")
                pending.append((idx, prompt, cache_key))

        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            chunk_prompts = [p for _, p, _ in chunk]
            encoded = tokenizer(chunk_prompts, return_tensors="pt", padding=True).to(device)
            input_lens = encoded["attention_mask"].sum(dim=1).tolist()
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            for row, (orig_idx, _, cache_key) in enumerate(chunk):
                text = tokenizer.decode(
                    generated[row][int(input_lens[row]) :], skip_special_tokens=True
                ).strip()
                cache[cache_key] = text
                outputs[orig_idx] = text
        return outputs

    # Sampling path: generate per-prompt to keep seed behaviour simple.
    for idx, prompt in enumerate(prompt_list):
        cache_key = f"{prompt}||{seed}||{max_new_tokens}||{temperature}||{top_p}"
        if cache_key in cache:
            outputs.append(cache[cache_key])
            continue

        torch.manual_seed(seed + idx)
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(
            generated[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        cache[cache_key] = text
        outputs.append(text)
    return outputs
