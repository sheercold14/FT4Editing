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
    outputs: List[str] = []
    if device is None:
        device = next(model.parameters()).device
    if cache is None:
        cache = {}

    max_new_tokens = int(gen_cfg.get("max_new_tokens", 32))
    temperature = float(gen_cfg.get("temperature", 0.0))
    top_p = float(gen_cfg.get("top_p", 1.0))

    for idx, prompt in enumerate(prompts):
        cache_key = f"{prompt}||{seed}||{max_new_tokens}||{temperature}||{top_p}"
        if cache_key in cache:
            outputs.append(cache[cache_key])
            continue

        torch.manual_seed(seed + idx)
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        generated = model.generate(
            **encoded,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 1e-6),
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(generated[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        cache[cache_key] = text
        outputs.append(text)
    return outputs

