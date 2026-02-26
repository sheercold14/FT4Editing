from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F


def _norm(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _sequence_logp(model, tokenizer, prompt: str, target: str, device: torch.device) -> float:
    joined = prompt + target
    batch = tokenizer(joined, return_tensors="pt").to(device)
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    with torch.no_grad():
        logits = model(**batch).logits[:, :-1, :]
    labels = batch["input_ids"][:, 1:]
    logp = F.log_softmax(logits, dim=-1)
    token_logp = torch.gather(logp, -1, labels.unsqueeze(-1)).squeeze(-1)
    target_logp = token_logp[:, prompt_len - 1 :].sum().item()
    return float(target_logp)


def compute_conflict_score(
    x_on: Iterable[str],
    y_hat: Iterable[str],
    y_pos: str,
    model=None,
    tokenizer=None,
    verifier: Optional[Callable[[str, str, str], float]] = None,
) -> Dict[str, float]:
    prompts = list(x_on)
    preds = list(y_hat)
    if not prompts or not preds:
        return {"mismatch_rate": 0.0, "margin": 0.0, "score": 0.0}

    mismatches = []
    margins: List[float] = []
    for prompt, pred in zip(prompts, preds):
        if verifier is None:
            mismatch = float(_norm(pred) != _norm(y_pos))
        else:
            mismatch = float(verifier(prompt, pred, y_pos) < 0.5)
        mismatches.append(mismatch)

        if model is not None and tokenizer is not None:
            device = next(model.parameters()).device
            pos_lp = _sequence_logp(model, tokenizer, prompt, y_pos, device)
            pred_lp = _sequence_logp(model, tokenizer, prompt, pred, device)
            margins.append(pos_lp - pred_lp)

    mismatch_rate = float(sum(mismatches) / len(mismatches))
    margin = float(sum(margins) / len(margins)) if margins else 0.0
    normalized_margin = 1.0 / (1.0 + torch.exp(torch.tensor(margin)).item()) if margins else 0.0
    score = float(0.7 * mismatch_rate + 0.3 * normalized_margin)
    return {"mismatch_rate": mismatch_rate, "margin": margin, "score": score}
