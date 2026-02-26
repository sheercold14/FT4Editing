from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


class EditorModel:
    def edit(self, x: str, y_hat: str, target_spec: Dict[str, Any], constraints: str = "minimal change") -> str:
        raise NotImplementedError


@dataclass
class RuleBasedEditor(EditorModel):
    fallback_field: str = "target_new"

    def edit(self, x: str, y_hat: str, target_spec: Dict[str, Any], constraints: str = "minimal change") -> str:
        value = (
            target_spec.get("target_new")
            or target_spec.get("alt")
            or target_spec.get(self.fallback_field)
            or ""
        )
        return str(value).strip()


@dataclass
class APIEditor(EditorModel):
    client: Optional[Any] = None

    def edit(self, x: str, y_hat: str, target_spec: Dict[str, Any], constraints: str = "minimal change") -> str:
        if self.client is None:
            return RuleBasedEditor().edit(x, y_hat, target_spec, constraints=constraints)
        prompt = (
            "You are a factual editor.\n"
            f"Prompt: {x}\n"
            f"Current answer: {y_hat}\n"
            f"Target spec: {target_spec}\n"
            "Return only the corrected short answer."
        )
        return str(self.client(prompt)).strip()


@dataclass
class LocalModelEditor(EditorModel):
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None
    max_new_tokens: int = 32

    def edit(self, x: str, y_hat: str, target_spec: Dict[str, Any], constraints: str = "minimal change") -> str:
        if self.model is None or self.tokenizer is None:
            return RuleBasedEditor().edit(x, y_hat, target_spec, constraints=constraints)
        prompt = (
            f"Question: {x}\n"
            f"Current answer: {y_hat}\n"
            f"Target: {target_spec.get('target_new', target_spec.get('alt', ''))}\n"
            "Corrected answer:"
        )
        encoded = self.tokenizer(prompt, return_tensors="pt").to(next(self.model.parameters()).device)
        generated = self.model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(generated[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True).strip()

