from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path
from typing import Dict, List

import regex
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.datasets import OffPolicySFTDataset, RecordMapper


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(prediction: str, target: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(target))

def contains_match(prediction: str, target: str) -> float:
    pred_n = normalize_answer(prediction)
    tgt_n = normalize_answer(target)
    if not tgt_n:
        return 0.0
    return float(tgt_n == pred_n or tgt_n in pred_n)


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 24) -> str:
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(generated[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True)
    return text.strip().split("\n")[0].strip()


def evaluate(model_path: str, data_path: str, num_samples: int = 100) -> Dict[str, float]:
    mapper = RecordMapper()
    dataset = OffPolicySFTDataset(data_path, mapper=mapper)
    records = dataset.records
    if len(records) > num_samples:
        random.seed(42)
        records = random.sample(records, num_samples)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    edit_em: List[float] = []
    edit_contains: List[float] = []
    locality_em: List[float] = []
    locality_contains: List[float] = []
    for record in records:
        pred = generate_answer(model, tokenizer, record["prompt"])
        edit_em.append(exact_match(pred, record["target_new"]))
        edit_contains.append(contains_match(pred, record["target_new"]))

        loc_prompt = record.get("locality_prompt")
        loc_target = record.get("locality_ground_truth")
        if loc_prompt and loc_target:
            loc_pred = generate_answer(model, tokenizer, loc_prompt)
            locality_em.append(exact_match(loc_pred, str(loc_target)))
            locality_contains.append(contains_match(loc_pred, str(loc_target)))

    return {
        "num_samples": len(records),
        "edit_success": sum(edit_em) / max(1, len(edit_em)),
        "edit_success_contains": sum(edit_contains) / max(1, len(edit_contains)),
        "locality": sum(locality_em) / max(1, len(locality_em)) if locality_em else 0.0,
        "locality_contains": sum(locality_contains) / max(1, len(locality_contains)) if locality_contains else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()

    metrics = evaluate(args.model_path, args.data_path, args.num_samples)
    print(json.dumps(metrics, indent=2))
    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
