import json
import argparse
import random
from datetime import datetime
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import regex
import string
from typing import Optional


def print_time(process_name):
    now = datetime.now()
    formatted_time = now.strftime("%m-%d %H:%M:%S")
    print(f'[{formatted_time}] {process_name}')

def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--model_path', required=True, type=str, help="The Path for edited LLMs")
    parser.add_argument('--tp_size', type=int, default=1, help="Tensor parallelism size")
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0, help="Seed for subsampling (when num_samples < dataset size)")
    parser.add_argument('--quiet', action='store_true', help="Disable per-example printing")
    parser.add_argument('--output_path', type=str, default=None, help="Optional path to save a JSON summary")
    args = parser.parse_args()

    # 1. Load Data for Editing
    with open(args.data_path, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    if len(source_data) > args.num_samples:
        random.seed(args.seed)
        data = random.sample(source_data, args.num_samples)
    else:
        data = source_data
    
    print(f"Loaded {len(data)} samples from {args.data_path}")

    # 2. Prepare Prompt Lists
    # Reliability: assess edit accuracy
    # Generalization: assess rephrase accuracy
    src_prompts = []
    rephrase_prompts = []
    targets = []

    for d in data:
        src_prompts.append(d.get('prompt', d.get('src')))
        rephrase_prompts.append(d.get('rephrase_prompt', d.get('rephrase')))
        targets.append(d.get('target_new', d.get('alt')))

    # 3. Initialize vLLM
    print_time("Initializing vLLM")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        # gpu_memory_utilization=0.8,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=0,  # Greedy search
        max_tokens=32,
        stop=[".", "\n", tokenizer.eos_token], # Set stop words based on dataset characteristics
    )

    # 4. Execute Batch Inference
    all_prompts = src_prompts + rephrase_prompts
    print_time(f"Starting Batch Inference (Total: {len(all_prompts)})")
    
    outputs = llm.generate(all_prompts, sampling_params)
    
    # Split the results
    src_outputs = outputs[:len(src_prompts)]
    rephrase_outputs = outputs[len(src_prompts):]

    # 5. Calculate Metrics
    src_em_list = []
    rephrase_em_list = []

    for i in range(len(data)):
        pred_src = src_outputs[i].outputs[0].text
        pred_rephrase = rephrase_outputs[i].outputs[0].text
        target = targets[i]

        src_em = float(exact_match_score(pred_src, target))
        rephrase_em = float(exact_match_score(pred_rephrase, target))

        src_em_list.append(src_em)
        rephrase_em_list.append(rephrase_em)

        if not args.quiet:
            print(f"Example {i+1}:")
            print(f"Rewrite Output: {pred_src}")
            print(f"Rephrase Output: {pred_rephrase}")
            print(f"Target: {target}")
            print(f"Source Exact Match: {src_em:.4f}")
            print(f"Rephrase Exact Match: {rephrase_em:.4f}")
            print("-" * 50)

    # 6. Summary Output
    print("\n" + "="*30)
    print(f"Evaluation Results for: {args.model_path}")
    reliability = sum(src_em_list) / len(src_em_list) if src_em_list else 0.0
    generalization = sum(rephrase_em_list) / len(rephrase_em_list) if rephrase_em_list else 0.0
    print(f"Reliability (Src) EM: {reliability:.4f}")
    print(f"Generalization (Rephrase) EM: {generalization:.4f}")
    print("="*30)
    
    if args.output_path:
        payload = {
            "data_path": args.data_path,
            "model_path": args.model_path,
            "num_samples": len(data),
            "seed": args.seed,
            "reliability_src_em": reliability,
            "generalization_rephrase_em": generalization,
        }
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print_time("Evaluation Finished")

if __name__ == "__main__":
    run_evaluation()
