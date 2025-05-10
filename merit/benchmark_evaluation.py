"""
Benchmark evaluation for the MERIT framework.
"""
import os
import sys
import json
import argparse
import time
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from merit.models.adapters.huggingface import HuggingFaceAdapter
from merit.models.adapters.gemini import GeminiAdapter
from merit.core.evaluation import ReasoningEvaluator
from merit.core.metrics import get_default_metric_registry

BENCHMARKS = {
    "hellaswag": {
        "load_fn": lambda limit: load_dataset("hellaswag", split=f"validation[:{limit}]", trust_remote_code=True),
        "format_fn": lambda example: {
            "prompt": f"{example['ctx']}\nChoose the most plausible continuation by selecting the option number (1, 2, 3, or 4) and provide a brief explanation:\n" + 
                     "\n".join([f"{idx+1}. {ending}" for idx, ending in enumerate(example['endings'])]),
            "reference": example['endings'][int(example['label'])],
            "label": int(example['label'])
        }
    },
    "arc": {
        "load_fn": lambda limit: load_dataset("ai2_arc", "ARC-Challenge", split=f"test[:{limit}]", trust_remote_code=True),
        "format_fn": lambda example: {
            "prompt": f"Question: {example['question']}\nChoose the correct answer by selecting the letter (A, B, C, D, or E):\n" + 
                     "\n".join([f"{choice}. {text}" for choice, text in zip(["A", "B", "C", "D", "E"][:len(example['choices']['text'])], example['choices']['text'])]),
            "reference": example['choices']['text'][example['choices']['label'].index(example['answerKey'])],
            "label": example['answerKey']
        }
    },
    "mmlu_math": {
        "load_fn": lambda limit: load_dataset("cais/mmlu", "high_school_mathematics", split=f"test[:{limit}]", trust_remote_code=True),
        "format_fn": lambda example: {
            "prompt": f"Question: {example['question']}\nChoose the correct answer by selecting the letter (A, B, C, or D):\n" + 
                     "\n".join([f"{choice}. {text}" for choice, text in zip(["A", "B", "C", "D"], example['choices'])]),
            "reference": example['choices'][example['answer']],
            "label": example['answer']
        }
    },
    "mmlu_logic": {
        "load_fn": lambda limit: load_dataset("cais/mmlu", "formal_logic", split=f"test[:{limit}]", trust_remote_code=True),
        "format_fn": lambda example: {
            "prompt": f"Question: {example['question']}\nTo answer, first explain your reasoning step-by-step, considering each option (A, B, C, D) and eliminating incorrect ones. Then, select the correct answer as a single letter (A, B, C, or D) and verify your choice by double-checking your logic. Provide your final answer in the format: **Final Answer: [letter]**\n" + 
                     "\n".join([f"{choice}. {text}" for choice, text in zip(["A", "B", "C", "D"], example['choices'])]),
            "reference": example['choices'][example['answer']],
            "label": example['answer']
        }
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description='Run MERIT benchmark evaluation')
    parser.add_argument('--benchmark', type=str, required=True, choices=BENCHMARKS.keys())
    parser.add_argument('--adapter', type=str, choices=['huggingface', 'gemini'], default='huggingface')
    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--local_path', type=str, default=None)
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--samples', type=int, default=50)
    parser.add_argument('--rpm_limit', type=int, default=14)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--temp', type=float, default=0.1)
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()

def rate_limit(start_time, i, rpm_limit=14):
    if i > 0 and rpm_limit > 0:
        elapsed = time.time() - start_time
        expected_time = (i / rpm_limit) * 60
        if elapsed < expected_time:
            time.sleep(expected_time - elapsed)

def extract_answer(response, benchmark):
    import re
    if benchmark == "hellaswag":
        patterns = [
            r'(?:^|\n)(\d+)[.)]',  
            r'[Tt]he best answer is \**(\d+)\**',
            r'[Tt]he most plausible continuation is \**(\d+)\**',
            r'[Oo]ption \**(\d+)\**',
            r'\**(\d+)\** is the most',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                try:
                    return int(matches[-1]) - 1  # 0-indexed
                except:
                    pass
    elif benchmark in ["arc", "mmlu_math", "mmlu_logic"]:
        patterns = [
            r'(?:^|\n|\s|^Answer:|^So the answer is|^Final Answer:)\s*([A-D])(?:\s|$|\.|,|\))',  # A, A., A), Answer: A
            r'[Tt]he correct answer is \**([A-D])\**',  # **A**
            r'[Oo]ption \**([A-D])\**',  # Option **A**
            r'\b([A-D])\b(?!\s*[\w])',  # Single A, B, C, D
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                letter_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
                return letter_to_index.get(matches[-1], 0)  # Fallback to 0 if invalid
        print(f"Warning: No valid answer extracted from response: {response[:100]}...")
    return 0  # Fallback to 0 if no match

def calculate_accuracy(predictions, references):
    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    return correct / len(predictions) if predictions else 0

def run_benchmark(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    if args.adapter == 'gemini':
        if not args.api_key and not os.environ.get("GOOGLE_API_KEY"):
            print("Error: Gemini adapter requires an API key.")
            return
        model_adapter = GeminiAdapter(api_key=args.api_key, model_name=args.model or "gemini-2.0-flash-001")
    else:
        if args.local_path:
            model_adapter = HuggingFaceAdapter(model_name=args.model, local_path=args.local_path)
        else:
            api_token = os.environ.get("HF_API_TOKEN")
            if not api_token:
                print("Please set the HF_API_TOKEN environment variable.")
                return
            model_adapter = HuggingFaceAdapter(api_token=api_token, model_name=args.model)
    
    evaluator = ReasoningEvaluator(metric_registry=get_default_metric_registry(), model_adapter=model_adapter)
    
    print(f"Loading {args.benchmark} benchmark dataset...")
    benchmark_config = BENCHMARKS[args.benchmark]
    dataset = benchmark_config["load_fn"](args.samples)
    
    print("Preparing examples...")
    examples = [benchmark_config["format_fn"](item) for item in dataset]
    
    print(f"Running evaluation on {len(examples)} examples...")
    print(f"Rate limiting set to {args.rpm_limit} requests per minute")
    start_time = time.time()
    loop_start_time = time.time()
    
    results = []
    predictions = []
    references = []
    
    for i, example in enumerate(tqdm(examples)):
        rate_limit(loop_start_time, i, rpm_limit=args.rpm_limit)
        try:
            result = evaluator.evaluate_prompt(prompt=example["prompt"], reference=example["reference"], temperature=args.temp)
            prediction = extract_answer(result["prediction"], args.benchmark)
            actual_label = example["label"]
            
            # Convert prediction to integer for MMLU and ARC (A=0, B=1, C=2, D=3)
            if args.benchmark in ["mmlu_logic", "mmlu_math", "arc"]:
                letter_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
                prediction_int = letter_to_index.get(prediction, prediction) if isinstance(prediction, str) else prediction
            else:
                prediction_int = prediction
            
            # Debug logging
            print(f"Example {i+1}: Prediction={prediction} (type={type(prediction)}), Actual={actual_label} (type={type(actual_label)})")
            
            # Ensure integer comparison
            correct = prediction_int == actual_label
            
            result_item = {
                "prompt": example["prompt"],
                "response": result["prediction"],
                "reference": example["reference"],
                "metrics": result["metrics"],
                "prediction": prediction_int,
                "actual_label": actual_label,
                "correct": correct
            }
            results.append(result_item)
            predictions.append(prediction_int)
            references.append(actual_label)
            
            if args.verbose:
                print(f"\nExample {i+1}:")
                print(f"Prompt: {example['prompt'][:100]}...")
                print(f"Prediction: {prediction}")
                print(f"Actual: {actual_label}")
                print(f"Correct: {correct}")
                print("-" * 40)
        except Exception as e:
            print(f"Error evaluating example {i}: {e}")
    
    end_time = time.time()
    accuracy = calculate_accuracy(predictions, references)
    
    metric_scores = {}
    for result in results:
        for metric_name, metric_result in result["metrics"].items():
            if isinstance(metric_result, dict) and "score" in metric_result:
                if metric_name not in metric_scores:
                    metric_scores[metric_name] = []
                metric_scores[metric_name].append(metric_result["score"])
    
    avg_metric_scores = {metric: sum(scores) / len(scores) if scores else 0 for metric, scores in metric_scores.items()}
    
    benchmark_results = {
        "benchmark": args.benchmark,
        "model": args.model,
        "adapter": args.adapter,
        "samples": len(examples),
        "accuracy": accuracy,
        "merit_metrics": avg_metric_scores,
        "execution_time": end_time - start_time,
        "results": results
    }
    
    with open(args.output, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    print("\n" + "=" * 50)
    print(f"Benchmark: {args.benchmark}")
    print(f"Model: {args.model}")
    print(f"Adapter: {args.adapter}")
    print(f"Samples: {len(examples)}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nMERIT Metrics:")
    for metric, score in avg_metric_scores.items():
        print(f"  {metric}: {score:.4f}")
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")
    print(f"Results saved to {args.output}")
    print("=" * 50)

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)