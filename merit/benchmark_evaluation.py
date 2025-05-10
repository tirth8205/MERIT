"""
Benchmark evaluation for the MERIT framework across multiple datasets.
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
            "prompt": f"Question: {example['ctx']}\nTo answer, explain your reasoning step-by-step, considering each option (1, 2, 3, 4) and eliminating incorrect ones. Then, select the correct answer as a number (1, 2, 3, or 4) and verify your choice. Provide your final answer in the format: **Final Answer: [number]**\n" + 
                     "\n".join([f"{idx+1}. {ending}" for idx, ending in enumerate(example['endings'])]),
            "reference": example['endings'][int(example['label'])],
            "label": str(int(example['label']))  # Store as string for consistency
        }
    },
    "arc": {
        "load_fn": lambda limit: load_dataset("ai2_arc", "ARC-Challenge", split=f"test[:{limit}]", trust_remote_code=True),
        "format_fn": lambda example: {
            "prompt": f"Question: {example['question']}\nTo answer, explain your reasoning step-by-step, considering each option (A, B, C, D, or E) and eliminating incorrect ones. Then, select the correct answer as a single letter (A, B, C, D, or E) and verify your choice. Provide your final answer in the format: **Final Answer: [letter]**\n" + 
                     "\n".join([f"{choice}. {text}" for choice, text in zip(["A", "B", "C", "D", "E"][:len(example['choices']['text'])], example['choices']['text'])]),
            "reference": example['choices']['text'][example['choices']['label'].index(example['answerKey'])],
            "label": str(example['answerKey'])  # Store as string
        }
    },
    "mmlu_math": {
        "load_fn": lambda limit: load_dataset("cais/mmlu", "high_school_mathematics", split=f"test[:{limit}]", trust_remote_code=True),
        "format_fn": lambda example: {
            "prompt": f"Question: {example['question']}\nTo answer, explain your reasoning step-by-step, considering each option (A, B, C, D) and eliminating incorrect ones. Then, select the correct answer as a single letter (A, B, C, or D) and verify your choice. Provide your final answer in the format: **Final Answer: [letter]**\n" + 
                     "\n".join([f"{choice}. {text}" for choice, text in zip(["A", "B", "C", "D"], example['choices'])]),
            "reference": example['choices'][example['answer']],
            "label": str(['A', 'B', 'C', 'D'][example['answer']])  # Convert index to letter
        }
    },
    "mmlu_logic": {
        "load_fn": lambda limit: load_dataset("cais/mmlu", "formal_logic", split=f"test[:{limit}]", trust_remote_code=True),
        "format_fn": lambda example: {
            "prompt": f"Question: {example['question']}\nTo answer, explain your reasoning step-by-step, considering each option (A, B, C, D) and eliminating incorrect ones. Then, select the correct answer as a single letter (A, B, C, or D) and verify your choice. Provide your final answer in the format: **Final Answer: [letter]**\n" + 
                     "\n".join([f"{choice}. {text}" for choice, text in zip(["A", "B", "C", "D"], example['choices'])]),
            "reference": example['choices'][example['answer']],
            "label": str(['A', 'B', 'C', 'D'][example['answer']])  # Convert index to letter
        }
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description='Run MERIT benchmark evaluation for multiple datasets')
    parser.add_argument('--benchmarks', type=str, nargs='+', default=['arc', 'mmlu_logic', 'mmlu_math', 'hellaswag'], choices=BENCHMARKS.keys(), help='List of benchmarks to evaluate')
    parser.add_argument('--adapter', type=str, choices=['huggingface', 'gemini'], default='gemini')
    parser.add_argument('--model', type=str, default='gemini-2.0-flash-001')
    parser.add_argument('--local_path', type=str, default=None)
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--rpm_limit', type=int, default=14)
    parser.add_argument('--output_prefix', type=str, default='results/gemini')
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
            r'**Final Answer: \[(\d+)\]**'
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                try:
                    return str(int(matches[-1]) - 1)  # 0-indexed, return as string
                except:
                    pass
    elif benchmark in ["arc", "mmlu_math", "mmlu_logic"]:
        patterns = [
            r'(?:^|\n|\s|^Answer:|^So the answer is|^Final Answer:)\s*([A-E])(?:\s|$|\.|,|\))',  # A, A., A), Answer: A
            r'[Tt]he correct answer is \**([A-E])\**',  # **A**
            r'[Oo]ption \**([A-E])\**',  # Option **A**
            r'\b([A-E])\b(?!\s*[\w])',  # Single A, B, C, D, E
            r'**Final Answer: \[([A-E])\]**'
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                return str(matches[-1])  # Return letter as string
        print(f"Warning: No valid answer extracted from response: {response[:100]}...")
    return "0"  # Fallback to "0" as string

def calculate_accuracy(predictions, references):
    correct = sum(1 for p, r in zip(predictions, references) if str(p) == str(r))
    return correct / len(predictions) if predictions else 0

def run_benchmark(args, benchmark):
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    output_file = f"{args.output_prefix}_{benchmark}_test.json"
    
    try:
        if args.adapter == 'gemini':
            if not args.api_key and not os.environ.get("GOOGLE_API_KEY"):
                print("Error: Gemini adapter requires an API key.")
                return
            model_adapter = GeminiAdapter(api_key=args.api_key, model_name=args.model)
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
        
        print(f"\nLoading {benchmark} benchmark dataset...")
        benchmark_config = BENCHMARKS[benchmark]
        dataset = benchmark_config["load_fn"](args.samples)
        
        print("Preparing examples...")
        examples = [benchmark_config["format_fn"](item) for item in dataset]
        
        print(f"Running evaluation on {len(examples)} examples...")
        print(f"Rate limiting set to {args.rpm_limit} requests per minute")
        start_time = time.time()
        loop_start_time = time.time()
        request_count = 0
        
        results = []
        predictions = []
        references = []
        
        for i, example in enumerate(tqdm(examples)):
            rate_limit(loop_start_time, i, rpm_limit=args.rpm_limit)
            try:
                result = evaluator.evaluate_prompt(prompt=example["prompt"], reference=example["reference"], temperature=args.temp)
                prediction = extract_answer(result["prediction"], benchmark)
                actual_label = example["label"]
                
                # Debug logging
                print(f"Example {i+1}: Prediction={prediction} (type={type(prediction)}), Actual={actual_label} (type={type(actual_label)})")
                
                # Ensure string comparison
                correct = str(prediction) == str(actual_label)
                
                result_item = {
                    "prompt": example["prompt"],
                    "response": result["prediction"],
                    "reference": example["reference"],
                    "metrics": result["metrics"],
                    "prediction": prediction,
                    "actual_label": actual_label,
                    "correct": correct
                }
                results.append(result_item)
                predictions.append(prediction)
                references.append(actual_label)
                request_count += 1
                
                if args.verbose:
                    print(f"\nExample {i+1}:")
                    print(f"Prompt: {example['prompt'][:100]}...")
                    print(f"Prediction: {prediction}")
                    print(f"Actual: {actual_label}")
                    print(f"Correct: {correct}")
                    print(f"Requests used: {request_count}")
                    print("-" * 40)
            except Exception as e:
                print(f"Error evaluating example {i+1} in {benchmark}: {e}")
                continue
        
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
            "benchmark": benchmark,
            "model": args.model,
            "adapter": args.adapter,
            "samples": len(examples),
            "accuracy": accuracy,
            "merit_metrics": avg_metric_scores,
            "execution_time": end_time - start_time,
            "requests_used": request_count,
            "results": results
        }
        
        with open(output_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        print("\n" + "=" * 50)
        print(f"Benchmark: {benchmark}")
        print(f"Model: {args.model}")
        print(f"Adapter: {args.adapter}")
        print(f"Samples: {len(examples)}")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nMERIT Metrics:")
        for metric, score in avg_metric_scores.items():
            print(f"  {metric}: {score:.4f}")
        print(f"\nExecution time: {end_time - start_time:.2f} seconds")
        print(f"Requests used: {request_count}")
        print(f"Results saved to {output_file}")
        print("=" * 50)
        
        return benchmark_results
    
    except Exception as e:
        print(f"Error in {benchmark} evaluation: {e}")
        return None

def main():
    args = parse_args()
    all_results = {}
    total_requests = 0
    
    for benchmark in args.benchmarks:
        print(f"\nStarting evaluation for {benchmark}...")
        results = run_benchmark(args, benchmark)
        if results:
            all_results[benchmark] = results
            total_requests += results.get("requests_used", 0)
    
    # Summary
    print("\n" + "=" * 50)
    print("Evaluation Summary:")
    for benchmark, results in all_results.items():
        print(f"{benchmark}: Accuracy = {results['accuracy']:.4f}, Execution Time = {results['execution_time']:.2f} seconds, Requests = {results['requests_used']}")
    print(f"Total Requests Used: {total_requests}")
    print("=" * 50)

if __name__ == "__main__":
    main()