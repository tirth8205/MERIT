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

# Fix imports - use proper datasets import
from datasets import load_dataset

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from merit.models.adapters.huggingface import HuggingFaceAdapter
from merit.models.adapters.gemini import GeminiAdapter
from merit.core.evaluation import ReasoningEvaluator
from merit.core.metrics import get_default_metric_registry

# Supported benchmarks
BENCHMARKS = {
    "hellaswag": {
        "load_fn": lambda limit: load_dataset("hellaswag", split=f"validation[:{limit}]", trust_remote_code=True),
        "format_fn": lambda example: {
            "prompt": f"{example['ctx']}\nChoose the most plausible continuation:\n" + 
                     "\n".join([f"{idx+1}. {ending}" for idx, ending in enumerate(example['endings'])]),
            "reference": example['endings'][int(example['label'])],
            "label": int(example['label'])
        }
    },
    "arc": {
        "load_fn": lambda limit: load_dataset("ai2_arc", "ARC-Challenge", split=f"test[:{limit}]", trust_remote_code=True),
        "format_fn": lambda example: {
            "prompt": f"Question: {example['question']}\nChoose the correct answer:\n" + 
                     "\n".join([f"{choice}. {text}" for choice, text in zip(["A", "B", "C", "D", "E"][:len(example['choices']['text'])], example['choices']['text'])]),
            "reference": example['choices']['text'][example['choices']['label'].index(example['answerKey'])],
            "label": example['answerKey']
        }
    },
    "mmlu_math": {
        "load_fn": lambda limit: load_dataset("cais/mmlu", "high_school_mathematics", split=f"test[:{limit}]", trust_remote_code=True),
        "format_fn": lambda example: {
            "prompt": f"Question: {example['question']}\nChoose the correct answer:\n" + 
                     "\n".join([f"{choice}. {text}" for choice, text in zip(["A", "B", "C", "D"], example['choices'])]),
            "reference": example['choices'][example['answer']],
            "label": example['answer']
        }
    },
    "mmlu_logic": {
        "load_fn": lambda limit: load_dataset("cais/mmlu", "formal_logic", split=f"test[:{limit}]", trust_remote_code=True),
        "format_fn": lambda example: {
            "prompt": f"Question: {example['question']}\nChoose the correct answer:\n" + 
                     "\n".join([f"{choice}. {text}" for choice, text in zip(["A", "B", "C", "D"], example['choices'])]),
            "reference": example['choices'][example['answer']],
            "label": example['answer']
        }
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description='Run MERIT benchmark evaluation')
    parser.add_argument('--benchmark', type=str, required=True, choices=BENCHMARKS.keys(),
                        help='Benchmark dataset to evaluate on')
    parser.add_argument('--adapter', type=str, choices=['huggingface', 'gemini'], default='huggingface',
                        help='Model adapter to use (default: huggingface)')
    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct',
                        help='Model to evaluate (model name for HF, model type for Gemini)')
    parser.add_argument('--local_path', type=str, default=None,
                        help='Path to local model directory (for HuggingFace adapter)')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key (required for Gemini adapter)')
    parser.add_argument('--samples', type=int, default=50,
                        help='Number of samples to evaluate (maximum)')
    parser.add_argument('--rpm_limit', type=int, default=14,
                        help='Rate limit in requests per minute (default: 14)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file path for results (JSON)')
    parser.add_argument('--temp', type=float, default=0.1,
                        help='Temperature for generation')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed progress')
    return parser.parse_args()

def rate_limit(start_time, i, rpm_limit=14):
    """Basic rate limiter to respect API limits."""
    if i > 0 and rpm_limit > 0:
        elapsed = time.time() - start_time
        expected_time = (i / rpm_limit) * 60
        if elapsed < expected_time:
            sleep_time = expected_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

def extract_answer(response, benchmark):
    """
    Extract the answer choice from model response based on benchmark format.
    This is a simple implementation - may need enhancement for better extraction.
    """
    if benchmark == "hellaswag":
        # Look for numbers like "1.", "2.", etc.
        import re
        matches = re.findall(r'(?:^|\n)(\d+)[.)]', response)
        if matches:
            try:
                # Convert to 0-indexed for hellaswag
                return int(matches[0]) - 1
            except:
                pass
    elif benchmark in ["arc", "mmlu_math", "mmlu_logic"]:
        # Look for letter choices like "A.", "B.", etc.
        import re
        matches = re.findall(r'(?:^|\n|\s)([A-D])[.)]', response)
        if matches:
            return matches[0]  # Return the letter
    
    # Fallback: just return the full response
    return response

def calculate_accuracy(predictions, references):
    """Calculate accuracy of predictions against references."""
    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    return correct / len(predictions) if predictions else 0

def run_benchmark(args):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize appropriate model adapter
    if args.adapter == 'gemini':
        if not args.api_key and not os.environ.get("GOOGLE_API_KEY"):
            print("Error: Gemini adapter requires an API key. Please provide it with --api_key or set GOOGLE_API_KEY environment variable.")
            return
            
        model_adapter = GeminiAdapter(
            api_key=args.api_key,
            model_name=args.model or "gemini-2.0-flash-001"
        )
    else:  # huggingface adapter
        if args.local_path:
            # Use local model
            model_adapter = HuggingFaceAdapter(
                model_name=args.model,
                local_path=args.local_path
            )
        else:
            # Use API
            api_token = os.environ.get("HF_API_TOKEN")
            if not api_token:
                print("Please set the HF_API_TOKEN environment variable.")
                return
            
            model_adapter = HuggingFaceAdapter(
                api_token=api_token,
                model_name=args.model
            )
    
    # Initialize evaluator
    evaluator = ReasoningEvaluator(
        metric_registry=get_default_metric_registry(),
        model_adapter=model_adapter
    )
    
    # Load benchmark dataset
    print(f"Loading {args.benchmark} benchmark dataset...")
    benchmark_config = BENCHMARKS[args.benchmark]
    dataset = benchmark_config["load_fn"](args.samples)
    
    # Format examples for evaluation
    print("Preparing examples...")
    examples = []
    for item in dataset:
        examples.append(benchmark_config["format_fn"](item))
    
    print(f"Running evaluation on {len(examples)} examples...")
    print(f"Rate limiting set to {args.rpm_limit} requests per minute")
    start_time = time.time()
    loop_start_time = time.time()
    
    results = []
    predictions = []
    references = []
    
    for i, example in enumerate(tqdm(examples)):
        # Apply rate limiting 
        rate_limit(loop_start_time, i, rpm_limit=args.rpm_limit)
        
        try:
            # Evaluate the example
            result = evaluator.evaluate_prompt(
                prompt=example["prompt"],
                reference=example["reference"],
                temperature=args.temp
            )
            
            # Extract answer from response
            prediction = extract_answer(result["prediction"], args.benchmark)
            actual_label = example["label"]
            
            # Add to results
            result_item = {
                "prompt": example["prompt"],
                "response": result["prediction"],
                "reference": example["reference"],
                "metrics": result["metrics"],
                "prediction": prediction,
                "actual_label": actual_label,
                "correct": prediction == actual_label
            }
            results.append(result_item)
            
            # Track for accuracy calculation
            predictions.append(prediction)
            references.append(actual_label)
            
            if args.verbose:
                print(f"\nExample {i+1}:")
                print(f"Prompt: {example['prompt'][:100]}...")
                print(f"Prediction: {prediction}")
                print(f"Actual: {actual_label}")
                print(f"Correct: {prediction == actual_label}")
                print("-" * 40)
        
        except Exception as e:
            print(f"Error evaluating example {i}: {e}")
    
    end_time = time.time()
    
    # Calculate overall statistics
    accuracy = calculate_accuracy(predictions, references)
    
    # Calculate average metric scores
    metric_scores = {}
    for result in results:
        for metric_name, metric_result in result["metrics"].items():
            if isinstance(metric_result, dict) and "score" in metric_result:
                if metric_name not in metric_scores:
                    metric_scores[metric_name] = []
                metric_scores[metric_name].append(metric_result["score"])
    
    avg_metric_scores = {
        metric: sum(scores) / len(scores) if scores else 0
        for metric, scores in metric_scores.items()
    }
    
    # Prepare final results
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
    
    # Save results
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
