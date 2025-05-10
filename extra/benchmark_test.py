"""
Benchmark testing for the MERIT framework.
"""
import os
import sys
import json
import argparse
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from merit.models.adapters.huggingface import HuggingFaceAdapter
from merit.core.evaluation import ReasoningEvaluator
from merit.core.metrics import get_default_metric_registry
from merit.datasets.loaders import get_default_dataset_registry
from merit.core.utils import BenchmarkRunner

def parse_args():
    parser = argparse.ArgumentParser(description='Run MERIT benchmark tests')
    parser.add_argument('--benchmark', type=str, default='hellaswag', choices=['hellaswag', 'arc', 'mmlu', 'bbh'], 
                        help='Benchmark to run')
    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct',
                        help='Model to evaluate')
    parser.add_argument('--samples', type=int, default=50,
                        help='Number of samples to evaluate')
    parser.add_argument('--output', type=str, default='benchmark_results',
                        help='Output directory for results')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with baseline results')
    return parser.parse_args()

def load_benchmark_dataset(benchmark_name, num_samples=50):
    """Load a standardized benchmark dataset."""
    if benchmark_name == 'hellaswag':
        dataset = load_dataset("hellaswag", split="validation[:{}]".format(num_samples))
        # Transform to MERIT format
        return [{"prompt": item["ctx"], 
                 "reference": item["endings"][item["label"]], 
                 "choices": item["endings"]} for item in dataset]
    
    elif benchmark_name == 'arc':
        dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test[:{}]".format(num_samples))
        return [{"prompt": item["question"], 
                 "reference": item["choices"]["text"][item["choices"]["label"].index(item["answerKey"])],
                 "choices": item["choices"]["text"]} for item in dataset]
    
    elif benchmark_name == 'mmlu':
        # Using a subset of MMLU
        dataset = load_dataset("cais/mmlu", "high_school_mathematics", split="test[:{}]".format(num_samples))
        return [{"prompt": item["question"], 
                 "reference": item["choices"][item["answer"]], 
                 "choices": item["choices"]} for item in dataset]
    
    elif benchmark_name == 'bbh':
        # Using a subset of BBH
        dataset = load_dataset("lukaemon/bbh", "logical_deduction", split="test[:{}]".format(num_samples))
        return [{"prompt": item["input"], 
                 "reference": item["target"]} for item in dataset]
    
    else:
        raise ValueError(f"Benchmark {benchmark_name} not supported")

def format_prompt_for_benchmark(benchmark_name, item):
    """Format prompts appropriately for each benchmark."""
    if benchmark_name in ['hellaswag', 'arc', 'mmlu']:
        # Multiple choice format
        prompt = f"{item['prompt']}\n\nChoose the most appropriate option:\n"
        for i, choice in enumerate(item['choices']):
            prompt += f"{chr(65+i)}. {choice}\n"
        return prompt
    else:
        # For other benchmarks, use the prompt directly
        return item['prompt']

def run_benchmark(args):
    # Initialize model adapter
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
    dataset = load_benchmark_dataset(args.benchmark, args.samples)
    
    # Format prompts for the benchmark
    formatted_dataset = []
    for item in dataset:
        formatted_item = item.copy()
        formatted_item['prompt'] = format_prompt_for_benchmark(args.benchmark, item)
        formatted_dataset.append(formatted_item)
    
    # Initialize benchmark runner
    runner = BenchmarkRunner()
    
    # Run benchmark
    print(f"Running benchmark on {args.model}...")
    benchmark_result = runner.run(
        evaluator, 
        formatted_dataset, 
        name=f"{args.benchmark}-{args.model.split('/')[-1]}"
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Save results
    result_path = os.path.join(args.output, f"{args.benchmark}_{args.model.split('/')[-1]}.json")
    with open(result_path, 'w') as f:
        json.dump(benchmark_result, f, indent=2)
    
    # Print summary
    print(f"\nBenchmark results saved to: {result_path}")
    print("\nMetric summary:")
    
    for metric_name, metric_value in benchmark_result['result']['aggregated'].items():
        print(f"  {metric_name}: {metric_value['mean']:.4f}")
    
    if args.compare:
        compare_with_baselines(args.benchmark, benchmark_result)

def compare_with_baselines(benchmark_name, result):
    """Compare results with published baselines."""
    # This would ideally load from a database of published results
    # Here we're using placeholder data
    baselines = {
        'hellaswag': {
            'gpt-4': 0.953,
            'claude-3-opus': 0.932,
            'llama-3-70b': 0.886
        },
        'arc': {
            'gpt-4': 0.961,
            'claude-3-opus': 0.952,
            'llama-3-70b': 0.904
        },
        'mmlu': {
            'gpt-4': 0.860,
            'claude-3-opus': 0.845,
            'llama-3-70b': 0.797
        },
        'bbh': {
            'gpt-4': 0.831,
            'claude-3-opus': 0.802,
            'llama-3-70b': 0.768
        }
    }
    
    if benchmark_name in baselines:
        print("\nComparison with baselines:")
        print(f"  MERIT framework: {result['result']['aggregated']['logical_consistency']['mean']:.4f}")
        for model, score in baselines[benchmark_name].items():
            print(f"  {model}: {score:.4f}")

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
