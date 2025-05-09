"""
Dataset evaluation example for the MERIT framework.
"""
import os
import sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from merit.models.adapters.huggingface import HuggingFaceAdapter
from merit.core.evaluation import ReasoningEvaluator
from merit.core.metrics import get_default_metric_registry
from merit.datasets.loaders import get_default_dataset_registry, LogicalReasoningDataset, MathReasoningDataset, EthicalReasoningDataset
from merit.core.visualization import ReasoningVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description="Run MERIT evaluation on datasets")
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Dataset name (logical_reasoning, math_reasoning, ethical_reasoning)')
    parser.add_argument('--samples', type=int, default=5,
                        help='Number of samples to evaluate (default: 5)')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='Temperature for generation (default: 0.3)')
    parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help='Model name (default: meta-llama/Meta-Llama-3-8B-Instruct)')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed metric information')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to specified file path')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get API token
    api_token = os.environ.get("HF_API_TOKEN")
    if not api_token:
        print("Please set the HF_API_TOKEN environment variable.")
        print("You can get a token from https://huggingface.co/settings/tokens")
        return
    
    # Initialize model adapter
    model_adapter = HuggingFaceAdapter(
        api_token=api_token,
        model_name=args.model
    )
    
    # Initialize evaluator
    evaluator = ReasoningEvaluator(
        metric_registry=get_default_metric_registry(),
        model_adapter=model_adapter
    )
    
    # Use direct instantiation instead of registry for now
    print(f"Requested dataset: '{args.dataset}'")
    
    # Manual dataset selection to bypass registry issues
    if args.dataset == 'logical_reasoning':
        dataset = LogicalReasoningDataset()
    elif args.dataset == 'math_reasoning':
        dataset = MathReasoningDataset()
    elif args.dataset == 'ethical_reasoning':
        dataset = EthicalReasoningDataset()
    else:
        print(f"Dataset '{args.dataset}' not found. Available datasets: ['logical_reasoning', 'math_reasoning', 'ethical_reasoning']")
        return
    
    dataset.load()  # Load the dataset
    
    # Limit to requested number of samples
    samples = dataset.get_sample(args.samples)
    print(f"Evaluating {len(samples)} samples from {args.dataset} dataset")
    print("-" * 70)
    
    # Run evaluation on each sample
    all_results = []
    for i, example in enumerate(samples):
        print(f"\nSample {i+1}/{len(samples)}")
        print(f"Prompt: {example['prompt'][:100]}..." if len(example['prompt']) > 100 else f"Prompt: {example['prompt']}")
        
        result = evaluator.evaluate_prompt(
            prompt=example["prompt"],
            reference=example.get("reference"),
            temperature=args.temperature
        )
        
        # Print basic result info
        print(f"Response length: {len(result['prediction'])} chars")
        
        # Show beginning of response
        preview_length = 150
        print(f"Response preview: {result['prediction'][:preview_length]}..." if len(result['prediction']) > preview_length else f"Response: {result['prediction']}")
        
        print("Metrics:")
        for metric_name, metric_result in result["metrics"].items():
            if isinstance(metric_result, dict):
                if "score" in metric_result:
                    print(f"  {metric_name}: {metric_result['score']:.4f}")
                    
                    # If detailed flag is set, show more metric information
                    if args.detailed:
                        for key, value in metric_result.items():
                            if key != "score" and not isinstance(value, list) and not isinstance(value, dict):
                                print(f"    - {key}: {value}")
        
        all_results.append(result)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    # Calculate average scores for each metric
    metrics_summary = {}
    for result in all_results:
        for metric_name, metric_result in result["metrics"].items():
            if isinstance(metric_result, dict) and "score" in metric_result:
                if metric_name not in metrics_summary:
                    metrics_summary[metric_name] = []
                metrics_summary[metric_name].append(metric_result["score"])
    
    # Print average scores
    print("\nAverage metric scores:")
    for metric_name, scores in metrics_summary.items():
        avg_score = sum(scores) / len(scores)
        print(f"  {metric_name}: {avg_score:.4f}")
        
    # Save results if output path is specified
    if args.output:
        import json
        output_data = {
            "dataset": args.dataset,
            "model": args.model,
            "samples": len(samples),
            "results": all_results,
            "summary": {metric: sum(scores)/len(scores) for metric, scores in metrics_summary.items()}
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
