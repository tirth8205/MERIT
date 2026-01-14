"""
Command Line Interface for MERIT.
"""
import argparse
import sys
import os
from pathlib import Path
import json

from .experiments import ExperimentRunner, ExperimentConfig
from .models.local_models import ModelManager


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    if hasattr(args, 'func'):
        try:
            args.func(args)
        except KeyboardInterrupt:
            print("\nCancelled")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        parser.print_help()


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        prog="merit",
        description="MERIT: Multi-dimensional Evaluation of Reasoning in Transformers"
    )

    parser.add_argument("--version", action="version", version="MERIT 2.0.0")

    subparsers = parser.add_subparsers(title="commands")

    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model")
    eval_parser.add_argument("--model", "-m", required=True, help="Model to evaluate")
    eval_parser.add_argument("--dataset", "-d", default="arc",
                            choices=["arc", "hellaswag", "mmlu_logic"],
                            help="Dataset (default: arc)")
    eval_parser.add_argument("--sample-size", "-s", type=int, default=50,
                            help="Sample size (0 for full dataset)")
    eval_parser.add_argument("--output", "-o", help="Output file")
    eval_parser.set_defaults(func=cmd_evaluate)

    # models command
    models_parser = subparsers.add_parser("models", help="Model commands")
    models_sub = models_parser.add_subparsers()

    list_parser = models_sub.add_parser("list", help="List available models")
    list_parser.set_defaults(func=cmd_list_models)

    test_parser = models_sub.add_parser("test", help="Test a model")
    test_parser.add_argument("model_name", help="Model to test")
    test_parser.add_argument("--prompt", "-p", default="What is 2+2?", help="Test prompt")
    test_parser.set_defaults(func=cmd_test_model)

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare experiment results")
    compare_parser.add_argument("result_files", nargs="+", help="Result files to compare")
    compare_parser.add_argument("--output", "-o", default="comparison.txt", help="Output file")
    compare_parser.set_defaults(func=cmd_compare)

    return parser


def cmd_evaluate(args):
    """Evaluate a model on a dataset"""
    print(f"Evaluating {args.model} on {args.dataset}")
    print(f"Sample size: {args.sample_size if args.sample_size > 0 else 'full dataset'}")

    config = ExperimentConfig(
        experiment_name=f"eval_{args.model}_{args.dataset}",
        models=[args.model],
        benchmarks=[args.dataset],
        sample_sizes=[args.sample_size],
        num_runs=1,
        temperature=0.7,
        max_tokens=200,
        random_seed=42,
        metrics=["logical_consistency", "factual_accuracy", "reasoning_steps", "alignment"],
        output_dir=f"results_{args.model}_{args.dataset}"
    )

    runner = ExperimentRunner(config)
    results = runner.run_full_experiment()

    # Print summary
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    model_results = results.get("model_results", {}).get(args.model, {})
    if "error" not in model_results:
        benchmark_results = model_results.get("benchmarks", {}).get(args.dataset, {})
        size_key = str(args.sample_size)
        size_results = benchmark_results.get("sample_sizes", {}).get(size_key, {})

        if "statistics" in size_results:
            stats = size_results["statistics"]
            acc = stats.get("task_accuracy", {}).get("mean", 0)
            print(f"Accuracy: {acc:.1%}")

            for metric, metric_stats in stats.get("metric_statistics", {}).items():
                score = metric_stats.get("mean_across_runs", 0)
                print(f"{metric}: {score:.3f}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


def cmd_list_models(args):
    """List available models"""
    manager = ModelManager()
    models = manager.list_available_models()

    print("Available Models:")
    print("-" * 40)
    for name, info in models.items():
        print(f"{name}")
        print(f"  Size: {info.get('parameters', '?')}")
        print(f"  Memory: {info.get('memory_requirement', '?')}")
        print()


def cmd_test_model(args):
    """Test a model with a prompt"""
    print(f"Testing {args.model_name}")
    print(f"Prompt: {args.prompt}")
    print("-" * 40)

    manager = ModelManager()
    adapter = manager.load_model(args.model_name)
    response = adapter.generate(args.prompt, max_length=200)

    print(f"Response: {response}")
    manager.unload_model(args.model_name)


def cmd_compare(args):
    """Compare experiment results"""
    print(f"Comparing {len(args.result_files)} result files")

    all_results = {}
    for f in args.result_files:
        if os.path.exists(f):
            with open(f) as file:
                all_results[Path(f).stem] = json.load(file)

    # Generate comparison
    lines = ["MERIT Comparison Report", "=" * 40, ""]

    for name, results in all_results.items():
        lines.append(f"Experiment: {name}")
        for model, model_data in results.get("model_results", {}).items():
            lines.append(f"  Model: {model}")
            for dataset, bench_data in model_data.get("benchmarks", {}).items():
                for size, size_data in bench_data.get("sample_sizes", {}).items():
                    stats = size_data.get("statistics", {})
                    acc = stats.get("task_accuracy", {}).get("mean", 0)
                    lines.append(f"    {dataset}: {acc:.1%} accuracy")
        lines.append("")

    with open(args.output, 'w') as f:
        f.write("\n".join(lines))

    print(f"Comparison saved to: {args.output}")


if __name__ == "__main__":
    main()
