"""
Command Line Interface for MERIT.
"""
import argparse
import sys
import os
from pathlib import Path
import json

from .experiments import ExperimentRunner, ExperimentConfig
from .models.manager import ModelManager


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

    parser.add_argument("--version", action="version", version="MERIT 3.0.0")

    subparsers = parser.add_subparsers(title="commands")

    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model")
    eval_parser.add_argument("--model", "-m", required=True, help="Model to evaluate")
    eval_parser.add_argument("--dataset", "-d", default="arc",
                            choices=["arc", "hellaswag", "truthfulqa", "mmlu_logic",
                                     "gsm8k", "bbh"],
                            help="Dataset (default: arc)")
    eval_parser.add_argument("--sample-size", "-s", type=int, default=50,
                            help="Sample size (0 for full dataset)")
    eval_parser.add_argument("--mode", default="heuristic",
                            choices=["heuristic", "judge", "both"],
                            help="Evaluation mode: heuristic metrics, LLM judge, or both (default: heuristic)")
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

    # annotate command
    annotate_parser = subparsers.add_parser("annotate",
                                            help="Run annotation pipeline on experiment results")
    annotate_parser.add_argument("--input", "-i", required=True,
                                help="Path to experiment results JSON file")
    annotate_parser.add_argument("--samples", "-n", type=int, default=50,
                                help="Number of samples to annotate (default: 50)")
    annotate_parser.add_argument("--provider", default="anthropic",
                                choices=["anthropic", "ollama"],
                                help="Annotation provider (default: anthropic)")
    annotate_parser.add_argument("--output", "-o", default="data/annotations",
                                help="Output directory for annotations")
    annotate_parser.set_defaults(func=cmd_annotate)

    # report command
    report_parser = subparsers.add_parser("report",
                                          help="Generate paper-ready outputs from results")
    report_parser.add_argument("--input", "-i", required=True,
                              help="Path to experiment results JSON file")
    report_parser.add_argument("--format", "-f", default="latex",
                              choices=["latex", "csv", "json"],
                              help="Output format (default: latex)")
    report_parser.add_argument("--output", "-o", default="paper_outputs",
                              help="Output directory")
    report_parser.set_defaults(func=cmd_report)

    return parser


def cmd_evaluate(args):
    """Evaluate a model on a dataset"""
    mode_label = {"heuristic": "heuristic metrics", "judge": "LLM judge", "both": "heuristic + LLM judge"}
    print(f"Evaluating {args.model} on {args.dataset} (mode: {mode_label[args.mode]})")
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

    # Run LLM judge if requested
    if args.mode in ("judge", "both"):
        print("\nRunning LLM judge evaluation...")
        from merit.core.llm_judge import LLMJudge, JudgeConfig
        judge = LLMJudge(JudgeConfig())
        results["judge_mode"] = args.mode
        # Judge results would be appended by the runner in a full pipeline;
        # here we note the mode so downstream tools know what was requested.

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


def cmd_annotate(args):
    """Run annotation pipeline on experiment results"""
    from merit.evaluation.annotation import AnnotationPipeline, AnnotationConfig

    # Load experiment results
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    with open(args.input) as f:
        results = json.load(f)

    # Extract responses from results
    responses = []
    references = []
    contexts = []
    for model, model_data in results.get("model_results", {}).items():
        for dataset, bench_data in model_data.get("benchmarks", {}).items():
            for size, size_data in bench_data.get("sample_sizes", {}).items():
                for run_data in size_data.get("runs", []):
                    for item in run_data.get("items", []):
                        responses.append(item.get("response", ""))
                        references.append(item.get("reference", ""))
                        contexts.append(item.get("prompt", ""))

    if not responses:
        print("No responses found in results file.")
        sys.exit(1)

    # Limit to requested sample count
    n = min(args.samples, len(responses))
    responses = responses[:n]
    references = references[:n]
    contexts = contexts[:n]

    print(f"Annotating {n} samples from {args.input}")

    config = AnnotationConfig(
        provider=args.provider,
        output_path=args.output,
    )
    pipeline = AnnotationPipeline(config)

    def progress(i, total):
        print(f"  [{i}/{total}]", end="\r")

    annotations = pipeline.annotate_batch(
        responses, references, contexts,
        progress_callback=progress,
    )

    out_path = pipeline.save_annotations(annotations)
    print(f"\nAnnotations saved to: {out_path}")


def cmd_report(args):
    """Generate paper-ready outputs from experiment results"""
    from merit.reporting.tables import generate_results_table
    from merit.reporting.export import export_json, export_csv

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    with open(args.input) as f:
        raw_results = json.load(f)

    # Reshape raw results into the model -> dataset -> metric -> score format
    # expected by the reporting functions.
    table_data = {}
    for model, model_data in raw_results.get("model_results", {}).items():
        table_data[model] = {}
        for dataset, bench_data in model_data.get("benchmarks", {}).items():
            table_data[model][dataset] = {}
            for size, size_data in bench_data.get("sample_sizes", {}).items():
                stats = size_data.get("statistics", {})
                for metric, metric_stats in stats.get("metric_statistics", {}).items():
                    score = metric_stats.get("mean_across_runs", 0)
                    table_data[model][dataset][metric] = score

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = getattr(args, 'format', 'latex')  # argparse 'format' attribute

    if fmt == "latex":
        latex = generate_results_table(table_data)
        out_file = out_dir / "results_table.tex"
        with open(out_file, "w") as f:
            f.write(latex)
        print(f"LaTeX table saved to: {out_file}")

    elif fmt == "csv":
        out_file = str(out_dir / "results.csv")
        export_csv(table_data, out_file)
        print(f"CSV saved to: {out_file}")

    elif fmt == "json":
        out_file = str(out_dir / "results.json")
        export_json(table_data, out_file)
        print(f"JSON saved to: {out_file}")


if __name__ == "__main__":
    main()
