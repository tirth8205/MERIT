"""
Command Line Interface for MERIT.
Provides easy access to all MERIT functionality.
"""
import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional
import json

# Local imports
from .config import (
    load_config, 
    create_default_config_file, 
    validate_config_file,
    MeritConfig
)
from .experiments import ExperimentRunner, ExperimentConfig
from .models.local_models import ModelManager, get_system_recommendations
from .validation.human_validation import HumanAnnotationCollector, MetricValidator
from .validation.baseline_comparison import BaselineComparator
from .knowledge import EnhancedKnowledgeBase


def main():
    """Main CLI entry point"""
    parser = create_main_parser()
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        parser.print_help()


def create_main_parser():
    """Create the main argument parser"""
    
    parser = argparse.ArgumentParser(
        prog="merit",
        description="MERIT: Multi-dimensional Evaluation of Reasoning in Transformers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  merit init                          # Create default configuration
  merit validate-config config.yaml  # Validate configuration file
  merit run experiment.yaml          # Run experiment from config
  merit evaluate --model gpt2         # Quick evaluation
  merit models list                   # List available models
  merit system-info                   # Show system capabilities
        """
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version="MERIT 2.0.0"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        title="commands",
        description="Available MERIT commands",
        help="Command to run"
    )
    
    # Configuration commands
    add_config_commands(subparsers)
    
    # Experiment commands  
    add_experiment_commands(subparsers)
    
    # Model commands
    add_model_commands(subparsers)
    
    # Evaluation commands
    add_evaluation_commands(subparsers)
    
    # Validation commands
    add_validation_commands(subparsers)
    
    # Utility commands
    add_utility_commands(subparsers)
    
    return parser


def add_config_commands(subparsers):
    """Add configuration-related commands"""
    
    # Init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize MERIT with default configuration"
    )
    init_parser.add_argument(
        "--config-file", "-c",
        default="merit_config.yaml",
        help="Configuration file to create (default: merit_config.yaml)"
    )
    init_parser.add_argument(
        "--format", "-f",
        choices=["yaml", "json"],
        default="yaml",
        help="Configuration file format"
    )
    init_parser.set_defaults(func=cmd_init)
    
    # Validate config command
    validate_parser = subparsers.add_parser(
        "validate-config",
        help="Validate configuration file"
    )
    validate_parser.add_argument(
        "config_file",
        help="Configuration file to validate"
    )
    validate_parser.set_defaults(func=cmd_validate_config)
    
    # Show config command
    show_config_parser = subparsers.add_parser(
        "show-config",
        help="Show current configuration"
    )
    show_config_parser.add_argument(
        "--config", "-c",
        help="Configuration file to load"
    )
    show_config_parser.set_defaults(func=cmd_show_config)


def add_experiment_commands(subparsers):
    """Add experiment-related commands"""
    
    # Run experiment command
    run_parser = subparsers.add_parser(
        "run",
        help="Run MERIT experiment"
    )
    run_parser.add_argument(
        "config_file",
        help="Configuration file for the experiment"
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually running"
    )
    run_parser.add_argument(
        "--resume",
        help="Resume experiment from checkpoint"
    )
    run_parser.set_defaults(func=cmd_run_experiment)
    
    # List experiments command
    list_exp_parser = subparsers.add_parser(
        "list-experiments",
        help="List completed experiments"
    )
    list_exp_parser.add_argument(
        "--experiments-dir",
        default="experiments", 
        help="Experiments directory to search"
    )
    list_exp_parser.set_defaults(func=cmd_list_experiments)


def add_model_commands(subparsers):
    """Add model-related commands"""
    
    models_parser = subparsers.add_parser(
        "models",
        help="Model management commands"
    )
    models_subparsers = models_parser.add_subparsers(
        title="model commands",
        help="Model management operations"
    )
    
    # List models
    list_models_parser = models_subparsers.add_parser(
        "list",
        help="List available models"
    )
    list_models_parser.set_defaults(func=cmd_list_models)
    
    # Test model
    test_model_parser = models_subparsers.add_parser(
        "test",
        help="Test a model with sample prompts"
    )
    test_model_parser.add_argument(
        "model_name",
        help="Name of the model to test"
    )
    test_model_parser.add_argument(
        "--prompt", "-p",
        default="What is the capital of France?",
        help="Test prompt to use"
    )
    test_model_parser.set_defaults(func=cmd_test_model)
    
    # Benchmark models
    benchmark_parser = models_subparsers.add_parser(
        "benchmark",
        help="Benchmark multiple models"
    )
    benchmark_parser.add_argument(
        "models",
        nargs="+",
        help="Models to benchmark"
    )
    benchmark_parser.add_argument(
        "--output", "-o",
        default="model_benchmark.json",
        help="Output file for results"
    )
    benchmark_parser.set_defaults(func=cmd_benchmark_models)


def add_evaluation_commands(subparsers):
    """Add evaluation commands"""
    
    # Quick evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Quick evaluation of a model"
    )
    eval_parser.add_argument(
        "--model", "-m",
        required=True,
        help="Model to evaluate"
    )
    eval_parser.add_argument(
        "--dataset", "-d",
        default="arc",
        choices=["arc", "hellaswag", "mmlu_logic"],
        help="Dataset to use for evaluation"
    )
    eval_parser.add_argument(
        "--sample-size", "-s",
        type=int,
        default=50,
        help="Number of samples to evaluate"
    )
    eval_parser.add_argument(
        "--output", "-o",
        help="Output file for results"
    )
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare results from different experiments"
    )
    compare_parser.add_argument(
        "result_files",
        nargs="+",
        help="Result files to compare"
    )
    compare_parser.add_argument(
        "--output", "-o",
        default="comparison_report.html",
        help="Output file for comparison report"
    )
    compare_parser.set_defaults(func=cmd_compare_results)


def add_validation_commands(subparsers):
    """Add validation commands"""
    
    validation_parser = subparsers.add_parser(
        "validate",
        help="Validation commands"
    )
    validation_subparsers = validation_parser.add_subparsers(
        title="validation commands",
        help="Validation operations"
    )
    
    # Create annotation task
    annotate_parser = validation_subparsers.add_parser(
        "create-annotations",
        help="Create human annotation task"
    )
    annotate_parser.add_argument(
        "results_file",
        help="MERIT results file to create annotations for"
    )
    annotate_parser.add_argument(
        "--annotators", "-a",
        type=int,
        default=3,
        help="Number of annotators needed"
    )
    annotate_parser.add_argument(
        "--task-name", "-n",
        help="Name for the annotation task"
    )
    annotate_parser.set_defaults(func=cmd_create_annotations)
    
    # Validate metrics
    validate_metrics_parser = validation_subparsers.add_parser(
        "metrics",
        help="Validate MERIT metrics against human annotations"
    )
    validate_metrics_parser.add_argument(
        "merit_results",
        help="MERIT evaluation results file"
    )
    validate_metrics_parser.add_argument(
        "human_annotations", 
        help="Human annotation results file"
    )
    validate_metrics_parser.add_argument(
        "--output", "-o",
        default="validation_report.txt",
        help="Output file for validation report"
    )
    validate_metrics_parser.set_defaults(func=cmd_validate_metrics)


def add_utility_commands(subparsers):
    """Add utility commands"""
    
    # System info command
    sysinfo_parser = subparsers.add_parser(
        "system-info",
        help="Show system information and recommendations"
    )
    sysinfo_parser.set_defaults(func=cmd_system_info)
    
    # Knowledge base commands
    kb_parser = subparsers.add_parser(
        "knowledge",
        help="Knowledge base management"
    )
    kb_subparsers = kb_parser.add_subparsers(
        title="knowledge base commands",
        help="Knowledge base operations"
    )
    
    # KB stats
    kb_stats_parser = kb_subparsers.add_parser(
        "stats",
        help="Show knowledge base statistics"
    )
    kb_stats_parser.set_defaults(func=cmd_kb_stats)
    
    # KB search
    kb_search_parser = kb_subparsers.add_parser(
        "search",
        help="Search knowledge base"
    )
    kb_search_parser.add_argument(
        "query",
        help="Search query"
    )
    kb_search_parser.set_defaults(func=cmd_kb_search)


# Command implementations

def cmd_init(args):
    """Initialize MERIT with default configuration"""
    
    config_file = args.config_file
    if os.path.exists(config_file):
        response = input(f"Configuration file {config_file} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Initialization cancelled.")
            return
    
    create_default_config_file(config_file)
    print(f"MERIT initialized with configuration file: {config_file}")
    print("Edit the configuration file to customize your evaluation settings.")


def cmd_validate_config(args):
    """Validate configuration file"""
    
    config_file = args.config_file
    
    if not os.path.exists(config_file):
        print(f"Error: Configuration file not found: {config_file}")
        sys.exit(1)
    
    print(f"Validating configuration file: {config_file}")
    
    if validate_config_file(config_file):
        print("✓ Configuration is valid!")
    else:
        print("✗ Configuration validation failed!")
        sys.exit(1)


def cmd_show_config(args):
    """Show current configuration"""
    
    try:
        config = load_config(args.config)
        
        print("\nCurrent MERIT Configuration:")
        print("=" * 40)
        
        # Create a manager to show summary
        from .config import ConfigurationManager
        manager = ConfigurationManager(args.config)
        manager.print_config_summary()
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def cmd_run_experiment(args):
    """Run MERIT experiment"""
    
    config_file = args.config_file
    
    if not os.path.exists(config_file):
        print(f"Error: Configuration file not found: {config_file}")
        sys.exit(1)
    
    print(f"Loading experiment configuration: {config_file}")
    
    try:
        # Load configuration
        config = load_config(config_file)
        
        if args.dry_run:
            print("\nDRY RUN MODE - No actual evaluation will be performed")
            print(f"Experiment: {config.experiment.name}")
            print(f"Models: {[m.name for m in config.experiment.models]}")
            print(f"Datasets: {[d.name for d in config.experiment.datasets]}")
            print(f"Metrics: {[m.name for m in config.experiment.metrics]}")
            print(f"Number of runs: {config.experiment.num_runs}")
            return
        
        # Convert to ExperimentConfig for runner
        from .experiments.robust_evaluation import ExperimentConfig as RunnerConfig
        
        runner_config = RunnerConfig(
            experiment_name=config.experiment.name,
            models=[m.name for m in config.experiment.models],
            benchmarks=[d.name for d in config.experiment.datasets],
            sample_sizes=[d.sample_size for d in config.experiment.datasets],
            num_runs=config.experiment.num_runs,
            temperature=config.experiment.models[0].temperature if config.experiment.models else 0.1,
            max_tokens=config.experiment.models[0].max_tokens if config.experiment.models else 1000,
            random_seed=config.experiment.random_seed,
            metrics=[m.name for m in config.experiment.metrics],
            baseline_methods=["bert_score", "rouge"],
            statistical_tests=["t_test"],
            output_dir=config.experiment.output_dir
        )
        
        # Run experiment
        runner = ExperimentRunner(runner_config)
        results = runner.run_full_experiment()
        
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {config.experiment.output_dir}")
        
    except Exception as e:
        print(f"Error running experiment: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def cmd_list_experiments(args):
    """List completed experiments"""
    
    experiments_dir = Path(args.experiments_dir)
    
    if not experiments_dir.exists():
        print(f"Experiments directory not found: {experiments_dir}")
        return
    
    print(f"Experiments in {experiments_dir}:")
    print("-" * 40)
    
    found_experiments = False
    
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir():
            results_file = exp_dir / "experiment_results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    
                    print(f"📊 {exp_dir.name}")
                    print(f"   ID: {results.get('experiment_id', 'Unknown')}")
                    print(f"   Models: {len(results.get('model_results', {}))}")
                    print(f"   Started: {results.get('start_time', 'Unknown')}")
                    print(f"   Status: {'Completed' if results.get('end_time') else 'Running'}")
                    print()
                    
                    found_experiments = True
                    
                except Exception as e:
                    print(f"⚠️  {exp_dir.name} (error reading results: {e})")
    
    if not found_experiments:
        print("No completed experiments found.")


def cmd_list_models(args):
    """List available models"""
    
    manager = ModelManager()
    models = manager.list_available_models()
    
    print("Available Models:")
    print("=" * 30)
    
    for model_name, info in models.items():
        print(f"📦 {model_name}")
        print(f"   Parameters: {info.get('parameters', 'Unknown')}")
        print(f"   Type: {info.get('type', 'Unknown')}")
        print(f"   Memory: {info.get('memory_requirement', 'Unknown')}")
        print(f"   License: {info.get('license', 'Unknown')}")
        print(f"   Description: {info.get('description', 'No description')}")
        print()


def cmd_test_model(args):
    """Test a model with sample prompts"""
    
    model_name = args.model_name
    prompt = args.prompt
    
    print(f"Testing model: {model_name}")
    print(f"Prompt: {prompt}")
    print("-" * 50)
    
    try:
        manager = ModelManager()
        adapter = manager.load_model(model_name)
        
        response = adapter.generate(prompt, max_length=200, temperature=0.7)
        
        print("Response:")
        print(response)
        print("-" * 50)
        
        # Unload model
        manager.unload_model(model_name)
        
    except Exception as e:
        print(f"Error testing model: {e}")
        sys.exit(1)


def cmd_benchmark_models(args):
    """Benchmark multiple models"""
    
    models = args.models
    output_file = args.output
    
    print(f"Benchmarking models: {', '.join(models)}")
    
    try:
        manager = ModelManager()
        
        test_prompts = [
            "What is the capital of France?",
            "Explain the process of photosynthesis.",
            "Solve: 2x + 5 = 15",
            "What are the benefits of renewable energy?"
        ]
        
        results = manager.benchmark_models(
            models=models,
            test_prompts=test_prompts,
            max_length=200,
            temperature=0.7
        )
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Benchmark results saved to: {output_file}")
        
        # Show summary
        print("\nPerformance Summary:")
        print("-" * 30)
        
        for model_name, metrics in results.get("performance_metrics", {}).items():
            print(f"{model_name}:")
            print(f"  Avg time: {metrics.get('average_generation_time', 0):.2f}s")
            print(f"  Tokens/sec: {metrics.get('average_tokens_per_second', 0):.1f}")
            print()
        
    except Exception as e:
        print(f"Error benchmarking models: {e}")
        sys.exit(1)


def cmd_evaluate(args):
    """Quick evaluation of a model"""
    
    model_name = args.model
    dataset = args.dataset
    sample_size = args.sample_size
    output_file = args.output
    
    print(f"Quick evaluation:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset}")
    print(f"  Sample size: {sample_size}")
    
    try:
        # Create temporary config for quick evaluation
        from .experiments.robust_evaluation import ExperimentConfig, ExperimentRunner
        
        quick_config = ExperimentConfig(
            experiment_name=f"quick_eval_{model_name}_{dataset}",
            models=[model_name],
            benchmarks=[dataset],
            sample_sizes=[sample_size],
            num_runs=1,
            temperature=0.1,
            max_tokens=500,
            random_seed=42,
            metrics=["logical_consistency", "factual_accuracy", "reasoning_steps", "alignment"],
            baseline_methods=[],
            statistical_tests=[],
            output_dir=f"quick_eval_{model_name}_{dataset}"
        )
        
        runner = ExperimentRunner(quick_config)
        results = runner.run_full_experiment()
        
        # Show quick summary
        print("\nResults Summary:")
        print("-" * 20)
        
        model_results = results.get("model_results", {}).get(model_name, {})
        if "error" not in model_results:
            benchmark_results = model_results.get("benchmarks", {}).get(dataset, {})
            size_results = benchmark_results.get("sample_sizes", {}).get(str(sample_size), {})
            
            if "statistics" in size_results:
                stats = size_results["statistics"]
                acc = stats.get("task_accuracy", {}).get("mean", 0)
                print(f"Task Accuracy: {acc:.3f}")
                
                for metric_name, metric_stats in stats.get("metric_statistics", {}).items():
                    score = metric_stats.get("mean_across_runs", 0)
                    print(f"{metric_name}: {score:.3f}")
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def cmd_compare_results(args):
    """Compare results from different experiments"""
    
    result_files = args.result_files
    output_file = args.output
    
    print(f"Comparing {len(result_files)} result files...")
    
    try:
        # Load all result files
        all_results = {}
        
        for result_file in result_files:
            if not os.path.exists(result_file):
                print(f"Warning: Result file not found: {result_file}")
                continue
            
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            exp_name = results.get("config", {}).get("experiment_name", Path(result_file).stem)
            all_results[exp_name] = results
        
        if not all_results:
            print("No valid result files found.")
            sys.exit(1)
        
        # Create comparison report
        report_lines = []
        report_lines.append("MERIT EXPERIMENT COMPARISON REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Comparing {len(all_results)} experiments")
        report_lines.append(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary table
        report_lines.append("EXPERIMENT SUMMARY")
        report_lines.append("-" * 20)
        
        for exp_name, results in all_results.items():
            report_lines.append(f"Experiment: {exp_name}")
            model_count = len(results.get("model_results", {}))
            report_lines.append(f"  Models evaluated: {model_count}")
            
            if "statistical_analysis" in results and "rankings" in results["statistical_analysis"]:
                rankings = results["statistical_analysis"]["rankings"]
                if "by_task_accuracy" in rankings and rankings["by_task_accuracy"]:
                    best_model = rankings["by_task_accuracy"][0]
                    report_lines.append(f"  Best model: {best_model[0]} ({best_model[1]:.3f} accuracy)")
            
            report_lines.append("")
        
        # Save report
        with open(output_file, 'w') as f:
            f.write("\n".join(report_lines))
        
        print(f"Comparison report saved to: {output_file}")
        
    except Exception as e:
        print(f"Error comparing results: {e}")
        sys.exit(1)


def cmd_create_annotations(args):
    """Create human annotation task"""
    
    results_file = args.results_file
    annotators = args.annotators
    task_name = args.task_name or f"annotation_task_{Path(results_file).stem}"
    
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)
    
    try:
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Extract examples
        examples = []
        for model_results in results.get("model_results", {}).values():
            for benchmark_results in model_results.get("benchmarks", {}).values():
                for size_results in benchmark_results.get("sample_sizes", {}).values():
                    for run_results in size_results.get("runs", []):
                        for individual_result in run_results.get("individual_results", []):
                            examples.append({
                                "prompt": individual_result["prompt"],
                                "response": individual_result["response"],
                                "reference": individual_result.get("reference", "")
                            })
                        break  # Only take first run
                    break  # Only take first size
                break  # Only take first benchmark
            break  # Only take first model
        
        if not examples:
            print("No examples found in results file.")
            sys.exit(1)
        
        # Create annotation task
        collector = HumanAnnotationCollector()
        task_id = collector.create_annotation_task(
            examples=examples[:50],  # Limit to 50 examples for annotation
            task_name=task_name,
            annotators_needed=annotators
        )
        
        print(f"✓ Annotation task created: {task_id}")
        print(f"✓ Annotation templates created for {annotators} annotators")
        print("✓ Distribute the annotation templates to human annotators")
        
    except Exception as e:
        print(f"Error creating annotation task: {e}")
        sys.exit(1)


def cmd_validate_metrics(args):
    """Validate MERIT metrics against human annotations"""
    
    merit_results_file = args.merit_results
    human_annotations_file = args.human_annotations
    output_file = args.output
    
    if not os.path.exists(merit_results_file):
        print(f"Error: MERIT results file not found: {merit_results_file}")
        sys.exit(1)
    
    if not os.path.exists(human_annotations_file):
        print(f"Error: Human annotations file not found: {human_annotations_file}")
        sys.exit(1)
    
    try:
        # Load data
        with open(merit_results_file, 'r') as f:
            merit_results = json.load(f)
        
        with open(human_annotations_file, 'r') as f:
            human_annotations = json.load(f)
        
        # Extract individual results for validation
        merit_individual_results = []
        for model_results in merit_results.get("model_results", {}).values():
            for benchmark_results in model_results.get("benchmarks", {}).values():
                for size_results in benchmark_results.get("sample_sizes", {}).values():
                    for run_results in size_results.get("runs", []):
                        merit_individual_results.extend(run_results.get("individual_results", []))
                        break
                    break
                break
            break
        
        # Validate metrics
        validator = MetricValidator()
        validation_results = validator.validate_metric_performance(
            merit_individual_results,
            human_annotations.get("examples", [])
        )
        
        # Create validation report
        validator.create_validation_report(validation_results, output_file)
        
        print(f"✓ Metric validation completed")
        print(f"✓ Validation report saved to: {output_file}")
        
        # Show summary
        if "overall_summary" in validation_results:
            summary = validation_results["overall_summary"]
            print(f"\nValidation Summary:")
            print(f"  Overall assessment: {summary.get('overall_assessment', 'Unknown')}")
            print(f"  Average correlation: {summary.get('average_correlation', 0):.3f}")
        
    except Exception as e:
        print(f"Error validating metrics: {e}")
        sys.exit(1)


def cmd_system_info(args):
    """Show system information and recommendations"""
    
    print("MERIT System Information")
    print("=" * 30)
    
    try:
        # Get system recommendations
        recommendations = get_system_recommendations()
        device_info = recommendations["device_info"]
        
        print(f"Device: {device_info['device']}")
        
        if device_info["device"] == "mps":
            print(f"Unified memory: {device_info.get('unified_memory', 'Unknown')}")
            print(f"Estimated memory: {device_info.get('estimated_memory_gb', 'Unknown')} GB")
        elif device_info["device"] == "cuda":
            total_gb = device_info["total_memory"] / (1024**3)
            print(f"Total GPU memory: {total_gb:.1f} GB")
        else:
            available_gb = device_info["available_memory"] / (1024**3)
            total_gb = device_info["total_memory"] / (1024**3)
            print(f"Available memory: {available_gb:.1f} GB / {total_gb:.1f} GB")
        
        print("\nRecommended Models:")
        for model in recommendations["recommended_models"]:
            print(f"  • {model}")
        
        print("\nPerformance Tips:")
        for tip in recommendations["performance_tips"]:
            print(f"  • {tip}")
        
        # Check for required dependencies
        print(f"\nDependency Status:")
        
        try:
            import torch
            print(f"  ✓ PyTorch: {torch.__version__}")
        except ImportError:
            print("  ✗ PyTorch: Not installed")
        
        try:
            import transformers
            print(f"  ✓ Transformers: {transformers.__version__}")
        except ImportError:
            print("  ✗ Transformers: Not installed")
        
        try:
            import datasets
            print(f"  ✓ Datasets: {datasets.__version__}")
        except ImportError:
            print("  ✗ Datasets: Not installed")
        
        try:
            import scipy
            print(f"  ✓ SciPy: {scipy.__version__}")
        except ImportError:
            print("  ✗ SciPy: Not installed (statistical analysis limited)")
        
        try:
            import spacy
            print(f"  ✓ spaCy: {spacy.__version__}")
        except ImportError:
            print("  ✗ spaCy: Not installed (NLP features limited)")
        
    except Exception as e:
        print(f"Error getting system information: {e}")


def cmd_kb_stats(args):
    """Show knowledge base statistics"""
    
    try:
        kb = EnhancedKnowledgeBase()
        stats = kb.get_knowledge_statistics()
        
        print("Knowledge Base Statistics")
        print("=" * 30)
        
        # Structured database stats
        db_stats = stats["structured_database"]
        print(f"Structured Facts: {db_stats['total_facts']}")
        
        conf_dist = db_stats["confidence_distribution"]
        print(f"  High confidence (≥0.8): {conf_dist['high']}")
        print(f"  Medium confidence (0.5-0.8): {conf_dist['medium']}")
        print(f"  Low confidence (<0.5): {conf_dist['low']}")
        
        print(f"\nFacts by Category:")
        for category, count in db_stats["categories"].items():
            print(f"  {category}: {count}")
        
        # Wikipedia cache stats
        wiki_stats = stats["wikipedia_cache"]
        print(f"\nWikipedia Cache:")
        print(f"  Cached queries: {wiki_stats['cached_queries']}")
        print(f"  Cache size: {wiki_stats['cache_size_mb']:.1f} MB")
        
    except Exception as e:
        print(f"Error getting knowledge base statistics: {e}")


def cmd_kb_search(args):
    """Search knowledge base"""
    
    query = args.query
    
    try:
        kb = EnhancedKnowledgeBase()
        results = kb.search_knowledge(query)
        
        print(f"Knowledge Base Search Results for: '{query}'")
        print("=" * 50)
        
        # Structured facts
        structured_facts = results["structured_facts"]
        if structured_facts:
            print(f"Structured Facts ({len(structured_facts)}):")
            for fact in structured_facts[:5]:  # Show top 5
                print(f"  • {fact['subject']} {fact['predicate']} {fact['object']} (confidence: {fact['confidence']:.2f})")
        
        # Wikipedia results
        wikipedia_results = results["wikipedia_results"]
        if wikipedia_results:
            print(f"\nWikipedia Results ({len(wikipedia_results)}):")
            for result in wikipedia_results[:3]:  # Show top 3
                print(f"  • {result['title']}")
                print(f"    {result['content_preview'][:100]}...")
                print(f"    URL: {result['url']}")
                print()
        
        if not structured_facts and not wikipedia_results:
            print("No results found.")
        
    except Exception as e:
        print(f"Error searching knowledge base: {e}")


if __name__ == "__main__":
    main()