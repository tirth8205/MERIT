"""Experiment runner for MERIT."""
import json
import uuid
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Any
from datetime import datetime
import numpy as np

# Import sibling modules
from .config import ExperimentConfig
from .datasets import load_dataset


class ExperimentRunner:
    """Runner for MERIT experiments"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.experiment_id = str(uuid.uuid4())[:8]
        self.output_dir = Path(config.output_dir) / f"{config.experiment_name}_{self.experiment_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components (lazy imports to avoid heavy deps at module level)
        from merit.models.manager import ModelManager
        self.model_manager = ModelManager()
        self.metrics = self._initialize_metrics()

        # Results storage
        self.results = {
            "experiment_id": self.experiment_id,
            "experiment_name": config.experiment_name,
            "config": asdict(config),
            "start_time": datetime.now().isoformat(),
            "model_results": {}
        }

    def _initialize_metrics(self) -> Dict[str, Any]:
        """Initialize metric instances"""
        from merit.core import (
            LogicalConsistencyMetric,
            FactualAccuracyMetric,
            ReasoningStepMetric,
            AlignmentMetric,
        )
        metric_map = {
            "logical_consistency": LogicalConsistencyMetric(),
            "factual_accuracy": FactualAccuracyMetric(),
            "reasoning_steps": ReasoningStepMetric(),
            "alignment": AlignmentMetric()
        }

        return {name: metric_map[name] for name in self.config.metrics if name in metric_map}

    def run_full_experiment(self) -> Dict[str, Any]:
        """Run the complete experiment"""
        print(f"\n{'='*60}")
        print(f"MERIT Experiment: {self.config.experiment_name}")
        print(f"Experiment ID: {self.experiment_id}")
        print(f"{'='*60}\n")

        # Set random seed
        np.random.seed(self.config.random_seed)

        # Run evaluation for each model
        for model_name in self.config.models:
            print(f"\nEvaluating model: {model_name}")
            print("-" * 60)

            try:
                model_results = self._evaluate_model(model_name)
                self.results["model_results"][model_name] = model_results
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                self.results["model_results"][model_name] = {"error": str(e)}

        # Save results
        self.results["end_time"] = datetime.now().isoformat()
        self._save_results()

        print(f"\n{'='*60}")
        print(f"Experiment completed!")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*60}\n")

        return self.results

    def _evaluate_model(self, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model"""
        model_results = {
            "model_name": model_name,
            "benchmarks": {}
        }

        # Load model
        print(f"Loading model: {model_name}")
        try:
            model_adapter = self.model_manager.load_model(model_name)
        except Exception as e:
            print(f"  Error loading model: {e}")
            return {"error": f"Failed to load model: {e}"}

        # Run on each benchmark
        for benchmark in self.config.benchmarks:
            print(f"\n  Benchmark: {benchmark}")

            benchmark_results = {
                "sample_sizes": {}
            }

            # Run for each sample size
            for sample_size in self.config.sample_sizes:
                print(f"    Sample size: {sample_size}")

                size_results = self._run_benchmark(
                    model_adapter,
                    benchmark,
                    sample_size
                )

                benchmark_results["sample_sizes"][str(sample_size)] = size_results

            model_results["benchmarks"][benchmark] = benchmark_results

        # Unload model to free memory
        self.model_manager.unload_model(model_name)

        return model_results

    def _run_benchmark(self, model_adapter, benchmark: str, sample_size: int) -> Dict[str, Any]:
        """Run evaluation on a benchmark"""
        results = {
            "runs": [],
            "statistics": {}
        }

        # Load dataset using standalone function
        dataset = load_dataset(
            benchmark,
            sample_size,
            seed=self.config.random_seed,
            use_instruction_format=self.config.use_instruction_format,
        )

        if not dataset:
            return {"error": "Failed to load dataset"}

        # Run multiple times for robustness
        for run_idx in range(self.config.num_runs):
            print(f"      Run {run_idx + 1}/{self.config.num_runs}")

            run_results = self._run_evaluation(model_adapter, dataset)
            results["runs"].append(run_results)

        # Calculate statistics
        results["statistics"] = self._calculate_statistics(results["runs"])

        return results

    def _run_evaluation(self, model_adapter, dataset: List[Dict]) -> Dict[str, Any]:
        """Run evaluation on dataset samples"""
        individual_results = []
        metric_scores = {metric_name: [] for metric_name in self.metrics.keys()}
        correct_count = 0

        for idx, item in enumerate(dataset):
            # Generate response
            try:
                response = model_adapter.generate(
                    item["prompt"],
                    max_length=self.config.max_tokens,
                    temperature=self.config.temperature
                    # Note: repetition_penalty is handled by each model adapter
                )
            except Exception as e:
                print(f"        Error generating for item {idx}: {e}")
                response = ""

            # Evaluate with metrics
            item_metrics = {}
            for metric_name, metric in self.metrics.items():
                try:
                    result = metric.compute(response, item.get("reference"))
                    score = result["score"] if isinstance(result, dict) else result
                    item_metrics[metric_name] = score
                    metric_scores[metric_name].append(score)
                except Exception as e:
                    print(f"        Error computing {metric_name}: {e}")
                    item_metrics[metric_name] = 0.0
                    metric_scores[metric_name].append(0.0)

            # Check task accuracy - improved for multiple choice
            is_correct = self._check_answer_correctness(
                response,
                item.get("reference", ""),
                item.get("reference_letter", ""),
                item.get("task_type", "default")
            )
            if is_correct:
                correct_count += 1

            individual_results.append({
                "prompt": item["prompt"],
                "response": response,
                "reference": item.get("reference", ""),
                "reference_letter": item.get("reference_letter", ""),
                "metrics": item_metrics,
                "correct": is_correct
            })

        # Calculate average scores
        avg_metrics = {
            name: np.mean(scores) if scores else 0.0
            for name, scores in metric_scores.items()
        }

        task_accuracy = correct_count / len(dataset) if dataset else 0.0

        return {
            "individual_results": individual_results,
            "average_metrics": avg_metrics,
            "task_accuracy": task_accuracy,
            "total_samples": len(dataset)
        }

    def _check_answer_correctness(self, response: str, reference: str, reference_letter: str, task_type: str) -> bool:
        """Check if the response contains the correct answer"""
        response_lower = response.lower().strip()
        reference_lower = reference.lower().strip()

        if task_type == "multiple_choice" and reference_letter:
            # For multiple choice, check if the correct letter appears at the start or prominently
            letter = reference_letter.upper()
            letter_lower = reference_letter.lower()

            # Check various patterns for answer selection
            patterns = [
                f"({letter})",           # (A)
                f"{letter})",            # A)
                f"{letter}.",            # A.
                f"{letter}:",            # A:
                f"answer is {letter_lower}",
                f"answer: {letter_lower}",
                f"choose {letter_lower}",
                f"select {letter_lower}",
                f"option {letter_lower}",
                f"{letter_lower} is correct",
                f"{letter_lower} is the",
            ]

            # Check if response starts with the letter
            first_chars = response_lower[:20]
            if letter_lower in first_chars or f"({letter_lower})" in first_chars:
                return True

            # Check for patterns
            for pattern in patterns:
                if pattern in response_lower:
                    return True

            # Also check if the full reference answer text is present
            if reference_lower in response_lower:
                return True

        else:
            # Default: check if reference text is in response
            if reference_lower in response_lower:
                return True

        return False

    def _calculate_statistics(self, runs: List[Dict]) -> Dict[str, Any]:
        """Calculate statistics across multiple runs"""
        if not runs:
            return {}

        # Extract metric scores across runs
        metric_names = list(runs[0]["average_metrics"].keys()) if runs else []

        stats = {
            "metric_statistics": {},
            "task_accuracy": {}
        }

        # Calculate statistics for each metric
        for metric_name in metric_names:
            scores = [run["average_metrics"][metric_name] for run in runs]

            stats["metric_statistics"][metric_name] = {
                "mean_across_runs": np.mean(scores),
                "std_across_runs": np.std(scores),
                "min_across_runs": np.min(scores),
                "max_across_runs": np.max(scores)
            }

        # Calculate task accuracy statistics
        accuracies = [run["task_accuracy"] for run in runs]
        stats["task_accuracy"] = {
            "mean": np.mean(accuracies),
            "std": np.std(accuracies),
            "min": np.min(accuracies),
            "max": np.max(accuracies)
        }

        return stats

    def _save_results(self):
        """Save experiment results to file"""
        try:
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)

            results_file = self.output_dir / "experiment_results.json"

            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)

            print(f"\nResults saved to: {results_file}")
        except Exception as e:
            print(f"\nWarning: Failed to save results: {e}")
            # Try to print results to console instead
            print("\nResults:")
            print(json.dumps(self.results, indent=2, default=str))


def create_default_config(experiment_name: str = "merit_experiment") -> ExperimentConfig:
    """Create a default experiment configuration"""
    return ExperimentConfig(
        experiment_name=experiment_name,
        models=["tinyllama-1b"],  # Use instruction-tuned model by default
        benchmarks=["arc"],
        sample_sizes=[10, 50],
        num_runs=3,
        temperature=0.7,
        max_tokens=200,
        random_seed=42,
        metrics=["logical_consistency", "factual_accuracy", "reasoning_steps", "alignment"],
        baseline_methods=["bert_score"],
        statistical_tests=["t_test"],
        output_dir="experiments",
        use_instruction_format=True
    )
