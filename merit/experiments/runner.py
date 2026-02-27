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
        self.baselines = self._initialize_baselines()

        # Results storage
        self.results = {
            "experiment_id": self.experiment_id,
            "experiment_name": config.experiment_name,
            "config": asdict(config),
            "start_time": datetime.now().isoformat(),
            "model_results": {}
        }

    def _initialize_metrics(self) -> Dict[str, Any]:
        """Initialize metric instances based on config.metric_mode.

        Returns a dict with up to two keys:
        - ``"heuristic"``: dict of heuristic metric instances (when mode is "heuristic" or "both")
        - ``"llm_judge"``: an LLMJudge instance (when mode is "llm_judge" or "both")
        """
        mode = getattr(self.config, "metric_mode", "heuristic")
        result: Dict[str, Any] = {}

        if mode in ("heuristic", "both"):
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
                "alignment": AlignmentMetric(),
            }
            result["heuristic"] = {
                name: metric_map[name] for name in self.config.metrics if name in metric_map
            }

        if mode in ("llm_judge", "both"):
            from merit.core.llm_judge import LLMJudge
            result["llm_judge"] = LLMJudge()

        return result

    def _initialize_baselines(self) -> Dict[str, Any]:
        """Initialize baseline evaluation methods from config.baseline_methods.

        Uses lazy imports so that optional dependencies (bert-score, anthropic)
        are only required when actually configured.
        """
        baselines: Dict[str, Any] = {}
        configured = getattr(self.config, "baseline_methods", [])
        if not configured:
            return baselines

        baseline_registry = {
            "bert_score": "_make_bertscore",
            "geval": "_make_geval",
        }

        for name in configured:
            factory = baseline_registry.get(name)
            if factory:
                try:
                    baselines[name] = getattr(self, factory)()
                except Exception as e:
                    print(f"Warning: could not initialize baseline '{name}': {e}")
        return baselines

    @staticmethod
    def _make_bertscore():
        from merit.baselines.bertscore import BERTScoreBaseline
        return BERTScoreBaseline()

    @staticmethod
    def _make_geval():
        from merit.baselines.geval import GEvalBaseline
        return GEvalBaseline()

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
        """Run evaluation on dataset samples.

        The returned dict uses a structured layout:
        - ``merit_heuristic``: per-metric averages when mode is ``"heuristic"`` or ``"both"``
        - ``merit_llm_judge``: per-dimension averages when mode is ``"llm_judge"`` or ``"both"``
        - ``baselines``: baseline scores when ``config.baseline_methods`` is non-empty
        - ``average_metrics``: flat dict kept for backward compatibility (same as ``merit_heuristic``)
        """
        heuristic_metrics = self.metrics.get("heuristic", {})
        llm_judge = self.metrics.get("llm_judge")

        individual_results = []
        heuristic_scores: Dict[str, List[float]] = {m: [] for m in heuristic_metrics}
        judge_scores: Dict[str, List[float]] = {}
        baseline_scores: Dict[str, List[float]] = {b: [] for b in self.baselines}
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

            reference = item.get("reference", "")
            item_metrics: Dict[str, Any] = {}

            # --- Heuristic metrics ---
            if heuristic_metrics:
                heuristic_item: Dict[str, float] = {}
                for metric_name, metric in heuristic_metrics.items():
                    try:
                        result = metric.compute(response, reference)
                        score = result["score"] if isinstance(result, dict) else result
                        heuristic_item[metric_name] = score
                        heuristic_scores[metric_name].append(score)
                    except Exception as e:
                        print(f"        Error computing {metric_name}: {e}")
                        heuristic_item[metric_name] = 0.0
                        heuristic_scores[metric_name].append(0.0)
                item_metrics["merit_heuristic"] = heuristic_item

            # --- LLM-judge metrics ---
            if llm_judge is not None:
                try:
                    judge_results = llm_judge.evaluate_all(response, reference)
                    judge_item = {
                        dim: r.score for dim, r in judge_results.items()
                    }
                except Exception as e:
                    print(f"        Error in LLM judge for item {idx}: {e}")
                    judge_item = {}
                item_metrics["merit_llm_judge"] = judge_item
                for dim, score in judge_item.items():
                    judge_scores.setdefault(dim, []).append(score)

            # --- Baselines ---
            if self.baselines:
                baseline_item: Dict[str, float] = {}
                for bl_name, bl in self.baselines.items():
                    try:
                        bl_result = bl.evaluate(response, reference)
                        bl_score = bl_result.get("f1", bl_result.get("score", 0.0))
                        baseline_item[bl_name] = bl_score
                        baseline_scores[bl_name].append(bl_score)
                    except Exception as e:
                        print(f"        Error computing baseline {bl_name}: {e}")
                        baseline_item[bl_name] = 0.0
                        baseline_scores[bl_name].append(0.0)
                item_metrics["baselines"] = baseline_item

            # Check task accuracy
            is_correct = self._check_answer_correctness(
                response,
                reference,
                item.get("reference_letter", ""),
                item.get("task_type", "default")
            )
            if is_correct:
                correct_count += 1

            individual_results.append({
                "prompt": item["prompt"],
                "response": response,
                "reference": reference,
                "reference_letter": item.get("reference_letter", ""),
                "metrics": item_metrics,
                "correct": is_correct
            })

        # --- Build aggregated results ---
        result: Dict[str, Any] = {
            "individual_results": individual_results,
            "task_accuracy": correct_count / len(dataset) if dataset else 0.0,
            "total_samples": len(dataset),
        }

        if heuristic_metrics:
            avg_heuristic = {
                name: float(np.mean(scores)) if scores else 0.0
                for name, scores in heuristic_scores.items()
            }
            result["merit_heuristic"] = avg_heuristic
            # Backward-compat: keep flat average_metrics identical to merit_heuristic
            result["average_metrics"] = avg_heuristic

        if llm_judge is not None:
            result["merit_llm_judge"] = {
                dim: float(np.mean(scores)) if scores else 0.0
                for dim, scores in judge_scores.items()
            }

        if self.baselines:
            result["baselines"] = {
                name: float(np.mean(scores)) if scores else 0.0
                for name, scores in baseline_scores.items()
            }

        return result

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
        """Calculate statistics across multiple runs.

        Handles the structured results layout (merit_heuristic, merit_llm_judge,
        baselines) as well as the legacy flat ``average_metrics`` key.
        """
        if not runs:
            return {}

        stats: Dict[str, Any] = {
            "metric_statistics": {},
            "task_accuracy": {},
        }

        # Helper: compute stats for a dict of scores keyed by metric name
        def _stats_for_section(section_key: str) -> Dict[str, Any]:
            section_stats: Dict[str, Any] = {}
            first = runs[0].get(section_key, {})
            if not first:
                return section_stats
            for name in first:
                scores = [
                    run[section_key][name]
                    for run in runs
                    if section_key in run and name in run[section_key]
                ]
                if scores:
                    section_stats[name] = {
                        "mean_across_runs": float(np.mean(scores)),
                        "std_across_runs": float(np.std(scores)),
                        "min_across_runs": float(np.min(scores)),
                        "max_across_runs": float(np.max(scores)),
                    }
            return section_stats

        # Heuristic metrics (also kept in legacy metric_statistics for compat)
        heuristic_stats = _stats_for_section("merit_heuristic")
        if heuristic_stats:
            stats["merit_heuristic"] = heuristic_stats
            # Backward compatibility: also write to top-level metric_statistics
            stats["metric_statistics"] = heuristic_stats

        # LLM-judge metrics
        judge_stats = _stats_for_section("merit_llm_judge")
        if judge_stats:
            stats["merit_llm_judge"] = judge_stats

        # Baselines
        baseline_stats = _stats_for_section("baselines")
        if baseline_stats:
            stats["baselines"] = baseline_stats

        # Legacy fallback: if metric_statistics is still empty, try average_metrics
        if not stats["metric_statistics"] and runs[0].get("average_metrics"):
            for name in runs[0]["average_metrics"]:
                scores = [run["average_metrics"][name] for run in runs if "average_metrics" in run]
                if scores:
                    stats["metric_statistics"][name] = {
                        "mean_across_runs": float(np.mean(scores)),
                        "std_across_runs": float(np.std(scores)),
                        "min_across_runs": float(np.min(scores)),
                        "max_across_runs": float(np.max(scores)),
                    }

        # Task accuracy statistics
        accuracies = [run["task_accuracy"] for run in runs]
        stats["task_accuracy"] = {
            "mean": float(np.mean(accuracies)),
            "std": float(np.std(accuracies)),
            "min": float(np.min(accuracies)),
            "max": float(np.max(accuracies)),
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
