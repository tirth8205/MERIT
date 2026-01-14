"""
Robust experimental design and evaluation framework for MERIT.
Provides ExperimentRunner and ExperimentConfig for running evaluations.
"""
import json
import uuid
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

# Import metrics from core
from merit.core.enhanced_metrics import (
    EnhancedLogicalConsistencyMetric,
    EnhancedFactualAccuracyMetric,
    EnhancedReasoningStepMetric,
    EnhancedAlignmentMetric
)

# Import model manager
from merit.models.local_models import ModelManager


# Instruction templates for different tasks
INSTRUCTION_TEMPLATES = {
    "multiple_choice": """Answer the following multiple choice question. Choose the best answer from the given options and explain your reasoning briefly.

Question: {question}

Instructions: Select ONE answer from the options above. Start your response with the letter of your choice (A, B, C, or D), then briefly explain why.""",

    "reasoning": """Please solve the following problem step by step:

{question}

Show your reasoning process clearly.""",

    "default": """Please answer the following question:

{question}

Provide a clear and concise answer."""
}


@dataclass
class ExperimentConfig:
    """Configuration for MERIT experiments"""
    experiment_name: str
    models: List[str]
    benchmarks: List[str]
    sample_sizes: List[int]
    num_runs: int = 3
    temperature: float = 0.7  # Higher temperature for more varied responses
    max_tokens: int = 200  # Shorter responses to avoid repetition
    random_seed: int = 42
    metrics: List[str] = field(default_factory=lambda: ["logical_consistency", "factual_accuracy", "reasoning_steps", "alignment"])
    baseline_methods: List[str] = field(default_factory=list)
    statistical_tests: List[str] = field(default_factory=list)
    output_dir: str = "experiments"
    use_instruction_format: bool = True  # Enable instruction formatting

    def save(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


class ExperimentRunner:
    """Runner for MERIT experiments"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.experiment_id = str(uuid.uuid4())[:8]
        self.output_dir = Path(config.output_dir) / f"{config.experiment_name}_{self.experiment_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
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
        metric_map = {
            "logical_consistency": EnhancedLogicalConsistencyMetric(),
            "factual_accuracy": EnhancedFactualAccuracyMetric(),
            "reasoning_steps": EnhancedReasoningStepMetric(),
            "alignment": EnhancedAlignmentMetric()
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

        # Load dataset
        dataset = self._load_dataset(benchmark, sample_size)

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

    def _load_dataset(self, benchmark: str, sample_size: int) -> List[Dict]:
        """Load and sample dataset"""
        try:
            from datasets import load_dataset

            # Load dataset based on benchmark name
            if benchmark == "arc":
                ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
            elif benchmark == "hellaswag":
                ds = load_dataset("Rowan/hellaswag", split="test")
            elif benchmark == "mmlu_logic":
                ds = load_dataset("cais/mmlu", "formal_logic", split="test")
            else:
                print(f"      Unknown benchmark: {benchmark}")
                return []

            # Sample the dataset
            import random
            random.seed(self.config.random_seed)

            total_size = len(ds)

            # If sample_size <= 0, use full dataset
            if sample_size <= 0:
                sample_size = total_size
            else:
                sample_size = min(sample_size, total_size)

            indices = random.sample(range(total_size), sample_size)
            sampled_data = [ds[i] for i in indices]

            # Convert to standard format with instruction templates
            formatted_data = []
            for item in sampled_data:
                if benchmark == "arc":
                    choices_text = "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(item["choices"]["text"])])
                    answer_key = item["answerKey"]
                    answer_idx = item["choices"]["label"].index(answer_key)
                    reference_text = item["choices"]["text"][answer_idx]

                    # Raw question with choices
                    raw_question = f"{item['question']}\n\nOptions:\n{choices_text}"

                    # Apply instruction template if enabled
                    if self.config.use_instruction_format:
                        prompt = INSTRUCTION_TEMPLATES["multiple_choice"].format(question=raw_question)
                    else:
                        prompt = raw_question

                    formatted_data.append({
                        "prompt": prompt,
                        "reference": reference_text,
                        "reference_letter": answer_key,
                        "task_type": "multiple_choice"
                    })

                elif benchmark == "hellaswag":
                    context = f"{item['ctx']}\n{item['activity_label']}"
                    endings = "\n".join([f"({chr(65+i)}) {ending}" for i, ending in enumerate(item["endings"])])
                    raw_question = f"{context}\n\nChoose the best continuation:\n{endings}"

                    if self.config.use_instruction_format:
                        prompt = INSTRUCTION_TEMPLATES["multiple_choice"].format(question=raw_question)
                    else:
                        prompt = raw_question

                    formatted_data.append({
                        "prompt": prompt,
                        "reference": item["endings"][int(item["label"])],
                        "reference_letter": chr(65 + int(item["label"])),
                        "task_type": "multiple_choice"
                    })

                elif benchmark == "mmlu_logic":
                    choices_text = "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(item["choices"])])
                    raw_question = f"{item['question']}\n\nOptions:\n{choices_text}"

                    if self.config.use_instruction_format:
                        prompt = INSTRUCTION_TEMPLATES["multiple_choice"].format(question=raw_question)
                    else:
                        prompt = raw_question

                    formatted_data.append({
                        "prompt": prompt,
                        "reference": item["choices"][item["answer"]],
                        "reference_letter": chr(65 + item["answer"]),
                        "task_type": "multiple_choice"
                    })

            return formatted_data

        except Exception as e:
            print(f"      Error loading dataset: {e}")
            import traceback
            traceback.print_exc()
            return []

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
