"""
Evaluation framework for reasoning and interpretation in LLMs.
"""
from typing import Dict, List, Optional, Union, Any, Callable
import json
import pandas as pd
from merit.core.metrics import MetricRegistry, get_default_metric_registry

class ReasoningEvaluator:
    """Main evaluator class for assessing reasoning capabilities."""
    
    def __init__(self, 
                 metric_registry: Optional[MetricRegistry] = None,
                 model_adapter = None,
                 task_adapters: Optional[Dict] = None):
        """Initialize the evaluator.
        
        Args:
            metric_registry: Registry of metrics to use for evaluation
            model_adapter: Adapter for the model to evaluate
            task_adapters: Dictionary of task-specific adapters
        """
        self.metric_registry = metric_registry or get_default_metric_registry()
        self.model_adapter = model_adapter
        self.task_adapters = task_adapters or {}
        self.results = []
    
    def evaluate_prompt(self, 
                        prompt: str, 
                        reference: Optional[str] = None,
                        task_type: Optional[str] = None,
                        metrics: Optional[List[str]] = None,
                        **kwargs) -> Dict:
        """Evaluate model response to a prompt.
        
        Args:
            prompt: Input prompt
            reference: Optional reference/ground truth
            task_type: Type of reasoning task
            metrics: List of metric names to compute
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with evaluation results
        """
        # Get model prediction
        if self.model_adapter is None:
            raise ValueError("Model adapter is required for evaluation")
        
        prediction = self.model_adapter.generate(prompt, **kwargs)
        
        # Evaluate with appropriate metrics
        result = self.evaluate_prediction(prediction, reference, task_type, metrics)
        
        # Store the evaluation context
        result["prompt"] = prompt
        result["prediction"] = prediction
        if reference:
            result["reference"] = reference
        if task_type:
            result["task_type"] = task_type
        
        # Add to results history
        self.results.append(result)
        
        return result
    
    def evaluate_prediction(self,
                           prediction: str,
                           reference: Optional[str] = None,
                           task_type: Optional[str] = None,
                           metrics: Optional[List[str]] = None,
                           **kwargs) -> Dict:
        """Evaluate a model prediction.
        
        Args:
            prediction: Model output to evaluate
            reference: Optional reference/ground truth
            task_type: Type of reasoning task
            metrics: List of metric names to compute
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with evaluation results
        """
        # Use task adapter if provided
        if task_type and task_type in self.task_adapters:
            task_adapter = self.task_adapters[task_type]
            return task_adapter.evaluate(prediction, reference, **kwargs)
        
        # Otherwise use the specified metrics
        if metrics:
            # Use only the specified metrics
            results = {}
            for metric_name in metrics:
                metric = self.metric_registry.get(metric_name)
                if metric:
                    results[metric_name] = metric.compute(prediction, reference, **kwargs)
                else:
                    raise ValueError(f"Metric '{metric_name}' not found")
        else:
            # Use all registered metrics
            results = self.metric_registry.compute_all(prediction, reference, **kwargs)
        
        return {
            "metrics": results,
            "prediction": prediction,
            "reference": reference
        }
    
    def evaluate_dataset(self,
                        dataset: List[Dict],
                        metrics: Optional[List[str]] = None,
                        task_type: Optional[str] = None,
                        **kwargs) -> Dict:
        """Evaluate model on a dataset.
        
        Args:
            dataset: List of dictionaries with prompts and references
            metrics: List of metric names to compute
            task_type: Type of reasoning task
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with evaluation results
        """
        results = []
        
        for item in dataset:
            prompt = item.get("prompt")
            reference = item.get("reference")
            
            # Evaluate individual example
            result = self.evaluate_prompt(
                prompt=prompt,
                reference=reference,
                task_type=task_type,
                metrics=metrics,
                **kwargs
            )
            
            results.append(result)
        
        # Aggregate results
        aggregated = self._aggregate_results(results)
        
        return {
            "results": results,
            "aggregated": aggregated
        }
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results from multiple evaluations.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary with aggregated results
        """
        # Extract metrics from results
        metrics_data = {}
        
        for result in results:
            metrics_result = result.get("metrics", {})
            for metric_name, metric_value in metrics_result.items():
                if isinstance(metric_value, dict) and "score" in metric_value:
                    if metric_name not in metrics_data:
                        metrics_data[metric_name] = []
                    metrics_data[metric_name].append(metric_value["score"])
        
        # Calculate statistics
        aggregated = {}
        for metric_name, scores in metrics_data.items():
            if scores:
                aggregated[metric_name] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "std": self._std(scores),
                    "count": len(scores)
                }
        
        return aggregated
    
    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation.
        
        Args:
            values: List of values
            
        Returns:
            Standard deviation
        """
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def save_results(self, path: str):
        """Save evaluation results to file.
        
        Args:
            path: Path to save results
        """
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def load_results(self, path: str):
        """Load evaluation results from file.
        
        Args:
            path: Path to load results from
        """
        with open(path, 'r') as f:
            self.results = json.load(f)
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """Get a pandas DataFrame summarizing the evaluation results.
        
        Returns:
            DataFrame with evaluation summary
        """
        summary_data = []
        
        for result in self.results:
            row = {
                "prompt": result.get("prompt", "")[:50] + "..." if len(result.get("prompt", "")) > 50 else result.get("prompt", ""),
                "task_type": result.get("task_type", "unknown")
            }
            
            # Extract scores from metrics
            metrics = result.get("metrics", {})
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, dict) and "score" in metric_value:
                    row[f"{metric_name}_score"] = metric_value["score"]
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
