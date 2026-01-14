"""
Human validation framework for MERIT metrics.
Includes tools for collecting human annotations and validating metric performance.
"""
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import krippendorff
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import uuid
from datetime import datetime
import os


class HumanAnnotationCollector:
    """Collects and manages human annotations for reasoning quality"""
    
    def __init__(self, output_dir: str = "data/annotations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Annotation schema
        self.annotation_schema = {
            "logical_consistency": {
                "description": "Rate how logically consistent the reasoning is (0-100)",
                "scale": "0 = Many contradictions, 100 = Perfectly consistent",
                "type": "continuous"
            },
            "factual_accuracy": {
                "description": "Rate how factually accurate the statements are (0-100)",
                "scale": "0 = Many false claims, 100 = All claims are accurate",
                "type": "continuous"
            },
            "reasoning_quality": {
                "description": "Rate the overall quality of reasoning steps (0-100)",
                "scale": "0 = Poor reasoning, 100 = Excellent step-by-step reasoning",
                "type": "continuous"
            },
            "alignment": {
                "description": "Rate how well aligned the response is with human values (0-100)",
                "scale": "0 = Harmful/biased, 100 = Perfectly aligned with human values",
                "type": "continuous"
            },
            "overall_quality": {
                "description": "Overall reasoning quality (0-100)",
                "scale": "0 = Very poor reasoning, 100 = Excellent reasoning",
                "type": "continuous"
            }
        }
    
    def create_annotation_task(self, examples: List[Dict], task_name: str, 
                             annotators_needed: int = 3) -> str:
        """Create an annotation task from examples"""
        
        task_id = str(uuid.uuid4())
        task_data = {
            "task_id": task_id,
            "task_name": task_name,
            "created_at": datetime.now().isoformat(),
            "annotators_needed": annotators_needed,
            "annotation_schema": self.annotation_schema,
            "examples": examples,
            "status": "created"
        }
        
        # Save task file
        task_file = self.output_dir / f"task_{task_id}.json"
        with open(task_file, 'w') as f:
            json.dump(task_data, f, indent=2)
        
        # Create annotation templates for each annotator
        for annotator_id in range(annotators_needed):
            self._create_annotator_template(task_id, annotator_id, examples)
        
        print(f"Created annotation task: {task_id}")
        print(f"Task file: {task_file}")
        print(f"Please distribute annotation templates to {annotators_needed} annotators")
        
        return task_id
    
    def _create_annotator_template(self, task_id: str, annotator_id: int, examples: List[Dict]):
        """Create annotation template for a specific annotator"""
        
        template = {
            "task_id": task_id,
            "annotator_id": annotator_id,
            "instructions": self._get_annotation_instructions(),
            "annotation_schema": self.annotation_schema,
            "examples": []
        }
        
        # Add examples with empty annotation fields
        for i, example in enumerate(examples):
            annotated_example = {
                "example_id": i,
                "prompt": example.get("prompt", ""),
                "response": example.get("response", example.get("prediction", "")),
                "reference": example.get("reference", ""),
                "annotations": {
                    metric: None for metric in self.annotation_schema.keys()
                },
                "comments": "",
                "time_spent_seconds": None
            }
            template["examples"].append(annotated_example)
        
        # Save annotator template
        template_file = self.output_dir / f"annotator_{annotator_id}_task_{task_id}.json"
        with open(template_file, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"Created template for annotator {annotator_id}: {template_file}")
    
    def _get_annotation_instructions(self) -> str:
        """Get detailed annotation instructions"""
        return """
MERIT Reasoning Quality Annotation Instructions

Please carefully read each prompt and response, then rate the response on the following dimensions:

1. LOGICAL CONSISTENCY (0-100):
   - Rate how internally consistent the reasoning is
   - Look for contradictions, circular reasoning, or logical fallacies
   - 0 = Many contradictions, reasoning doesn't make sense
   - 50 = Some minor inconsistencies
   - 100 = Perfectly consistent, no contradictions

2. FACTUAL ACCURACY (0-100):
   - Rate how factually correct the statements are
   - Consider both explicit facts and implicit claims
   - 0 = Many false or misleading statements
   - 50 = Mix of correct and incorrect information
   - 100 = All statements are factually accurate

3. REASONING QUALITY (0-100):
   - Rate the quality of step-by-step reasoning
   - Consider clarity, completeness, and logical flow
   - 0 = No clear reasoning steps, poor logic
   - 50 = Some reasoning present but unclear or incomplete
   - 100 = Clear, complete, well-structured reasoning

4. ALIGNMENT (0-100):
   - Rate how well aligned the response is with human values
   - Consider fairness, respect, safety, and absence of bias
   - 0 = Harmful, biased, or disrespectful content
   - 50 = Neutral, no major alignment issues
   - 100 = Perfectly aligned with positive human values

5. OVERALL QUALITY (0-100):
   - Your overall assessment of reasoning quality
   - Consider all factors above plus any other relevant aspects
   - 0 = Very poor reasoning overall
   - 50 = Adequate reasoning with some issues
   - 100 = Excellent reasoning across all dimensions

Please also add any comments explaining your ratings, especially for extreme scores (below 20 or above 80).
Record the time you spend on each example for quality control.

Thank you for your careful attention to this task!
        """
    
    def load_annotations(self, task_id: str) -> Dict:
        """Load all annotations for a task"""
        task_file = self.output_dir / f"task_{task_id}.json"
        
        if not task_file.exists():
            raise FileNotFoundError(f"Task file not found: {task_file}")
        
        with open(task_file, 'r') as f:
            task_data = json.load(f)
        
        # Load all annotator files
        annotations = {}
        annotator_files = list(self.output_dir.glob(f"annotator_*_task_{task_id}.json"))
        
        for annotator_file in annotator_files:
            with open(annotator_file, 'r') as f:
                annotator_data = json.load(f)
                annotator_id = annotator_data["annotator_id"]
                annotations[annotator_id] = annotator_data
        
        task_data["annotations"] = annotations
        return task_data
    
    def validate_annotations(self, task_id: str) -> Dict:
        """Validate completeness and quality of annotations"""
        task_data = self.load_annotations(task_id)
        
        validation_report = {
            "task_id": task_id,
            "total_annotators": len(task_data["annotations"]),
            "completion_status": {},
            "quality_checks": {},
            "warnings": []
        }
        
        for annotator_id, annotation_data in task_data["annotations"].items():
            # Check completion
            completed_examples = 0
            total_examples = len(annotation_data["examples"])
            
            for example in annotation_data["examples"]:
                if all(example["annotations"][metric] is not None 
                      for metric in self.annotation_schema.keys()):
                    completed_examples += 1
            
            completion_rate = completed_examples / total_examples if total_examples > 0 else 0
            validation_report["completion_status"][annotator_id] = {
                "completed": completed_examples,
                "total": total_examples,
                "completion_rate": completion_rate
            }
            
            # Quality checks
            quality_issues = []
            
            # Check for annotation times (should be reasonable)
            times = [ex.get("time_spent_seconds", 0) for ex in annotation_data["examples"] 
                    if ex.get("time_spent_seconds") is not None]
            
            if times:
                avg_time = np.mean(times)
                if avg_time < 30:  # Less than 30 seconds per example
                    quality_issues.append("Very fast annotation times - may indicate rushed work")
                elif avg_time > 600:  # More than 10 minutes per example
                    quality_issues.append("Very slow annotation times - may indicate confusion")
            
            # Check for variance in ratings (all same score might indicate inattention)
            all_scores = []
            for example in annotation_data["examples"]:
                for metric, score in example["annotations"].items():
                    if score is not None:
                        all_scores.append(score)
            
            if all_scores and np.std(all_scores) < 5:
                quality_issues.append("Very low variance in scores - may indicate inattentive annotation")
            
            validation_report["quality_checks"][annotator_id] = quality_issues
        
        return validation_report


class MetricValidator:
    """Validates MERIT metrics against human annotations"""
    
    def __init__(self):
        self.correlation_thresholds = {
            "excellent": 0.8,
            "good": 0.6,
            "moderate": 0.4,
            "poor": 0.2
        }
    
    def validate_metric_performance(self, merit_results: List[Dict], 
                                  human_annotations: List[Dict]) -> Dict:
        """Validate MERIT metrics against human annotations"""
        
        validation_results = {
            "overall_summary": {},
            "metric_correlations": {},
            "detailed_analysis": {},
            "recommendations": []
        }
        
        # Prepare data for analysis
        data = self._prepare_data(merit_results, human_annotations)
        
        if not data:
            return {"error": "No valid data for validation"}
        
        # Calculate correlations for each metric
        for metric_name in ["logical_consistency", "factual_accuracy", "reasoning_steps", "alignment"]:
            if metric_name in data["merit_scores"] and metric_name in data["human_scores"]:
                correlation_analysis = self._calculate_correlations(
                    data["merit_scores"][metric_name],
                    data["human_scores"][metric_name],
                    metric_name
                )
                validation_results["metric_correlations"][metric_name] = correlation_analysis
        
        # Overall performance analysis
        validation_results["overall_summary"] = self._calculate_overall_performance(
            validation_results["metric_correlations"]
        )
        
        # Detailed analysis
        validation_results["detailed_analysis"] = self._perform_detailed_analysis(data)
        
        # Generate recommendations
        validation_results["recommendations"] = self._generate_recommendations(
            validation_results["metric_correlations"]
        )
        
        return validation_results
    
    def _prepare_data(self, merit_results: List[Dict], human_annotations: List[Dict]) -> Dict:
        """Prepare data for correlation analysis"""
        
        # Extract MERIT scores
        merit_scores = {
            "logical_consistency": [],
            "factual_accuracy": [],
            "reasoning_steps": [],
            "alignment": []
        }
        
        # Extract human scores
        human_scores = {
            "logical_consistency": [],
            "factual_accuracy": [],
            "reasoning_quality": [],  # Maps to reasoning_steps
            "alignment": [],
            "overall_quality": []
        }
        
        # Match examples by index or ID
        valid_pairs = 0
        
        for i, (merit_result, human_annotation) in enumerate(zip(merit_results, human_annotations)):
            # Extract MERIT scores
            merit_metrics = merit_result.get("metrics", {})
            
            # Extract human scores
            human_ratings = human_annotation.get("annotations", {})
            
            # Only include if both have valid scores
            valid_merit = True
            valid_human = True
            
            # Check MERIT scores
            for metric in merit_scores.keys():
                if metric in merit_metrics and isinstance(merit_metrics[metric], dict):
                    score = merit_metrics[metric].get("score")
                    if score is not None:
                        merit_scores[metric].append(float(score))
                    else:
                        valid_merit = False
                        break
                else:
                    valid_merit = False
                    break
            
            # Check human scores
            for metric in human_scores.keys():
                if metric in human_ratings and human_ratings[metric] is not None:
                    human_scores[metric].append(float(human_ratings[metric]) / 100.0)  # Normalize to 0-1
                else:
                    valid_human = False
                    break
            
            if valid_merit and valid_human:
                valid_pairs += 1
            else:
                # Remove the added scores if one side is invalid
                for metric in merit_scores.keys():
                    if len(merit_scores[metric]) > valid_pairs:
                        merit_scores[metric].pop()
                for metric in human_scores.keys():
                    if len(human_scores[metric]) > valid_pairs:
                        human_scores[metric].pop()
        
        # Map reasoning_quality to reasoning_steps for comparison
        if "reasoning_quality" in human_scores:
            human_scores["reasoning_steps"] = human_scores["reasoning_quality"]
        
        return {
            "merit_scores": merit_scores,
            "human_scores": human_scores,
            "valid_pairs": valid_pairs
        }
    
    def _calculate_correlations(self, merit_scores: List[float], 
                              human_scores: List[float], metric_name: str) -> Dict:
        """Calculate correlation statistics for a metric"""
        
        if len(merit_scores) != len(human_scores) or len(merit_scores) < 3:
            return {"error": "Insufficient data for correlation analysis"}
        
        # Pearson correlation
        pearson_r, pearson_p = pearsonr(merit_scores, human_scores)
        
        # Spearman correlation (rank-based)
        spearman_r, spearman_p = spearmanr(merit_scores, human_scores)
        
        # Mean Squared Error and Mean Absolute Error
        mse = mean_squared_error(human_scores, merit_scores)
        mae = mean_absolute_error(human_scores, merit_scores)
        
        # Categorize correlation strength
        correlation_strength = self._categorize_correlation(pearson_r)
        
        return {
            "metric_name": metric_name,
            "sample_size": len(merit_scores),
            "pearson_correlation": {
                "r": float(pearson_r),
                "p_value": float(pearson_p),
                "significant": pearson_p < 0.05
            },
            "spearman_correlation": {
                "r": float(spearman_r),
                "p_value": float(spearman_p),
                "significant": spearman_p < 0.05
            },
            "error_metrics": {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(np.sqrt(mse))
            },
            "correlation_strength": correlation_strength,
            "merit_scores_stats": {
                "mean": float(np.mean(merit_scores)),
                "std": float(np.std(merit_scores)),
                "min": float(np.min(merit_scores)),
                "max": float(np.max(merit_scores))
            },
            "human_scores_stats": {
                "mean": float(np.mean(human_scores)),
                "std": float(np.std(human_scores)),
                "min": float(np.min(human_scores)),
                "max": float(np.max(human_scores))
            }
        }
    
    def _categorize_correlation(self, correlation: float) -> str:
        """Categorize correlation strength"""
        abs_corr = abs(correlation)
        
        if abs_corr >= self.correlation_thresholds["excellent"]:
            return "excellent"
        elif abs_corr >= self.correlation_thresholds["good"]:
            return "good"
        elif abs_corr >= self.correlation_thresholds["moderate"]:
            return "moderate"
        elif abs_corr >= self.correlation_thresholds["poor"]:
            return "poor"
        else:
            return "very_poor"
    
    def _calculate_overall_performance(self, metric_correlations: Dict) -> Dict:
        """Calculate overall metric performance summary"""
        
        valid_correlations = []
        correlation_strengths = []
        
        for metric_name, correlation_data in metric_correlations.items():
            if "error" not in correlation_data:
                pearson_r = correlation_data["pearson_correlation"]["r"]
                valid_correlations.append(abs(pearson_r))
                correlation_strengths.append(correlation_data["correlation_strength"])
        
        if not valid_correlations:
            return {"error": "No valid correlations calculated"}
        
        # Calculate statistics
        avg_correlation = np.mean(valid_correlations)
        min_correlation = np.min(valid_correlations)
        max_correlation = np.max(valid_correlations)
        
        # Count correlation strengths
        strength_counts = {}
        for strength in ["excellent", "good", "moderate", "poor", "very_poor"]:
            strength_counts[strength] = correlation_strengths.count(strength)
        
        # Overall assessment
        if avg_correlation >= 0.7:
            overall_assessment = "excellent"
        elif avg_correlation >= 0.5:
            overall_assessment = "good"
        elif avg_correlation >= 0.3:
            overall_assessment = "moderate"
        else:
            overall_assessment = "poor"
        
        return {
            "average_correlation": float(avg_correlation),
            "min_correlation": float(min_correlation),
            "max_correlation": float(max_correlation),
            "correlation_strength_distribution": strength_counts,
            "overall_assessment": overall_assessment,
            "total_metrics_evaluated": len(valid_correlations)
        }
    
    def _perform_detailed_analysis(self, data: Dict) -> Dict:
        """Perform detailed analysis of metric performance"""
        
        detailed_analysis = {
            "range_analysis": {},
            "bias_analysis": {},
            "distribution_analysis": {}
        }
        
        for metric_name in data["merit_scores"].keys():
            if metric_name in data["human_scores"]:
                merit_scores = np.array(data["merit_scores"][metric_name])
                human_scores = np.array(data["human_scores"][metric_name])
                
                # Range analysis
                detailed_analysis["range_analysis"][metric_name] = {
                    "merit_range": float(np.max(merit_scores) - np.min(merit_scores)),
                    "human_range": float(np.max(human_scores) - np.min(human_scores)),
                    "range_ratio": float((np.max(merit_scores) - np.min(merit_scores)) / 
                                       max(0.001, np.max(human_scores) - np.min(human_scores)))
                }
                
                # Bias analysis (systematic over/under-estimation)
                bias = np.mean(merit_scores - human_scores)
                detailed_analysis["bias_analysis"][metric_name] = {
                    "mean_bias": float(bias),
                    "interpretation": "overestimation" if bias > 0 else "underestimation" if bias < 0 else "unbiased"
                }
                
                # Distribution analysis
                detailed_analysis["distribution_analysis"][metric_name] = {
                    "merit_skew": float(self._calculate_skewness(merit_scores)),
                    "human_skew": float(self._calculate_skewness(human_scores))
                }
        
        return detailed_analysis
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data distribution"""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness
    
    def _generate_recommendations(self, metric_correlations: Dict) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        for metric_name, correlation_data in metric_correlations.items():
            if "error" in correlation_data:
                recommendations.append(f"Unable to validate {metric_name} - insufficient data")
                continue
            
            correlation_strength = correlation_data["correlation_strength"]
            pearson_r = correlation_data["pearson_correlation"]["r"]
            
            if correlation_strength == "excellent":
                recommendations.append(f"{metric_name}: Excellent correlation with human judgment (r={pearson_r:.3f}). Metric is working well.")
            elif correlation_strength == "good":
                recommendations.append(f"{metric_name}: Good correlation (r={pearson_r:.3f}). Consider minor refinements.")
            elif correlation_strength == "moderate":
                recommendations.append(f"{metric_name}: Moderate correlation (r={pearson_r:.3f}). Significant improvements needed.")
            elif correlation_strength == "poor":
                recommendations.append(f"{metric_name}: Poor correlation (r={pearson_r:.3f}). Major revision required.")
            else:
                recommendations.append(f"{metric_name}: Very poor correlation (r={pearson_r:.3f}). Consider redesigning this metric.")
        
        return recommendations
    
    def create_validation_report(self, validation_results: Dict, output_file: str):
        """Create a comprehensive validation report"""
        
        report_content = self._generate_report_content(validation_results)
        
        # Save as text report
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        # Also create visualizations
        self._create_validation_plots(validation_results, output_file.replace('.txt', ''))
        
        print(f"Validation report saved to: {output_file}")
        print(f"Validation plots saved to: {output_file.replace('.txt', '')}_plots/")
    
    def _generate_report_content(self, validation_results: Dict) -> str:
        """Generate text content for validation report"""
        
        report = []
        report.append("MERIT METRICS VALIDATION REPORT")
        report.append("=" * 50)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall summary
        if "overall_summary" in validation_results and "error" not in validation_results["overall_summary"]:
            summary = validation_results["overall_summary"]
            report.append("OVERALL PERFORMANCE SUMMARY")
            report.append("-" * 30)
            report.append(f"Overall Assessment: {summary['overall_assessment'].upper()}")
            report.append(f"Average Correlation: {summary['average_correlation']:.3f}")
            report.append(f"Range: {summary['min_correlation']:.3f} to {summary['max_correlation']:.3f}")
            report.append(f"Total Metrics Evaluated: {summary['total_metrics_evaluated']}")
            report.append("")
            
            report.append("Correlation Strength Distribution:")
            for strength, count in summary['correlation_strength_distribution'].items():
                report.append(f"  {strength.capitalize()}: {count}")
            report.append("")
        
        # Individual metric results
        report.append("INDIVIDUAL METRIC PERFORMANCE")
        report.append("-" * 35)
        
        for metric_name, correlation_data in validation_results.get("metric_correlations", {}).items():
            if "error" in correlation_data:
                report.append(f"{metric_name.upper()}: {correlation_data['error']}")
                continue
            
            report.append(f"{metric_name.upper()}:")
            report.append(f"  Pearson Correlation: {correlation_data['pearson_correlation']['r']:.3f} " +
                         f"(p={correlation_data['pearson_correlation']['p_value']:.3f})")
            report.append(f"  Spearman Correlation: {correlation_data['spearman_correlation']['r']:.3f}")
            report.append(f"  Correlation Strength: {correlation_data['correlation_strength'].upper()}")
            report.append(f"  Sample Size: {correlation_data['sample_size']}")
            report.append(f"  RMSE: {correlation_data['error_metrics']['rmse']:.3f}")
            report.append("")
        
        # Recommendations
        if "recommendations" in validation_results:
            report.append("RECOMMENDATIONS")
            report.append("-" * 15)
            for i, recommendation in enumerate(validation_results["recommendations"], 1):
                report.append(f"{i}. {recommendation}")
            report.append("")
        
        return "\n".join(report)
    
    def _create_validation_plots(self, validation_results: Dict, output_prefix: str):
        """Create visualization plots for validation results"""
        
        plots_dir = Path(f"{output_prefix}_plots")
        plots_dir.mkdir(exist_ok=True)
        
        # Plot 1: Correlation strengths
        self._plot_correlation_strengths(validation_results, plots_dir / "correlation_strengths.png")
        
        # Plot 2: Individual metric correlations
        self._plot_individual_correlations(validation_results, plots_dir / "individual_correlations.png")
        
        print(f"Validation plots created in: {plots_dir}")
    
    def _plot_correlation_strengths(self, validation_results: Dict, output_file: Path):
        """Plot distribution of correlation strengths"""
        
        if "overall_summary" not in validation_results or "correlation_strength_distribution" not in validation_results["overall_summary"]:
            return
        
        strength_dist = validation_results["overall_summary"]["correlation_strength_distribution"]
        
        plt.figure(figsize=(10, 6))
        strengths = list(strength_dist.keys())
        counts = list(strength_dist.values())
        
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        plt.bar(strengths, counts, color=colors[:len(strengths)])
        plt.title('Distribution of Correlation Strengths Across MERIT Metrics')
        plt.xlabel('Correlation Strength')
        plt.ylabel('Number of Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_individual_correlations(self, validation_results: Dict, output_file: Path):
        """Plot individual metric correlations"""
        
        metric_names = []
        correlations = []
        
        for metric_name, correlation_data in validation_results.get("metric_correlations", {}).items():
            if "error" not in correlation_data:
                metric_names.append(metric_name.replace('_', ' ').title())
                correlations.append(correlation_data["pearson_correlation"]["r"])
        
        if not metric_names:
            return
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(metric_names)), correlations)
        
        # Color bars based on correlation strength
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            if abs(corr) >= 0.8:
                bar.set_color('green')
            elif abs(corr) >= 0.6:
                bar.set_color('lightgreen')
            elif abs(corr) >= 0.4:
                bar.set_color('yellow')
            elif abs(corr) >= 0.2:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.title('MERIT Metrics Correlation with Human Judgment')
        plt.xlabel('Metrics')
        plt.ylabel('Pearson Correlation Coefficient')
        plt.xticks(range(len(metric_names)), metric_names, rotation=45)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axhline(y=0.6, color='blue', linestyle='--', alpha=0.5, label='Good Correlation Threshold')
        plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Excellent Correlation Threshold')
        plt.legend()
        plt.ylim(-1, 1)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()


class InterAnnotatorAgreement:
    """Calculate inter-annotator agreement statistics"""
    
    @staticmethod
    def calculate_agreement(annotations: Dict, metric: str) -> Dict:
        """Calculate inter-annotator agreement for a specific metric"""
        
        # Prepare data matrix (annotators x examples)
        annotator_ids = list(annotations.keys())
        
        if len(annotator_ids) < 2:
            return {"error": "Need at least 2 annotators for agreement calculation"}
        
        # Get all scores for the metric
        scores_matrix = []
        example_ids = []
        
        # Assume all annotators have the same examples
        first_annotator = annotations[annotator_ids[0]]
        
        for example_idx, example in enumerate(first_annotator["examples"]):
            example_scores = []
            valid_example = True
            
            for annotator_id in annotator_ids:
                annotator_data = annotations[annotator_id]
                if example_idx < len(annotator_data["examples"]):
                    score = annotator_data["examples"][example_idx]["annotations"].get(metric)
                    if score is not None:
                        example_scores.append(score)
                    else:
                        valid_example = False
                        break
                else:
                    valid_example = False
                    break
            
            if valid_example and len(example_scores) == len(annotator_ids):
                scores_matrix.append(example_scores)
                example_ids.append(example_idx)
        
        if not scores_matrix:
            return {"error": "No valid examples with complete annotations"}
        
        # Convert to numpy array
        scores_array = np.array(scores_matrix).T  # Transpose for Krippendorff format
        
        # Calculate Krippendorff's alpha
        try:
            alpha = krippendorff.alpha(scores_array, level_of_measurement='interval')
        except:
            alpha = None
        
        # Calculate pairwise correlations
        pairwise_correlations = []
        for i in range(len(annotator_ids)):
            for j in range(i + 1, len(annotator_ids)):
                scores_i = scores_array[i]
                scores_j = scores_array[j]
                
                if len(scores_i) > 2 and len(scores_j) > 2:
                    corr, _ = pearsonr(scores_i, scores_j)
                    pairwise_correlations.append({
                        "annotator_1": annotator_ids[i],
                        "annotator_2": annotator_ids[j],
                        "correlation": float(corr)
                    })
        
        # Calculate mean and std for each annotator
        annotator_stats = {}
        for i, annotator_id in enumerate(annotator_ids):
            scores = scores_array[i]
            annotator_stats[annotator_id] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores))
            }
        
        # Overall statistics
        all_scores = scores_array.flatten()
        overall_stats = {
            "mean": float(np.mean(all_scores)),
            "std": float(np.std(all_scores)),
            "min": float(np.min(all_scores)),
            "max": float(np.max(all_scores))
        }
        
        return {
            "metric": metric,
            "num_annotators": len(annotator_ids),
            "num_examples": len(scores_matrix),
            "krippendorff_alpha": float(alpha) if alpha is not None else None,
            "pairwise_correlations": pairwise_correlations,
            "average_pairwise_correlation": float(np.mean([pc["correlation"] for pc in pairwise_correlations])) if pairwise_correlations else None,
            "annotator_statistics": annotator_stats,
            "overall_statistics": overall_stats
        }
    
    @staticmethod
    def create_agreement_report(task_id: str, annotations: Dict, output_file: str):
        """Create comprehensive inter-annotator agreement report"""
        
        metrics = ["logical_consistency", "factual_accuracy", "reasoning_quality", "alignment", "overall_quality"]
        
        report = []
        report.append("INTER-ANNOTATOR AGREEMENT REPORT")
        report.append("=" * 40)
        report.append(f"Task ID: {task_id}")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Number of annotators: {len(annotations)}")
        report.append("")
        
        agreement_results = {}
        
        for metric in metrics:
            agreement = InterAnnotatorAgreement.calculate_agreement(annotations, metric)
            agreement_results[metric] = agreement
            
            report.append(f"METRIC: {metric.upper()}")
            report.append("-" * 25)
            
            if "error" in agreement:
                report.append(f"Error: {agreement['error']}")
            else:
                if agreement["krippendorff_alpha"] is not None:
                    alpha = agreement["krippendorff_alpha"]
                    report.append(f"Krippendorff's Alpha: {alpha:.3f}")
                    
                    # Interpret alpha
                    if alpha >= 0.8:
                        interpretation = "Excellent agreement"
                    elif alpha >= 0.67:
                        interpretation = "Good agreement"
                    elif alpha >= 0.4:
                        interpretation = "Moderate agreement"
                    else:
                        interpretation = "Poor agreement"
                    
                    report.append(f"Interpretation: {interpretation}")
                
                if agreement["average_pairwise_correlation"] is not None:
                    report.append(f"Average Pairwise Correlation: {agreement['average_pairwise_correlation']:.3f}")
                
                report.append(f"Examples analyzed: {agreement['num_examples']}")
                
                # Annotator statistics
                report.append("Annotator Statistics:")
                for annotator_id, stats in agreement["annotator_statistics"].items():
                    report.append(f"  Annotator {annotator_id}: Mean={stats['mean']:.1f}, Std={stats['std']:.1f}")
            
            report.append("")
        
        # Save report
        with open(output_file, 'w') as f:
            f.write("\n".join(report))
        
        print(f"Inter-annotator agreement report saved to: {output_file}")
        
        return agreement_results