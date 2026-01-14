"""
Baseline comparison methods for validating MERIT metrics.
Includes BERT-score, ROUGE, and local LLM judge implementations.
"""
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import re
from collections import Counter
import math

# Local imports that may not be available initially
try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    print("Warning: bert-score not available. Install with: pip install bert-score")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers")


class BaselineMethod:
    """Base class for baseline evaluation methods"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def evaluate(self, prediction: str, reference: str, **kwargs) -> Dict:
        """Evaluate prediction against reference"""
        raise NotImplementedError("Subclasses must implement evaluate method")
    
    def batch_evaluate(self, predictions: List[str], references: List[str], **kwargs) -> List[Dict]:
        """Evaluate multiple predictions"""
        results = []
        for pred, ref in zip(predictions, references):
            try:
                result = self.evaluate(pred, ref, **kwargs)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), "score": 0.0})
        return results


class BERTScoreBaseline(BaselineMethod):
    """BERT-score baseline for semantic similarity"""
    
    def __init__(self, model_type: str = "microsoft/deberta-xlarge-mnli"):
        super().__init__(
            name="bert_score",
            description="BERT-based semantic similarity using pre-trained models"
        )
        self.model_type = model_type
        self.device = self._get_device()
        
        if not BERT_SCORE_AVAILABLE:
            print("Warning: BERT-score not available. Scores will be zero.")
    
    def _get_device(self):
        """Get optimal device for computation"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def evaluate(self, prediction: str, reference: str, **kwargs) -> Dict:
        """Evaluate using BERT-score"""
        if not BERT_SCORE_AVAILABLE:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "score": 0.0,
                "error": "BERT-score not available"
            }
        
        try:
            # Calculate BERT-score
            P, R, F1 = bert_score(
                [prediction], 
                [reference], 
                model_type=self.model_type,
                device=self.device,
                verbose=False
            )
            
            return {
                "precision": float(P[0].item()),
                "recall": float(R[0].item()),
                "f1": float(F1[0].item()),
                "score": float(F1[0].item()),  # Use F1 as main score
                "model_type": self.model_type
            }
        
        except Exception as e:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "score": 0.0,
                "error": str(e)
            }
    
    def batch_evaluate(self, predictions: List[str], references: List[str], **kwargs) -> List[Dict]:
        """Batch evaluation for efficiency"""
        if not BERT_SCORE_AVAILABLE:
            return [{"score": 0.0, "error": "BERT-score not available"}] * len(predictions)
        
        try:
            # Calculate BERT-scores in batch
            P, R, F1 = bert_score(
                predictions,
                references,
                model_type=self.model_type,
                device=self.device,
                verbose=False
            )
            
            results = []
            for i in range(len(predictions)):
                results.append({
                    "precision": float(P[i].item()),
                    "recall": float(R[i].item()),
                    "f1": float(F1[i].item()),
                    "score": float(F1[i].item()),
                    "model_type": self.model_type
                })
            
            return results
        
        except Exception as e:
            return [{"score": 0.0, "error": str(e)}] * len(predictions)


class ROUGEBaseline(BaselineMethod):
    """ROUGE baseline for text overlap metrics.

    Uses rouge_score library if available for better accuracy,
    falls back to custom implementation otherwise.
    """

    def __init__(self):
        super().__init__(
            name="rouge",
            description="ROUGE-based text overlap metrics"
        )
        # Try to use rouge_score library for better accuracy
        self.use_library = False
        self.scorer = None
        try:
            from rouge_score import rouge_scorer
            self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.use_library = True
        except ImportError:
            pass  # Fall back to custom implementation

    def evaluate(self, prediction: str, reference: str, **kwargs) -> Dict:
        """Evaluate using ROUGE metrics"""
        try:
            if self.use_library and self.scorer:
                # Use rouge_score library
                scores = self.scorer.score(reference, prediction)
                rouge_1 = {
                    "precision": scores['rouge1'].precision,
                    "recall": scores['rouge1'].recall,
                    "f1": scores['rouge1'].fmeasure
                }
                rouge_2 = {
                    "precision": scores['rouge2'].precision,
                    "recall": scores['rouge2'].recall,
                    "f1": scores['rouge2'].fmeasure
                }
                rouge_l = {
                    "precision": scores['rougeL'].precision,
                    "recall": scores['rougeL'].recall,
                    "f1": scores['rougeL'].fmeasure
                }
            else:
                # Fall back to custom implementation
                rouge_1 = self._calculate_rouge_n(prediction, reference, n=1)
                rouge_2 = self._calculate_rouge_n(prediction, reference, n=2)
                rouge_l = self._calculate_rouge_l(prediction, reference)

            # Average score
            avg_score = (rouge_1["f1"] + rouge_2["f1"] + rouge_l["f1"]) / 3

            return {
                "rouge_1": rouge_1,
                "rouge_2": rouge_2,
                "rouge_l": rouge_l,
                "score": avg_score,
                "average_f1": avg_score,
                "using_library": self.use_library
            }

        except Exception as e:
            return {
                "rouge_1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "rouge_2": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "rouge_l": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "score": 0.0,
                "error": str(e)
            }
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and split on whitespace/punctuation
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _calculate_rouge_n(self, prediction: str, reference: str, n: int) -> Dict:
        """Calculate ROUGE-N score"""
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)
        
        if len(pred_tokens) < n or len(ref_tokens) < n:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Generate n-grams
        pred_ngrams = self._get_ngrams(pred_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        
        if not ref_ngrams:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Count overlapping n-grams
        overlap = sum((pred_ngrams & ref_ngrams).values())
        
        # Calculate precision, recall, F1
        precision = overlap / sum(pred_ngrams.values()) if sum(pred_ngrams.values()) > 0 else 0.0
        recall = overlap / sum(ref_ngrams.values()) if sum(ref_ngrams.values()) > 0 else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-grams from tokens"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
        return Counter(ngrams)
    
    def _calculate_rouge_l(self, prediction: str, reference: str) -> Dict:
        """Calculate ROUGE-L (Longest Common Subsequence)"""
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)
        
        if not pred_tokens or not ref_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Calculate LCS length
        lcs_length = self._lcs_length(pred_tokens, ref_tokens)
        
        # Calculate precision, recall, F1
        precision = lcs_length / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
        recall = lcs_length / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]


class LLMJudgeBaseline(BaselineMethod):
    """Local LLM judge baseline using small models"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", 
                 device: Optional[str] = None):
        super().__init__(
            name="llm_judge",
            description="Local LLM-based evaluation using small models"
        )
        
        self.model_name = model_name
        self.device = device or self._get_device()
        self.model = None
        self.tokenizer = None
        
        # Use a smaller, local model that can run on CPU/MPS
        self.available_models = {
            "tiny": "microsoft/DialoGPT-small",  # ~117M parameters
            "small": "microsoft/DialoGPT-medium",  # ~345M parameters
            "medium": "distilgpt2",  # ~82M parameters
            "efficient": "gpt2"  # ~124M parameters
        }
        
        if not TRANSFORMERS_AVAILABLE:
            print("Warning: transformers not available. LLM judge will not work.")
        else:
            self._load_model()
    
    def _get_device(self):
        """Get optimal device"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _load_model(self):
        """Load the evaluation model"""
        if not TRANSFORMERS_AVAILABLE:
            return
        
        try:
            print(f"Loading LLM judge model: {self.model_name} on {self.device}")
            
            # Use a simple model that works well for evaluation
            model_name = self.available_models.get("medium", "distilgpt2")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            if self.device != "cpu":
                self.model = self.model.to(self.device)
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("LLM judge model loaded successfully")
            
        except Exception as e:
            print(f"Error loading LLM judge model: {e}")
            self.model = None
            self.tokenizer = None
    
    def evaluate(self, prediction: str, reference: str, **kwargs) -> Dict:
        """Evaluate using local LLM judge"""
        if not TRANSFORMERS_AVAILABLE or self.model is None:
            return {
                "score": 0.0,
                "confidence": 0.0,
                "error": "LLM judge model not available"
            }
        
        try:
            # Create evaluation prompt
            prompt = self._create_evaluation_prompt(prediction, reference)
            
            # Generate evaluation
            evaluation_result = self._generate_evaluation(prompt)
            
            # Parse the evaluation result
            score, confidence = self._parse_evaluation(evaluation_result)
            
            return {
                "score": score,
                "confidence": confidence,
                "evaluation_text": evaluation_result[:200],  # Truncate for storage
                "model_name": self.model_name
            }
        
        except Exception as e:
            return {
                "score": 0.0,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _create_evaluation_prompt(self, prediction: str, reference: str) -> str:
        """Create evaluation prompt for the LLM"""
        prompt = f"""Evaluate the quality of reasoning in the following response.

Reference/Expected: {reference[:200]}...

Response to evaluate: {prediction[:200]}...

Rate the reasoning quality from 0 to 100 based on:
- Logical consistency
- Factual accuracy  
- Clarity of reasoning steps
- Overall coherence

Provide just a numerical score (0-100):"""
        
        return prompt
    
    def _generate_evaluation(self, prompt: str) -> str:
        """Generate evaluation using the model"""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the generated text
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the new part (after the prompt)
            evaluation = generated[len(prompt):].strip()
            
            return evaluation
        
        except Exception as e:
            return f"Error in generation: {e}"
    
    def _parse_evaluation(self, evaluation_text: str) -> Tuple[float, float]:
        """Parse evaluation text to extract score and confidence"""
        try:
            # Look for numerical scores
            numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', evaluation_text)
            
            if numbers:
                # Take the first reasonable score (0-100 range)
                for num_str in numbers:
                    score = float(num_str)
                    if 0 <= score <= 100:
                        normalized_score = score / 100.0
                        
                        # Confidence based on how explicit the score is
                        confidence = 0.8 if len(numbers) == 1 else 0.6
                        
                        return normalized_score, confidence
            
            # Fallback: look for qualitative indicators
            evaluation_lower = evaluation_text.lower()
            
            if any(word in evaluation_lower for word in ["excellent", "perfect", "outstanding"]):
                return 0.9, 0.5
            elif any(word in evaluation_lower for word in ["good", "solid", "well"]):
                return 0.7, 0.5
            elif any(word in evaluation_lower for word in ["average", "okay", "moderate"]):
                return 0.5, 0.4
            elif any(word in evaluation_lower for word in ["poor", "weak", "bad"]):
                return 0.3, 0.5
            elif any(word in evaluation_lower for word in ["terrible", "awful", "horrible"]):
                return 0.1, 0.5
            
            # Default neutral score with low confidence
            return 0.5, 0.2
        
        except Exception:
            return 0.5, 0.1


class SimpleLexicalBaseline(BaselineMethod):
    """Simple lexical overlap baseline"""
    
    def __init__(self):
        super().__init__(
            name="lexical_overlap",
            description="Simple lexical overlap and similarity metrics"
        )
    
    def evaluate(self, prediction: str, reference: str, **kwargs) -> Dict:
        """Evaluate using lexical overlap metrics"""
        try:
            # Tokenize
            pred_tokens = set(self._tokenize(prediction))
            ref_tokens = set(self._tokenize(reference))
            
            if not ref_tokens:
                return {"score": 0.0, "error": "Empty reference"}
            
            # Calculate Jaccard similarity
            intersection = len(pred_tokens & ref_tokens)
            union = len(pred_tokens | ref_tokens)
            jaccard = intersection / union if union > 0 else 0.0
            
            # Calculate overlap coefficient (Szymkiewicz–Simpson coefficient)
            overlap_coeff = intersection / min(len(pred_tokens), len(ref_tokens)) if min(len(pred_tokens), len(ref_tokens)) > 0 else 0.0
            
            # Calculate precision and recall
            precision = intersection / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
            recall = intersection / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
            
            # F1 score
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Length ratio
            length_ratio = len(prediction) / len(reference) if len(reference) > 0 else 0.0
            
            return {
                "jaccard_similarity": jaccard,
                "overlap_coefficient": overlap_coeff,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "length_ratio": length_ratio,
                "score": f1,  # Use F1 as main score
                "word_overlap": {
                    "prediction_words": len(pred_tokens),
                    "reference_words": len(ref_tokens),
                    "common_words": intersection
                }
            }
        
        except Exception as e:
            return {"score": 0.0, "error": str(e)}
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens


class BaselineComparator:
    """Compares MERIT metrics with baseline methods"""
    
    def __init__(self, baselines: Optional[List[BaselineMethod]] = None):
        self.baselines = baselines or self._get_default_baselines()
        self.results_cache = {}
    
    def _get_default_baselines(self) -> List[BaselineMethod]:
        """Get default baseline methods"""
        baselines = [
            SimpleLexicalBaseline(),
            ROUGEBaseline()
        ]
        
        # Add BERT-score if available
        if BERT_SCORE_AVAILABLE:
            baselines.append(BERTScoreBaseline())
        
        # Add LLM judge if transformers available
        if TRANSFORMERS_AVAILABLE:
            baselines.append(LLMJudgeBaseline())
        
        return baselines
    
    def compare_methods(self, examples: List[Dict], merit_results: List[Dict]) -> Dict:
        """Compare MERIT with baseline methods"""
        
        comparison_results = {
            "examples_count": len(examples),
            "baseline_results": {},
            "correlations": {},
            "performance_comparison": {},
            "statistical_tests": {}
        }
        
        # Run baseline evaluations
        for baseline in self.baselines:
            print(f"Running {baseline.name} baseline...")
            
            baseline_scores = []
            for example in examples:
                prediction = example.get("response", example.get("prediction", ""))
                reference = example.get("reference", "")
                
                result = baseline.evaluate(prediction, reference)
                baseline_scores.append(result.get("score", 0.0))
            
            comparison_results["baseline_results"][baseline.name] = {
                "scores": baseline_scores,
                "mean_score": np.mean(baseline_scores),
                "std_score": np.std(baseline_scores),
                "description": baseline.description
            }
        
        # Compare with MERIT metrics
        merit_metrics = ["logical_consistency", "factual_accuracy", "reasoning_steps", "alignment"]
        
        for metric_name in merit_metrics:
            merit_scores = []
            for result in merit_results:
                if "metrics" in result and metric_name in result["metrics"]:
                    metric_data = result["metrics"][metric_name]
                    if isinstance(metric_data, dict) and "score" in metric_data:
                        merit_scores.append(metric_data["score"])
                    else:
                        merit_scores.append(0.0)
                else:
                    merit_scores.append(0.0)
            
            if len(merit_scores) != len(examples):
                continue
            
            # Calculate correlations with baselines
            metric_correlations = {}
            for baseline_name, baseline_data in comparison_results["baseline_results"].items():
                baseline_scores = baseline_data["scores"]
                
                if len(baseline_scores) == len(merit_scores):
                    try:
                        correlation = np.corrcoef(merit_scores, baseline_scores)[0, 1]
                        if not np.isnan(correlation):
                            metric_correlations[baseline_name] = float(correlation)
                    except:
                        metric_correlations[baseline_name] = 0.0
            
            comparison_results["correlations"][metric_name] = metric_correlations
        
        # Performance comparison summary
        comparison_results["performance_comparison"] = self._summarize_performance(comparison_results)
        
        return comparison_results
    
    def _summarize_performance(self, comparison_results: Dict) -> Dict:
        """Summarize overall performance comparison"""
        
        summary = {
            "highest_correlations": {},
            "average_correlations": {},
            "baseline_rankings": {}
        }
        
        # Find highest correlations for each MERIT metric
        for metric_name, correlations in comparison_results["correlations"].items():
            if correlations:
                best_baseline = max(correlations.items(), key=lambda x: abs(x[1]))
                summary["highest_correlations"][metric_name] = {
                    "baseline": best_baseline[0],
                    "correlation": best_baseline[1]
                }
        
        # Calculate average correlations across all MERIT metrics
        baseline_names = list(comparison_results["baseline_results"].keys())
        for baseline_name in baseline_names:
            correlations = []
            for metric_correlations in comparison_results["correlations"].values():
                if baseline_name in metric_correlations:
                    correlations.append(abs(metric_correlations[baseline_name]))
            
            if correlations:
                summary["average_correlations"][baseline_name] = np.mean(correlations)
        
        # Rank baselines by average correlation
        if summary["average_correlations"]:
            ranked_baselines = sorted(
                summary["average_correlations"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            summary["baseline_rankings"] = {
                rank: {"baseline": name, "avg_correlation": corr}
                for rank, (name, corr) in enumerate(ranked_baselines, 1)
            }
        
        return summary
    
    def create_comparison_report(self, comparison_results: Dict, output_file: str):
        """Create a detailed comparison report"""
        
        report = []
        report.append("MERIT vs BASELINE METHODS COMPARISON REPORT")
        report.append("=" * 55)
        report.append(f"Examples evaluated: {comparison_results['examples_count']}")
        report.append(f"Baseline methods: {len(comparison_results['baseline_results'])}")
        report.append("")
        
        # Baseline method descriptions
        report.append("BASELINE METHODS")
        report.append("-" * 20)
        for name, data in comparison_results["baseline_results"].items():
            report.append(f"{name.upper()}: {data['description']}")
            report.append(f"  Mean Score: {data['mean_score']:.3f} (±{data['std_score']:.3f})")
        report.append("")
        
        # Correlation results
        report.append("CORRELATION ANALYSIS")
        report.append("-" * 22)
        for metric_name, correlations in comparison_results["correlations"].items():
            report.append(f"{metric_name.upper()}:")
            for baseline_name, correlation in correlations.items():
                correlation_strength = self._interpret_correlation(abs(correlation))
                report.append(f"  vs {baseline_name}: {correlation:.3f} ({correlation_strength})")
            report.append("")
        
        # Performance summary
        if "performance_comparison" in comparison_results:
            perf = comparison_results["performance_comparison"]
            
            report.append("PERFORMANCE SUMMARY")
            report.append("-" * 20)
            
            if "highest_correlations" in perf:
                report.append("Best correlations by MERIT metric:")
                for metric, best in perf["highest_correlations"].items():
                    report.append(f"  {metric}: {best['baseline']} (r={best['correlation']:.3f})")
                report.append("")
            
            if "baseline_rankings" in perf:
                report.append("Baseline method rankings (by average correlation):")
                for rank, data in perf["baseline_rankings"].items():
                    report.append(f"  {rank}. {data['baseline']}: {data['avg_correlation']:.3f}")
                report.append("")
        
        # Save report
        with open(output_file, 'w') as f:
            f.write("\n".join(report))
        
        print(f"Comparison report saved to: {output_file}")
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation strength"""
        if correlation >= 0.8:
            return "very strong"
        elif correlation >= 0.6:
            return "strong"
        elif correlation >= 0.4:
            return "moderate"
        elif correlation >= 0.2:
            return "weak"
        else:
            return "very weak"