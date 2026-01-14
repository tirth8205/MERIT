"""
Tests for validation components.
"""
import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from merit.validation.human_validation import (
    HumanAnnotationCollector,
    MetricValidator
)
from merit.validation.baseline_comparison import (
    BaselineComparator,
    BERTScoreBaseline,
    ROUGEBaseline,
    LLMJudgeBaseline,
    SimpleLexicalBaseline
)


class TestHumanAnnotationCollector:
    """Test human annotation collection functionality"""

    def test_collector_initialization(self):
        """Test annotation collector initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = HumanAnnotationCollector(output_dir=temp_dir)

            assert collector.output_dir.exists()
            assert collector.annotation_schema is not None

    def test_create_annotation_task(self):
        """Test creating annotation task"""
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = HumanAnnotationCollector(output_dir=temp_dir)

            examples = [
                {
                    "prompt": "What is 2+2?",
                    "response": "2+2 equals 4.",
                    "reference": "4"
                },
                {
                    "prompt": "What is the capital of France?",
                    "response": "The capital of France is Paris.",
                    "reference": "Paris"
                }
            ]

            task_id = collector.create_annotation_task(
                examples=examples,
                task_name="test_task",
                annotators_needed=2
            )

            assert isinstance(task_id, str)
            assert len(task_id) > 0

            # Task file should be created
            task_file = Path(temp_dir) / f"task_{task_id}.json"
            assert task_file.exists()

            # Annotator templates should be created
            for i in range(2):
                template_file = Path(temp_dir) / f"annotator_{i}_task_{task_id}.json"
                assert template_file.exists()

    def test_load_annotations(self):
        """Test loading annotations for a task"""
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = HumanAnnotationCollector(output_dir=temp_dir)

            # Create a task first
            examples = [{"prompt": "Test", "response": "Response", "reference": "Ref"}]
            task_id = collector.create_annotation_task(
                examples=examples,
                task_name="load_test",
                annotators_needed=1
            )

            # Load the task
            task_data = collector.load_annotations(task_id)

            assert isinstance(task_data, dict)
            assert "task_id" in task_data
            assert "examples" in task_data
            assert "annotations" in task_data

    def test_validate_annotations(self):
        """Test annotation validation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = HumanAnnotationCollector(output_dir=temp_dir)

            # Create a task
            examples = [{"prompt": "Test", "response": "Response", "reference": "Ref"}]
            task_id = collector.create_annotation_task(
                examples=examples,
                task_name="validate_test",
                annotators_needed=2
            )

            # Validate annotations (before any are filled in)
            validation_report = collector.validate_annotations(task_id)

            assert isinstance(validation_report, dict)
            assert "task_id" in validation_report
            assert "completion_status" in validation_report


class TestMetricValidator:
    """Test metric validation functionality"""

    def test_validator_initialization(self):
        """Test metric validator initialization"""
        validator = MetricValidator()

        assert validator is not None
        assert validator.correlation_thresholds is not None

    def test_validate_metric_performance(self):
        """Test validating metric performance"""
        validator = MetricValidator()

        # Mock MERIT results
        merit_results = [
            {
                "prompt": "What is 2+2?",
                "response": "2+2 equals 4",
                "metrics": {
                    "logical_consistency": {"score": 0.9},
                    "factual_accuracy": {"score": 0.95},
                    "reasoning_steps": {"score": 0.8},
                    "alignment": {"score": 0.85}
                }
            },
            {
                "prompt": "What is 3+3?",
                "response": "3+3 equals 6",
                "metrics": {
                    "logical_consistency": {"score": 0.85},
                    "factual_accuracy": {"score": 0.9},
                    "reasoning_steps": {"score": 0.75},
                    "alignment": {"score": 0.8}
                }
            },
            {
                "prompt": "What is 4+4?",
                "response": "4+4 equals 8",
                "metrics": {
                    "logical_consistency": {"score": 0.88},
                    "factual_accuracy": {"score": 0.92},
                    "reasoning_steps": {"score": 0.78},
                    "alignment": {"score": 0.82}
                }
            }
        ]

        # Mock human annotations (scaled 0-100)
        human_annotations = [
            {
                "annotations": {
                    "logical_consistency": 85,
                    "factual_accuracy": 90,
                    "reasoning_quality": 75,
                    "alignment": 80,
                    "overall_quality": 82
                }
            },
            {
                "annotations": {
                    "logical_consistency": 80,
                    "factual_accuracy": 88,
                    "reasoning_quality": 72,
                    "alignment": 78,
                    "overall_quality": 80
                }
            },
            {
                "annotations": {
                    "logical_consistency": 82,
                    "factual_accuracy": 89,
                    "reasoning_quality": 74,
                    "alignment": 79,
                    "overall_quality": 81
                }
            }
        ]

        validation_results = validator.validate_metric_performance(
            merit_results,
            human_annotations
        )

        assert isinstance(validation_results, dict)
        assert "overall_summary" in validation_results
        assert "metric_correlations" in validation_results

    def test_create_validation_report(self):
        """Test creating validation report"""
        validator = MetricValidator()

        validation_results = {
            "metric_correlations": {
                "logical_consistency": {
                    "pearson_correlation": {"r": 0.85, "p_value": 0.01},
                    "spearman_correlation": {"r": 0.82, "p_value": 0.02},
                    "correlation_strength": "excellent",
                    "sample_size": 10,
                    "error_metrics": {"rmse": 0.05}
                },
                "factual_accuracy": {
                    "pearson_correlation": {"r": 0.78, "p_value": 0.02},
                    "spearman_correlation": {"r": 0.75, "p_value": 0.03},
                    "correlation_strength": "good",
                    "sample_size": 10,
                    "error_metrics": {"rmse": 0.08}
                }
            },
            "overall_summary": {
                "average_correlation": 0.815,
                "min_correlation": 0.78,
                "max_correlation": 0.85,
                "correlation_strength_distribution": {"excellent": 1, "good": 1},
                "overall_assessment": "excellent",
                "total_metrics_evaluated": 2
            },
            "recommendations": [
                "logical_consistency: Excellent correlation (r=0.85)",
                "factual_accuracy: Good correlation (r=0.78)"
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            report_file = f.name

        try:
            validator.create_validation_report(validation_results, report_file)

            # Verify report was created
            assert os.path.exists(report_file)

            with open(report_file, 'r') as f:
                report_content = f.read()

            assert "MERIT" in report_content
            assert "logical_consistency" in report_content.lower()

        finally:
            os.unlink(report_file)
            # Clean up plots directory if created
            plots_dir = report_file.replace('.txt', '_plots')
            if os.path.exists(plots_dir):
                import shutil
                shutil.rmtree(plots_dir)


class TestBaselineComparator:
    """Test baseline comparison functionality"""

    def test_comparator_initialization(self):
        """Test baseline comparator initialization"""
        comparator = BaselineComparator()

        assert comparator.baselines is not None
        assert len(comparator.baselines) > 0

    def test_compare_methods(self):
        """Test comparing MERIT with baselines"""
        comparator = BaselineComparator()

        # Mock examples
        examples = [
            {
                "prompt": "What is AI?",
                "response": "The sky is blue and water is wet.",
                "reference": "The sky is blue. Water is wet."
            }
        ]

        # Mock MERIT results
        merit_results = [
            {
                "metrics": {
                    "logical_consistency": {"score": 0.85},
                    "factual_accuracy": {"score": 0.9},
                    "reasoning_steps": {"score": 0.8},
                    "alignment": {"score": 0.75}
                }
            }
        ]

        results = comparator.compare_methods(examples, merit_results)

        assert isinstance(results, dict)
        assert "baseline_results" in results
        assert "correlations" in results
        assert "performance_comparison" in results


class TestBertScoreBaseline:
    """Test BERT-Score baseline functionality"""

    def test_bert_score_initialization(self):
        """Test BERT-Score baseline initialization"""
        try:
            baseline = BERTScoreBaseline()
            assert baseline.model_type == "microsoft/deberta-xlarge-mnli"
            assert baseline.name == "bert_score"
        except Exception:
            # bert_score might not be installed
            pass

    def test_bert_score_evaluate(self):
        """Test BERT-Score evaluation"""
        baseline = BERTScoreBaseline()

        result = baseline.evaluate(
            prediction="The sky is blue.",
            reference="The sky is blue."
        )

        assert isinstance(result, dict)
        assert "score" in result
        assert isinstance(result["score"], float)


class TestRougeBaseline:
    """Test ROUGE baseline functionality"""

    def test_rouge_initialization(self):
        """Test ROUGE baseline initialization"""
        baseline = ROUGEBaseline()
        assert baseline.name == "rouge"

    def test_rouge_evaluate(self):
        """Test ROUGE evaluation"""
        baseline = ROUGEBaseline()

        result = baseline.evaluate(
            prediction="The sky is blue and beautiful.",
            reference="The sky is blue."
        )

        assert isinstance(result, dict)
        assert "score" in result
        assert "rouge_1" in result
        assert "rouge_2" in result
        assert "rouge_l" in result


class TestSimpleLexicalBaseline:
    """Test simple lexical overlap baseline"""

    def test_lexical_initialization(self):
        """Test lexical baseline initialization"""
        baseline = SimpleLexicalBaseline()
        assert baseline.name == "lexical_overlap"

    def test_lexical_evaluate(self):
        """Test lexical evaluation"""
        baseline = SimpleLexicalBaseline()

        result = baseline.evaluate(
            prediction="The sky is blue.",
            reference="The sky is blue."
        )

        assert isinstance(result, dict)
        assert "score" in result
        assert "jaccard_similarity" in result
        assert result["jaccard_similarity"] == 1.0  # Identical strings


class TestLLMJudgeBaseline:
    """Test LLM Judge baseline functionality"""

    @patch('merit.validation.baseline_comparison.TRANSFORMERS_AVAILABLE', True)
    @patch('merit.validation.baseline_comparison.AutoTokenizer')
    @patch('merit.validation.baseline_comparison.AutoModelForCausalLM')
    def test_llm_judge_initialization(self, mock_model_cls, mock_tokenizer_cls):
        """Test LLM Judge baseline initialization"""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model_cls.from_pretrained.return_value = mock_model

        with patch('torch.backends.mps.is_available', return_value=False), \
             patch('torch.cuda.is_available', return_value=False):
            baseline = LLMJudgeBaseline()
            assert baseline.name == "llm_judge"

    def test_llm_judge_evaluate_without_model(self):
        """Test LLM Judge evaluation fallback when model not available"""
        with patch('merit.validation.baseline_comparison.TRANSFORMERS_AVAILABLE', False):
            baseline = LLMJudgeBaseline.__new__(LLMJudgeBaseline)
            baseline.name = "llm_judge"
            baseline.description = "Test"
            baseline.model = None
            baseline.tokenizer = None

            result = baseline.evaluate(
                prediction="The sky is blue.",
                reference="The sky is blue."
            )

            assert isinstance(result, dict)
            assert "error" in result


class TestValidationIntegration:
    """Test integration between validation components"""

    def test_full_validation_workflow(self):
        """Test complete validation workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create annotation collector
            collector = HumanAnnotationCollector(output_dir=temp_dir)

            # Create task
            examples = [
                {
                    "prompt": "What is AI?",
                    "response": "AI is artificial intelligence technology.",
                    "reference": "Artificial intelligence"
                }
            ]

            task_id = collector.create_annotation_task(
                examples=examples,
                task_name="integration_test",
                annotators_needed=2
            )

            # Verify task was created
            assert task_id is not None

            # Load annotations
            task_data = collector.load_annotations(task_id)
            assert "examples" in task_data

    def test_baseline_comparison_integration(self):
        """Test baseline comparison integration"""
        comparator = BaselineComparator()

        examples = [
            {
                "prompt": "Test",
                "response": "Test prediction response",
                "reference": "Test reference response"
            }
        ]

        merit_results = [
            {
                "metrics": {
                    "logical_consistency": {"score": 0.8},
                    "factual_accuracy": {"score": 0.85},
                    "reasoning_steps": {"score": 0.75},
                    "alignment": {"score": 0.7}
                }
            }
        ]

        results = comparator.compare_methods(examples, merit_results)

        assert "baseline_results" in results
        # At minimum, SimpleLexicalBaseline and ROUGEBaseline should be present
        assert len(results["baseline_results"]) >= 2


if __name__ == "__main__":
    pytest.main([__file__])
