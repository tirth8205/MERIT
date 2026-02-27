"""Integration tests for the full MERIT pipeline."""
import pytest
from unittest.mock import patch, MagicMock
from merit.core.base import BaseMetric, MetricResult
from merit.core.consistency import LogicalConsistencyMetric
from merit.core.llm_judge import LLMJudge, JudgeConfig
from merit.experiments.config import ExperimentConfig
from merit.baselines.bertscore import BERTScoreBaseline
from merit.utils.stats import bootstrap_ci, cohens_d
from merit.reporting.tables import generate_results_table


class TestMetricInterface:
    """Verify all metrics follow the BaseMetric contract."""

    def test_heuristic_returns_metric_result(self):
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 384]

        with patch('merit.core.device.DeviceManager.get_optimal_device', return_value='cpu'), \
             patch('merit.core.consistency.SentenceTransformer', return_value=mock_model), \
             patch('merit.core.consistency.spacy.load', return_value=MagicMock()):
            metric = LogicalConsistencyMetric()
            result = metric.compute("The sky is blue.")
            assert isinstance(result, MetricResult)
            assert 0.0 <= result.score <= 1.0
            assert result.dimension == "consistency"

    def test_llm_judge_returns_metric_result(self):
        judge = LLMJudge(JudgeConfig())
        mock_response = {"score": 4, "explanation": "Good", "contradictions": []}
        with patch.object(judge, '_call_judge', return_value=mock_response):
            result = judge.evaluate_consistency("Test response")
            assert isinstance(result, MetricResult)
            assert result.score == 0.75


class TestEndToEnd:
    """End-to-end pipeline tests."""

    def test_results_to_latex(self):
        results = {
            "model_a": {"arc": {"consistency": 0.8, "factual": 0.7}},
            "model_b": {"arc": {"consistency": 0.6, "factual": 0.9}},
        }
        latex = generate_results_table(results)
        assert "model_a" in latex or "model\\_a" in latex
        assert "0.80" in latex or "0.8" in latex

    def test_stats_on_results(self):
        from merit.utils.stats import aggregate_runs
        runs = [[0.8, 0.7], [0.82, 0.71], [0.79, 0.69]]
        stats = aggregate_runs(runs)
        assert 0.7 < stats["mean"] < 0.8
        assert stats["n_runs"] == 3

    def test_config_roundtrip(self, tmp_path):
        config = ExperimentConfig(
            experiment_name="test",
            models=["tinyllama-1b"],
            benchmarks=["arc"],
            sample_sizes=[10],
        )
        path = str(tmp_path / "config.json")
        config.save(path)
        loaded = ExperimentConfig.load(path)
        assert loaded.experiment_name == "test"
        assert loaded.models == ["tinyllama-1b"]

    def test_imports_from_top_level(self):
        from merit import BaseMetric, MetricResult, DeviceManager, ExperimentConfig
        assert BaseMetric is not None
        assert MetricResult is not None
