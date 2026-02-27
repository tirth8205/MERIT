"""
Tests for experiment framework.
"""
import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from merit.experiments.config import ExperimentConfig
from merit.experiments.runner import ExperimentRunner


class TestExperimentConfig:
    """Test experiment configuration"""

    def test_config_creation(self):
        """Test creating experiment configuration"""
        config = ExperimentConfig(
            experiment_name="test_experiment",
            models=["gpt2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
            benchmarks=["arc", "hellaswag"],
            sample_sizes=[10, 20],
            num_runs=3,
            temperature=0.7,
            max_tokens=100,
            random_seed=42,
            metrics=["logical_consistency", "factual_accuracy"],
            baseline_methods=["bert_score", "rouge"],
            statistical_tests=["t_test", "wilcoxon"],
            output_dir="test_output"
        )

        assert config.experiment_name == "test_experiment"
        assert len(config.models) == 2
        assert len(config.benchmarks) == 2
        assert len(config.sample_sizes) == 2
        assert config.num_runs == 3
        assert config.temperature == 0.7
        assert config.max_tokens == 100
        assert config.random_seed == 42
        assert len(config.metrics) == 2
        assert len(config.baseline_methods) == 2
        assert len(config.statistical_tests) == 2
        assert config.output_dir == "test_output"

    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config should not raise
        valid_config = ExperimentConfig(
            experiment_name="valid_test",
            models=["gpt2"],
            benchmarks=["arc"],
            sample_sizes=[10],
            num_runs=1,
            temperature=0.7,
            max_tokens=100,
            random_seed=42,
            metrics=["logical_consistency"],
            baseline_methods=[],
            statistical_tests=[],
            output_dir="test_output"
        )

        # Should not raise any exceptions
        assert valid_config.experiment_name == "valid_test"

    def test_config_serialization(self):
        """Test configuration serialization"""
        config = ExperimentConfig(
            experiment_name="serialize_test",
            models=["gpt2"],
            benchmarks=["arc"],
            sample_sizes=[5],
            num_runs=2,
            temperature=0.7,
            max_tokens=100,
            random_seed=42,
            metrics=["logical_consistency"],
            baseline_methods=[],
            statistical_tests=[],
            output_dir="test_output"
        )

        # Should be serializable to dict using asdict
        config_dict = asdict(config)
        assert isinstance(config_dict, dict)
        assert config_dict["experiment_name"] == "serialize_test"

    def test_config_save_and_load(self):
        """Test saving and loading configuration"""
        config = ExperimentConfig(
            experiment_name="save_load_test",
            models=["gpt2"],
            benchmarks=["arc"],
            sample_sizes=[10],
            num_runs=2,
            temperature=0.7,
            max_tokens=100,
            random_seed=42,
            metrics=["logical_consistency"],
            baseline_methods=[],
            statistical_tests=[],
            output_dir="test_output"
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = f.name

        try:
            # Save config
            config.save(config_file)

            # Load config
            loaded_config = ExperimentConfig.load(config_file)

            assert loaded_config.experiment_name == config.experiment_name
            assert loaded_config.models == config.models
            assert loaded_config.num_runs == config.num_runs

        finally:
            os.unlink(config_file)


def _make_mock_runner(config):
    """Create an ExperimentRunner with heavy dependencies mocked out."""
    mock_manager_mod = MagicMock()
    with patch.dict('sys.modules', {'merit.models.manager': mock_manager_mod}):
        with patch.object(ExperimentRunner, '_initialize_metrics', return_value={}):
            with patch.object(ExperimentRunner, '_initialize_baselines', return_value={}):
                runner = ExperimentRunner(config)
    return runner


class TestExperimentRunner:
    """Test experiment runner functionality"""

    @pytest.fixture
    def mock_config(self):
        """Create mock experiment configuration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ExperimentConfig(
                experiment_name="test_run",
                models=["gpt2-medium"],
                benchmarks=["arc"],
                sample_sizes=[5],
                num_runs=2,
                temperature=0.1,
                max_tokens=50,
                random_seed=42,
                metrics=["logical_consistency", "factual_accuracy"],
                baseline_methods=["bert_score"],
                statistical_tests=["t_test"],
                output_dir=temp_dir
            )

    def test_runner_initialization(self, mock_config):
        """Test experiment runner initialization"""
        runner = _make_mock_runner(mock_config)

        assert runner.config == mock_config
        assert runner.experiment_id is not None
        assert runner.results is not None
        assert "experiment_id" in runner.results

    def test_runner_output_directory_creation(self, mock_config):
        """Test that runner creates output directory"""
        runner = _make_mock_runner(mock_config)

        # Output directory should be created
        assert runner.output_dir.exists()


class TestExperimentConfigDefaults:
    """Test ExperimentConfig with various configurations"""

    def test_config_with_single_model(self):
        """Test config with single model"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="single_model_test",
                models=["gpt2-medium"],
                benchmarks=["arc"],
                sample_sizes=[10],
                num_runs=1,
                temperature=0.7,
                max_tokens=100,
                random_seed=42,
                metrics=["logical_consistency"],
                baseline_methods=[],
                statistical_tests=[],
                output_dir=temp_dir
            )

            assert len(config.models) == 1
            assert config.models[0] == "gpt2-medium"

    def test_config_with_multiple_benchmarks(self):
        """Test config with multiple benchmarks"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="multi_benchmark_test",
                models=["gpt2-medium"],
                benchmarks=["arc", "hellaswag", "mmlu"],
                sample_sizes=[10],
                num_runs=1,
                temperature=0.7,
                max_tokens=100,
                random_seed=42,
                metrics=["logical_consistency"],
                baseline_methods=[],
                statistical_tests=[],
                output_dir=temp_dir
            )

            assert len(config.benchmarks) == 3

    def test_config_dict_conversion(self):
        """Test converting config to dict"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="dict_test",
                models=["gpt2-medium"],
                benchmarks=["arc"],
                sample_sizes=[10],
                num_runs=2,
                temperature=0.7,
                max_tokens=100,
                random_seed=42,
                metrics=["logical_consistency"],
                baseline_methods=[],
                statistical_tests=[],
                output_dir=temp_dir
            )

            config_dict = asdict(config)

            assert isinstance(config_dict, dict)
            assert config_dict["experiment_name"] == "dict_test"
            assert config_dict["num_runs"] == 2


class TestExperimentIntegration:
    """Test integration of experiment components"""

    def test_config_to_runner_integration(self):
        """Test that config integrates properly with runner"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="integration_test",
                models=["gpt2-medium"],
                benchmarks=["arc"],
                sample_sizes=[5],
                num_runs=1,
                temperature=0.7,
                max_tokens=50,
                random_seed=42,
                metrics=["logical_consistency"],
                baseline_methods=[],
                statistical_tests=[],
                output_dir=temp_dir
            )

            runner = _make_mock_runner(config)

            assert runner.config.experiment_name == "integration_test"
            assert runner.config.models == ["gpt2-medium"]
            assert runner.config.num_runs == 1


@pytest.mark.parametrize("num_runs", [1, 3, 5])
def test_experiment_with_different_runs(num_runs):
    """Test experiment config with different numbers of runs"""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ExperimentConfig(
            experiment_name=f"test_runs_{num_runs}",
            models=["gpt2-medium"],
            benchmarks=["arc"],
            sample_sizes=[5],
            num_runs=num_runs,
            temperature=0.7,
            max_tokens=100,
            random_seed=42,
            metrics=["logical_consistency"],
            baseline_methods=[],
            statistical_tests=[],
            output_dir=temp_dir
        )

        assert config.num_runs == num_runs


@pytest.mark.parametrize("sample_size", [1, 10, 50])
def test_experiment_with_different_sample_sizes(sample_size):
    """Test experiment config with different sample sizes"""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ExperimentConfig(
            experiment_name=f"test_samples_{sample_size}",
            models=["gpt2-medium"],
            benchmarks=["arc"],
            sample_sizes=[sample_size],
            num_runs=1,
            temperature=0.7,
            max_tokens=100,
            random_seed=42,
            metrics=["logical_consistency"],
            baseline_methods=[],
            statistical_tests=[],
            output_dir=temp_dir
        )

        assert sample_size in config.sample_sizes


class TestMetricModeConfig:
    """Test the metric_mode configuration field."""

    def test_default_metric_mode(self):
        """Default metric_mode should be 'heuristic'."""
        config = ExperimentConfig(
            experiment_name="mode_default",
            models=["gpt2"],
            benchmarks=["arc"],
            sample_sizes=[5],
        )
        assert config.metric_mode == "heuristic"

    @pytest.mark.parametrize("mode", ["heuristic", "llm_judge", "both"])
    def test_metric_mode_values(self, mode):
        """All three metric_mode values should be accepted."""
        config = ExperimentConfig(
            experiment_name="mode_test",
            models=["gpt2"],
            benchmarks=["arc"],
            sample_sizes=[5],
            metric_mode=mode,
        )
        assert config.metric_mode == mode

    def test_metric_mode_serialization_roundtrip(self):
        """metric_mode should survive save/load."""
        config = ExperimentConfig(
            experiment_name="mode_roundtrip",
            models=["gpt2"],
            benchmarks=["arc"],
            sample_sizes=[5],
            metric_mode="both",
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = f.name

        try:
            config.save(config_file)
            loaded = ExperimentConfig.load(config_file)
            assert loaded.metric_mode == "both"
        finally:
            os.unlink(config_file)


class TestRunnerMetricModes:
    """Test ExperimentRunner behavior under different metric modes."""

    def test_runner_heuristic_mode_initialises_heuristic_key(self):
        """In heuristic mode, metrics dict should contain 'heuristic' key."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="heuristic_runner",
                models=["gpt2"],
                benchmarks=["arc"],
                sample_sizes=[5],
                metric_mode="heuristic",
                output_dir=temp_dir,
            )
            mock_manager_mod = MagicMock()
            with patch.dict('sys.modules', {'merit.models.manager': mock_manager_mod}):
                with patch.object(ExperimentRunner, '_initialize_baselines', return_value={}):
                    runner = ExperimentRunner(config)

            assert "heuristic" in runner.metrics
            assert "llm_judge" not in runner.metrics

    def test_runner_llm_judge_mode_initialises_llm_judge_key(self):
        """In llm_judge mode, metrics dict should contain 'llm_judge' key."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="judge_runner",
                models=["gpt2"],
                benchmarks=["arc"],
                sample_sizes=[5],
                metric_mode="llm_judge",
                output_dir=temp_dir,
            )
            mock_judge = MagicMock()
            mock_manager_mod = MagicMock()
            with patch.dict('sys.modules', {'merit.models.manager': mock_manager_mod}):
                with patch.object(ExperimentRunner, '_initialize_baselines', return_value={}):
                    with patch('merit.core.llm_judge.LLMJudge', return_value=mock_judge):
                        runner = ExperimentRunner(config)

            assert "llm_judge" in runner.metrics
            assert "heuristic" not in runner.metrics

    def test_runner_both_mode_has_both_keys(self):
        """In 'both' mode, metrics dict should contain both keys."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="both_runner",
                models=["gpt2"],
                benchmarks=["arc"],
                sample_sizes=[5],
                metric_mode="both",
                output_dir=temp_dir,
            )
            mock_judge = MagicMock()
            both_metrics = {
                "heuristic": {"logical_consistency": MagicMock()},
                "llm_judge": mock_judge,
            }
            mock_manager_mod = MagicMock()
            with patch.dict('sys.modules', {'merit.models.manager': mock_manager_mod}):
                with patch.object(ExperimentRunner, '_initialize_baselines', return_value={}):
                    with patch.object(ExperimentRunner, '_initialize_metrics', return_value=both_metrics):
                        runner = ExperimentRunner(config)

            assert "heuristic" in runner.metrics
            assert "llm_judge" in runner.metrics

    def test_run_evaluation_heuristic_output_structure(self):
        """_run_evaluation in heuristic mode produces merit_heuristic and average_metrics."""
        import numpy as np

        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="eval_struct",
                models=["gpt2"],
                benchmarks=["arc"],
                sample_sizes=[2],
                metric_mode="heuristic",
                metrics=["logical_consistency"],
                baseline_methods=[],
                output_dir=temp_dir,
            )
            runner = _make_mock_runner(config)

            # Set up a mock heuristic metric
            mock_metric = MagicMock()
            mock_metric.compute.return_value = {"score": 0.75}
            runner.metrics = {"heuristic": {"logical_consistency": mock_metric}}
            runner.baselines = {}

            # Fake model adapter
            model_adapter = MagicMock()
            model_adapter.generate.return_value = "test response"

            dataset = [
                {"prompt": "q1", "reference": "a1"},
                {"prompt": "q2", "reference": "a2"},
            ]

            result = runner._run_evaluation(model_adapter, dataset)

            assert "merit_heuristic" in result
            assert "average_metrics" in result
            assert result["merit_heuristic"]["logical_consistency"] == 0.75
            assert result["average_metrics"]["logical_consistency"] == 0.75
            assert result["total_samples"] == 2

    def test_run_evaluation_baselines_output_structure(self):
        """When baselines are configured, results should contain 'baselines' key."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="baseline_struct",
                models=["gpt2"],
                benchmarks=["arc"],
                sample_sizes=[1],
                metric_mode="heuristic",
                metrics=["logical_consistency"],
                baseline_methods=["bert_score"],
                output_dir=temp_dir,
            )
            runner = _make_mock_runner(config)

            mock_metric = MagicMock()
            mock_metric.compute.return_value = {"score": 0.8}
            runner.metrics = {"heuristic": {"logical_consistency": mock_metric}}

            mock_bl = MagicMock()
            mock_bl.evaluate.return_value = {"f1": 0.65, "score": 0.65}
            runner.baselines = {"bert_score": mock_bl}

            model_adapter = MagicMock()
            model_adapter.generate.return_value = "test"

            dataset = [{"prompt": "q", "reference": "a"}]

            result = runner._run_evaluation(model_adapter, dataset)

            assert "baselines" in result
            assert result["baselines"]["bert_score"] == 0.65

    def test_calculate_statistics_structured(self):
        """_calculate_statistics should handle structured results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                experiment_name="stats_test",
                models=["gpt2"],
                benchmarks=["arc"],
                sample_sizes=[5],
                output_dir=temp_dir,
            )
            runner = _make_mock_runner(config)

            runs = [
                {
                    "merit_heuristic": {"logical_consistency": 0.8},
                    "average_metrics": {"logical_consistency": 0.8},
                    "baselines": {"bert_score": 0.6},
                    "task_accuracy": 0.5,
                    "total_samples": 5,
                    "individual_results": [],
                },
                {
                    "merit_heuristic": {"logical_consistency": 0.9},
                    "average_metrics": {"logical_consistency": 0.9},
                    "baselines": {"bert_score": 0.7},
                    "task_accuracy": 0.6,
                    "total_samples": 5,
                    "individual_results": [],
                },
            ]

            stats = runner._calculate_statistics(runs)

            assert "merit_heuristic" in stats
            assert "baselines" in stats
            assert "metric_statistics" in stats
            assert stats["merit_heuristic"]["logical_consistency"]["mean_across_runs"] == pytest.approx(0.85)
            assert stats["baselines"]["bert_score"]["mean_across_runs"] == pytest.approx(0.65)
            assert stats["task_accuracy"]["mean"] == pytest.approx(0.55)


if __name__ == "__main__":
    pytest.main([__file__])
