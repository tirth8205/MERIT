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


if __name__ == "__main__":
    pytest.main([__file__])
