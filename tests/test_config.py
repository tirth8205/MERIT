"""
Tests for configuration management system.
"""
import pytest
import os
import tempfile
import json
import yaml
from pathlib import Path

from merit.config import (
    ModelConfig,
    MetricConfig,
    DatasetConfig,
    ExperimentConfig,
    ValidationConfig,
    MeritConfig,
    ConfigurationManager,
    create_default_config_file,
    load_config,
    validate_config_file
)


class TestModelConfig:
    """Test ModelConfig dataclass"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = ModelConfig(name="test-model")
        
        assert config.name == "test-model"
        assert config.adapter_type == "local"
        assert config.device == "auto"
        assert config.max_tokens == 1000
        assert config.temperature == 0.1
    
    def test_custom_values(self):
        """Test custom configuration values"""
        config = ModelConfig(
            name="custom-model",
            adapter_type="api",
            device="cuda",
            max_tokens=2000,
            temperature=0.5
        )
        
        assert config.name == "custom-model"
        assert config.adapter_type == "api"
        assert config.device == "cuda"
        assert config.max_tokens == 2000
        assert config.temperature == 0.5
    
    def test_cache_dir_expansion(self):
        """Test cache directory path expansion"""
        config = ModelConfig(name="test", cache_dir="~/test_cache")
        assert config.cache_dir == os.path.expanduser("~/test_cache")


class TestMetricConfig:
    """Test MetricConfig dataclass"""
    
    def test_default_values(self):
        """Test default metric configuration"""
        config = MetricConfig(name="test_metric")
        
        assert config.name == "test_metric"
        assert config.enabled is True
        assert config.parameters == {}
        assert config.weight == 1.0
    
    def test_custom_parameters(self):
        """Test custom metric parameters"""
        params = {"threshold": 0.8, "max_length": 100}
        config = MetricConfig(
            name="custom_metric",
            enabled=False,
            parameters=params,
            weight=0.5
        )
        
        assert config.parameters == params
        assert config.enabled is False
        assert config.weight == 0.5


class TestExperimentConfig:
    """Test ExperimentConfig dataclass"""
    
    def test_default_experiment(self):
        """Test default experiment configuration"""
        config = ExperimentConfig(name="test_experiment")
        
        assert config.name == "test_experiment"
        assert config.description == ""
        assert config.models == []
        assert config.datasets == []
        assert config.metrics == []
        assert config.num_runs == 3
        assert config.random_seed == 42
    
    def test_with_components(self):
        """Test experiment with models, datasets, metrics"""
        models = [ModelConfig(name="model1")]
        datasets = [DatasetConfig(name="dataset1")]
        metrics = [MetricConfig(name="metric1")]
        
        config = ExperimentConfig(
            name="full_experiment",
            models=models,
            datasets=datasets,
            metrics=metrics,
            num_runs=5
        )
        
        assert len(config.models) == 1
        assert len(config.datasets) == 1
        assert len(config.metrics) == 1
        assert config.num_runs == 5


class TestMeritConfig:
    """Test main MeritConfig dataclass"""
    
    def test_default_config(self):
        """Test default MERIT configuration"""
        config = MeritConfig()
        
        assert config.version == "2.0.0"
        assert config.log_level == "INFO"
        assert isinstance(config.experiment, ExperimentConfig)
        assert isinstance(config.validation, ValidationConfig)
        assert isinstance(config.system, dict)
    
    def test_path_expansion(self):
        """Test path expansion in configuration"""
        config = MeritConfig(
            cache_dir="~/test_cache",
            data_dir="~/test_data"
        )
        
        assert config.cache_dir == os.path.expanduser("~/test_cache")
        assert config.data_dir == os.path.expanduser("~/test_data")


class TestConfigurationManager:
    """Test configuration manager"""
    
    def test_default_configuration(self):
        """Test loading default configuration"""
        manager = ConfigurationManager()
        config = manager.get_config()
        
        assert isinstance(config, MeritConfig)
        assert config.version == "2.0.0"
        assert "default" in manager.config_sources
    
    def test_yaml_config_loading(self):
        """Test loading YAML configuration file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'version': '2.1.0',
                'log_level': 'DEBUG',
                'experiment': {
                    'name': 'yaml_test',
                    'num_runs': 5
                }
            }
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            manager = ConfigurationManager(config_path)
            config = manager.get_config()
            
            assert config.version == '2.1.0'
            assert config.log_level == 'DEBUG'
            assert config.experiment.name == 'yaml_test'
            assert config.experiment.num_runs == 5
            assert f"file:{config_path}" in manager.config_sources
            
        finally:
            os.unlink(config_path)
    
    def test_json_config_loading(self):
        """Test loading JSON configuration file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                'version': '2.2.0',
                'log_level': 'WARNING',
                'experiment': {
                    'name': 'json_test',
                    'num_runs': 7
                }
            }
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            manager = ConfigurationManager(config_path)
            config = manager.get_config()
            
            assert config.version == '2.2.0'
            assert config.log_level == 'WARNING'
            assert config.experiment.name == 'json_test'
            assert config.experiment.num_runs == 7
            
        finally:
            os.unlink(config_path)
    
    def test_environment_variables(self):
        """Test loading configuration from environment variables"""
        env_vars = {
            'MERIT_LOG_LEVEL': 'ERROR',
            'MERIT_EXPERIMENT_NAME': 'env_test',
            'MERIT_NUM_RUNS': '10',
            'MERIT_TEMPERATURE': '0.8',
            'MERIT_MAX_WORKERS': '8'
        }
        
        # Set environment variables
        for key, value in env_vars.items():
            os.environ[key] = value
        
        try:
            manager = ConfigurationManager()
            config = manager.get_config()
            
            assert config.log_level == 'ERROR'
            assert config.experiment.name == 'env_test'
            assert config.experiment.num_runs == 10
            assert config.system['max_workers'] == 8
            assert "environment" in manager.config_sources
            
        finally:
            # Clean up environment variables
            for key in env_vars:
                os.environ.pop(key, None)
    
    def test_config_validation(self):
        """Test configuration validation"""
        manager = ConfigurationManager()
        issues = manager.validate_config()
        
        # Default config should have some issues (no models, datasets, metrics)
        assert len(issues) > 0
        assert any("model" in issue.lower() for issue in issues)
        assert any("dataset" in issue.lower() for issue in issues)
        assert any("metric" in issue.lower() for issue in issues)
    
    def test_config_saving(self):
        """Test saving configuration to file"""
        manager = ConfigurationManager()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            output_path = f.name
        
        try:
            manager.save_config(output_path, format="yaml")
            
            # Load saved config and verify
            with open(output_path, 'r') as f:
                saved_data = yaml.safe_load(f)
            
            assert 'version' in saved_data
            assert 'experiment' in saved_data
            assert saved_data['version'] == '2.0.0'
            
        finally:
            os.unlink(output_path)
    
    def test_invalid_config_file(self):
        """Test handling of invalid configuration file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            manager = ConfigurationManager(config_path)
            # Should fall back to default config
            config = manager.get_config()
            assert isinstance(config, MeritConfig)
            
        finally:
            os.unlink(config_path)


class TestConfigurationFunctions:
    """Test configuration utility functions"""
    
    def test_create_default_config_file(self):
        """Test creating default configuration file"""
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            create_default_config_file(config_path)
            
            # Verify file was created and is valid
            assert os.path.exists(config_path)
            
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            assert 'version' in config_data
            assert 'experiment' in config_data
            assert 'validation' in config_data
            
        finally:
            os.unlink(config_path)
    
    def test_load_config_function(self):
        """Test load_config utility function"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'version': '2.5.0',
                'log_level': 'DEBUG'
            }
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            
            assert isinstance(config, MeritConfig)
            assert config.version == '2.5.0'
            assert config.log_level == 'DEBUG'
            
        finally:
            os.unlink(config_path)
    
    def test_validate_config_file_function(self):
        """Test validate_config_file utility function"""
        # Create valid config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'experiment': {
                    'name': 'test_experiment',
                    'models': [{'name': 'test_model'}],
                    'datasets': [{'name': 'test_dataset'}],
                    'metrics': [{'name': 'test_metric'}],
                    'num_runs': 3
                }
            }
            yaml.dump(config_data, f)
            valid_config_path = f.name
        
        # Create invalid config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'experiment': {
                    'name': '',  # Invalid: empty name
                    'models': [],  # Invalid: no models
                    'num_runs': 0  # Invalid: zero runs
                }
            }
            yaml.dump(config_data, f)
            invalid_config_path = f.name
        
        try:
            # Valid config should pass
            assert validate_config_file(valid_config_path) is True
            
            # Invalid config should fail
            assert validate_config_file(invalid_config_path) is False
            
        finally:
            os.unlink(valid_config_path)
            os.unlink(invalid_config_path)
    
    def test_nonexistent_config_file(self):
        """Test handling of non-existent configuration file"""
        nonexistent_path = "/path/that/does/not/exist.yaml"
        
        # Should return False for non-existent file
        assert validate_config_file(nonexistent_path) is False


class TestConfigurationIntegration:
    """Test integration between different configuration components"""
    
    def test_full_configuration_workflow(self):
        """Test complete configuration workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test_config.yaml")
            
            # Create default config
            create_default_config_file(config_path)
            
            # Load and modify config
            config = load_config(config_path)
            config.experiment.name = "integration_test"
            config.experiment.num_runs = 5
            
            # Save modified config
            manager = ConfigurationManager()
            manager.config = config
            manager.save_config(config_path, format="yaml")
            
            # Reload and verify
            reloaded_config = load_config(config_path)
            assert reloaded_config.experiment.name == "integration_test"
            assert reloaded_config.experiment.num_runs == 5
            
            # Validate the configuration
            assert validate_config_file(config_path)
    
    def test_configuration_priority(self):
        """Test configuration loading priority (file > env > default)"""
        # Set environment variable
        os.environ['MERIT_LOG_LEVEL'] = 'WARNING'
        
        # Create config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {'log_level': 'ERROR'}
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            manager = ConfigurationManager(config_path)
            config = manager.get_config()
            
            # File should override environment
            assert config.log_level == 'ERROR'
            assert "environment" in manager.config_sources
            assert f"file:{config_path}" in manager.config_sources
            
        finally:
            os.unlink(config_path)
            os.environ.pop('MERIT_LOG_LEVEL', None)


if __name__ == "__main__":
    pytest.main([__file__])