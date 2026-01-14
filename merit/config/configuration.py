"""
Configuration management system for MERIT.
Supports YAML, JSON, and environment variable configuration.
"""
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime

try:
    import hydra
    from omegaconf import DictConfig, OmegaConfig
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    print("Note: hydra-core not available. Using basic YAML configuration.")


@dataclass
class ModelConfig:
    """Configuration for model settings"""
    name: str
    adapter_type: str = "local"  # local, api, huggingface
    model_path: str = ""
    api_key: str = ""
    device: str = "auto"  # auto, cpu, mps, cuda
    max_tokens: int = 1000
    temperature: float = 0.1
    cache_dir: str = "~/.cache/merit_models"
    
    def __post_init__(self):
        # Expand user path
        self.cache_dir = os.path.expanduser(self.cache_dir)


@dataclass
class MetricConfig:
    """Configuration for individual metrics"""
    name: str
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0


@dataclass  
class DatasetConfig:
    """Configuration for datasets"""
    name: str
    split: str = "test"
    sample_size: int = 100
    random_seed: int = 42
    cache_dir: str = "~/.cache/merit_datasets"
    
    def __post_init__(self):
        self.cache_dir = os.path.expanduser(self.cache_dir)


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    name: str
    description: str = ""
    models: List[ModelConfig] = field(default_factory=list)
    datasets: List[DatasetConfig] = field(default_factory=list)
    metrics: List[MetricConfig] = field(default_factory=list)
    num_runs: int = 3
    output_dir: str = "experiments"
    save_individual_results: bool = True
    random_seed: int = 42
    
    def __post_init__(self):
        self.output_dir = os.path.expanduser(self.output_dir)


@dataclass
class ValidationConfig:
    """Configuration for validation settings"""
    human_validation: bool = False
    baseline_comparison: bool = True
    statistical_tests: bool = True
    confidence_level: float = 0.95
    min_sample_size: int = 30


@dataclass
class KnowledgeBaseConfig:
    """Configuration for knowledge base settings"""
    db_path: str = "data/facts.db"
    cache_dir: str = "data/wikipedia_cache"
    cache_only: bool = True  # Default: no live Wikipedia API calls
    cache_duration_days: int = 7

    def __post_init__(self):
        self.db_path = os.path.expanduser(self.db_path)
        self.cache_dir = os.path.expanduser(self.cache_dir)


@dataclass
class MeritConfig:
    """Main MERIT configuration"""
    # General settings
    version: str = "2.0.0"
    log_level: str = "INFO"
    cache_dir: str = "~/.cache/merit"
    data_dir: str = "data"

    # Language setting (currently only English supported)
    # NOTE: MERIT metrics use English NLP models (spaCy en_core_web_sm, etc.)
    # Multilingual support would require different models
    language: str = "en"

    # Knowledge base configuration
    knowledge_base: KnowledgeBaseConfig = field(default_factory=KnowledgeBaseConfig)
    
    # Experiment configuration
    experiment: ExperimentConfig = field(default_factory=lambda: ExperimentConfig(
        name="default_experiment",
        description="Default MERIT experiment"
    ))
    
    # Validation configuration
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    
    # System configuration
    system: Dict[str, Any] = field(default_factory=lambda: {
        "parallel_processing": True,
        "max_workers": 4,
        "memory_limit_gb": 8,
        "timeout_seconds": 300
    })
    
    def __post_init__(self):
        self.cache_dir = os.path.expanduser(self.cache_dir)
        self.data_dir = os.path.expanduser(self.data_dir)


class ConfigurationManager:
    """Manages MERIT configuration from multiple sources"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = None
        self.config_sources = []
        
        # Default configuration locations
        self.default_config_paths = [
            "merit_config.yaml",
            "merit_config.json", 
            "config/merit.yaml",
            "config/merit.json",
            os.path.expanduser("~/.merit/config.yaml"),
            os.path.expanduser("~/.merit/config.json")
        ]
        
        # Load configuration
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from multiple sources"""
        
        # Start with default configuration
        self.config = MeritConfig()
        self.config_sources.append("default")
        
        # Override with file configuration
        config_file = self._find_config_file()
        if config_file:
            file_config = self._load_config_file(config_file)
            if file_config:
                self.config = self._merge_configs(self.config, file_config)
                self.config_sources.append(f"file:{config_file}")
        
        # Override with environment variables
        env_config = self._load_env_variables()
        if env_config:
            self.config = self._merge_configs(self.config, env_config)
            self.config_sources.append("environment")
        
        print(f"Configuration loaded from: {', '.join(self.config_sources)}")
    
    def _find_config_file(self) -> Optional[str]:
        """Find configuration file"""
        
        # Check explicit path first
        if self.config_path:
            if os.path.exists(self.config_path):
                return self.config_path
            else:
                print(f"Warning: Specified config file not found: {self.config_path}")
        
        # Check default locations
        for path in self.default_config_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _load_config_file(self, config_path: str) -> Optional[Dict]:
        """Load configuration from file"""
        
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                elif config_path.endswith('.json'):
                    return json.load(f)
                else:
                    print(f"Warning: Unknown config file format: {config_path}")
                    return None
        
        except Exception as e:
            print(f"Error loading config file {config_path}: {e}")
            return None
    
    def _load_env_variables(self) -> Dict:
        """Load configuration from environment variables"""
        
        env_config = {}
        
        # Map environment variables to config structure
        env_mappings = {
            "MERIT_LOG_LEVEL": ("log_level", str),
            "MERIT_CACHE_DIR": ("cache_dir", str),
            "MERIT_DATA_DIR": ("data_dir", str),
            "MERIT_EXPERIMENT_NAME": ("experiment.name", str),
            "MERIT_NUM_RUNS": ("experiment.num_runs", int),
            "MERIT_OUTPUT_DIR": ("experiment.output_dir", str),
            "MERIT_RANDOM_SEED": ("experiment.random_seed", int),
            "MERIT_TEMPERATURE": ("experiment.temperature", float),
            "MERIT_MAX_TOKENS": ("experiment.max_tokens", int),
            "MERIT_PARALLEL_PROCESSING": ("system.parallel_processing", bool),
            "MERIT_MAX_WORKERS": ("system.max_workers", int),
            "MERIT_MEMORY_LIMIT_GB": ("system.memory_limit_gb", int)
        }
        
        for env_var, (config_path, value_type) in env_mappings.items():
            if env_var in os.environ:
                try:
                    value = os.environ[env_var]
                    
                    # Convert value type
                    if value_type == bool:
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    elif value_type == int:
                        value = int(value)
                    elif value_type == float:
                        value = float(value)
                    
                    # Set nested value
                    self._set_nested_value(env_config, config_path, value)
                    
                except Exception as e:
                    print(f"Error parsing environment variable {env_var}: {e}")
        
        return env_config
    
    def _set_nested_value(self, config: Dict, path: str, value: Any):
        """Set nested dictionary value using dot notation"""
        
        keys = path.split('.')
        current = config
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set final value
        current[keys[-1]] = value
    
    def _merge_configs(self, base_config: MeritConfig, override_config: Dict) -> MeritConfig:
        """Merge configuration dictionaries"""
        
        # Convert base config to dict for easier manipulation
        base_dict = asdict(base_config)
        
        # Deep merge
        merged_dict = self._deep_merge(base_dict, override_config)
        
        # Convert back to MeritConfig
        try:
            # Handle nested dataclass conversion
            if 'experiment' in merged_dict:
                exp_dict = merged_dict['experiment']
                
                # Convert model configs
                if 'models' in exp_dict:
                    exp_dict['models'] = [
                        ModelConfig(**model) if isinstance(model, dict) else model
                        for model in exp_dict['models']
                    ]
                
                # Convert dataset configs
                if 'datasets' in exp_dict:
                    exp_dict['datasets'] = [
                        DatasetConfig(**dataset) if isinstance(dataset, dict) else dataset
                        for dataset in exp_dict['datasets']
                    ]
                
                # Convert metric configs
                if 'metrics' in exp_dict:
                    exp_dict['metrics'] = [
                        MetricConfig(**metric) if isinstance(metric, dict) else metric
                        for metric in exp_dict['metrics']
                    ]
                
                merged_dict['experiment'] = ExperimentConfig(**exp_dict)
            
            if 'validation' in merged_dict:
                merged_dict['validation'] = ValidationConfig(**merged_dict['validation'])
            
            return MeritConfig(**merged_dict)
            
        except Exception as e:
            print(f"Error merging configurations: {e}")
            return base_config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_config(self) -> MeritConfig:
        """Get the current configuration"""
        return self.config
    
    def save_config(self, output_path: str, format: str = "yaml"):
        """Save current configuration to file"""
        
        config_dict = asdict(self.config)
        
        try:
            with open(output_path, 'w') as f:
                if format.lower() == "yaml":
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif format.lower() == "json":
                    json.dump(config_dict, f, indent=2, default=str)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            print(f"Configuration saved to: {output_path}")
            
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        
        issues = []
        
        # Validate experiment configuration
        if not self.config.experiment.name:
            issues.append("Experiment name is required")
        
        if not self.config.experiment.models:
            issues.append("At least one model must be configured")
        
        if not self.config.experiment.datasets:
            issues.append("At least one dataset must be configured")
        
        if not self.config.experiment.metrics:
            issues.append("At least one metric must be configured")
        
        if self.config.experiment.num_runs < 1:
            issues.append("Number of runs must be at least 1")
        
        # Validate model configurations
        for i, model in enumerate(self.config.experiment.models):
            if not model.name:
                issues.append(f"Model {i} is missing a name")
            
            if model.adapter_type not in ["local", "api", "huggingface"]:
                issues.append(f"Model {i} has invalid adapter type: {model.adapter_type}")
        
        # Validate dataset configurations
        for i, dataset in enumerate(self.config.experiment.datasets):
            if not dataset.name:
                issues.append(f"Dataset {i} is missing a name")
            
            if dataset.sample_size < 1:
                issues.append(f"Dataset {i} sample size must be at least 1")
        
        # Validate metric configurations
        for i, metric in enumerate(self.config.experiment.metrics):
            if not metric.name:
                issues.append(f"Metric {i} is missing a name")
            
            if metric.weight < 0:
                issues.append(f"Metric {i} weight must be non-negative")
        
        # Validate system configuration
        if self.config.system.get("max_workers", 1) < 1:
            issues.append("max_workers must be at least 1")
        
        if self.config.system.get("memory_limit_gb", 1) < 1:
            issues.append("memory_limit_gb must be at least 1")
        
        return issues
    
    def print_config_summary(self):
        """Print a summary of the current configuration"""
        
        print("\nMERIT Configuration Summary")
        print("=" * 30)
        print(f"Version: {self.config.version}")
        print(f"Log Level: {self.config.log_level}")
        print(f"Cache Directory: {self.config.cache_dir}")
        print(f"Data Directory: {self.config.data_dir}")
        print()
        
        print("Experiment Configuration:")
        print(f"  Name: {self.config.experiment.name}")
        print(f"  Description: {self.config.experiment.description}")
        print(f"  Models: {len(self.config.experiment.models)}")
        print(f"  Datasets: {len(self.config.experiment.datasets)}")
        print(f"  Metrics: {len(self.config.experiment.metrics)}")
        print(f"  Number of runs: {self.config.experiment.num_runs}")
        print(f"  Output directory: {self.config.experiment.output_dir}")
        print()
        
        print("Validation Configuration:")
        print(f"  Human validation: {self.config.validation.human_validation}")
        print(f"  Baseline comparison: {self.config.validation.baseline_comparison}")
        print(f"  Statistical tests: {self.config.validation.statistical_tests}")
        print()
        
        print("System Configuration:")
        for key, value in self.config.system.items():
            print(f"  {key}: {value}")
        print()
        
        print(f"Configuration sources: {', '.join(self.config_sources)}")


def create_default_config_file(output_path: str = "merit_config.yaml"):
    """Create a default configuration file"""
    
    # Create default configuration with examples
    config = MeritConfig(
        experiment=ExperimentConfig(
            name="example_experiment",
            description="Example MERIT evaluation experiment",
            models=[
                ModelConfig(
                    name="gpt2-medium",
                    adapter_type="local",
                    device="auto",
                    max_tokens=500,
                    temperature=0.1
                ),
                ModelConfig(
                    name="tinyllama-1b",
                    adapter_type="local", 
                    device="auto",
                    max_tokens=500,
                    temperature=0.1
                )
            ],
            datasets=[
                DatasetConfig(
                    name="arc",
                    split="test",
                    sample_size=100,
                    random_seed=42
                ),
                DatasetConfig(
                    name="hellaswag",
                    split="validation",
                    sample_size=100,
                    random_seed=42
                )
            ],
            metrics=[
                MetricConfig(
                    name="logical_consistency",
                    enabled=True,
                    weight=1.0,
                    parameters={"similarity_threshold": 0.8}
                ),
                MetricConfig(
                    name="factual_accuracy",
                    enabled=True,
                    weight=1.0,
                    parameters={"knowledge_base_path": "data/knowledge_base.json"}
                ),
                MetricConfig(
                    name="reasoning_steps",
                    enabled=True,
                    weight=1.0
                ),
                MetricConfig(
                    name="alignment",
                    enabled=True,
                    weight=1.0
                )
            ],
            num_runs=3,
            output_dir="experiments/example_experiment"
        ),
        validation=ValidationConfig(
            human_validation=False,
            baseline_comparison=True,
            statistical_tests=True,
            confidence_level=0.95
        )
    )
    
    # Save to file
    config_dict = asdict(config)
    
    try:
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        print(f"Default configuration file created: {output_path}")
        print("Edit this file to customize your MERIT evaluation settings.")
        
    except Exception as e:
        print(f"Error creating config file: {e}")


def load_config(config_path: Optional[str] = None) -> MeritConfig:
    """Load MERIT configuration"""
    
    manager = ConfigurationManager(config_path)
    return manager.get_config()


def validate_config_file(config_path: str) -> bool:
    """Validate a configuration file"""
    
    try:
        manager = ConfigurationManager(config_path)
        issues = manager.validate_config()
        
        if issues:
            print(f"Configuration validation failed with {len(issues)} issues:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
            return False
        else:
            print("Configuration validation passed!")
            return True
            
    except Exception as e:
        print(f"Error validating configuration: {e}")
        return False