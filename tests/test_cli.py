"""
Tests for CLI functionality.
"""
import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from merit.cli import (
    create_main_parser,
    cmd_init,
    cmd_validate_config,
    cmd_show_config,
    cmd_run_experiment,
    cmd_list_models,
    cmd_test_model,
    cmd_evaluate,
    cmd_system_info
)


class TestCLIParser:
    """Test CLI argument parser"""
    
    def test_main_parser_creation(self):
        """Test creating main argument parser"""
        parser = create_main_parser()
        
        assert parser is not None
        assert parser.prog == "merit"
        
        # Test help doesn't crash
        help_text = parser.format_help()
        assert "MERIT" in help_text
        assert "commands" in help_text
    
    def test_parser_version(self):
        """Test version argument"""
        parser = create_main_parser()
        
        # Test version argument exists
        with pytest.raises(SystemExit):
            parser.parse_args(["--version"])
    
    def test_parser_verbose_flag(self):
        """Test verbose flag"""
        parser = create_main_parser()
        
        # Test without verbose
        args = parser.parse_args(["init"])
        assert args.verbose is False
        
        # Test with verbose
        args = parser.parse_args(["--verbose", "init"])
        assert args.verbose is True
    
    def test_parser_subcommands(self):
        """Test that all expected subcommands exist"""
        parser = create_main_parser()
        
        # Test init command
        args = parser.parse_args(["init"])
        assert hasattr(args, 'func')
        
        # Test validate-config command
        args = parser.parse_args(["validate-config", "test.yaml"])
        assert hasattr(args, 'func')
        assert args.config_file == "test.yaml"
        
        # Test run command
        args = parser.parse_args(["run", "config.yaml"])
        assert hasattr(args, 'func')
        assert args.config_file == "config.yaml"
        
        # Test models list command
        args = parser.parse_args(["models", "list"])
        assert hasattr(args, 'func')
        
        # Test evaluate command
        args = parser.parse_args(["evaluate", "--model", "gpt2"])
        assert hasattr(args, 'func')
        assert args.model == "gpt2"


class TestInitCommand:
    """Test init command functionality"""
    
    def test_init_command_basic(self):
        """Test basic init command"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "test_config.yaml")
            
            # Mock args
            args = Mock()
            args.config_file = config_file
            
            with patch('merit.cli.create_default_config_file') as mock_create:
                with patch('builtins.input', return_value='n'):  # Don't overwrite
                    cmd_init(args)
                    mock_create.assert_called_once_with(config_file)
    
    def test_init_command_overwrite_prompt(self):
        """Test init command with overwrite prompt"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "existing_config.yaml")
            
            # Create existing file
            with open(config_file, 'w') as f:
                f.write("existing: config")
            
            args = Mock()
            args.config_file = config_file
            
            # Test declining overwrite
            with patch('builtins.input', return_value='n'):
                with patch('merit.cli.create_default_config_file') as mock_create:
                    cmd_init(args)
                    mock_create.assert_not_called()
            
            # Test accepting overwrite
            with patch('builtins.input', return_value='y'):
                with patch('merit.cli.create_default_config_file') as mock_create:
                    cmd_init(args)
                    mock_create.assert_called_once_with(config_file)


class TestValidateConfigCommand:
    """Test validate-config command functionality"""
    
    def test_validate_config_success(self):
        """Test successful config validation"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_file = f.name
            f.write("test: config")
        
        try:
            args = Mock()
            args.config_file = config_file
            
            with patch('merit.cli.validate_config_file', return_value=True) as mock_validate:
                with patch('sys.exit') as mock_exit:
                    cmd_validate_config(args)
                    mock_validate.assert_called_once_with(config_file)
                    mock_exit.assert_not_called()
        finally:
            os.unlink(config_file)
    
    def test_validate_config_failure(self):
        """Test config validation failure"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_file = f.name
            f.write("invalid: yaml: content")
        
        try:
            args = Mock()
            args.config_file = config_file
            
            with patch('merit.cli.validate_config_file', return_value=False):
                with patch('sys.exit') as mock_exit:
                    cmd_validate_config(args)
                    mock_exit.assert_called_once_with(1)
        finally:
            os.unlink(config_file)
    
    def test_validate_config_nonexistent_file(self):
        """Test validation with non-existent file"""
        args = Mock()
        args.config_file = "/nonexistent/path/config.yaml"
        
        with patch('sys.exit') as mock_exit:
            cmd_validate_config(args)
            mock_exit.assert_called_once_with(1)


class TestShowConfigCommand:
    """Test show-config command functionality"""
    
    def test_show_config_success(self):
        """Test successful config display"""
        args = Mock()
        args.config = "test_config.yaml"
        
        mock_config = Mock()
        mock_config.experiment.name = "test_experiment"
        
        with patch('merit.cli.load_config', return_value=mock_config):
            with patch('merit.cli.ConfigurationManager') as mock_manager_class:
                mock_manager = Mock()
                mock_manager_class.return_value = mock_manager
                
                cmd_show_config(args)
                
                mock_manager.print_config_summary.assert_called_once()
    
    def test_show_config_error(self):
        """Test config display with error"""
        args = Mock()
        args.config = "invalid_config.yaml"
        
        with patch('merit.cli.load_config', side_effect=Exception("Config error")):
            with patch('sys.exit') as mock_exit:
                cmd_show_config(args)
                mock_exit.assert_called_once_with(1)


class TestRunExperimentCommand:
    """Test run experiment command functionality"""
    
    def test_run_experiment_dry_run(self):
        """Test dry run mode"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_file = f.name
            f.write("test: config")
        
        try:
            args = Mock()
            args.config_file = config_file
            args.dry_run = True
            args.verbose = False
            
            mock_config = Mock()
            mock_config.experiment.name = "test_experiment"
            mock_config.experiment.models = [Mock(name="gpt2")]
            mock_config.experiment.datasets = [Mock(name="arc")]
            mock_config.experiment.metrics = [Mock(name="logical_consistency")]
            mock_config.experiment.num_runs = 3
            
            with patch('merit.cli.load_config', return_value=mock_config):
                # Should not raise exception in dry run
                cmd_run_experiment(args)
        finally:
            os.unlink(config_file)
    
    def test_run_experiment_actual_run(self):
        """Test actual experiment run"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_file = f.name
            f.write("test: config")
        
        try:
            args = Mock()
            args.config_file = config_file
            args.dry_run = False
            args.verbose = False
            
            mock_config = Mock()
            mock_config.experiment.name = "test_experiment"
            mock_config.experiment.models = [Mock(name="gpt2", temperature=0.1, max_tokens=100)]
            mock_config.experiment.datasets = [Mock(name="arc", sample_size=10)]
            mock_config.experiment.metrics = [Mock(name="logical_consistency")]
            mock_config.experiment.num_runs = 1
            mock_config.experiment.random_seed = 42
            mock_config.experiment.output_dir = "test_output"
            
            mock_runner = Mock()
            mock_runner.run_full_experiment.return_value = {"results": "test"}
            
            with patch('merit.cli.load_config', return_value=mock_config):
                with patch('merit.cli.ExperimentRunner', return_value=mock_runner):
                    cmd_run_experiment(args)
                    mock_runner.run_full_experiment.assert_called_once()
        finally:
            os.unlink(config_file)


class TestModelCommands:
    """Test model-related commands"""
    
    def test_list_models_command(self):
        """Test list models command"""
        args = Mock()
        
        mock_manager = Mock()
        mock_manager.list_available_models.return_value = {
            "gpt2": {
                "parameters": "124M",
                "type": "causal_lm",
                "memory_requirement": "~500MB",
                "license": "MIT",
                "description": "Small GPT-2 model"
            }
        }
        
        with patch('merit.cli.ModelManager', return_value=mock_manager):
            cmd_list_models(args)
            mock_manager.list_available_models.assert_called_once()
    
    def test_test_model_command(self):
        """Test model testing command"""
        args = Mock()
        args.model_name = "gpt2"
        args.prompt = "What is AI?"
        
        mock_manager = Mock()
        mock_adapter = Mock()
        mock_adapter.generate.return_value = "AI is artificial intelligence."
        mock_manager.load_model.return_value = mock_adapter
        
        with patch('merit.cli.ModelManager', return_value=mock_manager):
            cmd_test_model(args)
            
            mock_manager.load_model.assert_called_once_with("gpt2")
            mock_adapter.generate.assert_called_once()
            mock_manager.unload_model.assert_called_once_with("gpt2")
    
    def test_test_model_command_error(self):
        """Test model testing with error"""
        args = Mock()
        args.model_name = "invalid_model"
        args.prompt = "Test prompt"
        
        mock_manager = Mock()
        mock_manager.load_model.side_effect = Exception("Model not found")
        
        with patch('merit.cli.ModelManager', return_value=mock_manager):
            with patch('sys.exit') as mock_exit:
                cmd_test_model(args)
                mock_exit.assert_called_once_with(1)


class TestEvaluateCommand:
    """Test evaluate command functionality"""
    
    def test_evaluate_command_basic(self):
        """Test basic evaluate command"""
        args = Mock()
        args.model = "gpt2"
        args.dataset = "arc"
        args.sample_size = 10
        args.output = None
        args.verbose = False
        
        mock_runner = Mock()
        mock_results = {
            "model_results": {
                "gpt2": {
                    "benchmarks": {
                        "arc": {
                            "sample_sizes": {
                                "10": {
                                    "statistics": {
                                        "task_accuracy": {"mean": 0.8},
                                        "metric_statistics": {
                                            "logical_consistency": {"mean_across_runs": 0.75}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        mock_runner.run_full_experiment.return_value = mock_results
        
        with patch('merit.cli.ExperimentRunner', return_value=mock_runner):
            cmd_evaluate(args)
            mock_runner.run_full_experiment.assert_called_once()
    
    def test_evaluate_command_with_output(self):
        """Test evaluate command with output file"""
        args = Mock()
        args.model = "gpt2"
        args.dataset = "arc"
        args.sample_size = 5
        args.output = "results.json"
        args.verbose = False
        
        mock_runner = Mock()
        mock_results = {"results": "test"}
        mock_runner.run_full_experiment.return_value = mock_results
        
        with patch('merit.cli.ExperimentRunner', return_value=mock_runner):
            with patch('builtins.open', create=True) as mock_open:
                mock_file = Mock()
                mock_open.return_value.__enter__.return_value = mock_file
                
                cmd_evaluate(args)
                
                # Verify file was opened for writing
                mock_open.assert_called_with("results.json", 'w')


class TestSystemInfoCommand:
    """Test system info command functionality"""
    
    def test_system_info_command(self):
        """Test system info command"""
        args = Mock()
        
        mock_recommendations = {
            "device_info": {
                "device": "mps",
                "unified_memory": True,
                "estimated_memory_gb": 16
            },
            "recommended_models": ["gpt2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
            "performance_tips": ["Use MPS for faster inference", "Consider batch processing"]
        }
        
        with patch('merit.cli.get_system_recommendations', return_value=mock_recommendations):
            with patch('torch.__version__', "2.0.0"):
                with patch('transformers.__version__', "4.30.0"):
                    cmd_system_info(args)
    
    def test_system_info_command_cpu_device(self):
        """Test system info command with CPU device"""
        args = Mock()
        
        mock_recommendations = {
            "device_info": {
                "device": "cpu",
                "available_memory": 8 * 1024**3,  # 8GB
                "total_memory": 16 * 1024**3      # 16GB
            },
            "recommended_models": ["gpt2"],
            "performance_tips": ["Consider smaller models for CPU inference"]
        }
        
        with patch('merit.cli.get_system_recommendations', return_value=mock_recommendations):
            cmd_system_info(args)
    
    def test_system_info_command_cuda_device(self):
        """Test system info command with CUDA device"""
        args = Mock()
        
        mock_recommendations = {
            "device_info": {
                "device": "cuda",
                "total_memory": 12 * 1024**3  # 12GB GPU
            },
            "recommended_models": ["gpt2", "microsoft/DialoGPT-medium"],
            "performance_tips": ["Use GPU memory efficiently"]
        }
        
        with patch('merit.cli.get_system_recommendations', return_value=mock_recommendations):
            cmd_system_info(args)


class TestCLIIntegration:
    """Test CLI integration functionality"""
    
    def test_main_function_with_args(self):
        """Test main function with valid arguments"""
        with patch('merit.cli.create_main_parser') as mock_parser:
            mock_args = Mock()
            mock_args.func = Mock()
            
            parser_instance = Mock()
            parser_instance.parse_args.return_value = mock_args
            mock_parser.return_value = parser_instance
            
            with patch('sys.argv', ['merit', 'init']):
                from merit.cli import main
                main()
                
                mock_args.func.assert_called_once_with(mock_args)
    
    def test_main_function_with_keyboard_interrupt(self):
        """Test main function handles keyboard interrupt"""
        with patch('merit.cli.create_main_parser') as mock_parser:
            mock_args = Mock()
            mock_args.func = Mock(side_effect=KeyboardInterrupt())
            
            parser_instance = Mock()
            parser_instance.parse_args.return_value = mock_args
            mock_parser.return_value = parser_instance
            
            with patch('sys.exit') as mock_exit:
                from merit.cli import main
                main()
                mock_exit.assert_called_once_with(1)
    
    def test_main_function_with_exception(self):
        """Test main function handles general exceptions"""
        with patch('merit.cli.create_main_parser') as mock_parser:
            mock_args = Mock()
            mock_args.func = Mock(side_effect=Exception("Test error"))
            
            parser_instance = Mock()
            parser_instance.parse_args.return_value = mock_args
            mock_parser.return_value = parser_instance
            
            with patch('sys.exit') as mock_exit:
                from merit.cli import main
                main()
                mock_exit.assert_called_once_with(1)
    
    def test_main_function_no_subcommand(self):
        """Test main function when no subcommand provided"""
        with patch('merit.cli.create_main_parser') as mock_parser:
            mock_args = Mock()
            delattr(mock_args, 'func')  # No func attribute
            
            parser_instance = Mock()
            parser_instance.parse_args.return_value = mock_args
            mock_parser.return_value = parser_instance
            
            from merit.cli import main
            main()
            
            parser_instance.print_help.assert_called_once()


class TestCLICommandArguments:
    """Test CLI command argument parsing"""
    
    def test_init_command_arguments(self):
        """Test init command arguments"""
        parser = create_main_parser()
        
        # Test default arguments
        args = parser.parse_args(["init"])
        assert args.config_file == "merit_config.yaml"
        assert args.format == "yaml"
        
        # Test custom arguments
        args = parser.parse_args(["init", "--config-file", "custom.yaml", "--format", "json"])
        assert args.config_file == "custom.yaml"
        assert args.format == "json"
    
    def test_evaluate_command_arguments(self):
        """Test evaluate command arguments"""
        parser = create_main_parser()
        
        # Test required arguments
        args = parser.parse_args(["evaluate", "--model", "gpt2"])
        assert args.model == "gpt2"
        assert args.dataset == "arc"  # default
        assert args.sample_size == 50  # default
        
        # Test custom arguments
        args = parser.parse_args([
            "evaluate", 
            "--model", "custom_model",
            "--dataset", "hellaswag",
            "--sample-size", "100",
            "--output", "results.json"
        ])
        assert args.model == "custom_model"
        assert args.dataset == "hellaswag"
        assert args.sample_size == 100
        assert args.output == "results.json"
    
    def test_run_command_arguments(self):
        """Test run command arguments"""
        parser = create_main_parser()
        
        # Test basic arguments
        args = parser.parse_args(["run", "config.yaml"])
        assert args.config_file == "config.yaml"
        assert args.dry_run is False
        assert args.resume is None
        
        # Test with flags
        args = parser.parse_args(["run", "config.yaml", "--dry-run", "--resume", "checkpoint.json"])
        assert args.dry_run is True
        assert args.resume == "checkpoint.json"


@pytest.mark.parametrize("command,args", [
    ("init", []),
    ("validate-config", ["test.yaml"]),
    ("show-config", []),
    ("run", ["config.yaml"]),
    ("models", ["list"]),
    ("evaluate", ["--model", "gpt2"]),
    ("system-info", [])
])
def test_all_commands_have_func_attribute(command, args):
    """Test that all commands have func attribute set"""
    parser = create_main_parser()
    full_args = [command] + args
    
    parsed_args = parser.parse_args(full_args)
    assert hasattr(parsed_args, 'func'), f"Command {command} missing func attribute"


def test_cli_error_handling():
    """Test CLI error handling for various scenarios"""
    # Test with invalid subcommand
    parser = create_main_parser()
    
    with pytest.raises(SystemExit):
        parser.parse_args(["invalid_command"])
    
    # Test missing required arguments
    with pytest.raises(SystemExit):
        parser.parse_args(["validate-config"])  # Missing config file
    
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate"])  # Missing --model


if __name__ == "__main__":
    pytest.main([__file__])