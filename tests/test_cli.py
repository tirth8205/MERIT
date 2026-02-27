"""
Tests for CLI functionality.

The CLI has been streamlined to: create_parser, main, cmd_evaluate,
cmd_list_models, cmd_test_model, cmd_compare.
"""
import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from merit.cli import (
    create_parser,
    main,
    cmd_evaluate,
    cmd_list_models,
    cmd_test_model,
    cmd_compare,
    cmd_annotate,
    cmd_report,
)


class TestCLIParser:
    """Test CLI argument parser"""

    def test_parser_creation(self):
        """Test creating argument parser"""
        parser = create_parser()

        assert parser is not None
        assert parser.prog == "merit"

        # Test help doesn't crash
        help_text = parser.format_help()
        assert "MERIT" in help_text

    def test_parser_version(self):
        """Test version argument"""
        parser = create_parser()

        # Test version argument exists
        with pytest.raises(SystemExit):
            parser.parse_args(["--version"])

    def test_parser_subcommands_evaluate(self):
        """Test evaluate subcommand parsing"""
        parser = create_parser()

        args = parser.parse_args(["evaluate", "--model", "gpt2"])
        assert hasattr(args, 'func')
        assert args.model == "gpt2"
        assert args.dataset == "arc"  # default
        assert args.sample_size == 50  # default
        assert args.mode == "heuristic"  # default

    def test_parser_evaluate_mode(self):
        """Test evaluate --mode argument"""
        parser = create_parser()

        for mode in ["heuristic", "judge", "both"]:
            args = parser.parse_args(["evaluate", "--model", "gpt2", "--mode", mode])
            assert args.mode == mode

    def test_parser_evaluate_new_datasets(self):
        """Test evaluate with GSM8K and BBH datasets"""
        parser = create_parser()

        for ds in ["gsm8k", "bbh"]:
            args = parser.parse_args(["evaluate", "--model", "gpt2", "--dataset", ds])
            assert args.dataset == ds

    def test_parser_subcommands_annotate(self):
        """Test annotate subcommand parsing"""
        parser = create_parser()

        args = parser.parse_args(["annotate", "--input", "results.json"])
        assert hasattr(args, 'func')
        assert getattr(args, 'input') == "results.json"
        assert args.samples == 50  # default

        args = parser.parse_args(["annotate", "-i", "results.json", "--samples", "100"])
        assert args.samples == 100

    def test_parser_subcommands_report(self):
        """Test report subcommand parsing"""
        parser = create_parser()

        args = parser.parse_args(["report", "--input", "results.json"])
        assert hasattr(args, 'func')
        assert getattr(args, 'input') == "results.json"
        assert getattr(args, 'format') == "latex"  # default
        assert args.output == "paper_outputs"  # default

        for fmt in ["latex", "csv", "json"]:
            args = parser.parse_args(["report", "-i", "r.json", "--format", fmt])
            assert getattr(args, 'format') == fmt

    def test_parser_subcommands_models_list(self):
        """Test models list subcommand parsing"""
        parser = create_parser()

        args = parser.parse_args(["models", "list"])
        assert hasattr(args, 'func')

    def test_parser_subcommands_models_test(self):
        """Test models test subcommand parsing"""
        parser = create_parser()

        args = parser.parse_args(["models", "test", "gpt2"])
        assert hasattr(args, 'func')
        assert args.model_name == "gpt2"

    def test_parser_subcommands_compare(self):
        """Test compare subcommand parsing"""
        parser = create_parser()

        args = parser.parse_args(["compare", "file1.json", "file2.json"])
        assert hasattr(args, 'func')
        assert args.result_files == ["file1.json", "file2.json"]


class TestEvaluateCommand:
    """Test evaluate command functionality"""

    def test_evaluate_command_basic(self):
        """Test basic evaluate command"""
        args = Mock()
        args.model = "gpt2"
        args.dataset = "arc"
        args.sample_size = 10
        args.mode = "heuristic"
        args.output = None

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
            with patch('merit.cli.ExperimentConfig') as mock_config_cls:
                cmd_evaluate(args)
                mock_runner.run_full_experiment.assert_called_once()

    def test_evaluate_command_with_output(self):
        """Test evaluate command with output file"""
        args = Mock()
        args.model = "gpt2"
        args.dataset = "arc"
        args.sample_size = 5
        args.mode = "heuristic"
        args.output = "results.json"

        mock_runner = Mock()
        mock_results = {"model_results": {"gpt2": {}}}
        mock_runner.run_full_experiment.return_value = mock_results

        with patch('merit.cli.ExperimentRunner', return_value=mock_runner), \
             patch('merit.cli.ExperimentConfig'), \
             patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__ = Mock(return_value=mock_file)
            mock_open.return_value.__exit__ = Mock(return_value=False)

            cmd_evaluate(args)

            # Verify file was opened for writing
            mock_open.assert_called_with("results.json", 'w')

    def test_evaluate_command_arguments(self):
        """Test evaluate command arguments parsing"""
        parser = create_parser()

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


class TestCompareCommand:
    """Test compare command functionality"""

    def test_compare_command(self):
        """Test compare command with result files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test result files
            file1 = os.path.join(temp_dir, "result1.json")
            file2 = os.path.join(temp_dir, "result2.json")
            output_file = os.path.join(temp_dir, "comparison.txt")

            with open(file1, 'w') as f:
                json.dump({
                    "model_results": {
                        "gpt2": {
                            "benchmarks": {
                                "arc": {
                                    "sample_sizes": {
                                        "50": {
                                            "statistics": {
                                                "task_accuracy": {"mean": 0.8}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }, f)

            with open(file2, 'w') as f:
                json.dump({
                    "model_results": {
                        "tinyllama": {
                            "benchmarks": {
                                "arc": {
                                    "sample_sizes": {
                                        "50": {
                                            "statistics": {
                                                "task_accuracy": {"mean": 0.6}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }, f)

            args = Mock()
            args.result_files = [file1, file2]
            args.output = output_file

            cmd_compare(args)

            assert os.path.exists(output_file)
            with open(output_file) as f:
                content = f.read()
            assert "Comparison Report" in content

    def test_compare_command_arguments(self):
        """Test compare command argument parsing"""
        parser = create_parser()

        args = parser.parse_args(["compare", "a.json", "b.json"])
        assert args.result_files == ["a.json", "b.json"]
        assert args.output == "comparison.txt"  # default

        args = parser.parse_args(["compare", "a.json", "-o", "custom.txt"])
        assert args.output == "custom.txt"


class TestCLIIntegration:
    """Test CLI integration functionality"""

    def test_main_function_with_args(self):
        """Test main function with valid arguments"""
        with patch('merit.cli.create_parser') as mock_parser:
            mock_args = Mock()
            mock_args.func = Mock()

            parser_instance = Mock()
            parser_instance.parse_args.return_value = mock_args
            mock_parser.return_value = parser_instance

            main()

            mock_args.func.assert_called_once_with(mock_args)

    def test_main_function_with_keyboard_interrupt(self):
        """Test main function handles keyboard interrupt"""
        with patch('merit.cli.create_parser') as mock_parser:
            mock_args = Mock()
            mock_args.func = Mock(side_effect=KeyboardInterrupt())

            parser_instance = Mock()
            parser_instance.parse_args.return_value = mock_args
            mock_parser.return_value = parser_instance

            with patch('sys.exit') as mock_exit:
                main()
                mock_exit.assert_called_once_with(1)

    def test_main_function_with_exception(self):
        """Test main function handles general exceptions"""
        with patch('merit.cli.create_parser') as mock_parser:
            mock_args = Mock()
            mock_args.func = Mock(side_effect=Exception("Test error"))

            parser_instance = Mock()
            parser_instance.parse_args.return_value = mock_args
            mock_parser.return_value = parser_instance

            with patch('sys.exit') as mock_exit:
                main()
                mock_exit.assert_called_once_with(1)

    def test_main_function_no_subcommand(self):
        """Test main function when no subcommand provided"""
        with patch('merit.cli.create_parser') as mock_parser:
            mock_args = Mock(spec=[])  # No 'func' attribute

            parser_instance = Mock()
            parser_instance.parse_args.return_value = mock_args
            mock_parser.return_value = parser_instance

            main()

            parser_instance.print_help.assert_called_once()


@pytest.mark.parametrize("command,args", [
    ("evaluate", ["--model", "gpt2"]),
    ("models", ["list"]),
    ("compare", ["file.json"]),
    ("annotate", ["--input", "results.json"]),
    ("report", ["--input", "results.json"]),
])
def test_all_commands_have_func_attribute(command, args):
    """Test that all commands have func attribute set"""
    parser = create_parser()
    full_args = [command] + args

    parsed_args = parser.parse_args(full_args)
    assert hasattr(parsed_args, 'func'), f"Command {command} missing func attribute"


def test_cli_error_handling():
    """Test CLI error handling for various scenarios"""
    parser = create_parser()

    # Test with invalid subcommand
    with pytest.raises(SystemExit):
        parser.parse_args(["invalid_command"])

    # Test missing required arguments
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate"])  # Missing --model


if __name__ == "__main__":
    pytest.main([__file__])
