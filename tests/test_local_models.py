"""
Tests for local model management system.
"""
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from merit.models.device import DeviceManager, get_system_recommendations
from merit.models.huggingface import LocalModelAdapter, TinyLlamaAdapter
from merit.models.manager import ModelManager
from merit.models.ollama import OllamaAdapter


class TestDeviceManager:
    """Test device management functionality"""

    def test_get_optimal_device_cpu_fallback(self):
        """Test CPU fallback when no GPU available"""
        with patch('torch.backends.mps.is_available', return_value=False), \
             patch('torch.cuda.is_available', return_value=False):
            device = DeviceManager.get_optimal_device()
            assert device == "cpu"

    def test_get_optimal_device_mps(self):
        """Test MPS device selection"""
        with patch('torch.backends.mps.is_available', return_value=True):
            device = DeviceManager.get_optimal_device()
            assert device == "mps"

    def test_get_optimal_device_cuda(self):
        """Test CUDA device selection"""
        with patch('torch.backends.mps.is_available', return_value=False), \
             patch('torch.cuda.is_available', return_value=True):
            device = DeviceManager.get_optimal_device()
            assert device == "cuda"

    def test_get_memory_info_cpu(self):
        """Test memory info for CPU"""
        mock_psutil = Mock()
        mock_vm = Mock()
        mock_vm.total = 16 * 1024**3
        mock_vm.available = 8 * 1024**3
        mock_vm.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_vm

        with patch('torch.backends.mps.is_available', return_value=False), \
             patch('torch.cuda.is_available', return_value=False), \
             patch.dict('sys.modules', {'psutil': mock_psutil}):

            memory_info = DeviceManager.get_memory_info()

            assert memory_info["device"] == "cpu"
            assert "available_memory_gb" in memory_info
            assert "total_memory_gb" in memory_info

    def test_get_memory_info_mps(self):
        """Test memory info for MPS"""
        mock_psutil = Mock()
        mock_vm = Mock()
        mock_vm.total = 16 * 1024**3
        mock_vm.available = 8 * 1024**3
        mock_vm.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_vm

        with patch('torch.backends.mps.is_available', return_value=True), \
             patch.dict('sys.modules', {'psutil': mock_psutil}):

            memory_info = DeviceManager.get_memory_info()

            assert memory_info["device"] == "mps"
            assert memory_info["unified_memory"] is True

    def test_check_memory_pressure_ok(self):
        """Test memory pressure check when memory is OK"""
        with patch.object(DeviceManager, 'get_memory_info') as mock_info:
            mock_info.return_value = {"memory_used_percent": 50}
            assert DeviceManager.check_memory_pressure() is True

    def test_check_memory_pressure_high(self):
        """Test memory pressure check when memory is high"""
        with patch.object(DeviceManager, 'get_memory_info') as mock_info:
            mock_info.return_value = {"memory_used_percent": 85}
            assert DeviceManager.check_memory_pressure() is False


class TestLocalModelAdapter:
    """Test local model adapter functionality"""

    def test_adapter_initialization(self):
        """Test adapter initialization with model_name"""
        with patch('merit.models.huggingface.TRANSFORMERS_AVAILABLE', True), \
             patch.object(DeviceManager, 'get_optimal_device', return_value='cpu'):

            with tempfile.TemporaryDirectory() as cache_dir:
                adapter = TinyLlamaAdapter(cache_dir=cache_dir)

                assert adapter.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                assert adapter.device == "cpu"
                assert adapter.model is None  # Not loaded yet

    @patch('merit.models.huggingface.AutoTokenizer')
    @patch('merit.models.huggingface.AutoModelForCausalLM')
    def test_generate_text(self, mock_model_cls, mock_tokenizer_cls):
        """Test text generation"""
        with patch.object(DeviceManager, 'get_optimal_device', return_value='cpu'):
            # Setup mocks
            mock_tokenizer = Mock()
            mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
            mock_tokenizer.decode.return_value = "Test prompt Generated response"
            mock_tokenizer.eos_token = "</s>"
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.pad_token = None
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

            mock_model = Mock()
            mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
            mock_model_cls.from_pretrained.return_value = mock_model

            with tempfile.TemporaryDirectory() as cache_dir:
                adapter = TinyLlamaAdapter(cache_dir=cache_dir)
                adapter.load_model()

                result = adapter.generate(
                    prompt="Test prompt",
                    max_length=50,
                    temperature=0.7
                )

                assert isinstance(result, str)
                mock_model.generate.assert_called_once()

    def test_get_embeddings(self):
        """Test that LocalModelAdapter is abstract (no get_embeddings by default)"""
        # LocalModelAdapter.generate is abstract
        with patch('merit.models.huggingface.TRANSFORMERS_AVAILABLE', True), \
             patch.object(DeviceManager, 'get_optimal_device', return_value='cpu'):

            with tempfile.TemporaryDirectory() as cache_dir:
                adapter = TinyLlamaAdapter(cache_dir=cache_dir)
                # get_embeddings not implemented - should not exist
                assert not hasattr(adapter, 'get_embeddings') or \
                       getattr(adapter, 'get_embeddings', None) is None

    @patch('merit.models.huggingface.AutoTokenizer')
    @patch('merit.models.huggingface.AutoModelForCausalLM')
    def test_generate_with_error_handling(self, mock_model_cls, mock_tokenizer_cls):
        """Test generation with error handling"""
        with patch.object(DeviceManager, 'get_optimal_device', return_value='cpu'):
            mock_tokenizer = Mock()
            mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "</s>"
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

            mock_model = Mock()
            mock_model.generate.side_effect = RuntimeError("CUDA out of memory")
            mock_model_cls.from_pretrained.return_value = mock_model

            with tempfile.TemporaryDirectory() as cache_dir:
                adapter = TinyLlamaAdapter(cache_dir=cache_dir)
                adapter.load_model()

                result = adapter.generate("Test prompt")

                # Should return error message
                assert "Error:" in result


class TestModelManager:
    """Test model manager functionality"""

    def test_model_manager_initialization(self):
        """Test model manager initialization"""
        with tempfile.TemporaryDirectory() as cache_dir:
            manager = ModelManager(cache_dir=cache_dir)

            assert manager.loaded_models == {}
            assert manager.cache_dir == cache_dir

    def test_list_available_models(self):
        """Test listing available models"""
        manager = ModelManager()
        models = manager.list_available_models()

        assert isinstance(models, dict)
        # Check model keys match our available models
        assert "tinyllama-1b" in models

        # Check model info structure for any model
        if models:
            first_model = list(models.values())[0]
            assert "parameters" in first_model
            assert "type" in first_model
            assert "memory_requirement" in first_model

    @patch('merit.models.huggingface.AutoTokenizer')
    @patch('merit.models.huggingface.AutoModelForCausalLM')
    def test_load_model_success(self, mock_model_cls, mock_tokenizer_cls):
        """Test successful model loading"""
        with patch.object(DeviceManager, 'get_optimal_device', return_value='cpu'):
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "</s>"
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

            with tempfile.TemporaryDirectory() as cache_dir:
                manager = ModelManager(cache_dir=cache_dir)
                adapter = manager.load_model("tinyllama-1b")

                assert adapter is not None
                assert "tinyllama-1b" in manager.loaded_models

    def test_load_model_invalid(self):
        """Test loading invalid model name"""
        manager = ModelManager()

        with pytest.raises(ValueError):
            manager.load_model("nonexistent_model_xyz")

    def test_unload_model(self):
        """Test model unloading"""
        manager = ModelManager()

        # Add a mock adapter to loaded models
        mock_adapter = Mock()
        manager.loaded_models["test_model"] = mock_adapter

        manager.unload_model("test_model")

        assert "test_model" not in manager.loaded_models
        mock_adapter.unload_model.assert_called_once()

    def test_unload_nonexistent_model(self):
        """Test unloading non-existent model (should not raise)"""
        manager = ModelManager()
        # Should not raise exception
        manager.unload_model("nonexistent_model")

    def test_unload_all_models(self):
        """Test unloading all models"""
        manager = ModelManager()

        # Add mock adapters
        manager.loaded_models["model1"] = Mock()
        manager.loaded_models["model2"] = Mock()

        manager.unload_all_models()

        assert len(manager.loaded_models) == 0

    @patch('merit.models.huggingface.AutoTokenizer')
    @patch('merit.models.huggingface.AutoModelForCausalLM')
    def test_benchmark_models(self, mock_model_cls, mock_tokenizer_cls):
        """Test model benchmarking"""
        with patch.object(DeviceManager, 'get_optimal_device', return_value='cpu'):
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "</s>"
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
            mock_tokenizer.decode.return_value = "Test response generated"
            mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

            with tempfile.TemporaryDirectory() as cache_dir:
                manager = ModelManager(cache_dir=cache_dir)

                results = manager.benchmark_models(
                    models=["tinyllama-1b"],
                    test_prompts=["Test prompt"],
                    max_length=50,
                    temperature=0.7
                )

                assert isinstance(results, dict)
                assert "performance_metrics" in results
                assert "tinyllama-1b" in results["performance_metrics"]


class TestSystemRecommendations:
    """Test system recommendation functionality"""

    def test_get_system_recommendations_high_memory(self):
        """Test recommendations for high-memory system"""
        with patch.object(DeviceManager, 'get_memory_info') as mock_info:
            mock_info.return_value = {
                "device": "mps",
                "unified_memory": True,
                "total_memory_gb": 32,
                "available_memory_gb": 24
            }

            recommendations = get_system_recommendations()

            assert isinstance(recommendations, dict)
            assert "device_info" in recommendations
            assert "recommended_models" in recommendations
            assert "performance_tips" in recommendations

    def test_get_system_recommendations_low_memory(self):
        """Test recommendations for low-memory system"""
        with patch.object(DeviceManager, 'get_memory_info') as mock_info:
            mock_info.return_value = {
                "device": "cpu",
                "available_memory_gb": 4,
                "total_memory_gb": 8,
                "memory_used_percent": 50
            }

            recommendations = get_system_recommendations()

            # Should recommend smaller models for low memory
            recommended = recommendations["recommended_models"]
            assert "gpt2-medium" in recommended or "tinyllama-1b" in recommended


class TestModelIntegration:
    """Test integration between model components"""

    @patch('merit.models.huggingface.AutoTokenizer')
    @patch('merit.models.huggingface.AutoModelForCausalLM')
    def test_full_model_workflow(self, mock_model_cls, mock_tokenizer_cls):
        """Test complete model workflow"""
        with patch.object(DeviceManager, 'get_optimal_device', return_value='cpu'):
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "</s>"
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
            mock_tokenizer.decode.return_value = "Test prompt This is a test response."
            mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

            with tempfile.TemporaryDirectory() as cache_dir:
                manager = ModelManager(cache_dir=cache_dir)

                # Load model
                adapter = manager.load_model("tinyllama-1b")
                assert adapter is not None

                # Generate text
                response = adapter.generate("What is AI?", max_length=50)
                assert isinstance(response, str)

                # Unload model
                manager.unload_model("tinyllama-1b")
                assert "tinyllama-1b" not in manager.loaded_models

    def test_device_compatibility_across_components(self):
        """Test device compatibility across all components"""
        devices = ["cpu", "mps", "cuda"]

        for device in devices:
            with patch.object(DeviceManager, 'get_optimal_device', return_value=device):
                optimal_device = DeviceManager.get_optimal_device()
                assert optimal_device == device

                # Manager should work with any device setting
                manager = ModelManager()
                assert manager is not None


@pytest.mark.parametrize("model_name", ["tinyllama-1b", "qwen2-0.5b"])
def test_model_availability(model_name):
    """Test that all supported models are properly configured"""
    manager = ModelManager()
    available_models = manager.list_available_models()

    assert model_name in available_models

    model_info = available_models[model_name]
    required_fields = ["parameters", "type", "memory_requirement", "license", "description"]

    for field in required_fields:
        assert field in model_info
        assert model_info[field] is not None


def test_error_handling_robustness():
    """Test robust error handling across model components"""
    manager = ModelManager()

    # Test with invalid model name
    with pytest.raises(ValueError):
        manager.load_model("invalid_model_name_12345")

    # Test unloading non-existent model (should not raise)
    manager.unload_model("non_existent_model")


if __name__ == "__main__":
    pytest.main([__file__])
