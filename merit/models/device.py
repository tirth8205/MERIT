"""Models-specific device management with memory info and pressure monitoring."""
import torch
from typing import Dict


class DeviceManager:
    """Enhanced device management for optimal performance on different hardware"""

    @staticmethod
    def get_optimal_device():
        """Get the best available device for model inference"""
        if torch.backends.mps.is_available():
            print("Using MPS (Metal Performance Shaders) for Apple Silicon")
            return "mps"
        elif torch.cuda.is_available():
            print("Using CUDA GPU")
            return "cuda"
        else:
            print("Using CPU (no GPU acceleration available)")
            return "cpu"

    @staticmethod
    def get_memory_info():
        """Get available memory information with actual system detection"""
        try:
            import psutil
            vm = psutil.virtual_memory()
            total_gb = vm.total / (1024**3)
            available_gb = vm.available / (1024**3)
            used_percent = vm.percent
        except ImportError:
            # Fallback if psutil not available
            total_gb = 8.0
            available_gb = 4.0
            used_percent = 50.0

        if torch.backends.mps.is_available():
            # MPS uses unified memory - get actual system RAM
            mps_allocated = 0
            try:
                # Try to get MPS allocated memory (PyTorch 2.0+)
                mps_allocated = torch.mps.current_allocated_memory() / (1024**3)
            except (AttributeError, RuntimeError):
                pass

            return {
                "device": "mps",
                "total_memory_gb": total_gb,
                "available_memory_gb": available_gb,
                "mps_allocated_gb": mps_allocated,
                "memory_used_percent": used_percent,
                "unified_memory": True,
                "warning": "high_memory" if used_percent > 80 else None
            }
        elif torch.cuda.is_available():
            return {
                "device": "cuda",
                "total_memory": torch.cuda.get_device_properties(0).total_memory,
                "allocated_memory": torch.cuda.memory_allocated(),
                "cached_memory": torch.cuda.memory_reserved(),
                "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "system_total_gb": total_gb
            }
        else:
            return {
                "device": "cpu",
                "available_memory_gb": available_gb,
                "total_memory_gb": total_gb,
                "memory_used_percent": used_percent
            }

    @staticmethod
    def check_memory_pressure() -> bool:
        """Check if system is under memory pressure (>80% used). Returns True if OK."""
        info = DeviceManager.get_memory_info()
        used_percent = info.get("memory_used_percent", 50)
        if used_percent > 80:
            print(f"WARNING: High memory usage ({used_percent:.1f}%). Consider unloading models.")
            return False
        return True

    @staticmethod
    def get_model_config(model_size: str, device: str):
        """Get optimal model configuration for device"""
        base_config = {
            "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True
        }

        if device == "mps":
            # MPS-specific optimizations
            base_config.update({
                "device_map": "mps",
                "max_memory": {0: "6GiB"},  # Conservative for unified memory
            })
        elif device == "cuda":
            base_config.update({
                "device_map": "auto",
                "max_memory": {0: "6GiB"}
            })
        else:
            # CPU optimizations
            base_config.update({
                "torch_dtype": torch.float32,
                "device_map": {"": "cpu"}
            })

        return base_config


# Utility functions
def get_system_recommendations() -> Dict:
    """Get system-specific model recommendations"""
    device_info = DeviceManager.get_memory_info()

    recommendations = {
        "device_info": device_info,
        "recommended_models": [],
        "performance_tips": []
    }

    if device_info["device"] == "mps":
        # Apple Silicon recommendations
        recommendations["recommended_models"] = [
            "tinyllama-1b",  # Fastest, lowest memory
            "gpt2-medium",   # Good balance
            "mistral-7b-instruct"  # Best quality if memory allows
        ]
        recommendations["performance_tips"] = [
            "Use smaller batch sizes for better MPS performance",
            "Consider using float16 precision for memory efficiency",
            "Monitor memory usage with Activity Monitor"
        ]

    elif device_info["device"] == "cuda":
        # CUDA GPU recommendations
        total_memory_gb = device_info["total_memory"] / (1024**3)

        if total_memory_gb >= 12:
            recommendations["recommended_models"] = ["llama3-3b", "mistral-7b-instruct"]
        elif total_memory_gb >= 6:
            recommendations["recommended_models"] = ["tinyllama-1b", "gpt2-large"]
        else:
            recommendations["recommended_models"] = ["gpt2-medium"]

        recommendations["performance_tips"] = [
            "Use CUDA for maximum performance",
            "Enable gradient checkpointing for memory efficiency",
            "Consider using quantization for larger models"
        ]

    else:
        # CPU recommendations
        recommendations["recommended_models"] = ["gpt2-medium", "tinyllama-1b"]
        recommendations["performance_tips"] = [
            "Use CPU-optimized models",
            "Consider using quantization",
            "Reduce sequence lengths for faster inference"
        ]

    return recommendations
