"""Model-specific device management with memory info and model configuration.

Extends the core DeviceManager with model-loading utilities.
"""
from merit.core.device import DeviceManager
from typing import Dict


def get_memory_info() -> Dict:
    """Get available memory information with actual system detection."""
    import torch

    try:
        import psutil
        vm = psutil.virtual_memory()
        total_gb = vm.total / (1024**3)
        available_gb = vm.available / (1024**3)
        used_percent = vm.percent
    except ImportError:
        total_gb = 8.0
        available_gb = 4.0
        used_percent = 50.0

    if torch.backends.mps.is_available():
        mps_allocated = 0
        try:
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
        }
    elif torch.cuda.is_available():
        return {
            "device": "cuda",
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "allocated_memory_gb": torch.cuda.memory_allocated() / (1024**3),
            "system_total_gb": total_gb,
        }
    else:
        return {
            "device": "cpu",
            "available_memory_gb": available_gb,
            "total_memory_gb": total_gb,
            "memory_used_percent": used_percent,
        }


def get_model_config(model_size: str, device: str) -> dict:
    """Get optimal model configuration for the given device."""
    import torch

    base_config = {
        "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }

    if device == "mps":
        base_config.update({
            "device_map": "mps",
            "max_memory": {0: "6GiB"},
        })
    elif device == "cuda":
        base_config.update({
            "device_map": "auto",
            "max_memory": {0: "6GiB"},
        })
    else:
        base_config.update({
            "torch_dtype": torch.float32,
            "device_map": {"": "cpu"},
        })

    return base_config


__all__ = ["DeviceManager", "get_memory_info", "get_model_config"]
