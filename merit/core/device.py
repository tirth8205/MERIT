"""Device management for optimal hardware utilization.

Note: torch is imported lazily inside methods since every consumer of
DeviceManager already depends on torch, but we avoid top-level import
to keep the module lightweight when only type-checking.
"""


class DeviceManager:
    """Manages device selection for optimal performance."""

    @staticmethod
    def get_optimal_device() -> str:
        """Get the best available compute device (MPS > CUDA > CPU)."""
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def get_model_kwargs(device: str) -> dict:
        """Get model kwargs optimized for the given device."""
        import torch

        if device in ("mps", "cuda"):
            return {"device": device, "torch_dtype": torch.float16}
        return {"device": device, "torch_dtype": torch.float32}
