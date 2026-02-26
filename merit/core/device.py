"""Device management for optimal hardware utilization."""
import torch


class DeviceManager:
    """Manages device selection for optimal performance."""

    @staticmethod
    def get_optimal_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def get_model_kwargs(device: str) -> dict:
        if device in ("mps", "cuda"):
            return {"device": device, "torch_dtype": torch.float16}
        return {"device": device, "torch_dtype": torch.float32}
