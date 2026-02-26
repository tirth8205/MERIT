"""Base model adapter abstraction."""
from abc import ABC, abstractmethod


class BaseModelAdapter(ABC):
    """Abstract base for all model adapters."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable identifier for this model."""
        ...

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory (CPU/GPU). Idempotent."""
        ...

    @abstractmethod
    def generate(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """Generate text from the given prompt. Model must be loaded first."""
        ...

    @abstractmethod
    def unload_model(self) -> None:
        """Release model resources and free GPU/CPU memory."""
        ...
