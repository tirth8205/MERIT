"""Base model adapter abstraction."""
from abc import ABC, abstractmethod
from typing import Optional


class BaseModelAdapter(ABC):
    """Abstract base for all model adapters."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def generate(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str: ...

    @abstractmethod
    def unload_model(self) -> None: ...
