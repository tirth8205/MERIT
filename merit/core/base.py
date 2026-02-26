"""Base abstractions for MERIT metrics."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class MetricResult:
    """Standardized result from any MERIT metric."""
    score: float
    dimension: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.score = max(0.0, min(1.0, self.score))

    def to_dict(self) -> Dict[str, Any]:
        return {"score": self.score, "dimension": self.dimension, "details": self.details}


class BaseMetric(ABC):
    """Abstract base class for all MERIT metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this metric."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> str:
        """Which MERIT dimension this metric evaluates."""
        ...

    @abstractmethod
    def compute(self, response: str, reference: Optional[str] = None, **kwargs) -> MetricResult:
        """Evaluate a response and return a MetricResult."""
        ...
