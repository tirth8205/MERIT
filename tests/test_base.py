"""Tests for base metric abstractions."""
import pytest
from merit.core.base import BaseMetric, MetricResult


class TestMetricResult:
    def test_creation(self):
        result = MetricResult(score=0.85, dimension="consistency", details={"contradictions": 0})
        assert result.score == 0.85
        assert result.dimension == "consistency"
        assert result.details == {"contradictions": 0}

    def test_score_clamped_high(self):
        result = MetricResult(score=1.5, dimension="test")
        assert result.score == 1.0

    def test_score_clamped_low(self):
        result = MetricResult(score=-0.5, dimension="test")
        assert result.score == 0.0

    def test_default_details(self):
        result = MetricResult(score=0.5, dimension="test")
        assert result.details == {}

    def test_to_dict(self):
        result = MetricResult(score=0.9, dimension="factual", details={"claims": 5})
        d = result.to_dict()
        assert d["score"] == 0.9
        assert d["dimension"] == "factual"
        assert d["details"]["claims"] == 5


class ConcreteMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "test_metric"

    @property
    def dimension(self) -> str:
        return "test"

    def compute(self, response, reference=None, **kwargs):
        return MetricResult(score=1.0, dimension=self.dimension)


class TestBaseMetric:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseMetric()

    def test_concrete_metric(self):
        m = ConcreteMetric()
        result = m.compute("hello")
        assert result.score == 1.0
        assert m.name == "test_metric"
        assert m.dimension == "test"


class TestBaseModelAdapter:
    def test_cannot_instantiate_abstract(self):
        from merit.models.base import BaseModelAdapter
        with pytest.raises(TypeError):
            BaseModelAdapter()
