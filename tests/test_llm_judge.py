"""Tests for LLM-as-judge metrics."""
import pytest
from unittest.mock import patch, MagicMock
from merit.core.llm_judge import LLMJudge, JudgeConfig, RUBRICS
from merit.core.base import MetricResult


class TestJudgeConfig:
    def test_default_config(self):
        config = JudgeConfig()
        assert config.judge_model == "claude-sonnet-4-20250514"
        assert config.temperature == 0.0
        assert config.max_tokens == 500
        assert config.provider == "anthropic"

    def test_custom_config(self):
        config = JudgeConfig(judge_model="claude-haiku-4-5-20251001", temperature=0.1, provider="ollama")
        assert config.judge_model == "claude-haiku-4-5-20251001"
        assert config.provider == "ollama"


class TestLLMJudge:
    @pytest.fixture
    def judge(self):
        return LLMJudge(JudgeConfig())

    def test_evaluate_consistency(self, judge):
        with patch.object(judge, '_call_judge', return_value={"score": 4, "explanation": "Consistent", "contradictions": []}):
            result = judge.evaluate_consistency("The sky is blue. Water reflects the sky.")
            assert isinstance(result, MetricResult)
            assert result.dimension == "consistency"
            assert result.score == 0.75  # (4-1)/4

    def test_evaluate_factual(self, judge):
        with patch.object(judge, '_call_judge', return_value={"score": 5, "explanation": "Accurate", "errors": []}):
            result = judge.evaluate_factual("Paris is the capital of France.", reference="France's capital is Paris.")
            assert isinstance(result, MetricResult)
            assert result.dimension == "factual"
            assert result.score == 1.0

    def test_evaluate_reasoning(self, judge):
        with patch.object(judge, '_call_judge', return_value={"score": 3, "explanation": "Adequate", "steps_identified": 2, "gaps": ["missing conclusion"]}):
            result = judge.evaluate_reasoning("Step 1: X. Step 2: Y.")
            assert isinstance(result, MetricResult)
            assert result.dimension == "reasoning"
            assert result.score == 0.5

    def test_evaluate_alignment(self, judge):
        with patch.object(judge, '_call_judge', return_value={"score": 4, "explanation": "Respectful", "concerns": []}):
            result = judge.evaluate_alignment("Here is a balanced perspective.")
            assert isinstance(result, MetricResult)
            assert result.dimension == "alignment"

    def test_evaluate_all(self, judge):
        mock_results = {"score": 4, "explanation": "Good", "contradictions": [], "errors": [], "gaps": [], "concerns": []}
        with patch.object(judge, '_call_judge', return_value=mock_results):
            results = judge.evaluate_all("Test response", reference="Test reference")
            assert len(results) == 4
            assert all(isinstance(r, MetricResult) for r in results.values())
            assert set(results.keys()) == {"consistency", "factual", "reasoning", "alignment"}

    def test_score_clamping(self, judge):
        with patch.object(judge, '_call_judge', return_value={"score": 1, "explanation": "Bad"}):
            result = judge.evaluate_consistency("Bad text")
            assert result.score == 0.0  # (1-1)/4 = 0

    def test_score_max(self, judge):
        with patch.object(judge, '_call_judge', return_value={"score": 5, "explanation": "Perfect"}):
            result = judge.evaluate_consistency("Perfect text")
            assert result.score == 1.0  # (5-1)/4 = 1

    def test_parse_failure_fallback(self, judge):
        with patch.object(judge, '_call_judge', return_value={"score": 3, "explanation": "Failed to parse judge response"}):
            result = judge.evaluate_consistency("Some text")
            assert result.score == 0.5  # Default score 3 -> (3-1)/4 = 0.5

    def test_invalid_provider(self):
        judge = LLMJudge(JudgeConfig(provider="invalid"))
        with pytest.raises(ValueError, match="Unknown provider"):
            judge._call_judge("test prompt")


class TestRubrics:
    def test_all_dimensions_have_rubrics(self):
        assert set(RUBRICS.keys()) == {"consistency", "factual", "reasoning", "alignment"}

    def test_rubrics_contain_placeholders(self):
        assert "{response}" in RUBRICS["consistency"]
        assert "{response}" in RUBRICS["factual"]
        assert "{reference}" in RUBRICS["factual"]
        assert "{response}" in RUBRICS["reasoning"]
        assert "{response}" in RUBRICS["alignment"]
