"""Tests for baseline comparison methods."""
import pytest
from unittest.mock import patch, MagicMock
from merit.baselines.bertscore import BERTScoreBaseline
from merit.baselines.geval import GEvalBaseline


class TestBERTScoreBaseline:
    def test_initialization(self):
        baseline = BERTScoreBaseline()
        assert baseline.name == "bert_score"
        assert baseline.model_type == "microsoft/deberta-xlarge-mnli"

    def test_custom_model_type(self):
        baseline = BERTScoreBaseline(model_type="roberta-large")
        assert baseline.model_type == "roberta-large"

    def test_evaluate_with_mocked_bertscore(self):
        baseline = BERTScoreBaseline()
        import torch
        mock_P = [torch.tensor(0.92)]
        mock_R = [torch.tensor(0.88)]
        mock_F1 = [torch.tensor(0.90)]
        with patch('merit.baselines.bertscore.BERT_SCORE_AVAILABLE', True), \
             patch('merit.baselines.bertscore.bert_score_fn', create=True, return_value=(mock_P, mock_R, mock_F1)):
            result = baseline.evaluate("hello world", "hello world")
            assert "precision" in result
            assert "recall" in result
            assert "f1" in result
            assert abs(result["precision"] - 0.92) < 0.01
            assert abs(result["recall"] - 0.88) < 0.01
            assert abs(result["f1"] - 0.90) < 0.01

    def test_evaluate_without_bertscore(self):
        baseline = BERTScoreBaseline()
        with patch('merit.baselines.bertscore.BERT_SCORE_AVAILABLE', False):
            result = baseline.evaluate("hello", "world")
            assert result["f1"] == 0.0
            assert result["precision"] == 0.0
            assert result["recall"] == 0.0
            assert "error" in result

    def test_batch_evaluate_with_mocked_bertscore(self):
        baseline = BERTScoreBaseline()
        import torch
        mock_P = [torch.tensor(0.9), torch.tensor(0.8)]
        mock_R = [torch.tensor(0.85), torch.tensor(0.75)]
        mock_F1 = [torch.tensor(0.87), torch.tensor(0.77)]
        with patch('merit.baselines.bertscore.BERT_SCORE_AVAILABLE', True), \
             patch('merit.baselines.bertscore.bert_score_fn', create=True, return_value=(mock_P, mock_R, mock_F1)):
            result = baseline.batch_evaluate(["a", "b"], ["c", "d"])
            assert len(result["f1"]) == 2
            assert len(result["precision"]) == 2
            assert len(result["recall"]) == 2
            assert abs(result["f1"][0] - 0.87) < 0.01
            assert abs(result["f1"][1] - 0.77) < 0.01

    def test_batch_evaluate_without_bertscore(self):
        baseline = BERTScoreBaseline()
        with patch('merit.baselines.bertscore.BERT_SCORE_AVAILABLE', False):
            result = baseline.batch_evaluate(["a", "b"], ["c", "d"])
            assert result["precision"] == []
            assert result["recall"] == []
            assert result["f1"] == []
            assert "error" in result

    def test_device_detection(self):
        baseline = BERTScoreBaseline()
        assert baseline.device in ("cpu", "cuda", "mps")


class TestGEvalBaseline:
    def test_initialization(self):
        geval = GEvalBaseline()
        assert geval.name == "geval"
        assert geval.provider == "anthropic"
        assert geval.model == "claude-sonnet-4-20250514"

    def test_custom_config(self):
        geval = GEvalBaseline(provider="ollama", model="llama3")
        assert geval.provider == "ollama"
        assert geval.model == "llama3"

    def test_evaluate_with_mock(self):
        geval = GEvalBaseline()
        with patch.object(geval, '_call_model', return_value="Analysis... Score: 4"):
            result = geval.evaluate("The answer is 42.", "42 is the answer.")
            assert "score" in result
            assert "raw_score" in result
            assert "explanation" in result
            assert result["score"] == 0.75  # (4-1)/4
            assert result["raw_score"] == 4

    def test_evaluate_score_1(self):
        geval = GEvalBaseline()
        with patch.object(geval, '_call_model', return_value="Terrible. Score: 1"):
            result = geval.evaluate("wrong", "right")
            assert result["score"] == 0.0  # (1-1)/4
            assert result["raw_score"] == 1

    def test_evaluate_score_5(self):
        geval = GEvalBaseline()
        with patch.object(geval, '_call_model', return_value="Perfect match. Score: 5"):
            result = geval.evaluate("perfect", "perfect")
            assert result["score"] == 1.0  # (5-1)/4
            assert result["raw_score"] == 5

    def test_evaluate_parse_failure(self):
        geval = GEvalBaseline()
        with patch.object(geval, '_call_model', return_value="No score here"):
            result = geval.evaluate("test", "test")
            assert result["raw_score"] == 3  # Default fallback
            assert result["score"] == 0.5  # (3-1)/4

    def test_invalid_provider(self):
        geval = GEvalBaseline(provider="invalid")
        with pytest.raises(ValueError, match="Unknown provider"):
            geval._call_model("test")

    def test_explanation_included(self):
        geval = GEvalBaseline()
        explanation_text = "Step 1: The response captures key info. Score: 4"
        with patch.object(geval, '_call_model', return_value=explanation_text):
            result = geval.evaluate("test", "test")
            assert result["explanation"] == explanation_text


class TestBaselineImports:
    """Test that the baselines package exports work correctly."""

    def test_import_from_package(self):
        from merit.baselines import BERTScoreBaseline, GEvalBaseline
        assert BERTScoreBaseline is not None
        assert GEvalBaseline is not None

    def test_all_exports(self):
        import merit.baselines
        assert "BERTScoreBaseline" in merit.baselines.__all__
        assert "GEvalBaseline" in merit.baselines.__all__
