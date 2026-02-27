"""Tests for annotation pipeline."""
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from merit.evaluation.annotation import AnnotationPipeline, AnnotationConfig, ANNOTATION_PROMPT


class TestAnnotationConfig:
    def test_defaults(self):
        config = AnnotationConfig()
        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-4-20250514"
        assert config.temperature == 0.0
        assert config.batch_delay == 0.5

    def test_custom(self):
        config = AnnotationConfig(provider="ollama", model="llama3", batch_delay=0.0)
        assert config.provider == "ollama"


class TestAnnotationPipeline:
    @pytest.fixture
    def pipeline(self):
        return AnnotationPipeline(AnnotationConfig(batch_delay=0.0))

    def test_annotate_single(self, pipeline):
        mock_result = {
            "consistency": 4, "factual": 3, "reasoning": 5, "alignment": 4,
            "explanations": {"consistency": "ok", "factual": "ok", "reasoning": "ok", "alignment": "ok"}
        }
        with patch.object(pipeline, '_call_annotator', return_value=mock_result):
            result = pipeline.annotate("Sample response", reference="Reference answer")
            assert result["consistency"] == 4
            assert result["factual"] == 3
            assert result["reasoning"] == 5
            assert result["alignment"] == 4

    def test_annotate_batch(self, pipeline):
        mock_result = {
            "consistency": 4, "factual": 3, "reasoning": 5, "alignment": 4,
            "explanations": {}
        }
        with patch.object(pipeline, '_call_annotator', return_value=mock_result):
            results = pipeline.annotate_batch(
                responses=["resp1", "resp2", "resp3"],
                references=["ref1", "ref2", "ref3"],
            )
            assert len(results) == 3
            assert all(r["consistency"] == 4 for r in results)
            assert results[0]["index"] == 0
            assert results[2]["index"] == 2

    def test_annotate_batch_with_progress(self, pipeline):
        mock_result = {"consistency": 4, "factual": 3, "reasoning": 5, "alignment": 4, "explanations": {}}
        progress_calls = []
        with patch.object(pipeline, '_call_annotator', return_value=mock_result):
            pipeline.annotate_batch(
                responses=["a", "b"],
                progress_callback=lambda i, t: progress_calls.append((i, t)),
            )
        assert progress_calls == [(1, 2), (2, 2)]

    def test_save_and_load_annotations(self, pipeline, tmp_path):
        pipeline.config.output_path = str(tmp_path)
        annotations = [
            {"consistency": 4, "factual": 3, "index": 0},
            {"consistency": 5, "factual": 4, "index": 1},
        ]
        path = pipeline.save_annotations(annotations, "test.json")
        assert path.exists()

        loaded = pipeline.load_annotations("test.json")
        assert len(loaded) == 2
        assert loaded[0]["consistency"] == 4

    def test_invalid_provider(self):
        pipeline = AnnotationPipeline(AnnotationConfig(provider="invalid"))
        with pytest.raises(ValueError, match="Unknown provider"):
            pipeline._call_annotator("test")

    def test_parse_error_fallback(self, pipeline):
        """Test that unparseable responses return defaults."""
        with patch.object(pipeline, '_call_annotator', return_value={
            "consistency": 3, "factual": 3, "reasoning": 3, "alignment": 3,
            "explanations": {}, "parse_error": True
        }):
            result = pipeline.annotate("bad response")
            assert result["consistency"] == 3  # Default


class TestAnnotationPrompt:
    def test_prompt_has_placeholders(self):
        assert "{response}" in ANNOTATION_PROMPT
        assert "{reference}" in ANNOTATION_PROMPT
        assert "{context}" in ANNOTATION_PROMPT

    def test_prompt_mentions_all_dimensions(self):
        assert "Consistency" in ANNOTATION_PROMPT
        assert "Factual" in ANNOTATION_PROMPT
        assert "Reasoning" in ANNOTATION_PROMPT
        assert "Alignment" in ANNOTATION_PROMPT
