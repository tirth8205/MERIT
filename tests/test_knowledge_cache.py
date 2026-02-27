"""Tests for knowledge cache."""
import pytest
import json
from pathlib import Path
from merit.core.knowledge_cache import KnowledgeCache


class TestKnowledgeCache:
    @pytest.fixture
    def tmp_cache_path(self, tmp_path):
        return str(tmp_path / "test_cache.json")

    def test_empty_cache(self, tmp_cache_path):
        cache = KnowledgeCache(tmp_cache_path)
        assert len(cache) == 0
        assert cache.get("test") is None

    def test_put_and_get(self, tmp_cache_path):
        cache = KnowledgeCache(tmp_cache_path)
        cache.put("Paris is in France", {"verified": True, "confidence": 0.9})
        result = cache.get("Paris is in France")
        assert result["verified"] is True
        assert result["confidence"] == 0.9

    def test_contains(self, tmp_cache_path):
        cache = KnowledgeCache(tmp_cache_path)
        cache.put("test claim", {"verified": True})
        assert "test claim" in cache
        assert "other claim" not in cache

    def test_save_and_reload(self, tmp_cache_path):
        cache = KnowledgeCache(tmp_cache_path)
        cache.put("claim1", {"verified": True, "confidence": 0.85})
        cache.put("claim2", {"verified": False, "confidence": 0.3})
        cache.save()

        # Reload from disk
        cache2 = KnowledgeCache(tmp_cache_path)
        assert len(cache2) == 2
        assert cache2.get("claim1")["confidence"] == 0.85

    def test_load_existing_cache(self, tmp_path):
        cache_path = str(tmp_path / "existing.json")
        # Pre-populate a cache file
        existing = {"pre-cached": {"verified": True, "confidence": 1.0}}
        with open(cache_path, "w") as f:
            json.dump(existing, f)

        cache = KnowledgeCache(cache_path)
        assert len(cache) == 1
        assert cache.get("pre-cached")["confidence"] == 1.0

    def test_nonexistent_path(self, tmp_path):
        cache_path = str(tmp_path / "deep" / "nested" / "cache.json")
        cache = KnowledgeCache(cache_path)
        assert len(cache) == 0
        cache.put("test", {"data": "value"})
        cache.save()
        assert Path(cache_path).exists()
