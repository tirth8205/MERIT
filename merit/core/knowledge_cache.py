"""Cached knowledge base for reproducible fact verification."""
import json
from pathlib import Path
from typing import Optional, Dict, Any


class KnowledgeCache:
    """File-backed cache for fact verification results.

    Used to make factual accuracy metric reproducible. When fact-checking
    runs in "live" mode, API responses are cached. In "reproducible" mode,
    only cached responses are used (no API calls).
    """

    def __init__(self, cache_path: str = "data/knowledge_cache.json"):
        self.cache_path = Path(cache_path)
        self._cache: Dict[str, Dict[str, Any]] = {}
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                self._cache = json.load(f)

    def get(self, claim: str) -> Optional[Dict[str, Any]]:
        """Get cached verification result for a claim."""
        return self._cache.get(claim)

    def put(self, claim: str, result: Dict[str, Any]) -> None:
        """Store a verification result."""
        self._cache[claim] = result

    def save(self) -> None:
        """Persist cache to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, claim: str) -> bool:
        return claim in self._cache
