"""
Tests for knowledge base system.
"""
import pytest
import sqlite3
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from merit.knowledge.enhanced_kb import (
    EnhancedKnowledgeBase,
    FactDatabase,
    WikipediaKnowledgeBase
)


class TestFactDatabase:
    """Test structured fact database functionality"""

    def test_database_initialization(self):
        """Test database initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_facts.db")

            db = FactDatabase(db_path)

            # Database file should be created
            assert os.path.exists(db_path)

            # Test connection
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check that facts table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='facts'")
            result = cursor.fetchone()
            assert result is not None

            conn.close()

    def test_add_fact(self):
        """Test adding facts to database"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_facts.db")
            db = FactDatabase(db_path)

            # Add a fact
            result = db.add_fact(
                subject="Earth",
                predicate="has_satellite",
                obj="Moon",
                confidence=0.99,
                source="astronomy_textbook",
                category="astronomy"
            )

            assert result is True

            # Verify fact was added using query_facts
            facts = db.query_facts(subject="Earth")
            assert len(facts) >= 1

            # Find the fact we added
            earth_moon = [f for f in facts if f["predicate"] == "has_satellite"]
            assert len(earth_moon) >= 1
            assert earth_moon[0]["object"] == "Moon"
            assert earth_moon[0]["confidence"] == 0.99

    def test_query_facts(self):
        """Test querying facts in database"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_facts.db")
            db = FactDatabase(db_path)

            # Add test facts
            db.add_fact("Water", "chemical_formula", "H2O", 1.0, "chemistry", "science")
            db.add_fact("Water", "boiling_point", "100C", 0.95, "physics", "science")
            db.add_fact("Ice", "state_of", "Water", 0.9, "physics", "science")

            # Query for water-related facts
            water_facts = db.query_facts(subject="Water")
            assert len(water_facts) >= 2

            # Query for H2O in object
            h2o_facts = db.query_facts(obj="H2O")
            assert len(h2o_facts) >= 1
            assert h2o_facts[0]["object"] == "H2O"

    def test_get_statistics(self):
        """Test getting database statistics"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_facts.db")
            db = FactDatabase(db_path)

            # Add test facts
            db.add_fact("Earth", "radius", "6371km", 0.99, "geography", "earth_science")
            db.add_fact("Mars", "radius", "3390km", 0.95, "astronomy", "space_science")
            db.add_fact("Sun", "type", "star", 1.0, "astronomy", "space_science")

            stats = db.get_statistics()

            assert isinstance(stats, dict)
            assert "total_facts" in stats
            assert "confidence_distribution" in stats
            assert "categories" in stats

            # Check totals (may include initial facts)
            assert stats["total_facts"] >= 3

            # Check confidence distribution
            conf_dist = stats["confidence_distribution"]
            assert "high" in conf_dist
            assert "medium" in conf_dist
            assert "low" in conf_dist


class TestWikipediaKnowledgeBase:
    """Test Wikipedia knowledge source functionality"""

    def test_source_initialization(self):
        """Test Wikipedia source initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, "wiki_cache")

            source = WikipediaKnowledgeBase(cache_dir=cache_dir, cache_only=True)

            # Cache directory should be created
            assert os.path.exists(cache_dir)
            assert source.cache_dir == Path(cache_dir)

    def test_search_wikipedia_cache_only(self):
        """Test Wikipedia search in cache_only mode returns empty for uncached queries"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, "wiki_cache")
            source = WikipediaKnowledgeBase(cache_dir=cache_dir, cache_only=True)

            # Without cached data, should return empty list
            results = source.search_wikipedia("artificial intelligence")
            assert isinstance(results, list)
            # In cache_only mode with no cache, returns empty
            assert len(results) == 0

    @patch('merit.knowledge.enhanced_kb.wikipedia.search')
    @patch('merit.knowledge.enhanced_kb.wikipedia.summary')
    @patch('merit.knowledge.enhanced_kb.wikipedia.page')
    def test_search_wikipedia_with_live(self, mock_page, mock_summary, mock_search):
        """Test Wikipedia search functionality with live mode"""
        # Mock Wikipedia API responses
        mock_search.return_value = ["Artificial intelligence"]
        mock_summary.return_value = "Artificial intelligence (AI) is intelligence demonstrated by machines..."

        mock_page_obj = Mock()
        mock_page_obj.title = "Artificial intelligence"
        mock_page_obj.url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
        mock_page.return_value = mock_page_obj

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, "wiki_cache")
            source = WikipediaKnowledgeBase(cache_dir=cache_dir, cache_only=False)

            results = source.search_wikipedia("artificial intelligence")

            assert isinstance(results, list)
            if len(results) > 0:
                result = results[0]
                assert "title" in result
                assert "summary" in result

    def test_verify_claim_cache_only(self):
        """Test claim verification in cache_only mode"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, "wiki_cache")
            source = WikipediaKnowledgeBase(cache_dir=cache_dir, cache_only=True)

            result = source.verify_claim("Paris", "capital_of", "France")
            assert isinstance(result, dict)
            assert "verified" in result
            assert "confidence" in result


class TestEnhancedKnowledgeBase:
    """Test enhanced knowledge base functionality"""

    def test_knowledge_base_initialization(self):
        """Test knowledge base initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "facts.db")
            cache_dir = os.path.join(temp_dir, "cache")

            kb = EnhancedKnowledgeBase(db_path=db_path, cache_dir=cache_dir, cache_only=True)

            assert kb.fact_db is not None
            assert kb.wikipedia_kb is not None
            assert kb.cache_only is True

    def test_verify_claim(self):
        """Test claim verification"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "facts.db")
            cache_dir = os.path.join(temp_dir, "cache")

            kb = EnhancedKnowledgeBase(db_path=db_path, cache_dir=cache_dir, cache_only=True)

            # Add some facts to database
            kb.fact_db.add_fact("Water", "formula", "H2O", 1.0, "chemistry", "science")

            # Test verification
            result = kb.verify_claim("Water", "formula", "H2O")

            assert isinstance(result, dict)
            assert "verified" in result
            assert "confidence" in result
            assert "claim" in result

    def test_search_knowledge(self):
        """Test knowledge search across sources"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "facts.db")
            cache_dir = os.path.join(temp_dir, "cache")

            kb = EnhancedKnowledgeBase(db_path=db_path, cache_dir=cache_dir, cache_only=True)

            # Add structured facts
            kb.fact_db.add_fact("Python", "type", "programming_language", 1.0, "programming", "tech")

            # Search knowledge base
            results = kb.search_knowledge("Python")

            assert isinstance(results, dict)
            assert "structured_facts" in results
            assert "wikipedia_results" in results
            assert len(results["structured_facts"]) > 0

    def test_get_knowledge_statistics(self):
        """Test knowledge base statistics"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "facts.db")
            cache_dir = os.path.join(temp_dir, "cache")

            kb = EnhancedKnowledgeBase(db_path=db_path, cache_dir=cache_dir, cache_only=True)

            # Add some data
            kb.fact_db.add_fact("AI", "field", "computer_science", 1.0, "tech", "science")

            stats = kb.get_knowledge_statistics()

            assert isinstance(stats, dict)
            assert "structured_database" in stats
            assert "wikipedia_cache" in stats

            # Check structured database stats
            db_stats = stats["structured_database"]
            assert "total_facts" in db_stats

    def test_batch_verify_claims(self):
        """Test batch claim verification"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "facts.db")
            cache_dir = os.path.join(temp_dir, "cache")

            kb = EnhancedKnowledgeBase(db_path=db_path, cache_dir=cache_dir, cache_only=True)

            # Add facts
            kb.fact_db.add_fact("Water", "formula", "H2O", 1.0, "chemistry", "science")
            kb.fact_db.add_fact("Earth", "shape", "sphere", 0.9, "geography", "science")

            claims = [
                {"subject": "Water", "predicate": "formula", "object": "H2O"},
                {"subject": "Earth", "predicate": "shape", "object": "sphere"}
            ]

            results = kb.batch_verify_claims(claims)

            assert isinstance(results, list)
            assert len(results) == 2
            for result in results:
                assert "verified" in result or "error" in result


class TestKnowledgeBaseIntegration:
    """Test integration between knowledge base components"""

    def test_full_knowledge_workflow(self):
        """Test complete knowledge base workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "facts.db")
            cache_dir = os.path.join(temp_dir, "cache")

            kb = EnhancedKnowledgeBase(db_path=db_path, cache_dir=cache_dir, cache_only=True)

            # Add structured knowledge
            kb.fact_db.add_fact(
                "Machine Learning", "subset_of", "Artificial Intelligence",
                confidence=0.95, source="textbook", category="AI"
            )

            # Search knowledge base
            search_results = kb.search_knowledge("machine learning")
            assert "structured_facts" in search_results

            # Verify a claim
            verification_result = kb.verify_claim(
                "Machine Learning", "subset_of", "Artificial Intelligence"
            )
            assert verification_result["verified"] is True
            assert verification_result["confidence"] >= 0.5

    def test_cross_source_verification(self):
        """Test verification using structured database"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "facts.db")
            cache_dir = os.path.join(temp_dir, "cache")

            kb = EnhancedKnowledgeBase(db_path=db_path, cache_dir=cache_dir, cache_only=True)

            # Add facts
            kb.fact_db.add_fact("Earth", "age", "4.5_billion_years", 0.9, "geology", "science")

            result = kb.verify_claim("Earth", "age", "4.5_billion_years")

            assert isinstance(result, dict)
            assert result["verified"] is True

    def test_knowledge_base_scaling(self):
        """Test knowledge base performance with larger datasets"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "facts.db")
            cache_dir = os.path.join(temp_dir, "cache")

            kb = EnhancedKnowledgeBase(db_path=db_path, cache_dir=cache_dir, cache_only=True)

            # Add many facts
            for i in range(100):
                kb.fact_db.add_fact(
                    f"Entity_{i}",
                    "property",
                    f"value_{i}",
                    confidence=0.8 + (i % 20) / 100,
                    source=f"source_{i % 5}",
                    category=f"category_{i % 10}"
                )

            # Test statistics
            stats = kb.get_knowledge_statistics()
            db_stats = stats["structured_database"]

            assert db_stats["total_facts"] >= 100

            # Test search performance
            facts = kb.fact_db.query_facts(subject="Entity_50")
            assert len(facts) > 0

    def test_error_handling_robustness(self):
        """Test robust error handling in knowledge base"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "facts.db")
            cache_dir = os.path.join(temp_dir, "cache")

            kb = EnhancedKnowledgeBase(db_path=db_path, cache_dir=cache_dir, cache_only=True)

            # Test with empty query
            results = kb.search_knowledge("")
            assert isinstance(results, dict)

            # Test verify with empty values
            result = kb.verify_claim("", "", "")
            assert isinstance(result, dict)


@pytest.mark.parametrize("confidence_level", [0.1, 0.5, 0.9, 1.0])
def test_facts_with_different_confidence_levels(confidence_level):
    """Test handling facts with different confidence levels"""
    with tempfile.TemporaryDirectory() as temp_dir:
        db = FactDatabase(os.path.join(temp_dir, "test.db"))

        # Add fact with specific confidence
        result = db.add_fact(
            "Test_Entity",
            "test_property",
            "test_value",
            confidence=confidence_level,
            source="test",
            category="test"
        )

        assert result is True

        # Retrieve and verify confidence
        facts = db.query_facts(subject="Test_Entity")
        # Filter for our specific fact
        our_facts = [f for f in facts if f["predicate"] == "test_property"]
        assert len(our_facts) >= 1
        assert our_facts[0]["confidence"] == confidence_level


def test_database_persistence():
    """Test that database persists data across sessions"""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "persistent_test.db")

        # First session - add data
        db1 = FactDatabase(db_path)
        db1.add_fact("Persistent", "test", "value", 1.0, "test", "test")

        # Second session - retrieve data
        db2 = FactDatabase(db_path)
        facts = db2.query_facts(subject="Persistent")

        # Filter for our specific fact (exclude initial facts)
        our_facts = [f for f in facts if f["predicate"] == "test"]
        assert len(our_facts) >= 1
        assert our_facts[0]["subject"] == "Persistent"
        assert our_facts[0]["object"] == "value"


if __name__ == "__main__":
    pytest.main([__file__])
