"""
Enhanced knowledge base system for factual accuracy checking.
Includes Wikipedia integration, structured fact databases, and caching.
"""
import json
import pickle
import sqlite3
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
import time
import re
from datetime import datetime, timedelta

try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False
    print("Warning: wikipedia not available. Install with: pip install wikipedia")

try:
    import requests
    from urllib.parse import quote
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not available. Install with: pip install requests")

try:
    import backoff
    BACKOFF_AVAILABLE = True
except ImportError:
    BACKOFF_AVAILABLE = False

# Default timeout for all network requests (seconds)
DEFAULT_REQUEST_TIMEOUT = 10


class FactDatabase:
    """SQLite-based fact database for structured knowledge storage"""
    
    def __init__(self, db_path: str = "data/facts.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database with fact tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main facts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT 'manual',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                verified_at TIMESTAMP,
                category TEXT,
                UNIQUE(subject, predicate, object)
            )
        ''')
        
        # Synonyms table for better matching
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS synonyms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_term TEXT NOT NULL,
                synonym TEXT NOT NULL,
                UNIQUE(canonical_term, synonym)
            )
        ''')
        
        # Categories table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                parent_category TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # Populate with initial data if empty
        self._populate_initial_facts()
    
    def _populate_initial_facts(self):
        """Populate database with initial facts if empty"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if we have any facts
        cursor.execute("SELECT COUNT(*) FROM facts")
        count = cursor.fetchone()[0]
        
        if count == 0:
            # Initial facts categorized by domain
            initial_facts = [
                # Scientific facts
                ("Earth", "revolves around", "Sun", 1.0, "scientific", "astronomy"),
                ("Sun", "revolves around", "Earth", 0.0, "scientific", "astronomy"),  # False fact
                ("water", "chemical formula", "H2O", 1.0, "scientific", "chemistry"),
                ("DNA", "contains", "genetic information", 1.0, "scientific", "biology"),
                ("speed of light", "approximately", "300000000 meters per second", 1.0, "scientific", "physics"),
                ("photosynthesis", "produces", "oxygen", 1.0, "scientific", "biology"),
                ("gravity", "pulls objects towards", "Earth", 1.0, "scientific", "physics"),
                ("atoms", "made of", "protons neutrons electrons", 1.0, "scientific", "chemistry"),
                
                # Mathematical facts
                ("pi", "approximately equals", "3.14159", 1.0, "mathematical", "constants"),
                ("square root of 16", "equals", "4", 1.0, "mathematical", "arithmetic"),
                ("2 plus 2", "equals", "4", 1.0, "mathematical", "arithmetic"),
                ("2 plus 2", "equals", "5", 0.0, "mathematical", "arithmetic"),  # False
                
                # Historical facts
                ("World War 2", "ended in", "1945", 1.0, "historical", "wars"),
                ("Christopher Columbus", "reached Americas in", "1492", 1.0, "historical", "exploration"),
                ("Berlin Wall", "fell in", "1989", 1.0, "historical", "events"),
                ("American Revolution", "began in", "1775", 1.0, "historical", "wars"),
                ("Moon landing", "occurred in", "1969", 1.0, "historical", "space"),
                
                # Geographical facts
                ("Mount Everest", "is", "highest mountain", 1.0, "geographical", "mountains"),
                ("Pacific Ocean", "is", "largest ocean", 1.0, "geographical", "oceans"),
                ("Sahara", "located in", "Africa", 1.0, "geographical", "deserts"),
                ("Nile River", "is", "longest river", 1.0, "geographical", "rivers"),
                ("Antarctica", "is", "coldest continent", 1.0, "geographical", "continents"),
                
                # Common knowledge
                ("humans", "have", "two legs", 1.0, "biology", "anatomy"),
                ("humans", "have", "three legs", 0.0, "biology", "anatomy"),  # False
                ("cats", "are", "mammals", 1.0, "biology", "animals"),
                ("birds", "can", "fly", 0.8, "biology", "animals"),  # Most but not all
                ("fish", "breathe through", "gills", 1.0, "biology", "animals"),
                
                # Technology facts
                ("Internet", "invented in", "1960s-1970s", 1.0, "technology", "computing"),
                ("World Wide Web", "created by", "Tim Berners-Lee", 1.0, "technology", "computing"),
                ("first computer", "was", "ENIAC", 0.8, "technology", "computing"),
                
                # Common misconceptions (false facts)
                ("Great Wall of China", "visible from", "space", 0.0, "misconception", "geography"),
                ("humans", "use only", "10 percent of brain", 0.0, "misconception", "biology"),
                ("lightning", "never strikes", "same place twice", 0.0, "misconception", "physics")
            ]
            
            # Insert facts
            for subject, predicate, obj, confidence, source, category in initial_facts:
                cursor.execute('''
                    INSERT OR IGNORE INTO facts 
                    (subject, predicate, object, confidence, source, category)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (subject, predicate, obj, confidence, source, category))
            
            # Add some synonyms
            synonyms = [
                ("Earth", "planet Earth"),
                ("Earth", "world"),
                ("Sun", "solar"),
                ("water", "H2O"),
                ("DNA", "deoxyribonucleic acid"),
                ("World War 2", "WWII"),
                ("World War 2", "Second World War"),
                ("United States", "USA"),
                ("United States", "America")
            ]
            
            for canonical, synonym in synonyms:
                cursor.execute('''
                    INSERT OR IGNORE INTO synonyms (canonical_term, synonym)
                    VALUES (?, ?)
                ''', (canonical, synonym))
            
            conn.commit()
            print(f"Populated fact database with {len(initial_facts)} initial facts")
        
        conn.close()
    
    def add_fact(self, subject: str, predicate: str, obj: str, 
                 confidence: float = 1.0, source: str = "manual", 
                 category: str = "general") -> bool:
        """Add a new fact to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO facts 
                (subject, predicate, object, confidence, source, category, verified_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (subject, predicate, obj, confidence, source, category))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error adding fact: {e}")
            return False
        finally:
            conn.close()
    
    def query_facts(self, subject: str = None, predicate: str = None, 
                   obj: str = None, min_confidence: float = 0.0) -> List[Dict]:
        """Query facts from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build query
        conditions = []
        params = []
        
        if subject:
            conditions.append("(subject LIKE ? OR subject IN (SELECT canonical_term FROM synonyms WHERE synonym LIKE ?))")
            params.extend([f"%{subject}%", f"%{subject}%"])
        
        if predicate:
            conditions.append("predicate LIKE ?")
            params.append(f"%{predicate}%")
        
        if obj:
            conditions.append("(object LIKE ? OR object IN (SELECT canonical_term FROM synonyms WHERE synonym LIKE ?))")
            params.extend([f"%{obj}%", f"%{obj}%"])
        
        if min_confidence > 0:
            conditions.append("confidence >= ?")
            params.append(min_confidence)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f'''
            SELECT subject, predicate, object, confidence, source, category, created_at, verified_at
            FROM facts 
            WHERE {where_clause}
            ORDER BY confidence DESC
        '''
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        # Convert to dictionaries
        facts = []
        for row in results:
            facts.append({
                "subject": row[0],
                "predicate": row[1],
                "object": row[2],
                "confidence": row[3],
                "source": row[4],
                "category": row[5],
                "created_at": row[6],
                "verified_at": row[7]
            })
        
        return facts
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total facts
        cursor.execute("SELECT COUNT(*) FROM facts")
        stats["total_facts"] = cursor.fetchone()[0]
        
        # Facts by confidence
        cursor.execute('''
            SELECT 
                SUM(CASE WHEN confidence >= 0.8 THEN 1 ELSE 0 END) as high_confidence,
                SUM(CASE WHEN confidence >= 0.5 AND confidence < 0.8 THEN 1 ELSE 0 END) as medium_confidence,
                SUM(CASE WHEN confidence < 0.5 THEN 1 ELSE 0 END) as low_confidence
            FROM facts
        ''')
        result = cursor.fetchone()
        stats["confidence_distribution"] = {
            "high": result[0],
            "medium": result[1], 
            "low": result[2]
        }
        
        # Facts by category
        cursor.execute('''
            SELECT category, COUNT(*) 
            FROM facts 
            GROUP BY category 
            ORDER BY COUNT(*) DESC
        ''')
        stats["categories"] = dict(cursor.fetchall())
        
        # Facts by source
        cursor.execute('''
            SELECT source, COUNT(*) 
            FROM facts 
            GROUP BY source 
            ORDER BY COUNT(*) DESC
        ''')
        stats["sources"] = dict(cursor.fetchall())
        
        conn.close()
        return stats


class WikipediaKnowledgeBase:
    """Wikipedia-based knowledge base with caching"""

    def __init__(self, cache_dir: str = "data/wikipedia_cache", cache_only: bool = True):
        """
        Initialize Wikipedia knowledge base.

        Args:
            cache_dir: Directory for caching Wikipedia results
            cache_only: If True (default), only use cached results - no live API calls.
                       This prevents rate limiting and ensures offline operation.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_duration = timedelta(days=7)  # Cache for 7 days
        self.cache_only = cache_only
        self.request_timeout = DEFAULT_REQUEST_TIMEOUT

        if not WIKIPEDIA_AVAILABLE:
            print("Warning: Wikipedia not available")

        # Set Wikipedia library timeout
        if WIKIPEDIA_AVAILABLE:
            wikipedia.set_rate_limiting(True)  # Enable rate limiting
    
    def _get_cache_path(self, query: str) -> Path:
        """Get cache file path for a query"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return self.cache_dir / f"{query_hash}.pkl"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache is valid (not expired)"""
        if not cache_path.exists():
            return False
        
        cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - cache_time < self.cache_duration
    
    def _fetch_wikipedia_with_retry(self, query: str, max_results: int = 3) -> List[Dict]:
        """Fetch Wikipedia results with retry logic (internal method)."""
        results = []

        try:
            # Search for pages
            search_results = wikipedia.search(query, results=max_results)

            for title in search_results:
                try:
                    # Get page summary
                    summary = wikipedia.summary(title, sentences=3, auto_suggest=False)

                    # Get page URL
                    page = wikipedia.page(title, auto_suggest=False)

                    results.append({
                        "title": title,
                        "summary": summary,
                        "url": page.url,
                        "content_preview": summary[:300] + "..." if len(summary) > 300 else summary
                    })

                except wikipedia.exceptions.DisambiguationError as e:
                    # Take the first disambiguation option
                    if e.options:
                        try:
                            summary = wikipedia.summary(e.options[0], sentences=2)
                            page = wikipedia.page(e.options[0])
                            results.append({
                                "title": e.options[0],
                                "summary": summary,
                                "url": page.url,
                                "disambiguation": True,
                                "content_preview": summary[:300] + "..." if len(summary) > 300 else summary
                            })
                        except:
                            continue

                except wikipedia.exceptions.PageError:
                    continue
                except Exception as e:
                    print(f"Error processing Wikipedia page {title}: {e}")
                    continue

        except Exception as e:
            print(f"Error searching Wikipedia: {e}")

        return results

    def search_wikipedia(self, query: str, max_results: int = 3) -> List[Dict]:
        """Search Wikipedia for information about a query.

        Respects cache_only setting - if True, only returns cached results.
        """
        if not WIKIPEDIA_AVAILABLE:
            return []

        cache_path = self._get_cache_path(f"search_{query}")

        # Check cache first
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                pass

        # If cache_only mode and no valid cache, return empty
        if self.cache_only:
            # Try to load expired cache as fallback
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
                except:
                    pass
            return []

        # Fetch with retry logic
        if BACKOFF_AVAILABLE:
            @backoff.on_exception(
                backoff.expo,
                (requests.exceptions.RequestException, Exception),
                max_tries=3,
                max_time=30
            )
            def fetch_with_backoff():
                return self._fetch_wikipedia_with_retry(query, max_results)
            results = fetch_with_backoff()
        else:
            results = self._fetch_wikipedia_with_retry(query, max_results)

        # Cache results
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(results, f)
        except Exception as e:
            print(f"Error caching Wikipedia results: {e}")

        return results
    
    def verify_claim(self, subject: str, predicate: str, obj: str) -> Dict:
        """Verify a claim against Wikipedia"""
        if not WIKIPEDIA_AVAILABLE:
            return {"verified": False, "confidence": 0.0, "error": "Wikipedia not available"}
        
        # Create cache key
        claim = f"{subject}_{predicate}_{obj}"
        cache_path = self._get_cache_path(f"verify_{claim}")
        
        # Check cache
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        
        verification_result = {
            "verified": False,
            "confidence": 0.0,
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "sources": []
        }
        
        try:
            # Search for subject
            search_results = self.search_wikipedia(subject, max_results=2)
            
            for result in search_results:
                content = result["content_preview"].lower()
                obj_lower = obj.lower()
                
                # Simple keyword matching (can be enhanced with NLP)
                if obj_lower in content:
                    # Check context around the object mention
                    sentences = re.split(r'[.!?]+', content)
                    
                    for sentence in sentences:
                        if obj_lower in sentence:
                            # Basic sentiment/verification check
                            if any(neg in sentence for neg in ["not", "never", "no", "false", "incorrect"]):
                                verification_result["contradicting_evidence"].append({
                                    "sentence": sentence.strip(),
                                    "source": result["title"],
                                    "url": result["url"]
                                })
                            else:
                                verification_result["supporting_evidence"].append({
                                    "sentence": sentence.strip(),
                                    "source": result["title"],
                                    "url": result["url"]
                                })
                
                verification_result["sources"].append({
                    "title": result["title"],
                    "url": result["url"]
                })
            
            # Calculate confidence based on evidence
            supporting_count = len(verification_result["supporting_evidence"])
            contradicting_count = len(verification_result["contradicting_evidence"])
            
            if supporting_count > 0 and contradicting_count == 0:
                verification_result["verified"] = True
                verification_result["confidence"] = min(0.8, 0.5 + supporting_count * 0.1)
            elif supporting_count > contradicting_count:
                verification_result["verified"] = True
                verification_result["confidence"] = 0.3 + (supporting_count - contradicting_count) * 0.1
            elif contradicting_count > supporting_count:
                verification_result["verified"] = False
                verification_result["confidence"] = 0.3 + (contradicting_count - supporting_count) * 0.1
            else:
                verification_result["confidence"] = 0.2  # Uncertain
        
        except Exception as e:
            verification_result["error"] = str(e)
        
        # Cache result
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(verification_result, f)
        except:
            pass
        
        return verification_result


class EnhancedKnowledgeBase:
    """Enhanced knowledge base combining structured facts and Wikipedia"""

    def __init__(self, db_path: str = "data/facts.db",
                 cache_dir: str = "data/wikipedia_cache",
                 cache_only: bool = True):
        """
        Initialize enhanced knowledge base.

        Args:
            db_path: Path to SQLite fact database
            cache_dir: Directory for Wikipedia cache
            cache_only: If True (default), Wikipedia only uses cached results (no live API)
        """
        self.fact_db = FactDatabase(db_path)
        self.wikipedia_kb = WikipediaKnowledgeBase(cache_dir, cache_only=cache_only)
        self.cache_only = cache_only

        # Configuration
        self.confidence_threshold = 0.5
        self.max_wikipedia_results = 3
    
    def verify_claim(self, subject: str, predicate: str, obj: str) -> Dict:
        """Verify a claim using both structured facts and Wikipedia"""
        
        result = {
            "claim": {"subject": subject, "predicate": predicate, "object": obj},
            "verified": False,
            "confidence": 0.0,
            "sources": [],
            "evidence": {
                "structured_facts": [],
                "wikipedia_evidence": {}
            },
            "explanation": ""
        }
        
        # First check structured facts database
        db_facts = self.fact_db.query_facts(
            subject=subject, 
            predicate=predicate, 
            obj=obj,
            min_confidence=0.1
        )
        
        if db_facts:
            # Found exact or similar match in database
            best_match = max(db_facts, key=lambda x: x["confidence"])
            result["verified"] = best_match["confidence"] >= self.confidence_threshold
            result["confidence"] = best_match["confidence"]
            result["evidence"]["structured_facts"] = db_facts
            result["sources"].append({
                "type": "structured_database",
                "confidence": best_match["confidence"],
                "category": best_match["category"]
            })
            
            if result["verified"]:
                result["explanation"] = f"Claim verified against structured fact database with {best_match['confidence']:.2f} confidence."
            else:
                result["explanation"] = f"Claim contradicted by structured fact database with {best_match['confidence']:.2f} confidence."
        
        else:
            # No exact match, check Wikipedia
            wiki_result = self.wikipedia_kb.verify_claim(subject, predicate, obj)
            result["evidence"]["wikipedia_evidence"] = wiki_result
            
            if "error" not in wiki_result:
                result["verified"] = wiki_result["verified"]
                result["confidence"] = wiki_result["confidence"]
                result["sources"].append({
                    "type": "wikipedia",
                    "confidence": wiki_result["confidence"],
                    "sources": wiki_result.get("sources", [])
                })
                
                if result["verified"]:
                    result["explanation"] = f"Claim supported by Wikipedia sources with {wiki_result['confidence']:.2f} confidence."
                else:
                    result["explanation"] = f"Claim not sufficiently supported by Wikipedia sources (confidence: {wiki_result['confidence']:.2f})."
            else:
                result["explanation"] = f"Could not verify claim: {wiki_result['error']}"
        
        # If we have high-confidence verification, add to structured database
        if result["confidence"] >= 0.8 and not db_facts:
            confidence_value = 1.0 if result["verified"] else 0.0
            self.fact_db.add_fact(
                subject, predicate, obj, 
                confidence=confidence_value, 
                source="wikipedia_verified",
                category="auto_verified"
            )
        
        return result
    
    def batch_verify_claims(self, claims: List[Dict]) -> List[Dict]:
        """Verify multiple claims efficiently"""
        results = []
        
        for claim in claims:
            if isinstance(claim, dict) and all(k in claim for k in ["subject", "predicate", "object"]):
                result = self.verify_claim(
                    claim["subject"], 
                    claim["predicate"], 
                    claim["object"]
                )
                results.append(result)
            else:
                results.append({
                    "error": "Invalid claim format",
                    "verified": False,
                    "confidence": 0.0
                })
        
        return results
    
    def search_knowledge(self, query: str) -> Dict:
        """Search across all knowledge sources"""
        search_results = {
            "query": query,
            "structured_facts": [],
            "wikipedia_results": []
        }
        
        # Search structured facts
        # Try different ways to match the query
        query_parts = query.lower().split()
        
        for part in query_parts:
            facts = self.fact_db.query_facts(subject=part)
            search_results["structured_facts"].extend(facts)
            
            facts = self.fact_db.query_facts(obj=part)
            search_results["structured_facts"].extend(facts)
        
        # Remove duplicates
        seen = set()
        unique_facts = []
        for fact in search_results["structured_facts"]:
            fact_key = (fact["subject"], fact["predicate"], fact["object"])
            if fact_key not in seen:
                seen.add(fact_key)
                unique_facts.append(fact)
        
        search_results["structured_facts"] = unique_facts
        
        # Search Wikipedia
        search_results["wikipedia_results"] = self.wikipedia_kb.search_wikipedia(query)
        
        return search_results
    
    def get_knowledge_statistics(self) -> Dict:
        """Get comprehensive knowledge base statistics"""
        stats = {
            "structured_database": self.fact_db.get_statistics(),
            "wikipedia_cache": self._get_wikipedia_cache_stats(),
            "total_knowledge_sources": 2
        }
        
        return stats
    
    def _get_wikipedia_cache_stats(self) -> Dict:
        """Get Wikipedia cache statistics"""
        cache_files = list(self.wikipedia_kb.cache_dir.glob("*.pkl"))
        
        return {
            "cached_queries": len(cache_files),
            "cache_size_mb": sum(f.stat().st_size for f in cache_files) / (1024 * 1024),
            "cache_directory": str(self.wikipedia_kb.cache_dir)
        }
    
    def export_knowledge_base(self, output_file: str):
        """Export the knowledge base to JSON"""
        export_data = {
            "structured_facts": self.fact_db.query_facts(),
            "statistics": self.get_knowledge_statistics(),
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Knowledge base exported to: {output_file}")
    
    def import_facts_from_json(self, json_file: str) -> int:
        """Import facts from JSON file"""
        imported = 0
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            facts = data.get("facts", [])
            
            for fact in facts:
                if all(k in fact for k in ["subject", "predicate", "object"]):
                    success = self.fact_db.add_fact(
                        fact["subject"],
                        fact["predicate"], 
                        fact["object"],
                        fact.get("confidence", 1.0),
                        fact.get("source", "imported"),
                        fact.get("category", "imported")
                    )
                    if success:
                        imported += 1
        
        except Exception as e:
            print(f"Error importing facts: {e}")
        
        print(f"Imported {imported} facts from {json_file}")
        return imported