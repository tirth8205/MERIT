"""Factual accuracy metric for MERIT."""
import re
from typing import Dict, List, Optional

import requests
import spacy
import wikipedia

from merit.core.base import BaseMetric, MetricResult
from merit.core.knowledge_cache import KnowledgeCache


class FactualAccuracyMetric(BaseMetric):
    """Measures factual accuracy using Wikipedia, Wikidata, and web search."""

    @property
    def name(self) -> str:
        return "factual_accuracy"

    @property
    def dimension(self) -> str:
        return "factuality"

    def __init__(
        self,
        reproducible: bool = False,
        cache_path: str = "data/knowledge_cache.json",
    ):
        """Initialize factual accuracy metric with web-based verification.

        Args:
            reproducible: If True, only cached results are used (no live API
                calls). Cache misses return an "unknown / not_cached" result.
            cache_path: Path to the JSON file used by :class:`KnowledgeCache`.
        """
        self.description = "Measures factual accuracy using Wikipedia, Wikidata, and web search"
        self.reproducible = reproducible

        # File-backed knowledge cache for reproducible fact verification
        self._knowledge_cache = KnowledgeCache(cache_path)

        # Initialize NLP model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Some features will be limited.")
            self.nlp = None

        # In-memory cache for raw API responses (avoids repeated calls within
        # a single session).  This is separate from the knowledge cache which
        # stores *verification results* keyed by claim text.
        self.cache = {}

        # Request session for web calls
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MERIT-FactChecker/2.0 (Academic Research)'
        })

    def compute(self, response: str, reference: Optional[str] = None, **kwargs) -> MetricResult:
        """Compute enhanced factual accuracy."""
        if not response.strip():
            return MetricResult(score=0.0, dimension=self.dimension, details={"analysis": "Empty prediction"})

        # Extract factual claims
        claims = self._extract_factual_claims(response)

        if not claims:
            return MetricResult(
                score=0.5,
                dimension=self.dimension,
                details={
                    "analysis": "No factual claims detected",
                    "claims_analysis": []
                }
            )

        # Verify each claim
        claims_analysis = []
        verified_count = 0
        total_verifiable = 0

        for claim in claims:
            verification = self._verify_claim(claim)
            claims_analysis.append(verification)

            if verification["verifiable"]:
                total_verifiable += 1
                if verification["verdict"] == "correct":
                    verified_count += 1

        # Calculate accuracy score
        if total_verifiable == 0:
            accuracy_score = 0.5  # Neutral for unverifiable claims
            coverage = 0.0
        else:
            accuracy_score = verified_count / total_verifiable
            coverage = total_verifiable / len(claims)

        # Combined score considering both accuracy and coverage
        combined_score = accuracy_score * (0.7 + 0.3 * coverage)

        details = {
            "accuracy_score": accuracy_score,
            "coverage": coverage,
            "total_claims": len(claims),
            "verifiable_claims": total_verifiable,
            "correct_claims": verified_count,
            "claims_analysis": claims_analysis
        }

        return MetricResult(score=combined_score, dimension=self.dimension, details=details)

    def _extract_factual_claims(self, text: str) -> List[Dict]:
        """Extract factual claims using NER and dependency parsing"""
        claims = []

        if self.nlp:
            doc = self.nlp(text)

            # Extract named entities and their relationships
            for sent in doc.sents:
                entities = [(ent.text, ent.label_) for ent in sent.ents]

                if entities:
                    # Extract subject-predicate-object triples
                    for token in sent:
                        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                            subject = token.text
                            predicate = token.head.text

                            # Find object
                            obj = None
                            for child in token.head.children:
                                if child.dep_ in ["dobj", "attr", "pobj"]:
                                    obj = child.text
                                    break

                            if obj:
                                claims.append({
                                    "subject": subject,
                                    "predicate": predicate,
                                    "object": obj,
                                    "full_sentence": sent.text,
                                    "entities": entities
                                })
        else:
            # Fallback: simple pattern-based extraction
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                if sentence.strip():
                    # Look for simple factual patterns
                    patterns = [
                        r'(\w+(?:\s+\w+)*)\s+is\s+(\w+(?:\s+\w+)*)',
                        r'(\w+(?:\s+\w+)*)\s+was\s+(\w+(?:\s+\w+)*)',
                        r'(\w+(?:\s+\w+)*)\s+has\s+(\w+(?:\s+\w+)*)',
                        r'(\w+(?:\s+\w+)*)\s+contains\s+(\w+(?:\s+\w+)*)'
                    ]

                    for pattern in patterns:
                        matches = re.findall(pattern, sentence, re.IGNORECASE)
                        for match in matches:
                            claims.append({
                                "subject": match[0],
                                "predicate": "is/was/has/contains",
                                "object": match[1],
                                "full_sentence": sentence.strip(),
                                "entities": []
                            })

        return claims

    def save_cache(self) -> None:
        """Persist the knowledge cache to disk.

        Call this after an evaluation run so that subsequent runs in
        ``reproducible=True`` mode can reuse the cached verification
        results without making live API calls.
        """
        self._knowledge_cache.save()

    def _verify_claim(self, claim: Dict) -> Dict:
        """Verify a factual claim using multiple web sources."""
        # Build a stable cache key from the claim's textual content
        cache_key = claim.get("full_sentence", f"{claim['subject']}|{claim['predicate']}|{claim['object']}")

        # 1. Check knowledge cache first
        cached = self._knowledge_cache.get(cache_key)
        if cached is not None:
            return cached

        # 2. In reproducible mode, do NOT make live API calls
        if self.reproducible:
            return {
                "claim": claim,
                "verifiable": False,
                "verdict": "unverifiable",
                "confidence": 0.0,
                "sources": [],
                "source": "not_cached",
            }

        # 3. Live verification via web APIs
        verification = {
            "claim": claim,
            "verifiable": False,
            "verdict": "unverifiable",
            "confidence": 0.0,
            "sources": []
        }

        # Try verification sources in order of reliability
        # 3a. Wikidata (structured, most reliable)
        wikidata_result = self._check_wikidata(claim)
        if wikidata_result["found"]:
            verification.update({
                "verifiable": True,
                "verdict": wikidata_result["verdict"],
                "confidence": wikidata_result["confidence"],
                "sources": [{"type": "wikidata", "result": wikidata_result}]
            })
            self._knowledge_cache.put(cache_key, verification)
            return verification

        # 3b. Wikipedia (detailed text)
        wikipedia_result = self._check_wikipedia(claim)
        if wikipedia_result["found"]:
            verification.update({
                "verifiable": True,
                "verdict": wikipedia_result["verdict"],
                "confidence": wikipedia_result["confidence"],
                "sources": [{"type": "wikipedia", "result": wikipedia_result}]
            })
            self._knowledge_cache.put(cache_key, verification)
            return verification

        # 3c. DuckDuckGo Instant Answers (broad coverage)
        ddg_result = self._check_duckduckgo(claim)
        if ddg_result["found"]:
            verification.update({
                "verifiable": True,
                "verdict": ddg_result["verdict"],
                "confidence": ddg_result["confidence"],
                "sources": [{"type": "duckduckgo", "result": ddg_result}]
            })
            self._knowledge_cache.put(cache_key, verification)
            return verification

        # Cache the "unverifiable" result too, so reproducible mode can
        # distinguish "looked up but not found" from "never looked up".
        self._knowledge_cache.put(cache_key, verification)
        return verification

    def _check_wikidata(self, claim: Dict) -> Dict:
        """Check claim against Wikidata structured knowledge base"""
        result = {"found": False, "verdict": "unverifiable", "confidence": 0.0}

        try:
            subject = claim["subject"]
            cache_key = f"wikidata:{subject}"

            if cache_key in self.cache:
                entity_data = self.cache[cache_key]
            else:
                # Search for entity in Wikidata
                search_url = "https://www.wikidata.org/w/api.php"
                search_params = {
                    "action": "wbsearchentities",
                    "search": subject,
                    "language": "en",
                    "format": "json",
                    "limit": 1
                }

                response = self.session.get(search_url, params=search_params, timeout=10)
                search_data = response.json()

                if not search_data.get("search"):
                    return result

                entity_id = search_data["search"][0]["id"]

                # Get entity details
                entity_url = "https://www.wikidata.org/w/api.php"
                entity_params = {
                    "action": "wbgetentities",
                    "ids": entity_id,
                    "languages": "en",
                    "format": "json"
                }

                response = self.session.get(entity_url, params=entity_params, timeout=10)
                entity_data = response.json()
                self.cache[cache_key] = entity_data

            # Extract claims from entity
            if "entities" in entity_data:
                entity = list(entity_data["entities"].values())[0]
                description = entity.get("descriptions", {}).get("en", {}).get("value", "").lower()
                label = entity.get("labels", {}).get("en", {}).get("value", "").lower()

                # Check if claim object matches entity description or label
                obj_lower = claim["object"].lower()
                if obj_lower in description or obj_lower in label:
                    result["found"] = True
                    result["verdict"] = "correct"
                    result["confidence"] = 0.85
                    result["entity_description"] = description
                    result["entity_label"] = label

        except Exception as e:
            pass  # Silently fail, try next source

        return result

    def _check_wikipedia(self, claim: Dict) -> Dict:
        """Check claim against Wikipedia"""
        result = {"found": False, "verdict": "unverifiable", "confidence": 0.0}

        try:
            subject = claim["subject"]
            cache_key = f"wikipedia:{subject}"

            if cache_key in self.cache:
                page_content = self.cache[cache_key]
            else:
                try:
                    page = wikipedia.page(subject, auto_suggest=True)
                    page_content = page.content.lower()
                    self.cache[cache_key] = page_content
                except wikipedia.DisambiguationError as e:
                    # Try first suggestion
                    try:
                        page = wikipedia.page(e.options[0])
                        page_content = page.content.lower()
                        self.cache[cache_key] = page_content
                    except:
                        return result
                except wikipedia.PageError:
                    # Try summary instead
                    try:
                        page_content = wikipedia.summary(subject, sentences=10).lower()
                        self.cache[cache_key] = page_content
                    except:
                        return result

            # Check if the claim is supported by Wikipedia content
            obj_lower = claim["object"].lower()

            if obj_lower in page_content:
                # Find supporting context
                sentences = page_content.split('.')
                supporting = [s.strip() for s in sentences if obj_lower in s][:2]

                if supporting:
                    result["found"] = True
                    result["verdict"] = "correct"
                    result["confidence"] = 0.75
                    result["supporting_text"] = supporting

        except Exception as e:
            pass  # Silently fail, try next source

        return result

    def _check_duckduckgo(self, claim: Dict) -> Dict:
        """Check claim using DuckDuckGo Instant Answer API"""
        result = {"found": False, "verdict": "unverifiable", "confidence": 0.0}

        try:
            # Construct search query from claim
            query = f"{claim['subject']} {claim['object']}"
            cache_key = f"ddg:{query}"

            if cache_key in self.cache:
                ddg_data = self.cache[cache_key]
            else:
                ddg_url = "https://api.duckduckgo.com/"
                params = {
                    "q": query,
                    "format": "json",
                    "no_html": 1,
                    "skip_disambig": 1
                }

                response = self.session.get(ddg_url, params=params, timeout=10)
                ddg_data = response.json()
                self.cache[cache_key] = ddg_data

            # Check Abstract, Answer, and Definition fields
            abstract = ddg_data.get("Abstract", "").lower()
            answer = ddg_data.get("Answer", "").lower()
            definition = ddg_data.get("Definition", "").lower()

            combined_text = f"{abstract} {answer} {definition}"
            obj_lower = claim["object"].lower()

            if obj_lower in combined_text:
                result["found"] = True
                result["verdict"] = "correct"
                result["confidence"] = 0.65
                result["source_url"] = ddg_data.get("AbstractURL", "")
                result["abstract"] = abstract[:200] if abstract else None

            # Also check Related Topics
            related = ddg_data.get("RelatedTopics", [])
            for topic in related[:5]:
                if isinstance(topic, dict):
                    topic_text = topic.get("Text", "").lower()
                    if obj_lower in topic_text:
                        result["found"] = True
                        result["verdict"] = "correct"
                        result["confidence"] = 0.55
                        result["related_topic"] = topic_text[:200]
                        break

        except Exception as e:
            pass  # Silently fail

        return result


# Backward compatibility
EnhancedFactualAccuracyMetric = FactualAccuracyMetric
