"""Logical consistency metric for MERIT."""
import re
from typing import Dict, List, Optional

import nltk
import numpy as np
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from merit.core.base import BaseMetric, MetricResult
from merit.core.device import DeviceManager

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class LogicalConsistencyMetric(BaseMetric):
    """Measures internal logical consistency using semantic analysis."""

    @property
    def name(self) -> str:
        return "logical_consistency"

    @property
    def dimension(self) -> str:
        return "consistency"

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.description = "Measures internal logical consistency using semantic analysis"

        # Initialize device and models
        self.device = DeviceManager.get_optimal_device()
        print(f"Using device: {self.device} for logical consistency metric")

        # Load sentence transformer for semantic similarity
        self.sentence_model = SentenceTransformer(model_name)
        if self.device != "cpu":
            self.sentence_model = self.sentence_model.to(self.device)

        # Load spaCy model for linguistic analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Logical fallacy patterns
        self.fallacy_patterns = {
            'circular_reasoning': [
                r'(.+)\s+because\s+(.+)\s+because\s+\1',
                r'(.+)\s+since\s+(.+)\s+since\s+\1'
            ],
            'contradiction': [
                r'(.+)\s+but\s+(.+\s+not\s+.+)',
                r'(.+\s+always\s+.+)\s+but\s+(.+\s+never\s+.+)'
            ],
            'false_dichotomy': [
                r'either\s+(.+)\s+or\s+(.+),?\s+(no|not)\s+(other|alternative)',
                r'only\s+two\s+(options|choices|ways)'
            ]
        }

    def compute(self, response: str, reference: Optional[str] = None, **kwargs) -> MetricResult:
        """Compute enhanced logical consistency."""
        if not response.strip():
            return MetricResult(score=0.0, dimension=self.dimension, details={"analysis": "Empty prediction"})

        # Split into sentences
        sentences = self._extract_sentences(response)
        if len(sentences) <= 1:
            return MetricResult(score=1.0, dimension=self.dimension, details={"analysis": "Single sentence - no consistency issues"})

        analysis = {
            "total_sentences": len(sentences),
            "semantic_contradictions": [],
            "logical_fallacies": {},
            "sentiment_contradictions": [],
            "dependency_issues": []
        }

        # Detect semantic contradictions
        if len(sentences) > 1:
            analysis["semantic_contradictions"] = self._detect_semantic_contradictions(sentences)

        # Detect logical fallacies
        analysis["logical_fallacies"] = self._detect_logical_fallacies(response)

        # Detect sentiment contradictions
        analysis["sentiment_contradictions"] = self._detect_sentiment_contradictions(sentences)

        # Analyze dependency structures if spaCy is available
        if self.nlp:
            analysis["dependency_issues"] = self._analyze_dependency_consistency(response)

        # Calculate overall consistency score
        consistency_score = self._calculate_consistency_score(analysis)

        details = {
            "analysis": analysis,
            "detailed_breakdown": {
                "semantic_consistency": 1.0 - (len(analysis["semantic_contradictions"]) / max(1, len(sentences) * (len(sentences) - 1) / 2)),
                "logical_fallacy_penalty": len(analysis["logical_fallacies"]) * 0.1,
                "sentiment_consistency": 1.0 - (len(analysis["sentiment_contradictions"]) / max(1, len(sentences))),
                "dependency_consistency": 1.0 - (len(analysis["dependency_issues"]) / max(1, len(sentences)))
            }
        }

        return MetricResult(score=consistency_score, dimension=self.dimension, details=details)

    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences using multiple methods"""
        # Use spaCy if available, otherwise use NLTK
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Fallback to NLTK
            try:
                from nltk.tokenize import sent_tokenize
                return [s.strip() for s in sent_tokenize(text) if s.strip()]
            except:
                # Final fallback to regex
                sentences = re.split(r'[.!?]+', text)
                return [s.strip() for s in sentences if s.strip()]

    def _detect_semantic_contradictions(self, sentences: List[str]) -> List[Dict]:
        """Detect contradictions using semantic similarity and sentiment analysis"""
        contradictions = []

        # Get embeddings for all sentences
        embeddings = self.sentence_model.encode(sentences)

        for i, sent1 in enumerate(sentences):
            for j, sent2 in enumerate(sentences[i+1:], i+1):
                emb1, emb2 = embeddings[i], embeddings[j]

                # Calculate semantic similarity
                similarity = cosine_similarity([emb1], [emb2])[0][0]

                # If sentences are semantically similar, check for opposite sentiments
                if similarity > 0.7:
                    sent1_sentiment = self.sentiment_analyzer.polarity_scores(sent1)
                    sent2_sentiment = self.sentiment_analyzer.polarity_scores(sent2)

                    # Check for opposite sentiment polarities
                    if (sent1_sentiment['compound'] > 0.1 and sent2_sentiment['compound'] < -0.1) or \
                       (sent1_sentiment['compound'] < -0.1 and sent2_sentiment['compound'] > 0.1):
                        contradictions.append({
                            "sentence1": sent1,
                            "sentence2": sent2,
                            "similarity": float(similarity),
                            "sentiment1": sent1_sentiment['compound'],
                            "sentiment2": sent2_sentiment['compound'],
                            "type": "semantic_sentiment_contradiction"
                        })

                # Check for explicit negations
                if self._are_explicit_negations(sent1, sent2, similarity):
                    contradictions.append({
                        "sentence1": sent1,
                        "sentence2": sent2,
                        "similarity": float(similarity),
                        "type": "explicit_negation"
                    })

        return contradictions

    def _are_explicit_negations(self, sent1: str, sent2: str, similarity: float) -> bool:
        """Check if sentences are explicit negations of each other"""
        if similarity < 0.6:  # Must be somewhat similar
            return False

        # Convert to lowercase for comparison
        s1_lower = sent1.lower()
        s2_lower = sent2.lower()

        # Check for negation patterns
        negation_words = ["not", "never", "no", "isn't", "aren't", "wasn't", "weren't",
                         "doesn't", "don't", "didn't", "won't", "wouldn't", "can't", "cannot"]

        # Count negations in each sentence
        s1_negations = sum(1 for neg in negation_words if neg in s1_lower)
        s2_negations = sum(1 for neg in negation_words if neg in s2_lower)

        # If one has negations and the other doesn't (or significantly different counts)
        return abs(s1_negations - s2_negations) >= 1

    def _detect_logical_fallacies(self, text: str) -> Dict:
        """Detect logical fallacies using pattern matching"""
        fallacies = {}
        text_lower = text.lower()

        for fallacy_type, patterns in self.fallacy_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text_lower, re.IGNORECASE)
                if found:
                    matches.extend(found)

            if matches:
                fallacies[fallacy_type] = matches

        return fallacies

    def _detect_sentiment_contradictions(self, sentences: List[str]) -> List[Dict]:
        """Detect sentences with contradictory sentiments about similar topics"""
        contradictions = []

        for i, sent1 in enumerate(sentences):
            for sent2 in sentences[i+1:]:
                # Get sentiment scores
                sent1_sentiment = self.sentiment_analyzer.polarity_scores(sent1)
                sent2_sentiment = self.sentiment_analyzer.polarity_scores(sent2)

                # Check if they have strongly opposite sentiments
                if abs(sent1_sentiment['compound'] - sent2_sentiment['compound']) > 1.0:
                    # Check if they're about similar topics (using keyword overlap)
                    if self._have_topic_overlap(sent1, sent2):
                        contradictions.append({
                            "sentence1": sent1,
                            "sentence2": sent2,
                            "sentiment1": sent1_sentiment['compound'],
                            "sentiment2": sent2_sentiment['compound'],
                            "sentiment_difference": abs(sent1_sentiment['compound'] - sent2_sentiment['compound'])
                        })

        return contradictions

    def _have_topic_overlap(self, sent1: str, sent2: str) -> bool:
        """Check if two sentences have significant topic overlap"""
        # Simple keyword-based approach
        words1 = set(re.findall(r'\w+', sent1.lower()))
        words2 = set(re.findall(r'\w+', sent2.lower()))

        # Remove common stop words
        stop_words = {"the", "is", "at", "which", "on", "and", "a", "to", "are", "as", "an", "be", "or", "will"}
        words1 -= stop_words
        words2 -= stop_words

        if not words1 or not words2:
            return False

        overlap = len(words1.intersection(words2))
        return overlap / min(len(words1), len(words2)) > 0.3

    def _analyze_dependency_consistency(self, text: str) -> List[Dict]:
        """Analyze dependency structures for consistency issues"""
        if not self.nlp:
            return []

        doc = self.nlp(text)
        issues = []

        # Look for dependency inconsistencies
        for sent in doc.sents:
            # Check for incomplete dependencies
            for token in sent:
                if token.dep_ == "ROOT" and token.pos_ not in ["VERB", "AUX"]:
                    issues.append({
                        "type": "incomplete_sentence",
                        "token": token.text,
                        "sentence": sent.text
                    })

                # Check for dangling modifiers
                if token.dep_ in ["amod", "advmod"] and not list(token.children):
                    # Find what it's modifying
                    head = token.head
                    if head.pos_ not in ["NOUN", "VERB", "ADJ", "ADV"]:
                        issues.append({
                            "type": "dangling_modifier",
                            "modifier": token.text,
                            "head": head.text,
                            "sentence": sent.text
                        })

        return issues

    def _calculate_consistency_score(self, analysis: Dict) -> float:
        """Calculate overall consistency score from analysis"""
        base_score = 1.0

        # Penalty for semantic contradictions
        semantic_penalty = len(analysis["semantic_contradictions"]) * 0.2

        # Penalty for logical fallacies
        fallacy_penalty = sum(len(matches) for matches in analysis["logical_fallacies"].values()) * 0.15

        # Penalty for sentiment contradictions
        sentiment_penalty = len(analysis["sentiment_contradictions"]) * 0.1

        # Penalty for dependency issues
        dependency_penalty = len(analysis["dependency_issues"]) * 0.05

        # Calculate final score
        final_score = base_score - semantic_penalty - fallacy_penalty - sentiment_penalty - dependency_penalty

        return max(0.0, min(1.0, final_score))


# Backward compatibility
EnhancedLogicalConsistencyMetric = LogicalConsistencyMetric
