"""Reasoning step metric for MERIT."""
import re
from typing import Dict, List, Optional

import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from merit.core.base import BaseMetric, MetricResult
from merit.core.device import DeviceManager


class ReasoningStepMetric(BaseMetric):
    """Evaluates quality and coherence of reasoning steps using NLP."""

    @property
    def name(self) -> str:
        return "reasoning_steps"

    @property
    def dimension(self) -> str:
        return "reasoning"

    def __init__(self):
        self.description = "Evaluates quality and coherence of reasoning steps using NLP"

        # Initialize NLP model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Some features will be limited.")
            self.nlp = None

        # Initialize sentence transformer for coherence analysis
        self.device = DeviceManager.get_optimal_device()
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        if self.device != "cpu":
            self.sentence_model = self.sentence_model.to(self.device)

        # Reasoning markers for different types of reasoning
        self.reasoning_markers = {
            "causal": ["because", "since", "due to", "owing to", "as a result", "therefore", "thus", "hence", "consequently"],
            "sequential": ["first", "second", "third", "next", "then", "finally", "lastly", "initially", "subsequently"],
            "conditional": ["if", "unless", "provided that", "assuming", "given that", "in case"],
            "contrastive": ["however", "nevertheless", "on the other hand", "in contrast", "but", "yet", "although"],
            "evidential": ["for example", "for instance", "such as", "namely", "specifically", "in particular"]
        }

    def compute(self, response: str, reference: Optional[str] = None, **kwargs) -> MetricResult:
        """Compute enhanced reasoning step quality."""
        if not response.strip():
            return MetricResult(score=0.0, dimension=self.dimension, details={"analysis": "Empty prediction"})

        # Extract reasoning steps using multiple methods
        steps = self._extract_reasoning_steps(response)

        if not steps:
            return MetricResult(
                score=0.2,
                dimension=self.dimension,
                details={
                    "analysis": "No clear reasoning steps found",
                    "steps": []
                }
            )

        # Analyze step quality
        step_analysis = self._analyze_step_quality(steps, response)

        # Calculate coherence between steps
        coherence_analysis = self._analyze_step_coherence(steps)

        # Evaluate completeness
        completeness_analysis = self._analyze_completeness(steps, response)

        # Calculate overall score
        overall_score = self._calculate_overall_score(step_analysis, coherence_analysis, completeness_analysis)

        details = {
            "steps": steps,
            "num_steps": len(steps),
            "step_quality": step_analysis,
            "coherence": coherence_analysis,
            "completeness": completeness_analysis,
            "detailed_scores": {
                "step_quality_score": step_analysis.get("average_quality", 0),
                "coherence_score": coherence_analysis.get("overall_coherence", 0),
                "completeness_score": completeness_analysis.get("completeness_score", 0)
            }
        }

        return MetricResult(score=overall_score, dimension=self.dimension, details=details)

    def _extract_reasoning_steps(self, text: str) -> List[Dict]:
        """Extract reasoning steps using multiple approaches"""
        steps = []

        # Method 1: Pattern-based extraction (enhanced)
        pattern_steps = self._extract_pattern_based_steps(text)
        steps.extend(pattern_steps)

        # Method 2: NLP-based extraction using dependency parsing
        if self.nlp:
            nlp_steps = self._extract_nlp_based_steps(text)
            steps.extend(nlp_steps)

        # Method 3: Sentence-based reasoning detection
        sentence_steps = self._extract_sentence_based_steps(text)
        steps.extend(sentence_steps)

        # Remove duplicates and merge similar steps
        steps = self._deduplicate_steps(steps)

        return steps

    def _extract_pattern_based_steps(self, text: str) -> List[Dict]:
        """Extract steps using improved pattern matching"""
        steps = []
        lines = text.split('\n')

        # Enhanced patterns for step detection
        patterns = [
            r'^\s*(\d+)[.)]\s*(.*)',  # 1. Step
            r'^\s*Step\s+(\d+)[:.]\s*(.*)',  # Step 1:
            r'^\s*[•*-]\s*(.*)',  # bullet Step
            r'^\s*(First|Second|Third|Fourth|Fifth|Finally|Lastly),\s*(.*)',  # First, Second, etc.
            r'^\s*(Initially|Subsequently|Then|Next|After that),\s*(.*)',  # Sequential markers
            r'^\s*(Therefore|Thus|Hence|Consequently),\s*(.*)',  # Logical connectors
        ]

        for line in lines:
            line = line.strip()
            if not line:
                continue

            for i, pattern in enumerate(patterns):
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    if len(match.groups()) == 2:
                        step_num, step_text = match.groups()
                        if isinstance(step_num, str) and not step_num.isdigit():
                            step_num = len(steps) + 1
                    else:
                        step_text = match.group(1)
                        step_num = len(steps) + 1

                    steps.append({
                        "number": step_num,
                        "text": step_text,
                        "type": "pattern_based",
                        "pattern_type": ["numbered", "explicit", "bulleted", "sequential", "temporal", "logical"][i],
                        "original_line": line
                    })
                    break

        return steps

    def _extract_nlp_based_steps(self, text: str) -> List[Dict]:
        """Extract steps using NLP dependency parsing"""
        if not self.nlp:
            return []

        steps = []
        doc = self.nlp(text)

        for i, sent in enumerate(doc.sents):
            # Check if sentence contains reasoning markers
            reasoning_type = self._classify_reasoning_type(sent.text)

            if reasoning_type:
                # Analyze the sentence structure
                root = [token for token in sent if token.dep_ == "ROOT"][0] if any(token.dep_ == "ROOT" for token in sent) else None

                if root:
                    steps.append({
                        "number": i + 1,
                        "text": sent.text,
                        "type": "nlp_based",
                        "reasoning_type": reasoning_type,
                        "root_verb": root.text,
                        "root_pos": root.pos_,
                        "dependencies": [(token.text, token.dep_, token.head.text) for token in sent]
                    })

        return steps

    def _extract_sentence_based_steps(self, text: str) -> List[Dict]:
        """Extract potential reasoning steps from individual sentences"""
        steps = []

        # Split text into sentences
        if self.nlp:
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sents]
        else:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

        for i, sentence in enumerate(sentences):
            # Check if sentence shows reasoning characteristics
            reasoning_score = self._score_reasoning_characteristics(sentence)

            if reasoning_score > 0.3:  # Threshold for considering it a reasoning step
                steps.append({
                    "number": i + 1,
                    "text": sentence,
                    "type": "sentence_based",
                    "reasoning_score": reasoning_score,
                    "characteristics": self._analyze_sentence_characteristics(sentence)
                })

        return steps

    def _classify_reasoning_type(self, text: str) -> Optional[str]:
        """Classify the type of reasoning in a sentence"""
        text_lower = text.lower()

        for reasoning_type, markers in self.reasoning_markers.items():
            if any(marker in text_lower for marker in markers):
                return reasoning_type

        return None

    def _score_reasoning_characteristics(self, sentence: str) -> float:
        """Score how much a sentence exhibits reasoning characteristics"""
        score = 0.0
        sentence_lower = sentence.lower()

        # Check for reasoning markers
        for reasoning_type, markers in self.reasoning_markers.items():
            if any(marker in sentence_lower for marker in markers):
                score += 0.3
                break

        # Check for logical structures
        logical_patterns = [
            r'if\s+.+\s+then',
            r'given\s+.+\s+we\s+can',
            r'since\s+.+\s+therefore',
            r'because\s+.+\s+thus'
        ]

        for pattern in logical_patterns:
            if re.search(pattern, sentence_lower):
                score += 0.4
                break

        # Check for analytical language
        analytical_words = ["analyze", "conclude", "infer", "deduce", "assume", "hypothesis", "evidence", "supports", "indicates"]
        if any(word in sentence_lower for word in analytical_words):
            score += 0.2

        # Check for quantitative reasoning
        if re.search(r'\d+', sentence) and any(word in sentence_lower for word in ["percent", "ratio", "proportion", "increase", "decrease"]):
            score += 0.2

        return min(1.0, score)

    def _analyze_sentence_characteristics(self, sentence: str) -> Dict:
        """Analyze characteristics of a sentence that indicate reasoning"""
        characteristics = {
            "has_reasoning_markers": False,
            "reasoning_types": [],
            "has_logical_structure": False,
            "has_quantitative_elements": False,
            "complexity_score": 0
        }

        sentence_lower = sentence.lower()

        # Check reasoning markers
        for reasoning_type, markers in self.reasoning_markers.items():
            if any(marker in sentence_lower for marker in markers):
                characteristics["has_reasoning_markers"] = True
                characteristics["reasoning_types"].append(reasoning_type)

        # Check logical structure
        logical_patterns = [r'if\s+.+\s+then', r'either\s+.+\s+or', r'not\s+only\s+.+\s+but\s+also']
        characteristics["has_logical_structure"] = any(re.search(pattern, sentence_lower) for pattern in logical_patterns)

        # Check quantitative elements
        characteristics["has_quantitative_elements"] = bool(re.search(r'\d+', sentence))

        # Calculate complexity (based on sentence length and structure)
        characteristics["complexity_score"] = min(1.0, len(sentence.split()) / 20.0)

        return characteristics

    def _deduplicate_steps(self, steps: List[Dict]) -> List[Dict]:
        """Remove duplicate and highly similar steps"""
        if not steps:
            return steps

        # Get embeddings for all step texts
        step_texts = [step["text"] for step in steps]
        embeddings = self.sentence_model.encode(step_texts)

        # Find duplicates using similarity threshold
        to_remove = set()
        for i, emb1 in enumerate(embeddings):
            for j, emb2 in enumerate(embeddings[i+1:], i+1):
                similarity = cosine_similarity([emb1], [emb2])[0][0]
                if similarity > 0.9:  # Very similar
                    # Keep the more detailed one
                    if len(steps[i]["text"]) >= len(steps[j]["text"]):
                        to_remove.add(j)
                    else:
                        to_remove.add(i)

        # Return non-duplicate steps
        return [step for i, step in enumerate(steps) if i not in to_remove]

    def _analyze_step_quality(self, steps: List[Dict], full_text: str) -> Dict:
        """Analyze the quality of individual reasoning steps"""
        if not steps:
            return {"average_quality": 0, "step_scores": []}

        step_scores = []

        for step in steps:
            quality_score = 0.0

            # Length and detail score (not too short, not too long)
            text_length = len(step["text"].split())
            if 5 <= text_length <= 30:
                quality_score += 0.3
            elif text_length > 30:
                quality_score += 0.2
            else:
                quality_score += 0.1

            # Reasoning marker presence
            if self._classify_reasoning_type(step["text"]):
                quality_score += 0.3

            # Specificity (presence of concrete details)
            if re.search(r'\d+', step["text"]) or any(word in step["text"].lower() for word in ["specific", "example", "such as", "namely"]):
                quality_score += 0.2

            # Logical structure
            logical_words = ["therefore", "because", "since", "if", "then", "given", "assuming"]
            if any(word in step["text"].lower() for word in logical_words):
                quality_score += 0.2

            step_scores.append(min(1.0, quality_score))

        return {
            "average_quality": sum(step_scores) / len(step_scores),
            "step_scores": step_scores,
            "quality_distribution": {
                "high_quality": sum(1 for score in step_scores if score >= 0.7),
                "medium_quality": sum(1 for score in step_scores if 0.4 <= score < 0.7),
                "low_quality": sum(1 for score in step_scores if score < 0.4)
            }
        }

    def _analyze_step_coherence(self, steps: List[Dict]) -> Dict:
        """Analyze coherence between reasoning steps using semantic similarity"""
        if len(steps) <= 1:
            return {"overall_coherence": 1.0, "pairwise_coherences": []}

        # Get embeddings for step texts
        step_texts = [step["text"] for step in steps]
        embeddings = self.sentence_model.encode(step_texts)

        # Calculate pairwise coherences
        coherences = []
        for i in range(len(embeddings) - 1):
            similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            coherences.append(float(similarity))

        # Check for logical connectors between adjacent steps
        connector_scores = []
        for i in range(len(steps) - 1):
            current_step = steps[i]["text"].lower()
            next_step = steps[i + 1]["text"].lower()

            # Check if next step has logical connectors referring to current step
            connectors = ["therefore", "thus", "hence", "so", "consequently", "as a result", "this means", "this shows"]
            has_connector = any(conn in next_step for conn in connectors)
            connector_scores.append(1.0 if has_connector else 0.5)

        # Combine semantic and logical coherence
        combined_coherences = []
        for i in range(len(coherences)):
            combined = (coherences[i] * 0.7) + (connector_scores[i] * 0.3)
            combined_coherences.append(combined)

        overall_coherence = sum(combined_coherences) / len(combined_coherences) if combined_coherences else 0

        return {
            "overall_coherence": overall_coherence,
            "semantic_coherences": coherences,
            "logical_coherences": connector_scores,
            "combined_coherences": combined_coherences,
            "coherence_statistics": {
                "mean": overall_coherence,
                "min": min(combined_coherences) if combined_coherences else 0,
                "max": max(combined_coherences) if combined_coherences else 0
            }
        }

    def _analyze_completeness(self, steps: List[Dict], full_text: str) -> Dict:
        """Analyze completeness of the reasoning chain"""
        # Calculate coverage of steps in the full text
        step_text = " ".join([step["text"] for step in steps])

        # Remove step markers from full text for fair comparison
        cleaned_full_text = re.sub(r'\d+[.)]|\s*[•*-]\s*|Step\s+\d+[:.]\s*|First,\s*|Second,\s*|Third,\s*|Finally,\s*', '', full_text)

        # Calculate word coverage
        full_words = set(re.findall(r'\b\w+\b', cleaned_full_text.lower()))
        step_words = set(re.findall(r'\b\w+\b', step_text.lower()))

        if not full_words:
            word_coverage = 0.0
        else:
            word_coverage = len(step_words.intersection(full_words)) / len(full_words)

        # Estimate expected number of steps based on text complexity
        sentences = re.split(r'[.!?]+', cleaned_full_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Heuristic: expect one reasoning step per 2-3 sentences for complex reasoning
        expected_steps = max(1, len(sentences) // 3)

        # Calculate completeness metrics
        step_ratio = len(steps) / expected_steps if expected_steps > 0 else 0
        step_completeness = min(1.0, step_ratio)

        # Overall completeness combines coverage and step ratio
        completeness_score = (word_coverage * 0.4) + (step_completeness * 0.6)

        return {
            "completeness_score": completeness_score,
            "word_coverage": word_coverage,
            "step_completeness": step_completeness,
            "actual_steps": len(steps),
            "expected_steps": expected_steps,
            "step_ratio": step_ratio,
            "total_sentences": len(sentences)
        }

    def _calculate_overall_score(self, step_analysis: Dict, coherence_analysis: Dict, completeness_analysis: Dict) -> float:
        """Calculate overall reasoning step score"""
        quality_score = step_analysis.get("average_quality", 0)
        coherence_score = coherence_analysis.get("overall_coherence", 0)
        completeness_score = completeness_analysis.get("completeness_score", 0)

        # Weighted combination
        overall_score = (quality_score * 0.4) + (coherence_score * 0.4) + (completeness_score * 0.2)

        return min(1.0, max(0.0, overall_score))


# Backward compatibility
EnhancedReasoningStepMetric = ReasoningStepMetric
