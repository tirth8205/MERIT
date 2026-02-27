"""G-Eval baseline: LLM-based evaluation with chain-of-thought.

Reference: Liu et al., 2023 - "G-Eval: NLG Evaluation using GPT-4 with Chain-of-Thought"
"""
from typing import Dict
import re


class GEvalBaseline:
    """G-Eval: NLG evaluation using LLM with chain-of-thought.

    Supports Anthropic (Claude) and Ollama providers.
    """

    def __init__(self, provider: str = "anthropic", model: str = "claude-sonnet-4-20250514"):
        self.name = "geval"
        self.provider = provider
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None and self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def _call_model(self, prompt: str) -> str:
        """Call the evaluation model."""
        if self.provider == "anthropic":
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=300,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        elif self.provider == "ollama":
            import requests
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
            return resp.json()["response"]
        raise ValueError(f"Unknown provider: {self.provider}")

    def evaluate(self, prediction: str, reference: str, **kwargs) -> Dict:
        """Evaluate prediction quality compared to reference using G-Eval approach.

        Uses chain-of-thought prompting to score the prediction on a 1-5 scale,
        then normalizes to [0, 1].

        Returns dict with ``score`` (normalized), ``raw_score`` (1-5), and
        ``explanation`` (full model output).
        """
        prompt = f"""You are evaluating the quality of a generated response compared to a reference answer.

**Generated Response:**
{prediction}

**Reference Answer:**
{reference}

Think step by step about the quality of the generated response:
1. Does it capture the key information from the reference?
2. Is it factually consistent with the reference?
3. Is it well-written and coherent?

After your analysis, provide a final score from 1 to 5:
1 = Poor quality, misses key information
2 = Below average, captures some information
3 = Average, captures most key information
4 = Good quality, accurate and coherent
5 = Excellent, fully captures reference with good writing

Output your final score as: Score: <number>"""

        text = self._call_model(prompt)
        match = re.search(r"Score:\s*(\d)", text)
        raw_score = int(match.group(1)) if match else 3
        normalized_score = (raw_score - 1) / 4.0

        return {
            "score": normalized_score,
            "raw_score": raw_score,
            "explanation": text,
        }
