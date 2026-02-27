"""BERTScore baseline for semantic similarity evaluation."""
import torch
from typing import Dict, List

try:
    from bert_score import score as bert_score_fn
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False


class BERTScoreBaseline:
    """BERT-based semantic similarity using pre-trained models."""

    def __init__(self, model_type: str = "microsoft/deberta-xlarge-mnli"):
        self.name = "bert_score"
        self.model_type = model_type
        self.device = self._get_device()

    def _get_device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def evaluate(self, prediction: str, reference: str, **kwargs) -> Dict:
        """Evaluate a single prediction against a reference using BERTScore.

        Returns dict with precision, recall, f1.  If bert-score is not
        installed the scores are 0.0 and an ``error`` key is included.
        """
        if not BERT_SCORE_AVAILABLE:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "error": "bert-score not installed"}

        P, R, F1 = bert_score_fn(
            [prediction], [reference],
            model_type=self.model_type,
            device=self.device,
            verbose=False,
        )
        return {
            "precision": P[0].item(),
            "recall": R[0].item(),
            "f1": F1[0].item(),
        }

    def batch_evaluate(self, predictions: List[str], references: List[str], **kwargs) -> Dict:
        """Evaluate multiple predictions at once (more efficient than one-by-one).

        Returns dict with lists of precision, recall, f1 values.
        """
        if not BERT_SCORE_AVAILABLE:
            return {"precision": [], "recall": [], "f1": [], "error": "bert-score not installed"}

        P, R, F1 = bert_score_fn(
            predictions, references,
            model_type=self.model_type,
            device=self.device,
            verbose=False,
        )
        return {
            "precision": [p.item() for p in P],
            "recall": [r.item() for r in R],
            "f1": [f.item() for f in F1],
        }
