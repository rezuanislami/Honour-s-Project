"""Supervised CNP baselines and retrieval-evidence feature extraction."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from consearch.core import SearchHit

FEATURE_NAMES = (
    "top_score",
    "top_margin",
    "score_mean",
    "score_std",
    "score_entropy",
    "sparse_dense_top_overlap",
    "rank_order_agreement",
    "dense_top_score",
    "sparse_top_score",
)


def retrieval_features(hits: Sequence[SearchHit]) -> dict[str, float]:
    """Summarize top-k strength, dispersion and sparse/dense agreement."""
    if not hits:
        return {name: 0.0 for name in FEATURE_NAMES}
    fused = np.asarray([max(0.0, h.score) for h in hits], dtype=float)
    sparse = np.asarray([max(0.0, h.sparse_score) for h in hits], dtype=float)
    dense = np.asarray([max(0.0, h.dense_score) for h in hits], dtype=float)
    probs = fused / fused.sum() if fused.sum() > 0 else np.full(len(fused), 1 / len(fused))
    entropy = float(-(probs * np.log(probs + 1e-12)).sum() / np.log(max(2, len(probs))))
    sparse_order = np.argsort(-sparse)
    dense_order = np.argsort(-dense)
    top_n = min(3, len(hits))
    sparse_top = set(sparse_order[:top_n].tolist())
    dense_top = set(dense_order[:top_n].tolist())
    overlap = len(sparse_top & dense_top) / max(1, len(sparse_top | dense_top))
    agreements = sum(int(sparse_order[i] == dense_order[i]) for i in range(len(hits))) / len(hits)
    return {
        "top_score": float(fused[0]),
        "top_margin": float(fused[0] - fused[1]) if len(fused) > 1 else float(fused[0]),
        "score_mean": float(fused.mean()),
        "score_std": float(fused.std()),
        "score_entropy": entropy,
        "sparse_dense_top_overlap": float(overlap),
        "rank_order_agreement": float(agreements),
        "dense_top_score": float(dense.max()),
        "sparse_top_score": float(sparse.max()),
    }


@dataclass
class CNPModel:
    """TF-IDF logistic regression; retrieval feature columns are optional by mode."""

    mode: str = "retrieval_informed"
    ngram_max: int = 2
    C: float = 2.0

    def __post_init__(self):
        if self.mode not in {"query_only", "retrieval_informed"}:
            raise ValueError("mode must be query_only or retrieval_informed")
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, self.ngram_max), min_df=1, sublinear_tf=True, strip_accents="unicode"
        )
        self.classifier = LogisticRegression(
            C=self.C, class_weight="balanced", max_iter=1000, random_state=42
        )
        self._fitted = False

    def _matrix(
        self, queries: Sequence[str], signals: Sequence[dict[str, float]] | None, fit: bool
    ):
        text = self.vectorizer.fit_transform(queries) if fit else self.vectorizer.transform(queries)
        if self.mode == "query_only":
            return text
        signals = signals or [{} for _ in queries]
        numeric = np.asarray(
            [[float(row.get(key, 0.0)) for key in FEATURE_NAMES] for row in signals],
            dtype="float32",
        )
        return hstack([text, csr_matrix(numeric)], format="csr")

    def fit(
        self,
        queries: Sequence[str],
        labels: Sequence[int],
        signals: Sequence[dict[str, float]] | None = None,
    ):
        labels = np.asarray(labels, dtype=int)
        if len(set(labels.tolist())) < 2:
            raise ValueError("Training requires examples from both classes (0 and 1)")
        self.classifier.fit(self._matrix(queries, signals, fit=True), labels)
        self._fitted = True
        return self

    def predict_proba(
        self, queries: Sequence[str], signals: Sequence[dict[str, float]] | None = None
    ) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("CNP model has not been trained")
        return self.classifier.predict_proba(self._matrix(queries, signals, fit=False))[:, 1]

    def predict(
        self,
        queries: Sequence[str],
        signals: Sequence[dict[str, float]] | None = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        return (self.predict_proba(queries, signals) >= threshold).astype(int)

    def save(self, path: Path) -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save an untrained CNP model")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> CNPModel:
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Artifact at {path} is not a CNPModel")
        return model


def classification_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float = 0.5
) -> dict:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(scores, dtype=float)
    predicted = (p >= threshold).astype(int)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lower, upper in itertools.pairwise(bins):
        mask = (p >= lower) & (p < upper if upper < 1 else p <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return {
        "n": len(y),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, predicted, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y, predicted)),
        "auroc": float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else None,
        "brier_score": float(brier_score_loss(y, p)),
        "expected_calibration_error_10_bins": float(ece),
        "threshold": float(threshold),
    }
