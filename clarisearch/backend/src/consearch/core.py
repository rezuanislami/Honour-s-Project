"""Core retrieval, confidence estimation, and routing logic."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class Document:
    id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] | None = None


@dataclass
class SearchHit:
    document: Document
    score: float
    sparse_score: float
    dense_score: float


@dataclass
class RouteResult:
    route: str
    confidence: float
    threshold: float
    clarification_question: str | None
    hits: list[SearchHit]
    reason: str


class HybridRetriever:
    """BM25 + SentenceTransformer retrieval with optional FAISS acceleration.

    Dense-model weights are downloaded by sentence-transformers on first use. Set
    CONSEARCH_DENSE_MODEL to use a different compatible model. If dense setup fails,
    the retriever continues in sparse-only mode and reports the failure in health output.
    """

    def __init__(
        self,
        documents: list[Document],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        enable_dense: bool = True,
    ):
        if not documents:
            raise ValueError("At least one document is required")
        self.documents = documents
        self._tokens = [tokenize(f"{d.title} {d.text}") for d in documents]
        self.bm25 = BM25Okapi(self._tokens)
        self.model_name = model_name
        self.encoder = None
        self.index = None
        self.dense_error: str | None = None
        try:
            if not enable_dense:
                raise RuntimeError("Dense retrieval disabled by configuration")
            import faiss
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(model_name)
            vectors = self.encoder.encode(
                [self._content(d) for d in documents], normalize_embeddings=True
            )
            vectors = np.asarray(vectors, dtype="float32")
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)
        except Exception as exc:  # noqa: BLE001 - dense setup errors intentionally fall back to BM25
            self.dense_error = f"{type(exc).__name__}: {exc}"
            self.encoder = None
            self.index = None

    @staticmethod
    def _content(document: Document) -> str:
        return f"{document.title}\n{document.text}".strip()

    def search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        n = len(self.documents)
        k = min(max(1, top_k), n)
        sparse = np.asarray(self.bm25.get_scores(tokenize(query)), dtype="float32")
        sparse = self._minmax(sparse)
        dense = np.zeros(n, dtype="float32")
        if self.encoder is not None and self.index is not None:
            vector = np.asarray(
                self.encoder.encode([query], normalize_embeddings=True), dtype="float32"
            )
            scores, ids = self.index.search(vector, n)
            raw = np.zeros(n, dtype="float32")
            raw[ids[0]] = scores[0]
            dense = self._minmax(raw)
        effective_alpha = alpha if self.encoder is not None else 1.0
        fused = effective_alpha * sparse + (1.0 - effective_alpha) * dense
        order = np.argsort(-fused)[:k]
        return [
            SearchHit(self.documents[int(i)], float(fused[i]), float(sparse[i]), float(dense[i]))
            for i in order
        ]

    @staticmethod
    def _minmax(values: np.ndarray) -> np.ndarray:
        low, high = float(np.min(values)), float(np.max(values))
        if math.isclose(low, high):
            return np.zeros_like(values)
        return (values - low) / (high - low)


class RetrievalDensityEstimator:
    """Turn-1 confidence heuristic derived from top-score strength and separation.

    This is a baseline calibration heuristic, not a calibrated probability. Tune its
    parameters using validation data before drawing research conclusions.
    """

    def __init__(self, threshold: float = 0.35, top_weight: float = 0.7):
        if not 0 <= threshold <= 1 or not 0 <= top_weight <= 1:
            raise ValueError("threshold and top_weight must be between 0 and 1")
        self.threshold = threshold
        self.top_weight = top_weight

    def estimate(self, hits: list[SearchHit]) -> float:
        if not hits:
            return 0.0
        top = max(0.0, min(1.0, hits[0].score))
        runner_up = hits[1].score if len(hits) > 1 else 0.0
        separation = max(0.0, min(1.0, top - runner_up))
        return self.top_weight * top + (1.0 - self.top_weight) * separation

    def route(self, query: str, hits: list[SearchHit], clarifier=None) -> RouteResult:
        confidence = self.estimate(hits)
        if confidence >= self.threshold:
            question = clarifier(query, hits) if clarifier else None
            question = question or self.default_question(query)
            return RouteResult(
                "clarify",
                confidence,
                self.threshold,
                question,
                hits,
                "retrieval confidence meets configured threshold",
            )
        return RouteResult(
            "retrieve",
            confidence,
            self.threshold,
            None,
            hits,
            "low retrieval confidence; abstain from clarification and retrieve directly",
        )

    @staticmethod
    def default_question(query: str) -> str:
        return f"Could you clarify what you mean by ‘{query.strip()}’ so I can narrow the search?"
