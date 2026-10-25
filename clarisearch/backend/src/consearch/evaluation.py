
"""Offline information-retrieval evaluation.

Topics JSONL: query_id, query.
Qrels JSONL: query_id, doc_id, relevance.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from consearch.core import HybridRetriever


def dcg(relevances: list[int], k: int) -> float:
    """Calculate discounted cumulative gain at cutoff k."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if any(rel < 0 for rel in relevances):
        raise ValueError("Relevance values must be non-negative")

    return sum(
        (2**rel - 1) / math.log2(rank + 1)
        for rank, rel in enumerate(relevances[:k], start=1)
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON Lines and validate that each row is an object."""
    rows = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue

        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {path} at line {line_number}"
            ) from exc

        if not isinstance(row, dict):
            raise ValueError(
                f"Expected a JSON object in {path} at line {line_number}"
            )

        rows.append(row)

    return rows


def evaluate(
    retriever: HybridRetriever,
    topics_path: Path,
    qrels_path: Path,
    top_k: int = 100,
) -> dict[str, int | float]:
    """Evaluate retrieval using query-level relevance judgements."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    topics = _read_jsonl(topics_path)
    qrel_rows = _read_jsonl(qrels_path)

    # Validate topics and ensure each query ID is unique.
    seen_queries: set[str] = set()

    for row in topics:
        if "query_id" not in row or "query" not in row:
            raise ValueError("Each topic must contain query_id and query")

        qid = str(row["query_id"])
        if qid in seen_queries:
            raise ValueError(f"Duplicate topic query_id: {qid}")
        seen_queries.add(qid)

        if not isinstance(row["query"], str) or not row["query"].strip():
            raise ValueError(f"Empty query for query_id: {qid}")

    # Validate qrels and reject duplicate query-document pairs.
    qrels: dict[str, dict[str, int]] = {}

    for row in qrel_rows:
        if not {"query_id", "doc_id", "relevance"}.issubset(row):
            raise ValueError(
                "Each qrel must contain query_id, doc_id, and relevance"
            )

        qid = str(row["query_id"])
        doc_id = str(row["doc_id"])

        try:
            relevance = int(row["relevance"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid relevance for query {qid}, document {doc_id}"
            ) from exc

        if relevance < 0:
            raise ValueError("Relevance values must be non-negative")

        if doc_id in qrels.setdefault(qid, {}):
            raise ValueError(
                f"Duplicate qrel for query {qid}, document {doc_id}"
            )

        qrels[qid][doc_id] = relevance

    # Require at least one judgement set for every evaluated query.
    for topic in topics:
        qid = str(topic["query_id"])
        if qid not in qrels:
            raise ValueError(f"Missing relevance judgements for query {qid}")

    metric_names = (
        "P@1",
        "P@5",
        "P@10",
        "MRR",
        "nDCG@5",
        "nDCG@10",
        f"Recall@{top_k}",
    )
    metrics: dict[str, list[float]] = {name: [] for name in metric_names}

    for topic in topics:
        qid = str(topic["query_id"])
        relevant = qrels[qid]

        hits = retriever.search(topic["query"], top_k=top_k)
        ranked = [str(hit.document.id) for hit in hits]
        rels = [relevant.get(doc_id, 0) for doc_id in ranked]

        # Precision uses fixed cutoff denominators, even for short result lists.
        for k in (1, 5, 10):
            metrics[f"P@{k}"].append(
                sum(rel > 0 for rel in rels[:k]) / k
            )

        metrics["MRR"].append(
            next(
                (
                    1.0 / rank
                    for rank, rel in enumerate(rels, start=1)
                    if rel > 0
                ),
                0.0,
            )
        )

        ideal = sorted(relevant.values(), reverse=True)

        for k in (5, 10):
            ideal_dcg = dcg(ideal, k)
            actual_dcg = dcg(rels, k)
            metrics[f"nDCG@{k}"].append(
                actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            )

        total_relevant = sum(value > 0 for value in relevant.values())
        retrieved_relevant = sum(rel > 0 for rel in rels[:top_k])

        metrics[f"Recall@{top_k}"].append(
            retrieved_relevant / total_relevant
            if total_relevant > 0
            else 0.0
        )

    return {
        "num_queries": len(topics),
        **{
            name: sum(values) / len(values) if values else 0.0
            for name, values in metrics.items()
        },
    }
