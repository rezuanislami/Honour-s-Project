"""Offline information-retrieval evaluation. Qrels JSONL: query_id, doc_id, relevance."""

from __future__ import annotations

import json
import math
from pathlib import Path

from consearch.core import HybridRetriever


def dcg(relevances: list[int], k: int) -> float:
    return sum((2**rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(relevances[:k], 1))


def evaluate(
    retriever: HybridRetriever, topics_path: Path, qrels_path: Path, top_k: int = 100
) -> dict:
    topics = [
        json.loads(line)
        for line in topics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    qrels: dict[str, dict[str, int]] = {}
    for line in qrels_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            qrels.setdefault(str(row["query_id"]), {})[str(row["doc_id"])] = int(row["relevance"])
    metrics = {
        name: [] for name in ("P@1", "P@5", "P@10", "MRR", "nDCG@5", "nDCG@10", "Recall@100")
    }
    for topic in topics:
        qid = str(topic["query_id"])
        relevant = qrels.get(qid, {})
        ranked = [hit.document.id for hit in retriever.search(topic["query"], top_k=top_k)]
        rels = [relevant.get(doc_id, 0) for doc_id in ranked]
        for k in (1, 5, 10):
            metrics[f"P@{k}"].append(sum(rel > 0 for rel in rels[:k]) / k)
        metrics["MRR"].append(
            next((1.0 / rank for rank, rel in enumerate(rels, 1) if rel > 0), 0.0)
        )
        ideal = sorted(relevant.values(), reverse=True)
        for k in (5, 10):
            ideal_dcg = dcg(ideal, k)
            metrics[f"nDCG@{k}"].append(dcg(rels, k) / ideal_dcg if ideal_dcg else 0.0)
        total = sum(value > 0 for value in relevant.values())
        metrics["Recall@100"].append(sum(rel > 0 for rel in rels[:100]) / total if total else 0.0)
    return {
        "num_queries": len(topics),
        **{key: sum(vals) / len(vals) if vals else 0.0 for key, vals in metrics.items()},
    }
