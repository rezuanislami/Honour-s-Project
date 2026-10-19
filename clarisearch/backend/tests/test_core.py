import json
from pathlib import Path

import pytest

from consearch.core import Document, HybridRetriever, RetrievalDensityEstimator, SearchHit
from consearch.evaluation import evaluate


@pytest.fixture
def retriever():
    docs = [
        Document("a", "Matariki is a Māori public holiday in Aotearoa", "Matariki"),
        Document("b", "BM25 and dense embeddings combine in hybrid search", "Hybrid retrieval"),
        Document("c", "A clarification question resolves ambiguous search intent", "Clarification"),
    ]
    return HybridRetriever(docs, enable_dense=False)


def test_search_returns_ranked_hits(retriever):
    hits = retriever.search("Matariki public holiday", top_k=2)
    assert len(hits) == 2
    assert hits[0].document.id == "a"
    assert hits[0].score >= hits[1].score


def test_sparse_only_fallback(retriever):
    assert retriever.encoder is None
    assert retriever.search("hybrid search")[0].document.id == "b"


def test_empty_query_is_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.search("   ")


def test_threshold_routes_low_confidence_to_retrieval():
    estimator = RetrievalDensityEstimator(threshold=0.6)
    hit = SearchHit(Document("x", "text"), 0.1, 0.1, 0)
    result = estimator.route("obscure term", [hit])
    assert result.route == "retrieve"
    assert result.clarification_question is None


def test_confident_route_asks_question():
    estimator = RetrievalDensityEstimator(threshold=0.3)
    hit = SearchHit(Document("x", "text"), 0.9, 0.9, 0.9)
    result = estimator.route("jaguar", [hit])
    assert result.route == "clarify"
    assert result.clarification_question


def test_evaluation_metrics(tmp_path: Path, retriever):
    topics = tmp_path / "topics.jsonl"
    qrels = tmp_path / "qrels.jsonl"
    topics.write_text(json.dumps({"query_id": "q", "query": "Matariki holiday"}) + "\n")
    qrels.write_text(json.dumps({"query_id": "q", "doc_id": "a", "relevance": 2}) + "\n")
    result = evaluate(retriever, topics, qrels, top_k=3)
    assert result["P@1"] == 1
    assert result["MRR"] == 1
    assert result["nDCG@5"] == 1
    assert result["Recall@100"] == 1
