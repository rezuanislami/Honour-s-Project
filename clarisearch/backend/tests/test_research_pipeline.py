import json
from pathlib import Path

import pytest

from consearch.cli import assert_no_topic_leakage
from consearch.cnp import CNPModel, classification_metrics, retrieval_features
from consearch.core import Document, HybridRetriever
from consearch.datasets import load_ambignq_json, load_clariq_tsv
from consearch.policies import evaluate_policy_suite


def test_clariq_deduplicates_topics_and_preserves_severity(tmp_path: Path):
    file = tmp_path / "train.tsv"
    file.write_text(
        "topic_id\tinitial_request\tclarification_need\tquestion\n"
        "1\tJaguar\t3\tWhich one?\n"
        "1\tJaguar\t3\tWhat context?\n"
        "2\tWeather today\t1\t\n",
        encoding="utf-8",
    )
    rows = load_clariq_tsv(file, "train")
    assert len(rows) == 2
    assert rows[0]["severity"] == 3 and rows[0]["label"] == 1
    assert rows[1]["severity"] == 1 and rows[1]["label"] == 0


def test_clariq_conflicting_topic_labels_error(tmp_path: Path):
    file = tmp_path / "bad.tsv"
    file.write_text("topic_id\tinitial_request\tclarification_need\n1\tJaguar\t1\n1\tJaguar\t4\n")
    with pytest.raises(ValueError, match="Conflicting"):
        load_clariq_tsv(file, "train")


def test_ambignq_labels_are_derived_and_marked(tmp_path: Path):
    file = tmp_path / "dev.json"
    file.write_text(
        json.dumps(
            [
                {
                    "id": "1",
                    "question": "Who is Jordan?",
                    "annotations": [{"type": "multipleQAs", "qaPairs": []}],
                },
                {
                    "id": "2",
                    "question": "When was X born?",
                    "annotations": [{"type": "singleAnswer", "answer": ["2000"]}],
                },
            ]
        )
    )
    rows = load_ambignq_json(file, "dev")
    assert [row["label"] for row in rows] == [1, 0]
    assert all("derived" in row["label_provenance"] for row in rows)


def test_query_and_retrieval_cnp_models_train_and_score():
    queries = ["Jaguar animal", "Jaguar car", "What is 2 plus 2", "Capital of France"]
    labels = [1, 1, 0, 0]
    query_model = CNPModel("query_only").fit(queries, labels)
    scores = query_model.predict_proba(["Jaguar"])
    assert len(scores) == 1 and 0 <= scores[0] <= 1
    docs = [Document("d1", "Jaguar is a big cat"), Document("d2", "Jaguar is a car brand")]
    retriever = HybridRetriever(docs, enable_dense=False)
    evidence = [retrieval_features(retriever.search(query)) for query in queries]
    retrieval_model = CNPModel("retrieval_informed").fit(queries, labels, evidence)
    assert 0 <= retrieval_model.predict_proba(["Jaguar"], [evidence[0]])[0] <= 1


def test_policy_suite_has_four_policies_and_cost_fields():
    result = evaluate_policy_suite([0, 1], [0.1, 0.9], [0.2, 0.8], threshold=0.5)
    assert set(result["policies"]) == {
        "direct_retrieval",
        "always_clarify",
        "query_only_cnp",
        "retrieval_informed_cnp",
    }
    assert result["policies"]["query_only_cnp"]["macro_f1"] == 1.0
    assert result["policies"]["always_clarify"]["false_clarification_rate"] == 1.0


def test_cnp_metrics_include_calibration():
    metrics = classification_metrics([0, 1], [0.1, 0.9])
    assert metrics["auroc"] == 1.0
    assert metrics["brier_score"] >= 0
    assert metrics["expected_calibration_error_10_bins"] >= 0


def test_split_leakage_guard():
    with pytest.raises(ValueError, match="overlap"):
        assert_no_topic_leakage(
            [{"id": "1", "source": "clariq"}], [{"id": "1", "source": "clariq"}]
        )
    assert_no_topic_leakage([{"id": "1", "source": "clariq"}], [{"id": "1", "source": "ambignq"}])
