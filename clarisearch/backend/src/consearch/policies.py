"""Evaluation of the four routing policies specified in the literature review."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix

from consearch.cnp import classification_metrics

POLICIES = ("direct_retrieval", "always_clarify", "query_only_cnp", "retrieval_informed_cnp")


def evaluate_policy_suite(
    labels: Sequence[int],
    query_only_scores: Sequence[float],
    retrieval_informed_scores: Sequence[float],
    threshold: float = 0.5,
    direct_ndcg: Sequence[float] | None = None,
    clarified_ndcg: Sequence[float] | None = None,
    clarify_cost: float = 0.1,
) -> dict:
    """Compare four policies using one target set and a consistent threshold.

    `labels` is a binary clarification-need target. For AmbigNQ, it must be described
    as derived weak labels, not human CNP labels. Supply paired per-query nDCG@5 values
    to compute utility; absent values are never fabricated.
    """
    y = np.asarray(labels, dtype=int)
    q = np.asarray(query_only_scores, dtype=float)
    r = np.asarray(retrieval_informed_scores, dtype=float)
    if not (len(y) == len(q) == len(r)):
        raise ValueError("labels and both score arrays must have equal lengths")
    if len(y) == 0 or not set(y.tolist()).issubset({0, 1}):
        raise ValueError("labels must be a non-empty binary array")
    predictions = {
        "direct_retrieval": np.zeros(len(y), dtype=int),
        "always_clarify": np.ones(len(y), dtype=int),
        "query_only_cnp": (q >= threshold).astype(int),
        "retrieval_informed_cnp": (r >= threshold).astype(int),
    }
    utility_inputs = direct_ndcg is not None and clarified_ndcg is not None
    if utility_inputs and not (len(direct_ndcg) == len(y) == len(clarified_ndcg)):
        raise ValueError("nDCG arrays must have the same length as labels")
    reports = {}
    for name, predicted in predictions.items():
        scores = (
            predicted.astype(float)
            if name in {"direct_retrieval", "always_clarify"}
            else (q if name == "query_only_cnp" else r)
        )
        report = classification_metrics(y, scores, threshold)
        tn, fp, fn, _tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        report.update(
            {
                "false_clarification_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
                "clarification_coverage": float(predicted.mean()),
                "unnecessary_clarification_count": int(fp),
                "missed_clarification_count": int(fn),
            }
        )
        if utility_inputs:
            direct = np.asarray(direct_ndcg, dtype=float)
            clarified = np.asarray(clarified_ndcg, dtype=float)
            outcomes = np.where(predicted == 1, clarified, direct)
            report["mean_ndcg5"] = float(outcomes.mean())
            report["mean_clarify_cost"] = float(predicted.mean() * clarify_cost)
            report["net_utility"] = float(outcomes.mean() - predicted.mean() * clarify_cost)
        else:
            report.update({"mean_ndcg5": None, "mean_clarify_cost": None, "net_utility": None})
        reports[name] = report
    return {
        "threshold": float(threshold),
        "clarify_cost_lambda": float(clarify_cost),
        "utility_available": utility_inputs,
        "policies": reports,
    }
