"""Command-line interface for serving, dataset preparation, model fitting and evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from consearch.cnp import CNPModel, classification_metrics, retrieval_features
from consearch.core import HybridRetriever, RetrievalDensityEstimator
from consearch.dataio import load_documents
from consearch.datasets import load_ambignq_json, load_clariq_tsv, write_jsonl
from consearch.evaluation import evaluate
from consearch.policies import evaluate_policy_suite

load_dotenv()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def make_retriever(path: str | None) -> HybridRetriever | None:
    if not path:
        return None
    docs = load_documents(Path(path))
    return HybridRetriever(
        docs, os.getenv("CONSEARCH_DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )


def assert_no_topic_leakage(train_rows: list[dict], test_rows: list[dict]) -> None:
    """Reject overlapping IDs when train and evaluation examples share a source."""
    train_sources = {row.get("source") for row in train_rows}
    test_sources = {row.get("source") for row in test_rows}
    if len(train_sources) == 1 and train_sources == test_sources:
        overlap = {str(row.get("id")) for row in train_rows} & {
            str(row.get("id")) for row in test_rows
        }
        if overlap:
            raise ValueError(f"Training and evaluation topic IDs overlap: {sorted(overlap)[:5]}")


def main():
    parser = argparse.ArgumentParser(prog="consearch")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the REST API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8000, type=int)

    route = sub.add_parser("route", help="Run routing for one query")
    route.add_argument("query")
    route.add_argument("--data", default="data/documents.jsonl")
    route.add_argument(
        "--threshold",
        default=float(
            os.getenv("CONSEARCH_DIAGRAM_THRESHOLD", os.getenv("CONSEARCH_THRESHOLD", "0.35"))
        ),
        type=float,
    )
    route.add_argument("--top-k", default=5, type=int)

    ev = sub.add_parser(
        "evaluate-ir", help="Compute P@1, MRR, nDCG@5 and Recall@100 from topics/qrels"
    )
    ev.add_argument("--data", default="data/documents.jsonl")
    ev.add_argument("--topics", default="data/topics.jsonl")
    ev.add_argument("--qrels", default="data/qrels.jsonl")
    ev.add_argument("--top-k", default=100, type=int)

    clariq = sub.add_parser("prepare-clariq", help="Normalize official ClariQ TSV to CNP JSONL")
    clariq.add_argument("--input", required=True)
    clariq.add_argument("--split", choices=["train", "dev", "test"], required=True)
    clariq.add_argument("--output", required=True)

    ambig = sub.add_parser(
        "prepare-ambignq", help="Normalize official AmbigNQ JSON to transfer-evaluation JSONL"
    )
    ambig.add_argument("--input", required=True)
    ambig.add_argument("--split", choices=["train", "dev", "test"], required=True)
    ambig.add_argument("--output", required=True)

    train = sub.add_parser("train-cnp", help="Train a query-only or retrieval-informed CNP model")
    train.add_argument(
        "--train", required=True, help="Normalized training JSONL, normally ClariQ train"
    )
    train.add_argument("--mode", choices=["query_only", "retrieval_informed"], required=True)
    train.add_argument(
        "--documents",
        help="Authorized fixed retrieval corpus JSONL; required in retrieval-informed mode",
    )
    train.add_argument("--output", required=True)
    train.add_argument("--top-k", default=10, type=int)

    comp = sub.add_parser(
        "compare-policies", help="Train both CNP variants and compare all four research policies"
    )
    comp.add_argument("--train", required=True, help="Normalized ClariQ train JSONL")
    comp.add_argument("--test", required=True, help="ClariQ held-out OR AmbigNQ transfer JSONL")
    comp.add_argument("--documents", required=True, help="Fixed authorized retrieval corpus JSONL")
    comp.add_argument("--threshold", default=0.5, type=float)
    comp.add_argument("--top-k", default=10, type=int)
    comp.add_argument("--clarify-cost", default=0.1, type=float)
    comp.add_argument("--output", required=True)

    evaluate_cnp = sub.add_parser(
        "evaluate-cnp", help="Evaluate a fitted CNP model on a normalized labelled split"
    )
    evaluate_cnp.add_argument("--model", required=True)
    evaluate_cnp.add_argument("--data", required=True)
    evaluate_cnp.add_argument("--documents", help="Required for retrieval-informed artifacts")
    evaluate_cnp.add_argument("--threshold", default=0.5, type=float)
    evaluate_cnp.add_argument("--top-k", default=10, type=int)

    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        uvicorn.run("consearch.api:app", host=args.host, port=args.port, reload=False)
        return
    if args.command == "prepare-clariq":
        records = load_clariq_tsv(Path(args.input), args.split)
        write_jsonl(records, Path(args.output))
        print(
            json.dumps(
                {
                    "source": "ClariQ",
                    "split": args.split,
                    "records": len(records),
                    "output": args.output,
                }
            )
        )
        return
    if args.command == "prepare-ambignq":
        records = load_ambignq_json(Path(args.input), args.split)
        write_jsonl(records, Path(args.output))
        print(
            json.dumps(
                {
                    "source": "AmbigNQ",
                    "split": args.split,
                    "records": len(records),
                    "warning": "Labels are derived weak labels from annotation type, not native CNP judgments.",
                    "output": args.output,
                }
            )
        )
        return
    if args.command == "train-cnp":
        records = [r for r in read_jsonl(Path(args.train)) if r.get("label") is not None]
        retriever = make_retriever(args.documents)
        if args.mode == "retrieval_informed" and retriever is None:
            parser.error("--documents is required for retrieval_informed mode")
        signals = (
            [retrieval_features(retriever.search(r["query"], args.top_k)) for r in records]
            if retriever
            else None
        )
        model = CNPModel(args.mode).fit(
            [r["query"] for r in records], [r["label"] for r in records], signals
        )
        model.save(Path(args.output))
        print(
            json.dumps({"mode": args.mode, "train_records": len(records), "artifact": args.output})
        )
        return
    if args.command == "compare-policies":
        train_rows = [r for r in read_jsonl(Path(args.train)) if r.get("label") is not None]
        test_rows = [r for r in read_jsonl(Path(args.test)) if r.get("label") is not None]
        assert_no_topic_leakage(train_rows, test_rows)
        retriever = make_retriever(args.documents)
        queries = [r["query"] for r in train_rows]
        y_train = [r["label"] for r in train_rows]
        test_queries = [r["query"] for r in test_rows]
        y_test = [r["label"] for r in test_rows]
        test_signals = [retrieval_features(retriever.search(q, args.top_k)) for q in test_queries]
        q_model = CNPModel("query_only").fit(queries, y_train)
        r_train_signals = [retrieval_features(retriever.search(q, args.top_k)) for q in queries]
        r_model = CNPModel("retrieval_informed").fit(queries, y_train, r_train_signals)
        q_scores = q_model.predict_proba(test_queries)
        r_scores = r_model.predict_proba(test_queries, test_signals)
        report = evaluate_policy_suite(
            y_test, q_scores, r_scores, args.threshold, clarify_cost=args.clarify_cost
        )
        report.update(
            {
                "train_source": train_rows[0].get("source") if train_rows else None,
                "test_source": test_rows[0].get("source") if test_rows else None,
                "test_label_provenance": sorted(
                    {r.get("label_provenance", "unknown") for r in test_rows}
                ),
                "train_n": len(train_rows),
                "test_n": len(test_rows),
                "note": "Classification/policy report only. Add paired per-query direct/clarified nDCG@5 to report utility; no retrieval gain is inferred from CNP labels.",
            }
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": args.output,
                    "policies": list(report["policies"]),
                    "test_n": len(test_rows),
                }
            )
        )
        return
    if args.command == "evaluate-cnp":
        model = CNPModel.load(Path(args.model))
        rows = [r for r in read_jsonl(Path(args.data)) if r.get("label") is not None]
        retriever = make_retriever(args.documents)
        if model.mode == "retrieval_informed" and retriever is None:
            parser.error("--documents is required for a retrieval-informed artifact")
        signals = (
            [retrieval_features(retriever.search(r["query"], args.top_k)) for r in rows]
            if retriever
            else None
        )
        scores = model.predict_proba([r["query"] for r in rows], signals)
        print(
            json.dumps(
                {
                    "mode": model.mode,
                    "source": rows[0].get("source") if rows else None,
                    "label_provenance": sorted(
                        {r.get("label_provenance", "unknown") for r in rows}
                    ),
                    **classification_metrics([r["label"] for r in rows], scores, args.threshold),
                },
                indent=2,
            )
        )
        return

    retriever = make_retriever(args.data)
    if args.command == "route":
        hits = retriever.search(args.query, args.top_k)
        result = RetrievalDensityEstimator(args.threshold).route(args.query, hits)
        print(
            json.dumps(
                {
                    "route": result.route,
                    "confidence": result.confidence,
                    "threshold": result.threshold,
                    "clarification_question": result.clarification_question,
                    "reason": result.reason,
                    "results": [
                        {"id": h.document.id, "title": h.document.title, "score": h.score}
                        for h in hits
                    ],
                },
                indent=2,
            )
        )
    else:
        print(
            json.dumps(
                evaluate(retriever, Path(args.topics), Path(args.qrels), args.top_k), indent=2
            )
        )


if __name__ == "__main__":
    main()
