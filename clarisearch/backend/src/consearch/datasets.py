"""Adapters for ClariQ and AmbigNQ benchmark files.

No benchmark data is bundled. ClariQ labels are native 1..4 clarification-need
ratings; AmbigNQ labels are explicitly weak labels derived from annotation type.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_clariq_tsv(path: Path, split: str) -> list[dict[str, Any]]:
    """Load ClariQ train/dev/test TSV, deduplicated to one record per topic.

    Repeated topic rows correspond to facets/questions. Their severity must agree;
    inconsistent topic labels raise an error rather than being silently collapsed.
    Test-with-labels can be loaded for offline analysis only where license permits.
    """
    by_id: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row_number, row in enumerate(csv.DictReader(stream, delimiter="\t"), 2):
            topic_id = str(row.get("topic_id", "")).strip()
            query = str(row.get("initial_request", "")).strip()
            if not topic_id or not query:
                raise ValueError(f"Missing topic_id or initial_request at {path}:{row_number}")
            raw_severity = row.get("clarification_need", "")
            severity = int(raw_severity) if str(raw_severity).strip() else None
            if severity is not None and severity not in (1, 2, 3, 4):
                raise ValueError(
                    f"ClariQ severity must be 1..4, got {severity} at row {row_number}"
                )
            record = by_id.get(topic_id)
            if record and record["severity"] != severity:
                raise ValueError(f"Conflicting ClariQ severity for topic {topic_id}")
            if not record:
                by_id[topic_id] = {
                    "id": topic_id,
                    "query": query,
                    "severity": severity,
                    "label": None if severity is None else int(severity > 1),
                    "source": "clariq",
                    "split": split,
                    "label_provenance": "native_clariq_rating"
                    if severity is not None
                    else "unlabelled_test",
                }
    return list(by_id.values())


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, dict):
        # Permit simple HuggingFace export wrappers, but never silently guess nested data.
        value = value.get("data", value.get("examples", value))
    if not isinstance(value, list) or not all(isinstance(x, dict) for x in value):
        raise ValueError(f"Expected a JSON list of AmbigNQ records in {path}")
    return value


def load_ambignq_json(path: Path, split: str) -> list[dict[str, Any]]:
    """Load AmbigNQ full/light JSON and derive transparent weak CNP labels.

    A record containing a `multipleQAs` annotation is marked ambiguous/clarify=1;
    a record containing only `singleAnswer` annotations is labeled 0. This is not
    a native CNP judgment and should be reported as cross-task transfer evaluation.
    """
    normalized = []
    for row in _read_json_records(path):
        query = str(row.get("question", "")).strip()
        if not query:
            continue
        annotations = row.get("annotations") or []
        types = {str(a.get("type", "")).lower() for a in annotations if isinstance(a, dict)}
        if not annotations or not types:
            label, severity, provenance = None, None, "unlabelled_or_missing_annotation"
        else:
            label = int("multipleqas" in types)
            severity = 4 if label else 1
            provenance = "derived_from_ambignq_annotation_type"
        normalized.append(
            {
                "id": str(row.get("id", len(normalized))),
                "query": query,
                "severity": severity,
                "label": label,
                "source": "ambignq",
                "split": split,
                "label_provenance": provenance,
                "gold_annotations": annotations,
            }
        )
    return normalized


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
