"""JSONL corpus loader."""

import json
from pathlib import Path
from typing import Any

from consearch.core import Document


def load_documents(path: Path) -> list[Document]:
    docs: list[Document] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item: dict[str, Any] = json.loads(line)
                docs.append(
                    Document(
                        str(item["id"]),
                        str(item["text"]),
                        str(item.get("title", "")),
                        item.get("metadata", {}),
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid JSONL document at {path}:{line_no}: {exc}") from exc
    return docs
