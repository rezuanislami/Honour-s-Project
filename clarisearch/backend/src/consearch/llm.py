"""Qwen-compatible clarification-question generation.

Configure QWEN_BASE_URL, QWEN_MODEL and QWEN_API_KEY (or DASHSCOPE_API_KEY).
The provider account/key is deliberately supplied by the project operator, not bundled.
"""

from __future__ import annotations

import os

from consearch.core import SearchHit

DEFAULT_QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen-plus"


def qwen_configured() -> bool:
    return bool((os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")) and qwen_settings()[0] and qwen_settings()[1])


def qwen_settings() -> tuple[str, str]:
    """Return configurable provider settings with safe Qwen-compatible defaults."""
    return (
        os.getenv("QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL).strip(),
        os.getenv("QWEN_MODEL", DEFAULT_QWEN_MODEL).strip(),
    )


def generate_clarification(query: str, hits: list[SearchHit]) -> str | None:
    """Generate one relevant, concise clarification question; None signals fallback."""
    base_url, model = qwen_settings()
    api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not (base_url and model and api_key):
        return None
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=float(os.getenv("QWEN_TIMEOUT_SECONDS", "20")),
            max_retries=1,
        )
        context = "\n".join(
            f"{rank}. {hit.document.title}: {hit.document.text[:600]}"
            for rank, hit in enumerate(hits[:4], 1)
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate exactly one concise clarification question for a conversational "
                        "search system. Ask about the most consequential unresolved intent or facet. "
                        "Use retrieved snippets only as context, not as facts to assert. Do not answer "
                        "the query, invent alternatives unsupported by the query, or ask multiple "
                        "questions. Return only the question."
                    ),
                },
                {"role": "user", "content": f"User query: {query}\nRetrieved context:\n{context}"},
            ],
            temperature=0.2,
            max_tokens=100,
        )
        text = (response.choices[0].message.content or "").strip().strip('"')
        if not text:
            return None
        text = " ".join(text.split())
        if len(text) > 500:
            text = text[:497].rsplit(" ", 1)[0] + "...?"
        return text if "?" in text else text + "?"
    except Exception:  # noqa: BLE001 - external LLM failure degrades to local fallback
        return None
