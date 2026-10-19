"""FastAPI service for hybrid retrieval, CNP routing, Qwen clarification, and follow-up search."""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from consearch.cnp import CNPModel, retrieval_features
from consearch.core import HybridRetriever, RetrievalDensityEstimator
from consearch.dataio import load_documents
from consearch.llm import generate_clarification, qwen_configured

load_dotenv()
PACKAGE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PACKAGE_DIR.parents[1]
DATA_PATH = Path(os.getenv("CONSEARCH_DATA", str(BACKEND_DIR / "data" / "documents.jsonl")))
MODEL_NAME = os.getenv("CONSEARCH_DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DIAGRAM_THRESHOLD = float(os.getenv("CONSEARCH_DIAGRAM_THRESHOLD", os.getenv("CONSEARCH_THRESHOLD", "0.35")))
CNP_THRESHOLD = float(os.getenv("CONSEARCH_CNP_THRESHOLD", "0.5"))
TOP_K = int(os.getenv("CONSEARCH_TOP_K", "5"))
ALPHA = float(os.getenv("CONSEARCH_BM25_WEIGHT", "0.5"))
STATIC_DIR = Path(os.getenv("CONSEARCH_STATIC_DIR", str(BACKEND_DIR.parent / "dist" / "public")))
MAX_BODY_BYTES = 64 * 1024
RATE_WINDOW_SECONDS = 60
RATE_LIMITS = {"/api/route": 12, "/api/follow-up": 12, "/api/search": 30}
_rate_windows: dict[str, deque[float]] = defaultdict(deque)
_rate_lock = Lock()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=TOP_K, ge=1, le=50)
    bm25_weight: float = Field(default=ALPHA, ge=0, le=1)
    policy: Literal["diagram", "direct_retrieval", "always_clarify", "query_only_cnp", "retrieval_informed_cnp"] = "retrieval_informed_cnp"


class FollowUpRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    clarification_question: str = Field(min_length=1, max_length=1000)
    user_answer: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=TOP_K, ge=1, le=50)
    bm25_weight: float = Field(default=ALPHA, ge=0, le=1)


def create_app(data_path: Path = DATA_PATH) -> FastAPI:
    app = FastAPI(
        title="ClariQ Conversational Search",
        version="0.3.0",
        description="Hybrid BM25 and SBERT/FAISS retrieval, clarification-need routing, optional Qwen generation, and research transparency.",
    )
    try:
        retriever = HybridRetriever(load_documents(data_path), MODEL_NAME)
    except Exception as exc:
        raise RuntimeError(f"Could not initialize corpus from {data_path}: {exc}") from exc
    estimator = RetrievalDensityEstimator(DIAGRAM_THRESHOLD)
    models: dict[str, CNPModel] = {}
    for mode, env_name in (
        ("query_only", "CONSEARCH_QUERY_CNP_MODEL"),
        ("retrieval_informed", "CONSEARCH_RETRIEVAL_CNP_MODEL"),
    ):
        artifact = os.getenv(env_name)
        if artifact:
            models[mode] = CNPModel.load(Path(artifact))

    @app.middleware("http")
    async def guard_public_requests(request: Request, call_next):
        path = request.url.path
        if request.method == "POST" and path.startswith("/api/"):
            length = request.headers.get("content-length")
            if length and int(length) > MAX_BODY_BYTES:
                return JSONResponse(status_code=413, content={"detail": "Request body exceeds the 64 KB limit."})
            limit = RATE_LIMITS.get(path)
            if limit:
                # The managed edge supplies forwarded client address metadata.
                forwarded = request.headers.get("x-forwarded-for", "")
                client_id = (forwarded.split(",", 1)[0].strip() if forwarded else "") or (request.client.host if request.client else "unknown")
                now = time.monotonic()
                with _rate_lock:
                    window = _rate_windows[client_id]
                    while window and now - window[0] >= RATE_WINDOW_SECONDS:
                        window.popleft()
                    if len(window) >= limit:
                        return JSONResponse(status_code=429, content={"detail": "Rate limit reached. Wait a minute before trying again."}, headers={"Retry-After": "60"})
                    window.append(now)
        response = await call_next(request)
        if path.startswith("/api/") or path in {"/", "/research"}:
            response.headers["Cache-Control"] = "no-store"
        return response

    def present_hits(hits):
        return [
            {
                "id": hit.document.id,
                "title": hit.document.title,
                "text": hit.document.text,
                "score": hit.score,
                "sparse_score": hit.sparse_score,
                "dense_score": hit.dense_score,
                "metadata": hit.document.metadata or {},
            }
            for hit in hits
        ]

    @app.get("/api/health")
    @app.get("/health", include_in_schema=False)
    def health():
        return {
            "status": "ok",
            "documents": len(retriever.documents),
            "sparse_enabled": True,
            "dense_enabled": retriever.encoder is not None,
            "dense_error": retriever.dense_error,
            "qwen_configured": qwen_configured(),
            "loaded_cnp_models": sorted(models),
            "default_route": "diagram",
            "corpus_kind": "synthetic-demo",
            "persistent_session_storage": False,
        }

    @app.post("/api/search")
    @app.post("/search", include_in_schema=False)
    def search(request: SearchRequest):
        try:
            hits = retriever.search(request.query, request.top_k, request.bm25_weight)
            return {"query": request.query, "results": present_hits(hits)}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/route")
    @app.post("/route", include_in_schema=False)
    def route(request: SearchRequest):
        try:
            hits = retriever.search(request.query, request.top_k, request.bm25_weight)
            diagram_score = estimator.estimate(hits)
            chosen_policy = request.policy
            fallback_reason = None
            if chosen_policy == "direct_retrieval":
                ask, confidence = False, 0.0
            elif chosen_policy == "always_clarify":
                ask, confidence = True, 1.0
            elif chosen_policy in {"query_only_cnp", "retrieval_informed_cnp"}:
                mode = "query_only" if chosen_policy == "query_only_cnp" else "retrieval_informed"
                model = models.get(mode)
                if model is None:
                    ask, confidence = diagram_score >= DIAGRAM_THRESHOLD, diagram_score
                    fallback_reason = f"No trained {mode} model loaded; used the architecture diagram threshold rule."
                else:
                    signals = [retrieval_features(hits)] if mode == "retrieval_informed" else None
                    confidence = float(model.predict_proba([request.query], signals)[0])
                    ask = confidence >= CNP_THRESHOLD
            else:
                confidence = diagram_score
                ask = confidence >= DIAGRAM_THRESHOLD
            question = None
            question_source = None
            if ask:
                question = generate_clarification(request.query, hits)
                question_source = "qwen" if question else "local_fallback"
                question = question or estimator.default_question(request.query)
            used_policy = chosen_policy if chosen_policy in {"diagram", "direct_retrieval", "always_clarify"} or chosen_policy in models_as_policy(models) else "diagram_fallback"
            return {
                "query": request.query,
                "requested_policy": request.policy,
                "policy_used": used_policy,
                "route": "clarify" if ask else "retrieve",
                "clarification_probability_or_diagram_score": confidence,
                "diagram_score": diagram_score,
                "diagram_threshold": DIAGRAM_THRESHOLD,
                "cnp_threshold": CNP_THRESHOLD,
                "clarification_question": question,
                "question_generator": question_source,
                "fallback_reason": fallback_reason,
                "reason": "policy predicts clarification need" if ask else "policy selects direct retrieval / abstention from clarification",
                "retrieval_signals": retrieval_features(hits),
                "results": present_hits(hits),
            }
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/follow-up")
    @app.post("/follow-up", include_in_schema=False)
    def follow_up(request: FollowUpRequest):
        """Incorporate the user's answer into the next retrieval query."""
        resolved_query = f"{request.query}\nClarification: {request.clarification_question}\nUser answer: {request.user_answer}"
        try:
            hits = retriever.search(resolved_query, request.top_k, request.bm25_weight)
            return {"original_query": request.query, "resolved_query": resolved_query, "results": present_hits(hits)}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    if STATIC_DIR.is_dir() and (STATIC_DIR / "index.html").is_file():
        static_root = STATIC_DIR.resolve()

        @app.get("/{page_path:path}", include_in_schema=False)
        def serve_site(page_path: str = ""):
            candidate = (static_root / page_path).resolve()
            if not candidate.is_relative_to(static_root):
                raise HTTPException(status_code=404, detail="Not found")
            if candidate.is_file():
                if page_path.startswith("api/") and page_path != "api/platform/config.js":
                    raise HTTPException(status_code=404, detail="Not found")
                return FileResponse(candidate)
            if page_path.startswith("api/") or Path(page_path).suffix:
                raise HTTPException(status_code=404, detail="Not found")
            return FileResponse(static_root / "index.html")

    return app


def models_as_policy(models: dict[str, CNPModel]) -> set[str]:
    """Map model artifacts to their public policy names."""
    return {"query_only_cnp" if mode == "query_only" else "retrieval_informed_cnp" for mode in models}


app = create_app()
