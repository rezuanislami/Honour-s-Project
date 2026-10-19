# Research notes and implementation assumptions

## Research question alignment

This repository operationalizes one proposed mechanism: use retrieval evidence from the initial query to decide between direct retrieval and a clarification question. The present prototype is an engineering baseline, not evidence that retrieval density reliably predicts clarification need or model knowledge boundaries.

## Current baseline decisions

- Sparse retrieval uses BM25 via `rank-bm25`.
- Dense retrieval uses normalized SentenceTransformer embeddings, indexed with FAISS inner product (cosine similarity for normalized vectors).
- Each channel is min-max normalized over the current corpus for each query, then combined with a configurable BM25 weight.
- The confidence heuristic combines top fused score and top-vs-runner-up separation.
- Low confidence maps to `retrieve` (abstain from asking); sufficient confidence maps to `clarify`.
- If no LLM endpoint is configured, a generic deterministic clarification is returned.

## Decisions requiring empirical validation

1. Is per-query min-max normalization stable as corpus size and composition change?
2. Should confidence incorporate score distribution entropy, rank gaps, query clarity, or explicit CNP prediction?
3. Does the proposed polarity (low density => retrieve, sufficient density => clarify) align with user intent in each query class? Treat this as a testable design hypothesis.
4. How should the system distinguish low retrieval quality from genuinely out-of-boundary regional language?
5. Which LLM prompt/model and evidence constraints best reduce irrelevant or unsupported questions?
6. How will query outcomes, cognitive burden, and task completion be measured?

## Required experiment controls

- Keep train/development/test query sets separate; fit calibration only on development.
- Compare retrieval-gated routing with ask-always, retrieve-always, and a justified ambiguity-only baseline.
- Report system quality and routing quality separately, including uncertainty and subgroup results.
- Manually audit LLM relevance judgments and any generated questions; do not treat synthetic qrels as research evidence.
- Track versions, seeds, prompt templates, model identifiers, thresholds, and corpus snapshot hashes.

## Data and cultural safeguards

Only use datasets with documented license/permission and permitted processing. Regional and Māori-language material requires provenance and context; consult relevant communities and institutional ethics guidance where appropriate. Avoid asserting representativeness from a small or convenience sample. Remove personal or sensitive information, define retention controls, and document any external LLM processing before transmitting queries or documents.
