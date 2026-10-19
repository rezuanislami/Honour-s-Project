# Website Outcomes

These items track the user-approved plan for the hosted ClariQ Conversational Search website. Items remain open until the corresponding behavior is implemented and verified.

## Search, retrieval, routing, and follow-up

- [ ] The public search workspace accepts a visitor’s first-turn search query and displays the evolving conversational-search session. It runs sparse BM25 and dense SBERT/FAISS retrieval, fuses the ranked results, computes retrieval density/confidence against the configured threshold, and exposes the retrieval signals and selected branch. Low confidence follows system abstention and direct document retrieval; sufficient confidence follows proactive clarification. Visitors can inspect retrieved documents, rankings, sparse/dense/fused scores, and the branch selected at each turn. A visitor can answer a clarification and continue retrieval with the enriched query context.

## Dataset and evaluation transparency

- [ ] The research view explains support for ClariQ and AmbigNQ, preserves ClariQ’s graded clarification-need label provenance, and clearly identifies AmbigNQ labels as derived. It describes and compares the direct-retrieval, always-clarify, query-only CNP, and retrieval-informed CNP policies and the project’s evaluation measures. The site does not commit licensed ClariQ/AmbigNQ copies or present the synthetic demo corpus as benchmark evidence. Every built-in example is marked synthetic; numerical results are shown as benchmark evidence only when they were produced from supplied benchmark inputs and an authorized retrieval corpus.

## Qwen clarification and public API safety

- [ ] Clarification generation calls the configured OpenAI-compatible Qwen endpoint from the server only. Provider credentials are stored as protected project secrets and never appear in frontend assets, source control, or logs. If Qwen is not configured or a call fails, the response clearly identifies that a local fallback question was used. Public model-backed requests validate query length and `top_k`, enforce request-size limits and per-client rate limits, and do not persist visitors’ queries or answers.

## Permanent hosted website and project identity

- [ ] The approved React/Vite interface and Python/FastAPI retrieval API run from one managed container with the configured `PORT`, an unauthenticated `/api/health` route, and the application's build and production startup steps included in the Dockerfile. The managed WebDev project serves the SPA and API from the same hosted origin. The public site uses the dark computational-instrument design derived from the supplied architecture image, including visible pipeline stages, confidence threshold, route branches, responsive behavior and reduced-motion support. The supplied diagram is used as a project asset. The site has a synchronized route manifest for `/` and `/research`, a matching app/project logo, and remains on the platform’s default on-demand hosting mode unless the user later requests a different setting. `publishing.auto_publish` remains disabled; the finished version is published explicitly.
