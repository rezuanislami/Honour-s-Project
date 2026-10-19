# ClariQ Conversational Search

A research prototype for retrieval-gated conversational search: hybrid retrieval first, then a confidence-based decision about whether to retrieve directly or ask a focused follow-up question.

The public demo follows the supplied architecture: first-turn query → sparse BM25 and dense SBERT/FAISS retrieval → hybrid score fusion → retrieval-density/confidence threshold → system routing → direct retrieval or proactive clarification → answer-aware follow-up retrieval.

## Run locally

Requirements: Python 3.11–3.13, Node.js 22, pnpm 10.18, and internet access for the first SBERT model download. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
pnpm install --frozen-lockfile
cp .env.example .env
pnpm dev
```

`pnpm dev` starts the FastAPI service on `127.0.0.1:8001` and the Vite interface on port 3000 (or `PORT`). Open `http://localhost:3000`. The first retrieval may download `sentence-transformers/all-MiniLM-L6-v2`; if dense-model initialization is unavailable, the API reports the issue and continues with sparse retrieval. The local development setup keeps the frontend and API same-origin through a Vite proxy.

For live Qwen clarification, provide your own provider key in an untracked local `.env` as `QWEN_API_KEY`. The default OpenAI-compatible endpoint is `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` with model `qwen-plus`; override `QWEN_BASE_URL` and `QWEN_MODEL` for your provider region/account as required. In the hosted application the key is configured as a protected Manus project secret and is only read by the server. Never commit credentials.

## What the interface demonstrates

- **Workspace:** query, selectable routing policy, visible pipeline, score/confidence evidence, results, generated clarification, and answer-aware follow-up retrieval.
- **Research notes:** architecture, label semantics, policy baselines, metric definitions, optional browser-local ClariQ/AmbigNQ label inspection, and JSON comparison-report display.
- **Transient sessions:** visitor queries and clarification answers are not persisted. The browser-only file inspection never uploads dataset files.
- **Synthetic-data boundary:** the bundled seven-document corpus is illustrative synthetic content (`synthetic-demo`), not ClariQ/AmbigNQ evidence. The repository intentionally contains no licensed benchmark release or domain retrieval corpus. No benchmark metric is pre-populated or fabricated.

## Routing policies

The live API supports `diagram`, `direct_retrieval`, `always_clarify`, `query_only_cnp`, and `retrieval_informed_cnp`. The two learned policies require trained model artifacts configured through `CONSEARCH_QUERY_CNP_MODEL` and `CONSEARCH_RETRIEVAL_CNP_MODEL`. Until such artifacts are loaded, the API reports that it used the architecture threshold fallback. The architecture threshold is independently configurable with `CONSEARCH_DIAGRAM_THRESHOLD`.

## Datasets and evaluation

ClariQ TSV and AmbigNQ JSON/JSONL are supported by the package adapters under `backend/src/consearch/datasets.py`. ClariQ's native graded `clarification_need` rating (1–4) is preserved; the binary target is explicitly derived. AmbigNQ has no native clarification-need labels here: the weak positive/negative target is derived from `multipleQAs` versus `singleAnswer` annotation types and must be reported as cross-task transfer, not as human CNP judgment.

The backend also retains the project's hybrid retrieval, CNP training, information-retrieval metrics, and four-policy evaluation modules. Use only permitted dataset files, strict train/test topic separation, and an authorized retrieval corpus when preparing research results. The research page can display a JSON report but does not invent missing rankings, relevance judgments, policy predictions, or net-utility inputs. See [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) for the methodological guardrails and [`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md) for the research log.

## API

- `GET /api/health` — service/model/data status; no credential values are returned.
- `POST /api/route` — accepts `{ "query": "...", "policy": "diagram", "top_k": 5 }`.
- `POST /api/search` — returns ranked hybrid results without routing.
- `POST /api/follow-up` — accepts the original query, clarification question and user answer, then retrieves again with the combined context.

Query length, `top_k`, JSON body size, and public request rates are bounded. API credentials never enter frontend assets or responses.

## Build and production

```bash
pnpm build:static
```

The managed WebDev deployment uses the repository `Dockerfile`: it builds the React site, installs pinned Python dependencies, and runs FastAPI on the platform-provided `PORT` (default 3000). The app serves the SPA and API from one origin, including `/api/health` for unauthenticated readiness. Runtime storage is ephemeral; the site does not retain visitor sessions or uploaded benchmark data.

## Main project layout

```text
backend/src/consearch/     Hybrid retrieval, Qwen, datasets, CNP and metrics
backend/data/              Synthetic demo corpus and toy fixtures only
backend/tests/             Offline research-pipeline tests
client/src/pages/          Search workspace and research page
client/public/assets/      User-supplied architecture reference
Dockerfile                 React build + Python production container
docs/                      Experiment protocol and research notes
plan.md / TODO.md           Approved website plan and implementation outcomes
```
