# ClariQ Conversational Search — Website Plan

## Product and implementation

Deliver a public, permanently hosted research website that makes the current conversational-search prototype usable through a browser. The UI will follow the supplied architecture image and expose the complete first-turn flow: query → parallel sparse BM25 and dense SBERT/FAISS retrieval → hybrid score fusion → retrieval-density/confidence estimate → threshold-based system abstention/direct retrieval or proactive clarification → answer-aware follow-up retrieval. The user can inspect ranked documents and sparse, dense, and fused evidence as well as the selected branch.

The site will retain the existing Python retrieval and research modules rather than replace their retrieval semantics with a browser-only approximation. The React/Vite frontend will be built in the existing Node project. A Python/FastAPI application will provide the production API and serve the compiled single-page app from the same container. A multi-stage Docker build will compile the frontend with the pinned pnpm toolchain, install the Python runtime dependencies, and start the API on the configured `PORT`; `/api/health` remains the unauthenticated deployment health route. The current Express/tRPC template code will not be used as a competing second production server.

The public interface will include:

- **Search workspace (`/`)**: submit an initial query; inspect its route/confidence, threshold, retrieval signals, ranked evidence, and generated clarification; answer a clarifying question and continue retrieval with that context.
- **Research notes (`/research`)**: explain ClariQ and AmbigNQ support, preserve the provenance of ClariQ's graded clarification-need labels, mark AmbigNQ labels as derived, and describe the four comparison policies (direct retrieval, always clarify, query-only CNP, retrieval-informed CNP) and research metrics. Display results only when they come from supplied benchmark inputs; label every built-in example as synthetic rather than benchmark evidence.
- **Data boundary**: do not commit licensed ClariQ/AmbigNQ copies or represent the small synthetic demo corpus as research evidence. The website remains usable with its marked demo corpus. Real benchmark results require the operator to supply the permitted dataset files and a separate authorized retrieval corpus.
- **Qwen**: call the configured OpenAI-compatible Qwen endpoint only from the server. Store the provider key as a protected project secret; never expose it in frontend assets, source control, or logs. When the key/model endpoint is not configured or a call fails, clearly identify the local fallback question.
- **Public API safeguards**: validate query length and `top_k`, bound dataset/evaluation request size, and apply per-client rate limits to public model-backed calls. Keep sessions transient; do not persist visitors' queries or answers.

## Project structure

- `client/src/pages/Home.tsx` — live query, clarification, and retrieval workspace.
- `client/src/pages/Research.tsx` — architecture, dataset-label provenance, policy descriptions, and metric definitions.
- `client/src/components/` — reusable pipeline, confidence/threshold indicator, result rows, and synthetic-data badge.
- `client/src/index.css` — responsive application styling based on the user's architecture-diagram visual language.
- `client/public/manus-routes.json` — declared website page routes for the platform.
- `client/public/assets/` — supplied architecture diagram and site assets.
- `src/consearch/` — existing Python retrieval, Qwen adapter, dataset adapters, CNP and evaluation modules.
- `data/` — small, explicitly synthetic demo corpus and demo fixtures only.
- `api.py` (or the package API module) — FastAPI routes and static frontend serving.
- `Dockerfile` and Python dependency lock/requirements — reproducible multi-stage container build and production entrypoint.
- `plan.md` and `TODO.md` — approved design/implementation decisions and tracked product outcomes.

## Design direction

**Design movement:** a dark computational instrument / editorial research console, directly derived from the supplied diagram rather than a generic chatbot template.

**Core principles:** make the decision path visible; keep retrieval evidence inspectable; distinguish measured, derived, and synthetic data; use color to encode system states consistently.

**Color philosophy:** near-black charcoal and cool grey create a quiet lab-like canvas. Electric teal signals query/dense-retrieval activity, violet marks hybrid fusion, green marks sufficient confidence and clarification, and amber/coral marks threshold and abstention. Color supplements labels and icons so route meaning is not color-only.

**Layout paradigm:** a restrained top navigation above an asymmetric working surface. The query and conversation take the primary column; a vertical connected pipeline and diagnostics rail expose the system path on wide screens, collapsing into a clear stacked sequence on mobile. Results remain visually subordinate to the query but immediately inspectable.

**Signature elements:** thin neon-outline stage cards echo the architecture boxes; a live confidence rail shows the threshold crossing; branching connector lines trace the selected route. Motion is limited to a subtle active-node pulse and short state transitions, with `prefers-reduced-motion` respected.

**Typography system:** Space Grotesk (or a metrically similar sans fallback) for editorial headings and interface text; IBM Plex Mono (or system monospace fallback) for scores, model labels, and pipeline annotations. Use a compact type scale with clear contrast between query, evidence and instrumentation.

**Brand essence:** “A transparent research instrument for search that knows when to ask.” Personality: rigorous, legible, exploratory.

**Brand voice:** concise and evidence-first. Examples: “Every route comes with retrieval evidence.” “Ask only when the missing facet can change the search.”

**Wordmark & logo:** a compact CLARiQ wordmark paired with a three-node branching path that resolves into a search lens; use the same mark for the site favicon and platform project logo.

**Signature brand color:** electric teal, approximately `#35D6C4`.

## Deployment and material constraints

Use the already initialized Manus WebDev project and its enabled server capability. Publish the website through the managed container workflow; keep `publishing.auto_publish` disabled and publish the user-requested finished version explicitly. The default hosted residency policy can sleep when idle, so “permanent” means the site remains deployed at a stable hosted URL, not that every request is guaranteed a warm process. No managed database is needed for transient demo sessions. Qwen calls require the operator's own provider credential and can incur provider usage charges. Benchmark data and a licensed retrieval corpus are not included in the repository; any numerical result shown without those inputs must be explicitly marked synthetic.
