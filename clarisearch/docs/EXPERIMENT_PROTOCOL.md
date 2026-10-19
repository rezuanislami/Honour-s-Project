# Experiment protocol

This protocol operationalizes the SOC10101 IPO and the submitted literature review. It is a research scaffold; no substantive benchmark result is claimed until licensed dataset files, an appropriate retrieval corpus, and judged retrieval outcomes are supplied.

## Dataset roles and labels

- **ClariQ** is the supervised CNP benchmark. Preserve its native clarification-need score 1–4 in every normalized row. For the standard binary task, map score 1 to `label=0` and scores 2–4 to `label=1`. Repeated TSV rows describe multiple facets/questions for a topic, so the adapter deduplicates by topic and rejects conflicting topic severity.
- **AmbigNQ** is a transfer set, not a native CNP dataset. The adapter derives a transparent weak label: any `multipleQAs` annotation means `label=1`; only `singleAnswer` annotations means `label=0`. Report these as annotation-derived transfer labels, never as human clarification-need judgments. The full annotations stay attached for later analysis.
- Keep topic IDs and split boundaries. Fit on ClariQ train, tune thresholds only on ClariQ dev, and report once on ClariQ held-out data. Evaluate the frozen ClariQ model on AmbigNQ dev as cross-task transfer, clearly identifying the construct difference. Do not train on AmbigNQ transfer examples when reporting transfer results.

## Retrieval corpus

ClariQ and AmbigNQ are CNP/ambiguity datasets; they do not, by themselves, constitute the document collection needed for a defensible end-to-end search benchmark. Supply a separately licensed, versioned document corpus as JSONL for BM25 and dense indexing, plus query-document relevance judgments for retrieval evaluation. Do not index test answers or annotations as documents. Record the corpus snapshot, preprocessing, model ID, and qrels provenance. The small `data/documents.jsonl` shipped here is only synthetic smoke-test content.

## Retrieval and routing

The engine runs BM25 and SentenceTransformer embeddings indexed by FAISS. The system exposes two interpretations without hiding their distinction:

1. `diagram` reproduces the attached IPO routing direction: confidence at/above the configured diagram threshold routes to proactive clarification; below threshold abstains from questioning and retrieves directly.
2. `query_only_cnp` and `retrieval_informed_cnp` apply learned probabilities of clarification need from ClariQ labels. Retrieval-informed features include top-score/margin, score dispersion, sparse/dense top-k overlap, and rank agreement. Higher predicted *clarification need* routes to a question.

The first is the faithful prototype of the submitted diagram. The latter two are the policy hypotheses motivated by the literature review. Do not conflate high retrieval relevance with high clarification need. Compare both mechanisms and document which policy is used for each result.

The API also supports `direct_retrieval` and `always_clarify` policy baselines. A saved CNP model artifact is required to use the learned policies; otherwise the API reports that it used the diagram-rule fallback.

## Qwen's role

Qwen is used only after a policy elects to clarify. The model receives the original query and a small set of retrieved snippets, and is instructed to return one concise evidence-aware clarification question, not an answer. The API accepts an OpenAI-compatible provider endpoint via environment variables and never includes the key in API responses. A deterministic local fallback keeps the service functional when Qwen is not configured or unavailable. Record the actual Qwen model ID, provider/region, prompt version, temperature, latency, and failures in experiment logs.

The user must supply a provider-issued API key and the correct base URL/model name for their account and region. Do not commit credentials or transmit licensed/private query data to a hosted endpoint without permission and institutional approval.

## Four-policy evaluation

The `compare-policies` command reports CNP precision, recall, macro-F1, weighted-F1, AUROC, Brier score, 10-bin expected calibration error, false-clarification rate, clarification coverage, unnecessary-question count, and missed-clarification count for:

- direct retrieval (never asks);
- always clarify (asks every time);
- query-only CNP;
- retrieval-informed CNP.

The separate IR evaluator reports P@1, P@5, P@10, MRR, nDCG@5, nDCG@10, and Recall@100 from qrels. Those retrieval metrics are not automatically the same as end-to-end policy utility. To compute `Uλ = nDCG@5 − λ × Cost(clarify)`, provide a valid paired evaluation design that measures direct and post-clarification retrieval outcomes with the same retriever, question-selection procedure, answer-incorporation method, and query set. Do not infer clarified-query nDCG from CNP labels or generated questions alone.

Use ClariQ's official per-topic evaluation resources or a documented human/controlled answer protocol where licensing permits. For AmbigNQ, use its official answer/disambiguation evaluation separately; any derived CNP scores are secondary transfer analysis. Report confidence intervals and paired significance tests in the dissertation; the current starter evaluator reports point estimates only.

## Running the dataset pipeline

Obtain the official files from the source repositories cited in the README. The adapter needs the ClariQ `train.tsv`/`dev.tsv` style fields (`topic_id`, `initial_request`, `clarification_need`) and AmbigNQ `train.json`/`dev.json` or light files (`id`, `question`, `annotations`).

```bash
consearch prepare-clariq --input /path/to/train.tsv --split train --output data/processed/clariq_train.jsonl
consearch prepare-clariq --input /path/to/dev.tsv --split dev --output data/processed/clariq_dev.jsonl
consearch prepare-ambignq --input /path/to/dev.json --split dev --output data/processed/ambignq_dev.jsonl
consearch train-cnp --train data/processed/clariq_train.jsonl --mode query_only --output models/query_only.joblib
consearch train-cnp --train data/processed/clariq_train.jsonl --mode retrieval_informed --documents /path/to/licensed_corpus.jsonl --output models/retrieval_informed.joblib
consearch compare-policies --train data/processed/clariq_train.jsonl --test data/processed/clariq_dev.jsonl --documents /path/to/licensed_corpus.jsonl --output reports/clariq_dev.json
consearch compare-policies --train data/processed/clariq_train.jsonl --test data/processed/ambignq_dev.jsonl --documents /path/to/licensed_corpus.jsonl --output reports/ambignq_transfer.json
```

Do not use train metrics as test metrics. Calibrate decision thresholds on development data and freeze them before evaluating held-out topics.
