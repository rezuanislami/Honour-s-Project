import { useMemo, useState } from "react";
import type { ChangeEvent } from "react";
import { Link } from "wouter";
import { ArrowLeft, ArrowUpRight, BadgeCheck, BookOpen, CheckCircle2, FileJson2, FileSpreadsheet, FlaskConical, ShieldCheck, Upload, Waypoints } from "lucide-react";

type DatasetSummary = { source: "ClariQ" | "AmbigNQ"; split: string; total: number; labelled: number; positive: number; negative: number; missing: number; provenance: string };
type PolicyReport = { policies: Record<string, Record<string, unknown>>; test_n?: number; threshold?: number; test_label_provenance?: unknown; utility_available?: boolean; clarify_cost_lambda?: number };
const MAX_FILE_BYTES = 20 * 1024 * 1024;
const POLICY_LABELS: Record<string, string> = {
  direct_retrieval: "Direct retrieval", always_clarify: "Always clarify", query_only_cnp: "Query-only CNP", retrieval_informed_cnp: "Retrieval-informed CNP",
};
const POLICY_DESCRIPTIONS: Record<string, string> = {
  direct_retrieval: "Never asks a question; routes every query straight to document retrieval.",
  always_clarify: "Asks for clarification on every query, regardless of retrieval evidence.",
  query_only_cnp: "Predicts clarification need from first-turn query text alone.",
  retrieval_informed_cnp: "Predicts clarification need from query text plus retrieval-density signals.",
};

function parseTsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted && ch === '"' && text[i + 1] === '"') { cell += '"'; i++; }
    else if (ch === '"') quoted = !quoted;
    else if (ch === "\t" && !quoted) { row.push(cell); cell = ""; }
    else if ((ch === "\n" || ch === "\r") && !quoted) {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(cell); if (row.some((value) => value.length)) rows.push(row); row = []; cell = "";
    } else cell += ch;
  }
  row.push(cell); if (row.some((value) => value.length)) rows.push(row);
  return rows;
}

function inspectClariQ(text: string, split: string): DatasetSummary {
  const rows = parseTsv(text.replace(/^\uFEFF/, ""));
  if (!rows.length) throw new Error("The TSV file is empty.");
  const header = rows[0].map((value) => value.trim());
  const idAt = header.indexOf("topic_id");
  const queryAt = header.indexOf("initial_request");
  const severityAt = header.indexOf("clarification_need");
  if (idAt < 0 || queryAt < 0 || severityAt < 0) throw new Error("Expected ClariQ columns: topic_id, initial_request and clarification_need.");
  const topics = new Map<string, number | null>();
  for (const values of rows.slice(1)) {
    const id = (values[idAt] ?? "").trim();
    const query = (values[queryAt] ?? "").trim();
    const raw = (values[severityAt] ?? "").trim();
    if (!id || !query) continue;
    const severity = raw ? Number(raw) : null;
    if (severity !== null && ![1, 2, 3, 4].includes(severity)) throw new Error(`ClariQ clarification_need must be 1–4; found “${raw}”.`);
    if (topics.has(id) && topics.get(id) !== severity) throw new Error(`Conflicting graded labels found for topic ${id}.`);
    topics.set(id, severity);
  }
  const values = [...topics.values()];
  const labelled = values.filter((value) => value !== null);
  return { source: "ClariQ", split, total: values.length, labelled: labelled.length, positive: labelled.filter((value) => (value ?? 1) > 1).length, negative: labelled.filter((value) => value === 1).length, missing: values.length - labelled.length, provenance: "Native ClariQ clarification_need rating (1–4); binary positive label is derived by the project's documented >1 threshold." };
}

function inspectAmbigNQ(text: string, split: string): DatasetSummary {
  let parsed: unknown;
  try { parsed = JSON.parse(text); }
  catch { parsed = text.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line)); }
  if (parsed && !Array.isArray(parsed) && typeof parsed === "object") {
    const record = parsed as { data?: unknown; examples?: unknown };
    parsed = record.data ?? record.examples;
  }
  if (!Array.isArray(parsed)) throw new Error("Expected a JSON array, JSONL, or an object containing data/examples.");
  let positive = 0; let negative = 0; let missing = 0;
  for (const item of parsed) {
    if (!item || typeof item !== "object") { missing++; continue; }
    const annotations = (item as { annotations?: unknown }).annotations;
    if (!Array.isArray(annotations) || !annotations.length) { missing++; continue; }
    const types = annotations.filter((a) => a && typeof a === "object").map((a) => String((a as { type?: unknown }).type ?? "").toLowerCase());
    if (!types.length) missing++;
    else if (types.includes("multipleqas")) positive++;
    else negative++;
  }
  const total = parsed.length;
  return { source: "AmbigNQ", split, total, labelled: total - missing, positive, negative, missing, provenance: "Derived weak clarification label from AmbigNQ annotation type (multipleQAs vs singleAnswer); this is not a native human CNP judgment." };
}

function niceMetric(key: string) { return key.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }
function metric(value: unknown) { return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "—"; }

function Header() {
  return <header className="site-header"><Link href="/" className="brand"><span className="brand-mark"><Waypoints size={19} /></span><span className="brand-word">CLAR<span>i</span>Q</span><span className="brand-divider" /><span className="brand-subtitle">CONVERSATIONAL SEARCH</span></Link><nav className="top-nav" aria-label="Main navigation"><Link href="/" className="nav-link">Workspace</Link><Link href="/research" className="nav-link nav-active">Research notes <ArrowUpRight size={13} /></Link></nav><div className="header-status"><span className="status-dot dot-live" /> METHODOLOGY</div></header>;
}

export default function Research() {
  const [datasetKind, setDatasetKind] = useState<"clariq" | "ambignq">("clariq");
  const [split, setSplit] = useState("train");
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [datasetError, setDatasetError] = useState("");
  const [report, setReport] = useState<PolicyReport | null>(null);
  const [reportError, setReportError] = useState("");
  const policyRows = useMemo(() => report?.policies ? Object.entries(report.policies) : [], [report]);

  async function handleDataset(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    setDatasetError(""); setSummary(null);
    if (!file) return;
    if (file.size > MAX_FILE_BYTES) { setDatasetError("For browser-only inspection, choose a file smaller than 20 MB."); event.target.value = ""; return; }
    try {
      const text = await file.text();
      const result = datasetKind === "clariq" ? inspectClariQ(text, split) : inspectAmbigNQ(text, split);
      setSummary(result);
    } catch (caught) { setDatasetError(caught instanceof Error ? caught.message : "The file could not be read."); }
    event.target.value = "";
  }

  async function handleReport(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    setReportError(""); setReport(null);
    if (!file) return;
    if (file.size > 2 * 1024 * 1024) { setReportError("Choose a report smaller than 2 MB."); event.target.value = ""; return; }
    try {
      const parsed = JSON.parse(await file.text()) as PolicyReport;
      if (!parsed || typeof parsed.policies !== "object" || !parsed.policies) throw new Error("Expected a JSON policy report with a policies object.");
      setReport(parsed);
    } catch (caught) { setReportError(caught instanceof Error ? caught.message : "Invalid report JSON."); }
    event.target.value = "";
  }

  return <div className="app-shell"><Header /><main className="research-shell">
    <div className="page-kicker"><span className="eyebrow-mark" /> RESEARCH SYSTEM / METHOD <span className="kicker-rule" /> <span>01—05</span></div>
    <Link href="/" className="back-link"><ArrowLeft size={14} /> Back to workspace</Link>
    <section className="research-heading"><div><span className="eyebrow">PROJECT METHOD</span><h1>Make the decision<br /><em>inspectable.</em></h1><p>ClariQ Search turns the clarification decision into a visible, evidence-backed step in the retrieval process.</p></div><div className="research-ref"><span className="ref-number">01</span><span>IPO / SYSTEM<br />ARCHITECTURE</span></div></section>
    <section className="architecture-section"><div className="section-title-row"><div><span className="eyebrow">SUPPLIED REFERENCE</span><h2>System architecture</h2></div><span className="reference-pill"><BadgeCheck size={13} /> PROJECT DIAGRAM</span></div><div className="architecture-frame"><img src="/assets/project-architecture.png" alt="User-supplied architecture diagram showing hybrid BM25 and SBERT/FAISS retrieval, confidence threshold routing, clarification and evaluation" /><div className="architecture-caption"><span>Architecture diagram supplied with the honours project</span><span>BM25 · SBERT / FAISS · CNP · QWEN</span></div></div></section>

    <section className="method-grid"><article className="method-card method-wide"><span className="method-number">02 / RETRIEVAL</span><h2>Two views of relevance.</h2><p>BM25 finds lexical matches; SBERT encodes meaning as vectors, with FAISS indexing the dense representations. Hybrid score fusion combines the two ranking signals before the density/confidence gate evaluates whether the retrieved evidence is sufficient to proceed without a follow-up.</p><div className="signal-pair"><div><span className="signal-tag signal-sparse">SPARSE</span><strong>BM25</strong><small>Token overlap & term weight</small></div><div><span className="signal-tag signal-dense">DENSE</span><strong>SBERT / FAISS</strong><small>Embedding similarity & vector search</small></div></div></article><article className="method-card"><span className="method-number">03 / ROUTING</span><h2>Ask only when useful.</h2><p>Low confidence follows system abstention and direct document retrieval. Sufficient confidence triggers a targeted clarification, then the visitor’s answer enriches the next retrieval.</p><div className="branch-key"><span><i className="route-low" /> Low confidence <b>Retrieve</b></span><span><i className="route-high" /> Gate met <b>Clarify</b></span></div></article></section>

    <section className="dataset-section"><div className="section-title-row"><div><span className="eyebrow">DATASET PROVENANCE</span><h2>Two benchmarks. Distinct label semantics.</h2></div><ShieldCheck size={18} className="section-icon" /></div><div className="dataset-cards"><article className="dataset-card"><div className="dataset-card-head"><span className="dataset-logo clariq-logo">CQ</span><div><h3>ClariQ</h3><span>Native clarification-need judgments</span></div><span className="provenance-badge native-badge">NATIVE</span></div><p>Preserve the original graded `clarification_need` rating (1–4). The binary CNP label used for classification is explicitly derived from the documented threshold; the source rating remains available for audit.</p><div className="label-scale"><span>1 · low</span><i /><i /><i /><span>4 · high</span></div></article><article className="dataset-card"><div className="dataset-card-head"><span className="dataset-logo ambig-logo">AQ</span><div><h3>AmbigNQ</h3><span>Cross-task transfer evaluation</span></div><span className="provenance-badge derived-badge">DERIVED</span></div><p>AmbigNQ does not provide a native CNP judgment. The prototype derives a weak label from annotation type (`multipleQAs` vs `singleAnswer`) and reports it as derived—not as a human clarification annotation.</p><div className="label-scale label-derived"><span>Annotation type</span><ArrowUpRight size={13} /><span>Weak CNP proxy</span></div></article></div></section>

    <section className="policy-section"><div className="section-title-row"><div><span className="eyebrow">EXPERIMENTAL COMPARISON</span><h2>Four policy baselines.</h2></div><span className="policy-count">04 POLICIES</span></div><div className="policy-list">{Object.entries(POLICY_LABELS).map(([key, label], index) => <article className="policy-row" key={key}><span className="policy-index">0{index + 1}</span><div><h3>{label}</h3><p>{POLICY_DESCRIPTIONS[key]}</p></div><ArrowUpRight size={15} className="policy-arrow" /></article>)}</div><div className="metrics-strip"><div><strong>nDCG@5</strong><span>Rank quality</span></div><div><strong>MRR</strong><span>First relevant hit</span></div><div><strong>P@1</strong><span>Top result precision</span></div><div><strong>Recall@100</strong><span>Relevant coverage</span></div><div><strong>Net utility</strong><span>Gain minus ask cost</span></div></div></section>

    <section className="local-data-section"><div className="section-title-row"><div><span className="eyebrow">OPTIONAL LOCAL INSPECTION</span><h2>Inspect your dataset files.</h2></div><span className="local-only-tag"><ShieldCheck size={13} /> STAYS IN THIS BROWSER</span></div><p className="local-data-intro">Choose a ClariQ TSV or AmbigNQ JSON/JSONL file to inspect topic counts and label provenance. The file is parsed locally and is never sent to this website. Official dataset files are not bundled or uploaded by the site.</p><div className="local-data-controls"><label className="field-label">DATASET<select value={datasetKind} onChange={(event) => { setDatasetKind(event.target.value as "clariq" | "ambignq"); setSummary(null); setDatasetError(""); }}><option value="clariq">ClariQ (TSV)</option><option value="ambignq">AmbigNQ (JSON / JSONL)</option></select></label><label className="field-label">SPLIT<select value={split} onChange={(event) => setSplit(event.target.value)}><option value="train">train</option><option value="dev">dev</option><option value="test">test</option></select></label><label className="file-picker"><input type="file" accept={datasetKind === "clariq" ? ".tsv,.txt,text/tab-separated-values" : ".json,.jsonl,application/json"} onChange={handleDataset} /><Upload size={15} /><span>Choose file</span></label></div>
      {datasetError && <p className="inline-error" role="alert">{datasetError}</p>}
      {summary && <div className="dataset-summary"><div className="summary-head"><CheckCircle2 size={16} /><strong>{summary.source} {summary.split} · {summary.total} unique topics</strong><span>LOCAL PARSE COMPLETE</span></div><div className="summary-stats"><div><b>{summary.labelled}</b><span>labelled</span></div><div><b>{summary.positive}</b><span>clarification positive</span></div><div><b>{summary.negative}</b><span>clarification negative</span></div><div><b>{summary.missing}</b><span>unlabelled</span></div></div><p>{summary.provenance}</p></div>}
    </section>

    <section className="report-section"><div className="section-title-row"><div><span className="eyebrow">BENCHMARK OUTPUTS</span><h2>Bring a real evaluation report.</h2></div><FileJson2 size={18} className="section-icon" /></div><p className="local-data-intro">Run the repository's evaluation command with permitted benchmark files and an authorized retrieval corpus. Load its JSON output here to inspect real policy metrics. Without a supplied report, no benchmark scores are displayed.</p><label className="report-drop"><input type="file" accept="application/json,.json" onChange={handleReport} /><FileSpreadsheet size={17} /><span><strong>Load policy-comparison JSON</strong><small>Parsed in this browser · maximum 2 MB</small></span><Upload size={15} /></label>
      {reportError && <p className="inline-error" role="alert">{reportError}</p>}
      {report && <div className="report-table-wrap"><div className="report-status"><span className="status-dot dot-live" /><strong>Imported report</strong><span>{report.test_n ?? "—"} test topics</span><span>Threshold {metric(report.threshold)}</span><span>{report.utility_available ? "Utility included" : "No net-utility inputs"}</span></div><table className="report-table"><thead><tr><th>POLICY</th><th>F1</th><th>FALSE-CLARIFY RATE</th><th>nDCG@5</th><th>NET UTILITY</th></tr></thead><tbody>{policyRows.map(([key, values]) => <tr key={key}><td>{POLICY_LABELS[key] ?? key.replaceAll("_", " ")}</td><td>{metric(values.f1)}</td><td>{metric(values.false_clarification_rate)}</td><td>{metric(values.mean_ndcg5)}</td><td>{metric(values.net_utility)}</td></tr>)}</tbody></table><p className="report-provenance">Label provenance: {Array.isArray(report.test_label_provenance) ? report.test_label_provenance.join(", ") : String(report.test_label_provenance ?? "not specified in report")} · Report values are shown as supplied; verify source, split and corpus provenance before academic use.</p></div>}
    </section>

    <div className="research-note"><BookOpen size={16} /><p><strong>Research integrity note.</strong> This public demo does not contain licensed benchmark records or a domain retrieval corpus. The seven bundled documents are synthetic. The website will not invent metric values or label synthetic results as benchmark findings.</p></div>
    <div className="research-bottom"><Link href="/" className="back-link"><ArrowLeft size={14} /> Return to search</Link><span>RETRIEVAL-GATED CLARIFICATION · RESEARCH PROTOTYPE</span></div>
  </main><footer className="site-footer"><span>CLARiQ <i>·</i> HONOURS RESEARCH PROTOTYPE</span><span>DATA PROVENANCE IS PART OF THE RESULT</span></footer></div>;
}
