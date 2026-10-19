import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import { Link } from "wouter";
import {
  Activity, ArrowDown, ArrowRight, ArrowUpRight, Check, ChevronDown, CircleHelp,
  Database, FlaskConical, GitBranch, LoaderCircle, MessageSquareText, Search,
  ShieldCheck, Sparkles, Waypoints,
} from "lucide-react";

type Hit = {
  id: string;
  title: string;
  text: string;
  score: number;
  sparse_score: number;
  dense_score: number;
  metadata: Record<string, unknown>;
};
type RouteResult = {
  query: string;
  requested_policy: string;
  policy_used: string;
  route: "clarify" | "retrieve";
  clarification_probability_or_diagram_score: number;
  diagram_score: number;
  diagram_threshold: number;
  cnp_threshold: number;
  clarification_question: string | null;
  question_generator: "qwen" | "local_fallback" | null;
  fallback_reason: string | null;
  reason: string;
  retrieval_signals: Record<string, number>;
  results: Hit[];
};
type FollowUp = { original_query: string; resolved_query: string; results: Hit[] };
type Health = { status: string; documents: number; dense_enabled: boolean; dense_error: string | null; qwen_configured: boolean; loaded_cnp_models: string[] };

const policies = [
  { value: "diagram", label: "Architecture threshold", note: "Your diagram's confidence gate" },
  { value: "direct_retrieval", label: "Direct retrieval", note: "Never asks a clarification" },
  { value: "always_clarify", label: "Always clarify", note: "Asks on every query" },
  { value: "query_only_cnp", label: "Query-only CNP", note: "Learned query signal" },
  { value: "retrieval_informed_cnp", label: "Retrieval-informed CNP", note: "Learned query + retrieval signals" },
];
const examples = ["Matariki public holiday", "Who is jaguar?", "Hybrid information retrieval"];
const stageClasses = ["stage-query", "stage-retrieval", "stage-fusion", "stage-confidence", "stage-routing"];

function pct(value: number | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
}
function fixed(value: number | undefined, digits = 3) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function Header({ qwen }: { qwen?: boolean }) {
  return (
    <header className="site-header">
      <Link href="/" className="brand" aria-label="ClariQ home">
        <span className="brand-mark" aria-hidden="true"><Waypoints size={19} strokeWidth={1.8} /></span>
        <span className="brand-word">CLAR<span>i</span>Q</span>
        <span className="brand-divider" />
        <span className="brand-subtitle">CONVERSATIONAL SEARCH</span>
      </Link>
      <nav className="top-nav" aria-label="Main navigation">
        <Link href="/" className="nav-link nav-active">Workspace</Link>
        <Link href="/research" className="nav-link">Research notes <ArrowUpRight size={13} /></Link>
      </nav>
      <div className="header-status" title={qwen ? "Qwen clarification is configured" : "Local clarification fallback is active"}>
        <span className={`status-dot ${qwen ? "dot-live" : "dot-muted"}`} />
        <span>{qwen ? "QWEN CONNECTED" : "LOCAL FALLBACK"}</span>
      </div>
    </header>
  );
}

function Pipeline({ result }: { result: RouteResult | null }) {
  const labels = ["First-turn query", "Hybrid retrieval", "Score fusion", "Density & confidence", "System routing"];
  const details = [
    "Visitor intent enters the pipeline",
    "BM25 sparse + SBERT / FAISS dense",
    "Ranked evidence is combined",
    result ? `Score ${fixed(result.diagram_score)} · threshold ${fixed(result.diagram_threshold)}` : "Awaiting retrieval evidence",
    result ? result.route === "clarify" ? "Proactive clarification" : "Abstain · retrieve directly" : "Awaiting a query",
  ];
  return (
    <div className="pipeline-list" aria-label="Conversational search pipeline">
      {labels.map((label, index) => {
        const active = Boolean(result) && index < 5;
        const branch = index === 4 && result ? result.route === "clarify" ? "branch-clarify" : "branch-retrieve" : "";
        return (
          <div className={`pipeline-stage ${stageClasses[index]} ${active ? "stage-complete" : ""} ${branch}`} key={label}>
            <span className="stage-index">0{index + 1}</span>
            <div className="stage-copy"><strong>{label}</strong><small>{details[index]}</small></div>
            {active ? <Check size={14} className="stage-check" /> : <span className="stage-pending" />}
          </div>
        );
      })}
    </div>
  );
}

function ScoreBar({ label, value, tone }: { label: string; value: number; tone: string }) {
  const width = `${Math.max(0, Math.min(100, value * 100))}%`;
  return (
    <div className="score-line">
      <span>{label}</span><span className="score-track"><i className={tone} style={{ width }} /></span><b>{fixed(value)}</b>
    </div>
  );
}

function Results({ hits, heading = "Retrieved documents" }: { hits: Hit[]; heading?: string }) {
  if (!hits.length) return <div className="empty-result">No matching documents were returned.</div>;
  return (
    <section className="results-block">
      <div className="section-title-row"><h3>{heading}</h3><span className="result-count">{hits.length} DOCUMENTS</span></div>
      <div className="result-list">
        {hits.map((hit, index) => (
          <article className="result-card" key={`${hit.id}-${index}`}>
            <div className="result-rank">{String(index + 1).padStart(2, "0")}</div>
            <div className="result-content">
              <div className="result-title-row"><h4>{hit.title || hit.id}</h4><span className="synthetic-tag">SYNTHETIC DEMO</span></div>
              <p>{hit.text}</p>
              <div className="score-bars">
                <ScoreBar label="Fused" value={hit.score} tone="bar-fused" />
                <ScoreBar label="BM25" value={hit.sparse_score} tone="bar-sparse" />
                <ScoreBar label="Dense" value={hit.dense_score} tone="bar-dense" />
              </div>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [policy, setPolicy] = useState("diagram");
  const [result, setResult] = useState<RouteResult | null>(null);
  const [followUp, setFollowUp] = useState<FollowUp | null>(null);
  const [answer, setAnswer] = useState("");
  const [health, setHealth] = useState<Health | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    fetch("/api/health").then((response) => response.json()).then((data: Health) => { if (alive) setHealth(data); }).catch(() => undefined);
    return () => { alive = false; };
  }, []);

  const selectedPolicy = useMemo(() => policies.find((item) => item.value === policy)?.label ?? policy, [policy]);

  async function submitQuery(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const value = query.trim();
    if (!value || busy) return;
    setBusy(true); setError(""); setFollowUp(null); setResult(null); setAnswer("");
    try {
      const response = await fetch("/api/route", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: value, policy, top_k: 5 }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? "Search could not be completed.");
      setResult(payload as RouteResult);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Search could not be completed. Please retry.");
    } finally { setBusy(false); }
  }

  async function submitAnswer(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!result?.clarification_question || !answer.trim() || busy) return;
    setBusy(true); setError("");
    try {
      const response = await fetch("/api/follow-up", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: result.query, clarification_question: result.clarification_question, user_answer: answer.trim(), top_k: 5 }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? "Follow-up retrieval could not be completed.");
      setFollowUp(payload as FollowUp);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Follow-up retrieval could not be completed.");
    } finally { setBusy(false); }
  }

  return (
    <div className="app-shell">
      <Header qwen={health?.qwen_configured} />
      <main className="workspace-shell">
        <div className="page-kicker"><span className="eyebrow-mark" /> RESEARCH SYSTEM / LIVE DEMO <span className="kicker-rule" /> <span>v0.3</span></div>
        <div className="workspace-heading">
          <div><h1>Search that knows<br /><em>when to ask.</em></h1><p>Explore a retrieval-gated conversation. Every route is paired with the evidence that caused it.</p></div>
          <div className="privacy-note"><ShieldCheck size={15} /><span>Queries aren’t saved<br />between sessions</span></div>
        </div>
        <div className="demo-banner"><span className="demo-indicator" /><strong>Illustrative demo corpus</strong><span>These seven documents are synthetic examples, not ClariQ/AmbigNQ benchmark evidence.</span><Link href="/research">Data & methodology <ArrowRight size={13} /></Link></div>

        <div className="workspace-grid">
          <section className="conversation-column">
            <form className="query-panel" onSubmit={submitQuery}>
              <div className="panel-topline"><span><Search size={15} /> FIRST-TURN SEARCH</span><span className="mono-label">TURN 01</span></div>
              <label className="sr-only" htmlFor="query-box">Search query</label>
              <textarea id="query-box" value={query} onChange={(event) => setQuery(event.target.value)} maxLength={2000} placeholder="What are you trying to find?" rows={3} />
              <div className="query-tools">
                <label className="policy-select"><span className="sr-only">Routing policy</span><select value={policy} onChange={(event) => setPolicy(event.target.value)} aria-label="Routing policy">{policies.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select><ChevronDown size={13} /></label>
                <button className="primary-button" type="submit" disabled={busy || !query.trim()}>{busy ? <LoaderCircle size={16} className="spin" /> : <Search size={16} />}<span>{busy ? "Retrieving" : "Search"}</span><ArrowRight size={14} /></button>
              </div>
              <div className="query-footnote"><span>{selectedPolicy}</span><span>{query.length}/2000</span></div>
            </form>

            {!result && !busy && <div className="example-section"><span className="eyebrow">TRY A SYNTHETIC EXAMPLE</span><div className="example-chips">{examples.map((item) => <button key={item} type="button" onClick={() => setQuery(item)}>{item}<ArrowUpRight size={12} /></button>)}</div></div>}
            {busy && <div className="loading-state"><span className="loader-orbit"><Search size={18} /></span><div><strong>Following the evidence trail</strong><span>Running hybrid retrieval and confidence routing…</span></div></div>}
            {error && <div className="error-message" role="alert"><CircleHelp size={17} /><span>{error}</span></div>}

            {result && !busy && <div className="response-stack">
              <section className={`decision-card ${result.route === "clarify" ? "decision-clarify" : "decision-retrieve"}`}>
                <div className="decision-icon">{result.route === "clarify" ? <MessageSquareText size={18} /> : <Database size={18} />}</div>
                <div className="decision-copy"><span className="eyebrow">SYSTEM ROUTE · {result.policy_used.replaceAll("_", " ").toUpperCase()}</span><h2>{result.route === "clarify" ? "Clarify before retrieval" : "Retrieve without asking"}</h2><p>{result.route === "clarify" ? "Confidence met the clarification gate. The system is asking for the facet that can narrow the search." : "The system abstains from clarification and returns the strongest available documents."}</p></div>
                <div className="decision-score"><strong>{pct(result.clarification_probability_or_diagram_score)}</strong><small>{result.policy_used.includes("cnp") && !result.fallback_reason ? "CNP probability" : "confidence score"}</small></div>
              </section>
              {result.fallback_reason && <div className="notice-line"><CircleHelp size={14} /><span>{result.fallback_reason}</span></div>}
              {result.route === "clarify" && result.clarification_question && <section className="clarification-card">
                <div className="clarification-head"><span className="clarification-icon"><Sparkles size={16} /></span><div><span className="eyebrow">CLARIFYING QUESTION</span><span className="generator-tag">{result.question_generator === "qwen" ? "QWEN AI" : "LOCAL FALLBACK"}</span></div></div>
                <p className="clarification-text">{result.clarification_question}</p>
                <form className="answer-form" onSubmit={submitAnswer}><label htmlFor="clarification-answer" className="sr-only">Your answer</label><input id="clarification-answer" value={answer} onChange={(event) => setAnswer(event.target.value)} maxLength={2000} placeholder="Add a little more context…" /><button type="submit" className="answer-button" disabled={busy || !answer.trim()}>{busy ? <LoaderCircle size={16} className="spin" /> : <ArrowRight size={16} />}<span>Continue search</span></button></form>
              </section>}
              <Results hits={result.results} />
              {followUp && <section className="follow-up-result"><div className="follow-up-heading"><span><Check size={14} /> FOLLOW-UP RETRIEVAL</span><code>{followUp.resolved_query}</code></div><Results hits={followUp.results} heading="Updated results after clarification" /></section>}
            </div>}
          </section>

          <aside className="instrument-column">
            <section className="instrument-panel">
              <div className="instrument-heading"><div><span className="eyebrow">SYSTEM INSTRUMENTATION</span><h2>Decision path</h2></div><Activity size={17} className="instrument-pulse" /></div>
              <Pipeline result={result} />
              <div className="confidence-panel">
                <div className="confidence-title"><span>RETRIEVAL CONFIDENCE</span><span>{result ? fixed(result.diagram_threshold) : "—"} THRESHOLD</span></div>
                <div className="confidence-meter"><div className="meter-fill" style={{ width: result ? `${Math.min(result.diagram_score * 100, 100)}%` : "0%" }} /><span className="meter-threshold" style={{ left: `${Math.min((result?.diagram_threshold ?? 0.35) * 100, 100)}%` }} /></div>
                <div className="meter-values"><span>0.00</span><strong>{result ? fixed(result.diagram_score) : "Awaiting query"}</strong><span>1.00</span></div>
                <div className="confidence-legend"><span><i className="legend-low" /> Below gate · retrieve</span><span><i className="legend-high" /> Gate met · clarify</span></div>
              </div>
              <div className="retrieval-status"><div><span className={`status-dot ${health?.dense_enabled ? "dot-live" : "dot-muted"}`} /><span>Dense model</span></div><strong>{health?.dense_enabled ? "SBERT / FAISS" : "Not ready"}</strong></div>
              {health?.dense_error && <p className="model-warning">Dense initialization fallback: {health.dense_error}</p>}
              <details className="signals-details"><summary>Inspect retrieval signals <ChevronDown size={14} /></summary>
                {result?.retrieval_signals ? <div className="signal-grid">{Object.entries(result.retrieval_signals).map(([key, value]) => <div key={key}><span>{key.replaceAll("_", " ")}</span><b>{fixed(value)}</b></div>)}</div> : <p className="signals-empty">Run a query to reveal measured signals.</p>}
              </details>
              <div className="instrument-foot"><Waypoints size={14} /><span>{health ? `${health.documents} indexed demo documents` : "Checking retrieval service…"}</span><span className="foot-separator">·</span><span>{health?.loaded_cnp_models?.length ? "CNP models loaded" : "CNP models not loaded"}</span></div>
            </section>
            <Link href="/research" className="research-link-card"><span className="research-icon"><FlaskConical size={17} /></span><span><strong>Research & evaluation</strong><small>Dataset provenance, policies, metrics</small></span><ArrowUpRight size={16} /></Link>
            <div className="session-note"><ShieldCheck size={14} /><span>This session is transient. Search queries and clarification answers are not stored.</span></div>
          </aside>
        </div>
      </main>
      <footer className="site-footer"><span>CLARiQ <i>·</i> HONOURS RESEARCH PROTOTYPE</span><span>HYBRID RETRIEVAL <i>·</i> CONFIDENCE-GATED CLARIFICATION</span></footer>
    </div>
  );
}
