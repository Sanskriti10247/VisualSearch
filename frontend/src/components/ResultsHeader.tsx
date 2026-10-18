import type { SearchResponse } from "../api";
import { fmtInt, fmtMs } from "../lib/format";
import { FUSION_LABEL, RERANK_LABEL } from "../lib/search";
import { TimingBar } from "./TimingBar";

/** Count, latency, stage timings, strategy facts and warnings for one response. */
export function ResultsHeader({ res, compact }: { res: SearchResponse; compact?: boolean }) {
  const s = res.strategy;
  const facts: [string, string][] = [];
  if (s.mode) facts.push(["mode", s.mode]);
  if (s.model) facts.push(["model", s.model]);
  if (s.text_model) facts.push(["text model", s.text_model]);
  if (s.image_model) facts.push(["image model", s.image_model]);
  if (s.fusion) facts.push(["fusion", `${FUSION_LABEL[s.fusion] ?? s.fusion}${s.text_weight != null ? ` · w=${s.text_weight}` : ""}`]);
  if (s.index) facts.push(["index", s.index]);
  if (s.filter_strategy) facts.push(["filter", s.filter_strategy]);
  if (s.filter_matches != null) facts.push(["filter matches", fmtInt(s.filter_matches)]);
  if (s.candidates != null) facts.push(["candidates", fmtInt(s.candidates)]);
  if (s.rerank && s.rerank !== "none") facts.push(["rerank", RERANK_LABEL[s.rerank] ?? s.rerank]);
  if (s.rerank_params) facts.push(["rerank params", Object.entries(s.rerank_params).map(([k, v]) => `${k}=${String(v)}`).join(" ")]);

  return (
    <section className={`results-head${compact ? " compact" : ""}`} aria-label="Search summary">
      {res.warnings.length > 0 && (
        <div className="notice notice-warn" role="status">
          {res.warnings.map((w, i) => <p key={i}>{w}</p>)}
        </div>
      )}
      <div className="results-stats">
        <p className="results-count">
          <strong>{fmtInt(res.total)}</strong> result{res.total === 1 ? "" : "s"}
          <span className="muted"> in </span><strong>{fmtMs(res.timings_ms.total)}</strong>
        </p>
        <TimingBar timings={res.timings_ms} />
      </div>
      <dl className="facts">
        {facts.map(([k, v]) => (
          <div key={k}><dt>{k}</dt><dd>{v}</dd></div>
        ))}
      </dl>
    </section>
  );
}
