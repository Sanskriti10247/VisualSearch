import { useState } from "react";
import { api, type HealthResponse, type LatencyStats, type MetricsResponse, type ModelsResponse } from "../api";
import { ErrorView } from "../components/ErrorView";
import { JsonView } from "../components/JsonView";
import { fmtBytes, fmtDuration, fmtInt, fmtMs, fmtNumber, shortHash } from "../lib/format";
import { useInterval, usePageVisible } from "../lib/hooks";

interface Snapshot {
  health: HealthResponse | null;
  models: ModelsResponse | null;
  metrics: MetricsResponse | null;
  at: Date;
}

export function SystemView({ active }: { active: boolean }) {
  const visible = usePageVisible();
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [error, setError] = useState<unknown>(null);

  useInterval(async (signal) => {
    try {
      const [health, models, metrics] = await Promise.all([
        api.health(signal), api.models(signal), api.metrics(signal)]);
      setSnap({ health, models, metrics, at: new Date() });
      setError(null);
    } catch (e) {
      if (!signal.aborted) setError(e);
    }
  }, 5000, active && visible);

  const h = snap?.health;
  const status = error ? "unreachable" : h?.status ?? "loading";

  return (
    <div className="view">
      <section className="panel system-head">
        <div>
          <h2>System status</h2>
          <p className="hint">
            Auto-refreshes every 5 s while this tab is visible
            {snap && <> · last update {snap.at.toLocaleTimeString()}</>}
          </p>
        </div>
        <span className={`status-pill status-${status}`} role="status">{status}</span>
      </section>

      {error != null && <ErrorView error={error} />}

      {h && snap && (
        <section className="stats-row" aria-label="Overview">
          <Stat label="version" value={h.version} />
          <Stat label="catalog images" value={fmtInt(h.catalog_images)} />
          <Stat label="index type" value={h.index_type} />
          {h.read_only && <Stat label="mode" value="read-only" />}
          {snap.metrics && <Stat label="uptime" value={fmtDuration(snap.metrics.uptime_s)} />}
        </section>
      )}

      {snap?.models && (
        <section>
          <h2 className="section-title">Models</h2>
          <div className="model-cards">
            {snap.models.models.map((m) => {
              const hm = h?.models[m.key];
              const disk = snap.metrics?.indexes[m.key]?.disk_bytes;
              return (
                <article key={m.key} className="model-card">
                  <header>
                    <h3>{m.key}</h3>
                    <span className={`status-pill status-${m.stale ? "degraded" : hm?.ready === false ? "degraded" : "ok"}`}>
                      {m.stale ? "stale" : hm?.ready === false ? "not ready" : "ready"}
                    </span>
                  </header>
                  {m.description && <p className="hint">{m.description}</p>}
                  <dl className="kv-grid">
                    <dt>HF id</dt><dd className="mono wrap">{m.hf_id}</dd>
                    <dt>revision</dt><dd className="mono">{shortHash(m.revision ?? null, 10)}</dd>
                    <dt>dim</dt><dd>{m.dim}</dd>
                    <dt>modalities</dt><dd>{m.modalities.join(", ")}</dd>
                    <dt>device</dt><dd>{m.device}{m.loaded === false ? " (not loaded)" : ""}</dd>
                    {m.runtime && <><dt>runtime</dt><dd>{m.runtime}{m.precision ? ` · ${m.precision}` : ""}</dd></>}
                    <dt>index</dt>
                    <dd>{m.index ? <>{m.index.kind} · {fmtInt(m.index.ntotal)} vectors</> : <span className="muted">none</span>}</dd>
                    {typeof disk === "number" && <><dt>on disk</dt><dd>{fmtBytes(disk)}</dd></>}
                    <dt>fingerprint</dt><dd className="mono" title={m.fingerprint}>{shortHash(m.fingerprint, 12)}</dd>
                  </dl>
                  {m.stale && (
                    <p className="hint hint-warn">The index was built with a different embedding fingerprint; rebuild it.</p>
                  )}
                </article>
              );
            })}
          </div>
        </section>
      )}

      {snap?.models && (
        <section className="panel">
          <h2 className="section-title">Score calibrations</h2>
          {snap.models.calibrations.length === 0 ? (
            <p className="hint">No calibrations fitted: “top x%” and P(match) are unavailable.</p>
          ) : (
            <div className="table-scroll">
              <table className="data">
                <thead><tr><th scope="col">Space</th><th scope="col">Query</th><th className="num" scope="col">Mean</th><th className="num" scope="col">Std</th><th scope="col">Fitted on</th></tr></thead>
                <tbody>
                  {snap.models.calibrations.map((c) => (
                    <tr key={`${c.space}:${c.query_modality}`}>
                      <th scope="row">{c.space}</th>
                      <td>{c.query_modality}</td>
                      <td className="num">{fmtNumber(c.mean)}</td>
                      <td className="num">{fmtNumber(c.std)}</td>
                      <td className="small">{typeof c.fitted_on === "string" ? c.fitted_on : JSON.stringify(c.fitted_on)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      {snap?.metrics && (
        <div className="two-col-sections">
          <LatencyTable title="Route latency" rows={snap.metrics.routes} counts={snap.metrics.status_counts} />
          <LatencyTable title="Search stages" rows={snap.metrics.search_stages} />
        </div>
      )}

      {snap && <JsonView value={snap} summary="Raw responses" />}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
    </div>
  );
}

function LatencyTable({ title, rows, counts }: {
  title: string;
  rows: Record<string, LatencyStats>;
  counts?: Record<string, Record<string, number>>;
}) {
  const entries = Object.entries(rows).sort((a, b) => a[0].localeCompare(b[0]));
  return (
    <section className="panel">
      <h2 className="section-title">{title}</h2>
      {entries.length === 0 ? <p className="hint">No observations yet.</p> : (
        <div className="table-scroll">
          <table className="data">
            <thead>
              <tr>
                <th scope="col">Name</th><th className="num" scope="col">p50</th><th className="num" scope="col">p95</th>
                <th className="num" scope="col">count</th>{counts && <th scope="col">status</th>}
              </tr>
            </thead>
            <tbody>
              {entries.map(([name, s]) => (
                <tr key={name}>
                  <th scope="row" className="mono small">{name}</th>
                  <td className="num">{fmtMs(s.p50_ms)}</td>
                  <td className="num">{fmtMs(s.p95_ms)}</td>
                  <td className="num">{fmtInt(s.count)}</td>
                  {counts && (
                    <td className="small">
                      {Object.entries(counts[name] ?? {}).map(([code, n]) => (
                        <span key={code} className={`tag tag-${code[0]}xx`}>{code}×{n}</span>
                      ))}
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
