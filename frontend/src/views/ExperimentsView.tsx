import { useEffect, useState } from "react";
import { api, type ExperimentsResponse } from "../api";
import { Chart } from "../components/Chart";
import { DataTable } from "../components/DataTable";
import { ErrorView } from "../components/ErrorView";
import { JsonView } from "../components/JsonView";
import { humanize } from "../lib/format";

export function ExperimentsView() {
  const [data, setData] = useState<ExperimentsResponse | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    const ctrl = new AbortController();
    setError(null);
    api.experiments(ctrl.signal).then((d) => {
      setData(d);
      setSelected((s) => (s && s in d.results ? s : Object.keys(d.results)[0] ?? null));
    }).catch((e) => { if (!ctrl.signal.aborted) setError(e); });
    return () => ctrl.abort();
  }, [nonce]);

  if (error) return <div className="view"><ErrorView error={error} onRetry={() => setNonce((n) => n + 1)} /></div>;
  if (!data) return <div className="view"><p className="hint">Loading experiments…</p></div>;
  const names = Object.keys(data.results);
  if (!names.length) {
    return (
      <div className="view">
        <div className="empty">
          <h3>No experiment results yet</h3>
          <p>
            Run the benchmark suite to produce machine-readable results, then reload this page:
          </p>
          <pre className="cmd">python scripts/run_experiments.py</pre>
          <p className="hint">Each experiment writes a JSON file into the results directory; files with a
            {" "}<code>report</code> field are rendered as tables and charts here.</p>
        </div>
      </div>
    );
  }
  const current = selected ? data.results[selected] : undefined;
  const { report, ...rest } = current ?? {};

  return (
    <div className="view experiments">
      <nav className="exp-list" aria-label="Experiments">
        <div className="exp-list-head">
          <h2>Experiments</h2>
          <button className="btn btn-ghost btn-sm" onClick={() => setNonce((n) => n + 1)}>Reload</button>
        </div>
        <ul>
          {names.map((n) => (
            <li key={n}>
              <button type="button" className={n === selected ? "active" : ""} aria-current={n === selected ? "page" : undefined}
                onClick={() => setSelected(n)}>
                {data.results[n].report?.title ?? humanize(n)}
                <span className="hint mono">{n}</span>
              </button>
            </li>
          ))}
        </ul>
      </nav>
      <article className="exp-body">
        {current && (
          <>
            <h2>{report?.title ?? humanize(selected!)}</h2>
            {report?.description?.split(/\n\s*\n/).map((para, i) => <p key={i} className="exp-desc">{para}</p>)}
            {!report && <p className="hint">This result has no <code>report</code> section; showing the raw JSON.</p>}
            {report?.tables?.map((t, i) => <DataTable key={i} table={t} />)}
            {report?.charts && report.charts.length > 0 && (
              <div className="charts">
                {report.charts.map((c, i) => <Chart key={i} chart={c} />)}
              </div>
            )}
            {Object.keys(rest).length > 0 && <JsonView value={rest} summary={report ? "Raw result fields" : "Raw JSON"} open={!report} />}
          </>
        )}
      </article>
    </div>
  );
}
