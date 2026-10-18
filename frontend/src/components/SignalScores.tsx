import type { SignalScore } from "../api";
import { topPercent } from "../lib/format";

/** Per-signal scores: raw cosine, "top x%", calibrated P(match) meter. */
export function SignalScores({ signals }: { signals: Record<string, SignalScore> }) {
  return (
    <ul className="signals">
      {Object.entries(signals).map(([key, s]) => {
        const [model, kind] = key.split(":");
        const top = topPercent(s.percentile);
        return (
          <li key={key} className="signal">
            <span className="signal-name" title={key}>
              {model}{kind && <span className="muted"> · {kind}</span>}
            </span>
            <span className="signal-vals">
              <span title="cosine similarity" className="mono">{s.raw.toFixed(3)}</span>
              {top && <span className="tag" title="vs. random pairs on validation data">{top}</span>}
            </span>
            {s.prob != null && (
              <span className="prob" title={`calibrated P(match) = ${(s.prob * 100).toFixed(1)}%`}>
                <meter min={0} max={1} low={0.2} high={0.6} optimum={1} value={s.prob} aria-label="P(match)" />
                <span className="mono">{(s.prob * 100).toFixed(0)}%</span>
              </span>
            )}
          </li>
        );
      })}
    </ul>
  );
}
