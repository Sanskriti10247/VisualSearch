import { fmtMs } from "../lib/format";

const STAGES = ["embed_query", "search", "rerank", "fuse", "hydrate"] as const;

/** Small stacked bar of per-stage timings. */
export function TimingBar({ timings }: { timings: Record<string, number> }) {
  const parts = STAGES.filter((s) => (timings[s] ?? 0) > 0).map((s) => ({ stage: s, ms: timings[s] }));
  const extra = Object.keys(timings).filter((k) => k !== "total" && !(STAGES as readonly string[]).includes(k));
  for (const k of extra) if (timings[k] > 0) parts.push({ stage: k as (typeof STAGES)[number], ms: timings[k] });
  const sum = parts.reduce((a, p) => a + p.ms, 0) || 1;
  return (
    <div className="timing">
      <div className="timing-bar" role="img"
        aria-label={parts.map((p) => `${p.stage} ${fmtMs(p.ms)}`).join(", ")}>
        {parts.map((p) => (
          <span key={p.stage} className={`seg seg-${p.stage}`} style={{ flexGrow: p.ms / sum }}
            title={`${p.stage}: ${fmtMs(p.ms)}`} />
        ))}
      </div>
      <ul className="timing-legend">
        {parts.map((p) => (
          <li key={p.stage}><span className={`swatch seg-${p.stage}`} />{p.stage.replace("_", " ")} {fmtMs(p.ms)}</li>
        ))}
      </ul>
    </div>
  );
}
