/** Number / text formatting helpers (pure, unit-tested). */

export function fmtMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return "–";
  if (ms < 1) return `${ms.toFixed(2)} ms`;
  if (ms < 100) return `${ms.toFixed(1)} ms`;
  if (ms < 10_000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

export function fmtBytes(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "–";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  let i = 0;
  while (Math.abs(v) >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${i === 0 ? v : v.toFixed(v >= 100 ? 0 : 1)} ${units[i]}`;
}

export function fmtInt(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "–";
  return Math.round(n).toLocaleString("en-US");
}

export function fmtDuration(s: number): string {
  if (!Number.isFinite(s)) return "–";
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (d) return `${d}d ${h}h`;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${Math.floor(s % 60)}s`;
  return `${Math.floor(s)}s`;
}

/**
 * "top x%" label from a percentile (fraction of random pairs scoring lower).
 * percentile 0.977 -> "top 2.3%".
 */
export function topPercent(percentile: number | null | undefined): string | null {
  if (percentile === null || percentile === undefined || !Number.isFinite(percentile)) return null;
  const top = Math.min(1, Math.max(0, 1 - percentile)) * 100;
  if (top < 0.01) return "top <0.01%";
  if (top < 0.1) return `top ${top.toFixed(2)}%`;
  if (top < 10) return `top ${top.toFixed(1)}%`;
  return `top ${Math.round(top)}%`;
}

const METRIC_RE = /(^|[^a-z])(r@\d+|p@\d+|hits?@\d+|recall|precision|mrr|map|ndcg|acc(uracy)?|top-?\d+)(?![a-z])/i;

/** Does a table column name look like a retrieval metric (R@k, MRR, mAP, P@k, recall...)? */
export function isMetricColumn(name: string): boolean {
  return METRIC_RE.test(name.trim());
}

/** Generic number formatting with sensible rounding. */
export function fmtNumber(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  if (Number.isInteger(v)) return Math.abs(v) >= 10_000 ? v.toLocaleString("en-US") : String(v);
  const a = Math.abs(v);
  if (a >= 1000) return Math.round(v).toLocaleString("en-US");
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  if (a >= 0.001) return v.toFixed(3);
  return v.toPrecision(2);
}

/** Format one experiment-table cell given its column name. */
export function fmtCell(column: string, v: unknown): string {
  if (v === null || v === undefined) return "–";
  if (typeof v === "number") {
    if (isMetricColumn(column) && v >= 0 && v <= 1 && !Number.isInteger(v)) {
      return `${(v * 100).toFixed(1)}%`;
    }
    if (isMetricColumn(column) && (v === 0 || v === 1)) return `${(v * 100).toFixed(1)}%`;
    return fmtNumber(v);
  }
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

/** Indices of the rows holding the max numeric value in `col`. */
export function bestRows(rows: unknown[][], col: number): Set<number> {
  let best = -Infinity;
  const out = new Set<number>();
  rows.forEach((r, i) => {
    const v = r[col];
    if (typeof v !== "number" || !Number.isFinite(v)) return;
    if (v > best) {
      best = v;
      out.clear();
      out.add(i);
    } else if (v === best) {
      out.add(i);
    }
  });
  return out;
}

export function shortHash(s: string | null | undefined, n = 8): string {
  if (!s) return "–";
  return s.length > n ? s.slice(0, n) : s;
}

export function humanize(key: string): string {
  return key.replace(/_/g, " ");
}
