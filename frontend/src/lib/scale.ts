/** Axis scales and tick generation for the hand-rolled SVG charts. */

/** "Nice" linear ticks covering [min, max] (about `count` of them). */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) {
    const d = Math.abs(min) || 1;
    min -= d * 0.5;
    max += d * 0.5;
  }
  if (min > max) [min, max] = [max, min];
  const raw = (max - min) / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const start = Math.floor(min / step) * step;
  const end = Math.ceil(max / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= end + step * 1e-9; v += step) {
    ticks.push(Number(v.toPrecision(12)));
  }
  return ticks;
}

/** Powers-of-ten ticks for a log axis (falls back to 1-2-5 when the span is narrow). */
export function logTicks(min: number, max: number): number[] {
  if (!(min > 0) || !(max > 0)) return [];
  const lo = Math.floor(Math.log10(min));
  const hi = Math.ceil(Math.log10(max));
  const ticks: number[] = [];
  const mult = hi - lo <= 2 ? [1, 2, 5] : [1];
  for (let e = lo; e <= hi; e++) {
    for (const m of mult) {
      const v = m * Math.pow(10, e);
      if (v >= min / 1.0001 && v <= max * 1.0001) ticks.push(Number(v.toPrecision(12)));
    }
  }
  if (ticks.length < 2) return [min, max];
  return ticks;
}

export interface Scale {
  (v: number): number;
  domain: [number, number];
}

export function linearScale(d0: number, d1: number, r0: number, r1: number): Scale {
  const span = d1 - d0 || 1;
  const f = ((v: number) => r0 + ((v - d0) / span) * (r1 - r0)) as Scale;
  f.domain = [d0, d1];
  return f;
}

export function logScale(d0: number, d1: number, r0: number, r1: number): Scale {
  const l0 = Math.log10(d0);
  const l1 = Math.log10(d1);
  const span = l1 - l0 || 1;
  const f = ((v: number) => r0 + ((Math.log10(v) - l0) / span) * (r1 - r0)) as Scale;
  f.domain = [d0, d1];
  return f;
}

export function fmtTick(v: number): string {
  const a = Math.abs(v);
  if (a >= 1e6) return `${+(v / 1e6).toPrecision(3)}M`;
  if (a >= 1e3) return `${+(v / 1e3).toPrecision(3)}k`;
  return String(+v.toPrecision(4));
}
