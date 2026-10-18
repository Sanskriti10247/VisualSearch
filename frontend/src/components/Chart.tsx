import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import type { ExperimentChart } from "../api";
import { fmtNumber } from "../lib/format";
import { fmtTick, linearScale, logScale, logTicks, niceTicks } from "../lib/scale";

const H = 300;
const M = { top: 14, right: 18, bottom: 46, left: 58 };
const PH = H - M.top - M.bottom;

const color = (i: number) => `var(--series-${(i % 8) + 1})`;

interface Tip {
  x: number; // svg coords
  y: number;
  title: string;
  rows: { name: string; value: string; i: number }[];
}

/**
 * Small dependency-free SVG chart: line / scatter / bar, nice ticks, optional
 * log x-axis, legend for >= 2 series, hover tooltips.
 */
export function Chart({ chart }: { chart: ExperimentChart }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [tip, setTip] = useState<Tip | null>(null);
  // Track the container width so text renders at its true pixel size.
  const [W, setW] = useState(640);
  useEffect(() => {
    const el = wrapRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const PW = W - M.left - M.right;
  const series = chart.series ?? [];
  const categorical = chart.type === "bar" ||
    series.some((s) => s.points.some((p) => typeof p[0] !== "number"));
  const logX = !!chart.log_x && !categorical;

  const geo = useMemo(() => {
    const ys = series.flatMap((s) => s.points.map((p) => p[1])).filter((v) => Number.isFinite(v));
    let yMin = Math.min(...ys);
    let yMax = Math.max(...ys);
    if (!ys.length) { yMin = 0; yMax = 1; }
    if (chart.type === "bar") { yMin = Math.min(0, yMin); yMax = Math.max(0, yMax); }
    const yt = niceTicks(yMin, yMax, 5);
    const y = linearScale(yt[0], yt[yt.length - 1], PH, 0);

    if (categorical) {
      const cats: string[] = [];
      for (const s of series) for (const p of s.points) if (!cats.includes(String(p[0]))) cats.push(String(p[0]));
      const band = PW / Math.max(1, cats.length);
      return { categorical: true as const, cats, band, y, yt };
    }
    const xs = series.flatMap((s) => s.points.map((p) => p[0] as number))
      .filter((v) => Number.isFinite(v) && (!logX || v > 0));
    const xMin = xs.length ? Math.min(...xs) : 0;
    const xMax = xs.length ? Math.max(...xs) : 1;
    let xt: number[];
    let x;
    if (logX) {
      xt = logTicks(xMin, xMax);
      x = logScale(Math.min(xMin, xt[0]), Math.max(xMax, xt[xt.length - 1]), 0, PW);
    } else {
      xt = niceTicks(xMin, xMax, Math.max(2, Math.round(PW / 90)));
      x = linearScale(xt[0], xt[xt.length - 1], 0, PW);
    }
    const allX = Array.from(new Set(xs)).sort((a, b) => a - b);
    return { categorical: false as const, x, xt, y, yt, allX };
  }, [series, chart.type, categorical, logX, PW]);

  if (!series.length) return <p className="hint">No data.</p>;

  const svgPoint = (e: MouseEvent) => {
    const r = svgRef.current!.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * W - M.left, y: ((e.clientY - r.top) / r.height) * H - M.top };
  };

  // Crosshair for line charts: nearest x across all series.
  const onMove = (e: MouseEvent) => {
    if (geo.categorical || chart.type !== "line") return;
    const { x: px } = svgPoint(e);
    let best = geo.allX[0];
    for (const v of geo.allX) if (Math.abs(geo.x(v) - px) < Math.abs(geo.x(best) - px)) best = v;
    if (best === undefined) return;
    const rows = series.flatMap((s, i) => {
      const p = s.points.find((q) => q[0] === best);
      return p ? [{ name: s.name, value: fmtNumber(p[1]), i }] : [];
    });
    const ys = series.flatMap((s) => s.points.filter((q) => q[0] === best).map((q) => geo.y(q[1])));
    setTip({ x: geo.x(best), y: Math.min(...ys), title: `${chart.x_label ?? "x"} = ${fmtNumber(best)}`, rows });
  };

  const nS = series.length;
  return (
    <figure className="chart">
      {chart.title && <figcaption className="chart-title">{chart.title}</figcaption>}
      {nS >= 2 && (
        <ul className="legend">
          {series.map((s, i) => (
            <li key={s.name}><span className="swatch" style={{ background: color(i) }} />{s.name}</li>
          ))}
        </ul>
      )}
      <div className="chart-wrap" ref={wrapRef}>
        <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} role="img"
          aria-label={`${chart.type} chart${chart.title ? `: ${chart.title}` : ""}`}
          onMouseLeave={() => setTip(null)}>
          <g transform={`translate(${M.left},${M.top})`}>
            {/* y grid + ticks */}
            {geo.yt.map((t) => (
              <g key={`y${t}`} transform={`translate(0,${geo.y(t)})`}>
                <line x2={PW} className="gridline" />
                <text x={-8} dy="0.32em" textAnchor="end" className="tick">{fmtTick(t)}</text>
              </g>
            ))}
            {/* x ticks */}
            {geo.categorical
              ? geo.cats.map((c, i) => (
                <text key={c} x={geo.band * (i + 0.5)} y={PH + 18} textAnchor="middle" className="tick">
                  {c.length > 14 ? `${c.slice(0, 13)}…` : c}
                  <title>{c}</title>
                </text>
              ))
              : geo.xt.map((t) => (
                <g key={`x${t}`} transform={`translate(${geo.x(t)},0)`}>
                  <line y1={PH} y2={PH + 5} className="axis" />
                  <text y={PH + 18} textAnchor="middle" className="tick">{fmtTick(t)}</text>
                </g>
              ))}
            <line y1={PH} y2={PH} x2={PW} className="axis" />
            {chart.x_label && <text x={PW / 2} y={PH + 38} textAnchor="middle" className="axis-label">{chart.x_label}{logX ? " (log)" : ""}</text>}
            {chart.y_label && (
              <text transform={`translate(${-44},${PH / 2}) rotate(-90)`} textAnchor="middle" className="axis-label">{chart.y_label}</text>
            )}

            {/* hover capture for the line crosshair */}
            {chart.type === "line" && !geo.categorical && (
              <rect width={PW} height={PH} fill="transparent" onMouseMove={onMove} />
            )}
            {tip && chart.type === "line" && <line x1={tip.x} x2={tip.x} y2={PH} className="crosshair" />}

            {/* marks */}
            {chart.type === "bar" && geo.categorical && series.map((s, si) => {
              const inner = (geo.band * 0.8) / nS;
              return s.points.map((p) => {
                const ci = geo.cats.indexOf(String(p[0]));
                const x0 = geo.band * ci + geo.band * 0.1 + inner * si;
                const y0 = geo.y(Math.max(0, p[1]));
                const h = Math.abs(geo.y(p[1]) - geo.y(0));
                return (
                  <rect key={`${si}-${ci}`} x={x0 + 1} y={y0} width={Math.max(1, inner - 2)} height={Math.max(0.5, h)}
                    rx={Math.min(4, inner / 4)} fill={color(si)} className="bar"
                    onMouseEnter={() => setTip({ x: x0 + inner / 2, y: y0, title: String(p[0]), rows: [{ name: s.name, value: fmtNumber(p[1]), i: si }] })}
                    onMouseLeave={() => setTip(null)} />
                );
              });
            })}
            {chart.type !== "bar" && series.map((s, si) => {
              const pts = s.points
                .map((p) => {
                  const xv = geo.categorical ? geo.band * (geo.cats.indexOf(String(p[0])) + 0.5) : geo.x(p[0] as number);
                  return { xv, yv: geo.y(p[1]), p };
                })
                .filter((q) => Number.isFinite(q.xv) && Number.isFinite(q.yv));
              return (
                <g key={s.name}>
                  {chart.type === "line" && (
                    <polyline points={pts.map((q) => `${q.xv},${q.yv}`).join(" ")} fill="none"
                      stroke={color(si)} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round"
                      strokeDasharray={si >= 8 ? "5 3" : undefined} pointerEvents="none" />
                  )}
                  {pts.map((q, j) => (
                    <g key={j}>
                      <circle cx={q.xv} cy={q.yv} r={chart.type === "scatter" ? 5 : 3.5} fill={color(si)}
                        className="dot" pointerEvents="none" />
                      {chart.type === "scatter" && (
                        <circle cx={q.xv} cy={q.yv} r={11} fill="transparent"
                          onMouseEnter={() => setTip({ x: q.xv, y: q.yv, title: s.name,
                            rows: [{ name: chart.x_label ?? "x", value: typeof q.p[0] === "number" ? fmtNumber(q.p[0]) : String(q.p[0]), i: si },
                              { name: chart.y_label ?? "y", value: fmtNumber(q.p[1]), i: si }] })}
                          onMouseLeave={() => setTip(null)} />
                      )}
                    </g>
                  ))}
                </g>
              );
            })}
          </g>
        </svg>
        {tip && (
          <div className="chart-tip" role="status"
            style={{
              left: `${((tip.x + M.left) / W) * 100}%`,
              top: `${((tip.y + M.top) / H) * 100}%`,
            }}>
            <strong>{tip.title}</strong>
            {tip.rows.map((r) => (
              <span key={r.name} className="tip-row">
                <span className="swatch" style={{ background: color(r.i) }} />{r.name}<b>{r.value}</b>
              </span>
            ))}
          </div>
        )}
      </div>
    </figure>
  );
}
