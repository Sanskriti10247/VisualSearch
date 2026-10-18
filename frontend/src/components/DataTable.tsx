import type { ExperimentTable } from "../api";
import { bestRows, fmtCell } from "../lib/format";

/** Experiment results table with metric formatting and per-column best highlighting. */
export function DataTable({ table }: { table: ExperimentTable }) {
  const best = new Map<number, Set<number>>();
  for (const c of table.highlight_best ?? []) best.set(c, bestRows(table.rows, c));
  return (
    <figure className="table-fig">
      {table.title && <figcaption className="chart-title">{table.title}</figcaption>}
      <div className="table-scroll">
        <table className="data">
          <thead>
            <tr>{table.columns.map((c, i) => <th key={i} scope="col" className={i ? "num" : ""}>{c}</th>)}</tr>
          </thead>
          <tbody>
            {table.rows.map((r, ri) => (
              <tr key={ri}>
                {r.map((v, ci) => {
                  const isBest = best.get(ci)?.has(ri);
                  const col = table.columns[ci] ?? "";
                  return ci === 0
                    ? <th key={ci} scope="row">{fmtCell(col, v)}</th>
                    : <td key={ci} className={`${typeof v === "number" ? "num" : ""}${isBest ? " best" : ""}`}>
                      {isBest ? <strong>{fmtCell(col, v)}</strong> : fmtCell(col, v)}
                    </td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {table.notes && <p className="hint table-notes">{table.notes}</p>}
    </figure>
  );
}
