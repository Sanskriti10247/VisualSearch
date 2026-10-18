import type { FacetValue } from "../api";
import { fmtInt } from "../lib/format";

interface Props {
  label: string;
  values: FacetValue[];
  selected: string[];
  onToggle: (value: string) => void;
  limit?: number;
}

/** Multi-select chip list with counts (for facet filters). */
export function ChipGroup({ label, values, selected, onToggle, limit = 40 }: Props) {
  if (!values.length) return null;
  // Keep selected values visible even if they fall outside the limit.
  const shown = values.slice(0, limit);
  for (const s of selected) {
    if (!shown.some((v) => v.value === s)) shown.push(values.find((v) => v.value === s) ?? { value: s, count: 0 });
  }
  return (
    <fieldset className="chip-group">
      <legend>{label}</legend>
      <div className="chips">
        {shown.map((v) => {
          const on = selected.includes(v.value);
          return (
            <button
              type="button"
              key={v.value}
              className={`chip${on ? " on" : ""}`}
              aria-pressed={on}
              onClick={() => onToggle(v.value)}
            >
              {v.value}
              <span className="chip-count">{fmtInt(v.count)}</span>
            </button>
          );
        })}
      </div>
      {values.length > limit && <p className="hint">Showing the {limit} most common of {values.length}.</p>}
    </fieldset>
  );
}
