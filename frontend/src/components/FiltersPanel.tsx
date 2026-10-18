import { useEffect, useId, useState } from "react";
import type { FacetsResponse, Orientation } from "../api";
import { ChipGroup } from "./ChipGroup";
import { countActiveFilters, emptyFilters, parseCsv, toggleValue, type FilterState } from "../lib/filters";

interface Props {
  facets: FacetsResponse | null;
  value: FilterState;
  onChange: (f: FilterState) => void;
}

/** Metadata filters built from GET /facets. */
export function FiltersPanel({ facets, value, onChange }: Props) {
  const id = useId();
  const [tagText, setTagText] = useState(value.tags_any.join(", "));
  useEffect(() => {
    if (parseCsv(tagText).join(",") !== value.tags_any.join(",")) setTagText(value.tags_any.join(", "));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value.tags_any]);
  const set = (patch: Partial<FilterState>) => onChange({ ...value, ...patch });
  const n = countActiveFilters(value);
  const dim = (raw: string) => {
    const v = Math.round(Number(raw));
    return raw === "" || !Number.isFinite(v) || v < 1 ? null : v;
  };

  if (!facets) return <p className="hint">Facets unavailable.</p>;
  return (
    <div className="filters">
      <ChipGroup label="Collections" values={facets.collections} selected={value.collections}
        onToggle={(v) => set({ collections: toggleValue(value.collections, v) })} />
      <ChipGroup label="Splits" values={facets.splits} selected={value.splits}
        onToggle={(v) => set({ splits: toggleValue(value.splits, v) })} />
      <ChipGroup label="Orientation" values={facets.orientations} selected={value.orientations}
        onToggle={(v) => set({ orientations: toggleValue(value.orientations, v as Orientation) })} />
      <ChipGroup label="Labels" values={facets.labels} selected={value.labels} limit={30}
        onToggle={(v) => set({ labels: toggleValue(value.labels, v) })} />
      <div className="config-grid">
        <div className="field field-wide">
          <label htmlFor={`${id}-tags`}>
            Tags (any of){facets.tags.length > 0 && <span className="muted"> · e.g. {facets.tags.slice(0, 4).map((t) => t.value).join(", ")}</span>}
          </label>
          <input id={`${id}-tags`} type="text" value={tagText} placeholder="comma-separated"
            list={`${id}-taglist`}
            onChange={(e) => { setTagText(e.target.value); set({ tags_any: parseCsv(e.target.value) }); }} />
          <datalist id={`${id}-taglist`}>
            {facets.tags.map((t) => <option key={t.value} value={t.value} />)}
          </datalist>
        </div>
        <div className="field field-narrow">
          <label htmlFor={`${id}-mw`}>Min width</label>
          <input id={`${id}-mw`} type="number" min={1} placeholder="px" value={value.min_width ?? ""}
            onChange={(e) => set({ min_width: dim(e.target.value) })} />
        </div>
        <div className="field field-narrow">
          <label htmlFor={`${id}-mh`}>Min height</label>
          <input id={`${id}-mh`} type="number" min={1} placeholder="px" value={value.min_height ?? ""}
            onChange={(e) => set({ min_height: dim(e.target.value) })} />
        </div>
      </div>
      {n > 0 && (
        <button type="button" className="btn btn-ghost btn-sm" onClick={() => { setTagText(""); onChange(emptyFilters()); }}>
          Clear {n} filter{n === 1 ? "" : "s"}
        </button>
      )}
    </div>
  );
}
