import { useId } from "react";
import { ImageInput } from "./ImageInput";
import { Segmented } from "./Segmented";
import { MODE_LABEL, type QueryState, type SearchMode } from "../lib/search";

export const EXAMPLE_QUERIES = [
  "a red backpack on a wooden table near a window",
  "two dogs playing in the snow",
  "a man riding a bicycle on a city street",
  "a golden retriever lying on grass",
];

interface Props {
  query: QueryState;
  onChange: (q: QueryState) => void;
  onSubmit: (q?: QueryState) => void;
  maxMb: number;
  active: boolean;
  busy: boolean;
  submitLabel?: string;
}

/** Mode switch + text box + image input (shared by Search and Compare). */
export function QueryInput({ query, onChange, onSubmit, maxMb, active, busy, submitLabel = "Search" }: Props) {
  const textId = useId();
  const showText = query.mode !== "image";
  const showImage = query.mode !== "text";
  const set = (patch: Partial<QueryState>) => onChange({ ...query, ...patch });

  return (
    <form
      className="query"
      onSubmit={(e) => { e.preventDefault(); onSubmit(); }}
      aria-label="Search query"
    >
      <Segmented<SearchMode>
        label="Search mode"
        value={query.mode}
        onChange={(mode) => set({ mode })}
        options={(["text", "image", "hybrid"] as const).map((m) => ({ value: m, label: MODE_LABEL[m] }))}
      />
      <div className={`query-body${showText && showImage ? " two-col" : ""}`}>
        {showText && (
          <div className="query-text">
            <label htmlFor={textId} className="visually-hidden">Describe the image you are looking for</label>
            <div className="search-box">
              <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <circle cx="11" cy="11" r="7" /><path d="M20 20l-4-4" />
              </svg>
              <input
                id={textId}
                type="search"
                value={query.text}
                maxLength={500}
                placeholder={query.mode === "hybrid" ? "Describe what should change or match…" : "Describe the image you are looking for…"}
                onChange={(e) => set({ text: e.target.value })}
                autoComplete="off"
                enterKeyHint="search"
              />
              <button type="submit" className="btn btn-primary" disabled={busy}>
                {busy ? "Searching…" : submitLabel}
              </button>
            </div>
            {query.mode === "text" && (
              <div className="examples" aria-label="Example queries">
                <span className="hint">Try:</span>
                {EXAMPLE_QUERIES.map((q) => (
                  <button key={q} type="button" className="chip chip-example" onClick={() => {
                    const next = { ...query, text: q };
                    onChange(next);
                    onSubmit(next);
                  }}>
                    {q}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
        {showImage && (
          <div className="query-image">
            <ImageInput value={query.image} onChange={(image) => set({ image })} maxMb={maxMb}
              pasteActive={active} compact={showText} />
            {!showText && (
              <button type="submit" className="btn btn-primary btn-block" disabled={busy || !query.image}>
                {busy ? "Searching…" : submitLabel === "Search" ? "Search similar" : submitLabel}
              </button>
            )}
          </div>
        )}
      </div>
    </form>
  );
}
