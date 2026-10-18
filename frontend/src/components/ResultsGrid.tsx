import type { ReactNode } from "react";
import type { SearchHit } from "../api";
import { ResultCard } from "./ResultCard";

interface Props {
  hits: SearchHit[];
  onOpen: (hit: SearchHit) => void;
  badgeFor?: (hit: SearchHit) => ReactNode;
  compact?: boolean;
}

export function ResultsGrid({ hits, onOpen, badgeFor, compact }: Props) {
  return (
    <ol className={`grid${compact ? " grid-compact" : ""}`} aria-label="Results">
      {hits.map((h) => (
        <ResultCard key={h.image_id} hit={h} onOpen={onOpen} badge={badgeFor?.(h)} compact={compact} />
      ))}
    </ol>
  );
}

export function SkeletonGrid({ count = 12, compact }: { count?: number; compact?: boolean }) {
  return (
    <ol className={`grid${compact ? " grid-compact" : ""}`} aria-busy="true" aria-label="Loading results">
      {Array.from({ length: count }, (_, i) => (
        <li key={i} className="card skeleton">
          <div className="thumb" />
          <div className="card-body">
            <span className="sk-line" />
            <span className="sk-line short" />
          </div>
        </li>
      ))}
    </ol>
  );
}

export function EmptyResults({ filtersActive }: { filtersActive: number }) {
  return (
    <div className="empty">
      <h3>No results</h3>
      <p>
        Nothing matched this query{filtersActive ? ` with ${filtersActive} active filter${filtersActive === 1 ? "" : "s"}` : ""}.
        {filtersActive ? " Try loosening or clearing the filters" : " Try a broader description"}
        {", or lower the minimum P(match) if one is set."}
      </p>
    </div>
  );
}
