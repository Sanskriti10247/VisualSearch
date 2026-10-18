import type { ReactNode } from "react";
import { apiUrl, type SearchHit } from "../api";
import { SignalScores } from "./SignalScores";

interface Props {
  hit: SearchHit;
  onOpen: (hit: SearchHit) => void;
  badge?: ReactNode;
  compact?: boolean;
}

export function ResultCard({ hit, onOpen, badge, compact }: Props) {
  const m = hit.metadata;
  return (
    <li className={`card${compact ? " compact" : ""}`}>
      <button type="button" className="card-hit" onClick={() => onOpen(hit)}
        aria-label={`Result ${hit.rank}: image ${hit.image_id}${m.label ? `, ${m.label}` : ""}. Open details`}>
        <div className="thumb">
          <img src={apiUrl(hit.thumbnail_url)} alt="" loading="lazy" decoding="async" />
          <span className="rank">{hit.rank}</span>
          {badge}
        </div>
      </button>
      <div className="card-body">
        {!compact && <SignalScores signals={hit.signals} />}
        <p className="card-meta">
          <span>{m.collection}</span>
          {m.label && <span className="label" title={m.label}>{m.label}</span>}
          <span className="mono">{m.width}×{m.height}</span>
        </p>
      </div>
    </li>
  );
}
