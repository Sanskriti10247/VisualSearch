import { useEffect, useState } from "react";
import { api, apiUrl, ApiError, type ImageDetail, type SearchHit } from "../api";
import { useApp } from "../AppContext";
import { fmtBytes } from "../lib/format";
import { modelOfFamily } from "../lib/search";
import { ErrorView } from "./ErrorView";
import { Modal } from "./Modal";
import { SignalScores } from "./SignalScores";

interface Props {
  hit: SearchHit;
  onClose: () => void;
  onFindSimilar: (imageId: number, model: string) => void;
  onUseInHybrid: (imageId: number) => void;
}

export function DetailModal({ hit, onClose, onFindSimilar, onUseInHybrid }: Props) {
  const { models } = useApp();
  const [detail, setDetail] = useState<ImageDetail | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    setDetail(null);
    setError(null);
    api.image(hit.image_id, ctrl.signal).then(setDetail).catch((e) => {
      if (!ctrl.signal.aborted) setError(e instanceof ApiError ? e : new ApiError(0, "unknown", String(e)));
    });
    return () => ctrl.abort();
  }, [hit.image_id]);

  const m = detail?.metadata ?? hit.metadata;
  const clip = modelOfFamily(models, "clip");
  const dino = modelOfFamily(models, "dinov2");
  const indexed = (key: string | null) => !!key && (!detail || detail.indexed_in.includes(key));
  const rows: [string, string][] = [
    ["Image id", String(m.id)],
    ["Collection", m.collection],
    ["Split", m.split ?? "–"],
    ["Label", m.label ?? "–"],
    ["Size", `${m.width} × ${m.height} (${m.orientation})`],
    ["File", `${m.format ?? "?"} · ${fmtBytes(m.file_size)}`],
    ["Source name", m.source_name ?? "–"],
    ["Tags", m.tags.length ? m.tags.join(", ") : "–"],
    ["SHA-256", m.sha256.slice(0, 16) + "…"],
    ["Added", m.created_at ?? "–"],
  ];

  return (
    <Modal title={`Image #${hit.image_id}`} onClose={onClose} wide>
      <div className="detail">
        <div className="detail-image">
          <img src={apiUrl(hit.image_url)} alt={detail?.captions[0] ?? `Image ${hit.image_id}`} />
        </div>
        <div className="detail-side">
          <div className="detail-actions">
            {clip && (
              <button className="btn btn-primary" disabled={!indexed(clip)}
                title={indexed(clip) ? `Image search with ${clip}` : `Not indexed in ${clip}`}
                onClick={() => onFindSimilar(hit.image_id, clip)}>
                Find similar (CLIP)
              </button>
            )}
            {dino && (
              <button className="btn" disabled={!indexed(dino)}
                title={indexed(dino) ? `Image search with ${dino}` : `Not indexed in ${dino}`}
                onClick={() => onFindSimilar(hit.image_id, dino)}>
                Find similar (DINOv2)
              </button>
            )}
            <button className="btn" onClick={() => onUseInHybrid(hit.image_id)}>Use in hybrid</button>
          </div>
          <h3>Scores (rank {hit.rank})</h3>
          <SignalScores signals={hit.signals} />
          <h3>Metadata</h3>
          <table className="kv">
            <tbody>
              {rows.map(([k, v]) => <tr key={k}><th scope="row">{k}</th><td>{v}</td></tr>)}
            </tbody>
          </table>
          {error && <ErrorView error={error} compact />}
          {detail && (
            <>
              <h3>Indexed in</h3>
              <p className="chips">
                {detail.indexed_in.length
                  ? detail.indexed_in.map((k) => <span key={k} className="tag">{k}</span>)
                  : <span className="muted">No index contains this image yet.</span>}
              </p>
              {detail.captions.length > 0 && (
                <>
                  <h3>Captions</h3>
                  <ul className="captions">
                    {detail.captions.map((c, i) => <li key={i}>{c}</li>)}
                  </ul>
                </>
              )}
            </>
          )}
          {!detail && !error && <p className="hint">Loading details…</p>}
          {m.extra && Object.keys(m.extra).length > 0 && (
            <details className="json-view"><summary>Extra metadata</summary>
              <pre className="json">{JSON.stringify(m.extra, null, 2)}</pre>
            </details>
          )}
        </div>
      </div>
    </Modal>
  );
}
