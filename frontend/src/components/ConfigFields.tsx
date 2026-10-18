import { useId } from "react";
import type { Fusion, ModelsResponse, Reranker } from "../api";
import {
  earlyFusionAllowed,
  FUSION_LABEL,
  imageModels,
  RERANK_LABEL,
  textModels,
  type SearchConfig,
  type SearchMode,
} from "../lib/search";

interface Props {
  mode: SearchMode;
  config: SearchConfig;
  onChange: (c: SearchConfig) => void;
  models: ModelsResponse | null;
  /** Hide k / min_prob (Compare shares them across both columns). */
  strategyOnly?: boolean;
}

/** Model / reranker / fusion / k / min_prob fields for the given mode. */
export function ConfigFields({ mode, config, onChange, models, strategyOnly }: Props) {
  const id = useId();
  const set = (patch: Partial<SearchConfig>) => onChange({ ...config, ...patch });
  const maxK = models?.limits.max_k ?? 200;
  const tModels = textModels(models);
  const iModels = imageModels(models);
  const showText = mode !== "image";
  const showImage = mode !== "text";
  const earlyOk = earlyFusionAllowed(config);

  return (
    <div className="config-grid">
      {showText && (
        <div className="field">
          <label htmlFor={`${id}-tm`}>Text model</label>
          <select id={`${id}-tm`} value={config.textModel} onChange={(e) => {
            const textModel = e.target.value;
            const fusion = config.fusion === "early" && textModel !== config.imageModel ? "zscore" : config.fusion;
            set({ textModel, fusion });
          }}>
            {tModels.map((m) => <option key={m.key} value={m.key}>{m.key}</option>)}
          </select>
        </div>
      )}
      {showImage && (
        <div className="field">
          <label htmlFor={`${id}-im`}>Image model</label>
          <select id={`${id}-im`} value={config.imageModel} onChange={(e) => {
            const imageModel = e.target.value;
            const fusion = config.fusion === "early" && imageModel !== config.textModel ? "zscore" : config.fusion;
            set({ imageModel, fusion });
          }}>
            {iModels.map((m) => <option key={m.key} value={m.key}>{m.key}</option>)}
          </select>
        </div>
      )}
      {mode === "text" && (
        <div className="field">
          <label htmlFor={`${id}-rt`}>Reranker</label>
          <select id={`${id}-rt`} value={config.textRerank}
            onChange={(e) => set({ textRerank: e.target.value as Reranker })}>
            {(models?.rerankers.text ?? ["none"]).map((r) => <option key={r} value={r}>{RERANK_LABEL[r] ?? r}</option>)}
          </select>
        </div>
      )}
      {mode === "image" && (
        <div className="field">
          <label htmlFor={`${id}-ri`}>Reranker</label>
          <select id={`${id}-ri`} value={config.imageRerank}
            onChange={(e) => set({ imageRerank: e.target.value as Reranker })}>
            {(models?.rerankers.image ?? ["none"]).map((r) => <option key={r} value={r}>{RERANK_LABEL[r] ?? r}</option>)}
          </select>
        </div>
      )}
      {mode === "hybrid" && (
        <>
          <div className="field">
            <label htmlFor={`${id}-fu`}>Fusion</label>
            <select id={`${id}-fu`} value={config.fusion} onChange={(e) => set({ fusion: e.target.value as Fusion })}>
              {(models?.fusions ?? ["zscore", "minmax", "rrf", "early"]).map((f) => (
                <option key={f} value={f} disabled={f === "early" && !earlyOk}>
                  {FUSION_LABEL[f] ?? f}{f === "early" && !earlyOk ? " — needs same model" : ""}
                </option>
              ))}
            </select>
          </div>
          <div className="field field-wide">
            <label htmlFor={`${id}-tw`}>
              Text weight <span className="muted">{config.textWeight.toFixed(2)}</span>
            </label>
            <div className="slider">
              <span className="hint">image</span>
              <input id={`${id}-tw`} type="range" min={0} max={1} step={0.05} value={config.textWeight}
                onChange={(e) => set({ textWeight: Number(e.target.value) })} />
              <span className="hint">text</span>
            </div>
          </div>
          <p className={`hint field-wide${!earlyOk ? " hint-warn" : ""}`}>
            <strong>early</strong> fusion mixes the text and image query vectors before a single index
            search, so it only works when text and image use the same CLIP model
            {earlyOk ? "." : ` (currently ${config.textModel || "?"} vs ${config.imageModel || "?"}).`}
            {" "}The others fuse per-model scores after two searches.
          </p>
        </>
      )}
      {!strategyOnly && (
        <>
          <div className="field field-narrow">
            <label htmlFor={`${id}-k`}>Results (k)</label>
            <input id={`${id}-k`} type="number" min={1} max={maxK} value={config.k}
              onChange={(e) => {
                const v = Math.round(Number(e.target.value));
                set({ k: Number.isFinite(v) ? Math.max(1, Math.min(maxK, v)) : config.k });
              }} />
          </div>
          <div className="field field-narrow">
            <label htmlFor={`${id}-mp`} title="Drop results whose calibrated P(match) is lower">Min P(match)</label>
            <input id={`${id}-mp`} type="number" min={0} max={1} step={0.05} placeholder="off"
              value={config.minProb ?? ""}
              onChange={(e) => {
                const raw = e.target.value;
                const v = Number(raw);
                set({ minProb: raw === "" || !Number.isFinite(v) ? null : Math.max(0, Math.min(1, v)) });
              }} />
          </div>
        </>
      )}
    </div>
  );
}
