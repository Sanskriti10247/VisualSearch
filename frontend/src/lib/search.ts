import {
  api,
  type Fusion,
  type ModelInfo,
  type ModelsResponse,
  type Reranker,
  type SearchResponse,
  type TextSearchRequest,
} from "../api";
import { filtersFormValue, serializeFilters, type FilterState } from "./filters";

export type SearchMode = "text" | "image" | "hybrid";

/** The query image: an uploaded/pasted file, or an already indexed image. */
export type QueryImage =
  | { kind: "file"; file: File; previewUrl: string }
  | { kind: "id"; imageId: number; previewUrl: string };

export interface QueryState {
  mode: SearchMode;
  text: string;
  image: QueryImage | null;
}

/** Everything that defines *how* to search (as opposed to *what*). */
export interface SearchConfig {
  textModel: string;
  imageModel: string;
  k: number;
  textRerank: Reranker;
  imageRerank: Reranker;
  minProb: number | null;
  fusion: Fusion;
  textWeight: number;
}

export const DEFAULT_K = 24;

export function defaultConfig(models: ModelsResponse | null): SearchConfig {
  return {
    textModel: models?.default_text_model ?? "",
    imageModel: models?.default_image_model ?? "",
    k: Math.min(DEFAULT_K, models?.limits.max_k ?? DEFAULT_K),
    textRerank: "none",
    imageRerank: "none",
    minProb: null,
    fusion: "zscore",
    textWeight: 0.5,
  };
}

export const textModels = (m: ModelsResponse | null): ModelInfo[] =>
  (m?.models ?? []).filter((x) => x.modalities.includes("text"));

export const imageModels = (m: ModelsResponse | null): ModelInfo[] =>
  (m?.models ?? []).filter((x) => x.modalities.includes("image"));

/** Early fusion averages query vectors, so both must live in the same space. */
export const earlyFusionAllowed = (c: SearchConfig): boolean =>
  !!c.textModel && c.textModel === c.imageModel;

/** Returns a user-facing reason why this query cannot be run, or null. */
export function queryProblem(q: QueryState): string | null {
  const text = q.text.trim();
  if (q.mode === "text" && !text) return "Type a description to search for.";
  if (q.mode === "image" && !q.image) return "Add a query image (drop, browse, or paste).";
  if (q.mode === "hybrid" && !text && !q.image) return "Hybrid search needs text, an image, or both.";
  if (text.length > 500) return "Query text is limited to 500 characters.";
  return null;
}

export type BuiltRequest =
  | { endpoint: "text"; body: TextSearchRequest }
  | { endpoint: "image" | "hybrid"; form: FormData };

function setIf(form: FormData, key: string, v: string | number | null | undefined) {
  if (v === null || v === undefined || v === "") return;
  form.append(key, String(v));
}

/** Build the request for the chosen mode (pure; FormData for multipart routes). */
export function buildSearchRequest(q: QueryState, c: SearchConfig, f: FilterState): BuiltRequest {
  if (q.mode === "text") {
    const body: TextSearchRequest = { query: q.text.trim(), k: c.k, rerank: c.textRerank };
    if (c.textModel) body.model = c.textModel;
    const filters = serializeFilters(f);
    if (filters) body.filters = filters;
    if (c.minProb !== null) body.min_prob = c.minProb;
    return { endpoint: "text", body };
  }
  const form = new FormData();
  if (q.image?.kind === "file") form.append("file", q.image.file, q.image.file.name || "query");
  else if (q.image?.kind === "id") form.append("image_id", String(q.image.imageId));
  setIf(form, "k", c.k);
  setIf(form, "filters", filtersFormValue(f));
  if (c.minProb !== null) setIf(form, "min_prob", c.minProb);
  if (q.mode === "image") {
    setIf(form, "model", c.imageModel);
    setIf(form, "rerank", c.imageRerank);
    return { endpoint: "image", form };
  }
  setIf(form, "text", q.text.trim());
  setIf(form, "text_model", c.textModel);
  setIf(form, "image_model", c.imageModel);
  setIf(form, "text_weight", c.textWeight);
  setIf(form, "fusion", c.fusion);
  return { endpoint: "hybrid", form };
}

export function runSearch(
  q: QueryState,
  c: SearchConfig,
  f: FilterState,
  signal?: AbortSignal,
): Promise<SearchResponse> {
  const req = buildSearchRequest(q, c, f);
  if (req.endpoint === "text") return api.searchText(req.body, signal);
  if (req.endpoint === "image") return api.searchImage(req.form, signal);
  return api.searchHybrid(req.form, signal);
}

/** Pick a model of a family for "Find similar (CLIP / DINOv2)", preferring defaults.
 *  Only models that can embed images count (the slim server's CLIP is text-only). */
export function modelOfFamily(m: ModelsResponse | null, family: "clip" | "dinov2"): string | null {
  if (!m) return null;
  const ofFamily = imageModels(m).filter((x) => x.family?.toLowerCase().startsWith(family) ||
    x.key.toLowerCase().startsWith(family));
  if (!ofFamily.length) return null;
  const preferred = [m.default_image_model, m.default_text_model].find((k) =>
    ofFamily.some((x) => x.key === k));
  return preferred ?? ofFamily[0].key;
}

export const MODE_LABEL: Record<SearchMode, string> = {
  text: "Text",
  image: "Image",
  hybrid: "Hybrid",
};

export const RERANK_LABEL: Record<string, string> = {
  none: "None",
  prf: "Pseudo-relevance feedback",
  visual_prf: "Visual PRF (DINOv2)",
  cross_encoder: "Cross-encoder (BLIP ITM)",
};

export const FUSION_LABEL: Record<string, string> = {
  zscore: "z-score (calibrated)",
  minmax: "min-max",
  rrf: "reciprocal rank (RRF)",
  early: "early (query-vector mix)",
};
