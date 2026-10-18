/**
 * Typed client for the VisualSearch FastAPI backend (`/api/v1`).
 *
 * Types mirror `src/visualsearch/api/schemas.py` and the ad-hoc dict responses
 * in `src/visualsearch/api/app.py`. All error handling funnels through
 * `parseErrorEnvelope` / `ApiError`.
 */

export const API_BASE: string = (import.meta.env?.VITE_API_BASE ?? "").replace(/\/+$/, "");
export const API_PREFIX = "/api/v1";

// ------------------------------------------------------------------ schemas
export type Orientation = "landscape" | "portrait" | "square";
export type Reranker = "none" | "prf" | "visual_prf" | "cross_encoder";
export type Fusion = "zscore" | "minmax" | "rrf" | "early";

/** Values within a field are OR-ed; fields are AND-ed. */
export interface Filters {
  collections?: string[];
  splits?: string[];
  labels?: string[];
  tags_any?: string[];
  orientations?: Orientation[];
  min_width?: number | null;
  min_height?: number | null;
  exclude_ids?: number[];
}

export interface TextSearchRequest {
  query: string;
  k?: number | null;
  model?: string | null;
  filters?: Filters | null;
  rerank?: Reranker;
  candidates?: number | null;
  min_prob?: number | null;
}

export interface SignalScore {
  raw: number;
  z?: number | null;
  percentile?: number | null;
  prob?: number | null;
}

export interface ImageMeta {
  id: number;
  collection: string;
  split?: string | null;
  source_name?: string | null;
  width: number;
  height: number;
  orientation: string;
  file_size: number;
  format?: string | null;
  label?: string | null;
  tags: string[];
  sha256: string;
  created_at?: string | null;
  extra?: Record<string, unknown> | null;
}

export interface SearchHit {
  rank: number;
  image_id: number;
  score: number;
  signals: Record<string, SignalScore>;
  thumbnail_url: string;
  image_url: string;
  metadata: ImageMeta;
}

/** `strategy` is a free-form dict; these are the keys the engine sets today. */
export interface SearchStrategy {
  mode?: "text" | "image" | "hybrid";
  model?: string;
  text_model?: string;
  image_model?: string;
  fusion?: Fusion;
  text_weight?: number;
  index?: string;
  filter_strategy?: string;
  filter_matches?: number | null;
  candidates?: number;
  rerank?: string;
  rerank_params?: Record<string, unknown>;
  query_image_id?: number | null;
  [key: string]: unknown;
}

export interface SearchResponse {
  request_id: string;
  results: SearchHit[];
  total: number;
  k: number;
  timings_ms: Record<string, number>;
  strategy: SearchStrategy;
  warnings: string[];
}

export interface ImageDetail {
  metadata: ImageMeta;
  captions: string[];
  thumbnail_url: string;
  image_url: string;
  indexed_in: string[];
}

export type JobStatus = "queued" | "running" | "succeeded" | "failed";

export interface IngestCounts {
  added?: number;
  duplicate?: number;
  near_duplicate_skipped?: number;
  invalid?: number;
  [status: string]: number | undefined;
}

export interface JobInfo {
  id: string;
  kind: string;
  status: JobStatus;
  created_at: string;
  updated_at: string;
  progress: number;
  total: number;
  result?: ({ counts?: IngestCounts } & Record<string, unknown>) | null;
  error?: string | null;
}

export interface IndexJobAccepted {
  job_id: string;
  status_url: string;
  accepted_files: number;
}

export interface ErrorBody {
  code: string;
  message: string;
  details?: unknown;
}

// ------------------------------------------------------- ops / dict routes
export interface IndexStats {
  kind: string;
  dim: number;
  params?: Record<string, unknown>;
  ntotal: number;
  [key: string]: unknown;
}

export interface ModelInfo {
  key: string;
  family: string;
  hf_id: string;
  revision?: string | null;
  description?: string | null;
  dim: number;
  modalities: string[];
  pooling?: string | null;
  crop_mode?: string | null;
  normalized?: boolean;
  preprocessing?: Record<string, unknown>;
  fingerprint: string;
  device: string;
  loaded?: boolean;
  index: IndexStats | null;
  stale: boolean;
}

export interface CalibrationInfo {
  space: string;
  query_modality: string;
  mean: number;
  std: number;
  fitted_on?: unknown;
  diagnostics?: unknown;
}

export interface ModelsResponse {
  default_text_model: string;
  default_image_model: string;
  index_type: string;
  models: ModelInfo[];
  calibrations: CalibrationInfo[];
  rerankers: { text: Reranker[]; image: Reranker[] };
  fusions: Fusion[];
  limits: { max_k: number; max_upload_mb: number; max_batch_files: number };
}

export interface FacetValue {
  value: string;
  count: number;
}

export interface FacetsResponse {
  collections: FacetValue[];
  splits: FacetValue[];
  labels: FacetValue[];
  orientations: FacetValue[];
  tags: FacetValue[];
  total: number;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  catalog_images: number;
  index_type: string;
  models: Record<string, {
    ready: boolean;
    stale: boolean;
    indexed: number;
    modalities: string[];
    device: string;
  }>;
}

export interface LatencyStats {
  count: number;
  p50_ms: number;
  p95_ms: number;
  max_ms: number;
}

export interface MetricsResponse {
  uptime_s: number;
  routes: Record<string, LatencyStats>;
  status_counts: Record<string, Record<string, number>>;
  search_stages: Record<string, LatencyStats>;
  indexes: Record<string, Record<string, unknown>>;
  catalog_images: number;
}

export interface ExperimentTable {
  title?: string;
  columns: string[];
  rows: (string | number | null)[][];
  notes?: string;
  highlight_best?: number[];
}

export interface ChartSeries {
  name: string;
  points: [number | string, number][];
}

export interface ExperimentChart {
  title?: string;
  type: "line" | "bar" | "scatter";
  x_label?: string;
  y_label?: string;
  log_x?: boolean;
  series: ChartSeries[];
}

export interface ExperimentReport {
  title?: string;
  description?: string;
  tables?: ExperimentTable[];
  charts?: ExperimentChart[];
}

export interface ExperimentsResponse {
  results: Record<string, { report?: ExperimentReport } & Record<string, unknown>>;
}

// -------------------------------------------------------------------- errors
/** Every failure (HTTP error, network failure, malformed body) becomes this. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: unknown;

  constructor(status: number, code: string, message: string, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }

  /** True when the backend could not be reached at all. */
  get unreachable(): boolean {
    return this.code === "network_error" || this.status === 502 || this.status === 503 ||
      this.status === 504;
  }
}

/**
 * Turn an HTTP status + (possibly non-JSON) body into an ApiError.
 * Understands the `{"error": {code, message, details}}` envelope and falls back
 * to FastAPI's default `{"detail": ...}` and plain text.
 */
export function parseErrorEnvelope(status: number, body: unknown, statusText = ""): ApiError {
  if (body && typeof body === "object") {
    const env = (body as { error?: unknown }).error;
    if (env && typeof env === "object") {
      const e = env as Partial<ErrorBody>;
      return new ApiError(status, String(e.code ?? "error"),
        String(e.message ?? (statusText || `HTTP ${status}`)), e.details ?? null);
    }
    const detail = (body as { detail?: unknown }).detail;
    if (detail !== undefined) {
      return new ApiError(status, "http_error",
        typeof detail === "string" ? detail : `HTTP ${status}`, typeof detail === "string" ? null : detail);
    }
  }
  if (typeof body === "string" && body.trim()) {
    return new ApiError(status, "http_error", body.trim().slice(0, 300));
  }
  return new ApiError(status, "http_error", statusText || `HTTP ${status}`);
}

export interface ValidationIssue {
  loc: (string | number)[];
  msg: string;
  type?: string;
}

/** Validation details are a list of `{loc, msg, type}` for `validation_error`. */
export function validationIssues(details: unknown): ValidationIssue[] {
  if (!Array.isArray(details)) return [];
  return details.filter((d): d is ValidationIssue =>
    !!d && typeof d === "object" && "msg" in d && Array.isArray((d as ValidationIssue).loc));
}

export function isAbortError(e: unknown): boolean {
  return e instanceof DOMException && e.name === "AbortError";
}

// ------------------------------------------------------------------ transport
export function apiUrl(path: string): string {
  if (/^https?:\/\//.test(path)) return path;
  // URLs returned by the server (thumbnail_url, ...) already carry /api/v1.
  const p = path.startsWith(API_PREFIX) ? path : `${API_PREFIX}${path}`;
  return `${API_BASE}${p}`;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(apiUrl(path), { ...init, headers: { Accept: "application/json", ...init.headers } });
  } catch (e) {
    if (isAbortError(e)) throw e;
    throw new ApiError(0, "network_error",
      "Cannot reach the VisualSearch API. Is the backend running on port 8000?");
  }
  const text = await res.text();
  let body: unknown = text;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    /* keep raw text */
  }
  if (!res.ok) throw parseErrorEnvelope(res.status, body, res.statusText);
  if (body === null || typeof body !== "object") {
    throw new ApiError(res.status, "bad_response", "The server returned an unexpected response.");
  }
  return body as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

// ---------------------------------------------------------------- endpoints
export const api = {
  searchText: (body: TextSearchRequest, signal?: AbortSignal) =>
    request<SearchResponse>("/search/text", { ...json(body), signal }),
  searchImage: (form: FormData, signal?: AbortSignal) =>
    request<SearchResponse>("/search/image", { method: "POST", body: form, signal }),
  searchHybrid: (form: FormData, signal?: AbortSignal) =>
    request<SearchResponse>("/search/hybrid", { method: "POST", body: form, signal }),
  image: (id: number, signal?: AbortSignal) => request<ImageDetail>(`/images/${id}`, { signal }),
  indexImages: (form: FormData) =>
    request<IndexJobAccepted>("/index/images", { method: "POST", body: form }),
  job: (id: string, signal?: AbortSignal) =>
    request<JobInfo>(`/jobs/${encodeURIComponent(id)}`, { signal }),
  health: (signal?: AbortSignal) => request<HealthResponse>("/health", { signal }),
  models: (signal?: AbortSignal) => request<ModelsResponse>("/models", { signal }),
  metrics: (signal?: AbortSignal) => request<MetricsResponse>("/metrics", { signal }),
  facets: (signal?: AbortSignal) => request<FacetsResponse>("/facets", { signal }),
  experiments: (signal?: AbortSignal) => request<ExperimentsResponse>("/experiments", { signal }),
};

export const thumbnailUrl = (id: number) => apiUrl(`/images/${id}/thumbnail`);
export const imageFileUrl = (id: number) => apiUrl(`/images/${id}/file`);
