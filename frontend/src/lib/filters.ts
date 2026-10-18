import type { Filters, Orientation } from "../api";

/** UI-side filter state: always fully populated, never null arrays. */
export interface FilterState {
  collections: string[];
  splits: string[];
  labels: string[];
  tags_any: string[];
  orientations: Orientation[];
  min_width: number | null;
  min_height: number | null;
}

export const emptyFilters = (): FilterState => ({
  collections: [],
  splits: [],
  labels: [],
  tags_any: [],
  orientations: [],
  min_width: null,
  min_height: null,
});

const LIST_KEYS = ["collections", "splits", "labels", "tags_any", "orientations"] as const;

const validDim = (v: number | null): v is number =>
  v !== null && Number.isFinite(v) && v >= 1;

/** Number of active filter fields (each non-empty list or dimension counts once). */
export function countActiveFilters(f: FilterState): number {
  let n = LIST_KEYS.filter((k) => f[k].length > 0).length;
  if (validDim(f.min_width)) n++;
  if (validDim(f.min_height)) n++;
  return n;
}

/**
 * Convert UI state to the API `Filters` object, dropping empty fields.
 * Returns null when nothing is active (so the request omits `filters`).
 */
export function serializeFilters(f: FilterState, excludeIds: number[] = []): Filters | null {
  const out: Filters = {};
  for (const k of LIST_KEYS) {
    const vals = Array.from(new Set(f[k].map((v) => v.trim()).filter(Boolean)));
    if (vals.length) (out as Record<string, unknown>)[k] = vals;
  }
  if (validDim(f.min_width)) out.min_width = Math.round(f.min_width);
  if (validDim(f.min_height)) out.min_height = Math.round(f.min_height);
  if (excludeIds.length) out.exclude_ids = Array.from(new Set(excludeIds));
  return Object.keys(out).length ? out : null;
}

/** JSON form-field encoding for multipart endpoints (undefined = omit the field). */
export function filtersFormValue(f: FilterState, excludeIds: number[] = []): string | undefined {
  const s = serializeFilters(f, excludeIds);
  return s ? JSON.stringify(s) : undefined;
}

export function toggleValue<T>(list: T[], value: T): T[] {
  return list.includes(value) ? list.filter((v) => v !== value) : [...list, value];
}

/** Parse "a, b ,,c" into ["a","b","c"] (deduplicated, order kept). */
export function parseCsv(s: string): string[] {
  return Array.from(new Set(s.split(",").map((t) => t.trim()).filter(Boolean)));
}
