import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api, ApiError, type FacetsResponse, type ModelsResponse } from "./api";
import type { FilterState } from "./lib/filters";
import type { QueryState, SearchConfig } from "./lib/search";

/** A request from another view to open Search pre-filled (and run it). */
export interface SearchIntent {
  id: number;
  query?: Partial<QueryState>;
  config?: Partial<SearchConfig>;
  filters?: Partial<FilterState>;
  run: boolean;
}

interface AppState {
  models: ModelsResponse | null;
  facets: FacetsResponse | null;
  bootError: ApiError | null;
  loading: boolean;
  reload: () => void;
  refreshFacets: () => void;
  intent: SearchIntent | null;
  requestSearch: (i: Omit<SearchIntent, "id">) => void;
}

const Ctx = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [facets, setFacets] = useState<FacetsResponse | null>(null);
  const [bootError, setBootError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);
  const [intent, setIntent] = useState<SearchIntent | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    setLoading(true);
    Promise.all([api.models(ctrl.signal), api.facets(ctrl.signal)])
      .then(([m, f]) => {
        setModels(m);
        setFacets(f);
        setBootError(null);
      })
      .catch((e: unknown) => {
        if (ctrl.signal.aborted) return;
        setBootError(e instanceof ApiError ? e : new ApiError(0, "unknown", String(e)));
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setLoading(false);
      });
    return () => ctrl.abort();
  }, [nonce]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  const refreshFacets = useCallback(() => {
    api.facets().then(setFacets).catch(() => undefined);
  }, []);
  const requestSearch = useCallback((i: Omit<SearchIntent, "id">) => {
    setIntent({ ...i, id: Date.now() + Math.random() });
  }, []);

  const value = useMemo<AppState>(() => ({
    models, facets, bootError, loading, reload, refreshFacets, intent, requestSearch,
  }), [models, facets, bootError, loading, reload, refreshFacets, intent, requestSearch]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApp(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useApp must be used inside <AppProvider>");
  return v;
}
