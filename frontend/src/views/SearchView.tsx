import { useEffect, useRef, useState } from "react";
import { isAbortError, thumbnailUrl, type SearchHit, type SearchResponse } from "../api";
import { useApp } from "../AppContext";
import { ConfigFields } from "../components/ConfigFields";
import { DetailModal } from "../components/DetailModal";
import { ErrorView } from "../components/ErrorView";
import { FiltersPanel } from "../components/FiltersPanel";
import { QueryInput } from "../components/QueryInput";
import { EmptyResults, ResultsGrid, SkeletonGrid } from "../components/ResultsGrid";
import { ResultsHeader } from "../components/ResultsHeader";
import { ScoreInfo } from "../components/ScoreInfo";
import { countActiveFilters, emptyFilters, type FilterState } from "../lib/filters";
import {
  defaultConfig,
  earlyFusionAllowed,
  queryProblem,
  runSearch,
  type QueryState,
  type SearchConfig,
} from "../lib/search";

export function SearchView({ active }: { active: boolean }) {
  const { models, facets, intent } = useApp();
  const [query, setQuery] = useState<QueryState>({ mode: "text", text: "", image: null });
  const [config, setConfig] = useState<SearchConfig>(() => defaultConfig(models));
  const [filters, setFilters] = useState<FilterState>(emptyFilters);
  const [showOptions, setShowOptions] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [res, setRes] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [selected, setSelected] = useState<SearchHit | null>(null);
  const ctrlRef = useRef<AbortController | null>(null);
  const topRef = useRef<HTMLDivElement>(null);

  // Fill model defaults once /models has loaded.
  useEffect(() => {
    if (!models) return;
    setConfig((c) => (c.textModel && c.imageModel ? c : { ...defaultConfig(models), ...pickSet(c) }));
  }, [models]);

  const doSearch = (q = query, c = config, f = filters) => {
    const p = queryProblem(q) ??
      (q.mode === "hybrid" && c.fusion === "early" && !earlyFusionAllowed(c) && q.text.trim() && q.image
        ? "Early fusion needs the same CLIP model for text and image. Pick another fusion or matching models."
        : null);
    setProblem(p);
    if (p) return;
    ctrlRef.current?.abort();
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    setLoading(true);
    setError(null);
    runSearch(q, c, f, ctrl.signal)
      .then((r) => {
        setRes(r);
        setError(null);
      })
      .catch((e: unknown) => {
        if (isAbortError(e) || ctrl.signal.aborted) return;
        setError(e);
        setRes(null);
      })
      .finally(() => {
        if (ctrlRef.current === ctrl) setLoading(false);
      });
  };

  useEffect(() => () => ctrlRef.current?.abort(), []);

  // Requests from other views (Index: "Search this collection").
  const lastIntent = useRef<number | null>(null);
  useEffect(() => {
    if (!intent || intent.id === lastIntent.current) return;
    lastIntent.current = intent.id;
    const q = { ...query, ...intent.query };
    const c = { ...config, ...intent.config };
    const f = intent.filters ? { ...emptyFilters(), ...intent.filters } : filters;
    setQuery(q);
    setConfig(c);
    setFilters(f);
    if (intent.filters) setShowFilters(true);
    if (intent.run && !queryProblem(q)) doSearch(q, c, f);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intent]);

  const findSimilar = (imageId: number, model: string) => {
    setSelected(null);
    const q: QueryState = { ...query, mode: "image", image: { kind: "id", imageId, previewUrl: thumbnailUrl(imageId) } };
    const c = { ...config, imageModel: model };
    setQuery(q);
    setConfig(c);
    doSearch(q, c);
    topRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const useInHybrid = (imageId: number) => {
    setSelected(null);
    const q: QueryState = { ...query, mode: "hybrid", image: { kind: "id", imageId, previewUrl: thumbnailUrl(imageId) } };
    setQuery(q);
    if (q.text.trim()) doSearch(q);
    topRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const nFilters = countActiveFilters(filters);
  const maxMb = models?.limits.max_upload_mb ?? 10;

  return (
    <div className="view" ref={topRef}>
      <section className="panel">
        <QueryInput query={query} onChange={(q) => { setQuery(q); setProblem(null); }} onSubmit={(q) => doSearch(q ?? query)}
          maxMb={maxMb} active={active && !selected} busy={loading} />
        {problem && <p className="field-error" role="alert">{problem}</p>}
        <div className="toolbar">
          <button type="button" className={`btn btn-ghost btn-sm${showOptions ? " on" : ""}`}
            aria-expanded={showOptions} aria-controls="search-options" onClick={() => setShowOptions((v) => !v)}>
            Options <Caret open={showOptions} />
          </button>
          <button type="button" className={`btn btn-ghost btn-sm${showFilters ? " on" : ""}`}
            aria-expanded={showFilters} aria-controls="search-filters" onClick={() => setShowFilters((v) => !v)}>
            Filters{nFilters > 0 && <span className="count-badge" aria-label={`${nFilters} active`}>{nFilters}</span>}
            <Caret open={showFilters} />
          </button>
          <span className="toolbar-spacer" />
          <ScoreInfo />
        </div>
        {showOptions && (
          <div id="search-options" className="subpanel">
            <ConfigFields mode={query.mode} config={config} onChange={setConfig} models={models} />
          </div>
        )}
        {showFilters && (
          <div id="search-filters" className="subpanel">
            <FiltersPanel facets={facets} value={filters} onChange={setFilters} />
          </div>
        )}
        {!showOptions && query.mode === "hybrid" && (
          <div className="subpanel subpanel-inline">
            <ConfigFields mode="hybrid" config={config} onChange={setConfig} models={models} strategyOnly />
          </div>
        )}
      </section>

      <section className="results" aria-live="polite">
        {error != null && <ErrorView error={error} onRetry={() => doSearch()} />}
        {loading && !res && <SkeletonGrid count={Math.min(config.k, 12)} />}
        {res && (
          <div className={loading ? "stale" : ""}>
            <ResultsHeader res={res} />
            {res.results.length === 0
              ? <EmptyResults filtersActive={nFilters} />
              : <ResultsGrid hits={res.results} onOpen={setSelected} />}
          </div>
        )}
        {!res && !loading && error == null && <Intro />}
      </section>

      {selected && (
        <DetailModal hit={selected} onClose={() => setSelected(null)}
          onFindSimilar={findSimilar} onUseInHybrid={useInHybrid} />
      )}
    </div>
  );
}

function pickSet(c: SearchConfig): Partial<SearchConfig> {
  const out: Partial<SearchConfig> = {};
  if (c.textModel) out.textModel = c.textModel;
  if (c.imageModel) out.imageModel = c.imageModel;
  return out;
}

function Caret({ open }: { open: boolean }) {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="2.5"
      style={{ transform: open ? "rotate(180deg)" : undefined, transition: "transform .15s" }}>
      <path d="M6 9l6 6 6-6" />
    </svg>
  );
}

function Intro() {
  const { facets, bootError } = useApp();
  if (bootError) return null;
  return (
    <div className="empty intro">
      <h3>Search {facets ? facets.total.toLocaleString("en-US") : ""} images by meaning</h3>
      <p>
        Describe a scene in plain language, drop in an example image, or combine both in hybrid mode.
        Open any result to see its captions and find visually similar images.
      </p>
    </div>
  );
}

