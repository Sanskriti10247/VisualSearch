import { useEffect, useMemo, useRef, useState } from "react";
import { isAbortError, thumbnailUrl, type ModelsResponse, type SearchHit, type SearchResponse } from "../api";
import { useApp } from "../AppContext";
import { ConfigFields } from "../components/ConfigFields";
import { DetailModal } from "../components/DetailModal";
import { ErrorView } from "../components/ErrorView";
import { FiltersPanel } from "../components/FiltersPanel";
import { QueryInput } from "../components/QueryInput";
import { EmptyResults, ResultsGrid, SkeletonGrid } from "../components/ResultsGrid";
import { ResultsHeader } from "../components/ResultsHeader";
import { overlapAtK, rankChangeLabel, rankChanges, type RankChange } from "../lib/compare";
import { countActiveFilters, emptyFilters, type FilterState } from "../lib/filters";
import { fmtMs } from "../lib/format";
import { navigate } from "../lib/route";
import {
  defaultConfig,
  earlyFusionAllowed,
  modelOfFamily,
  queryProblem,
  runSearch,
  type QueryState,
  type SearchConfig,
} from "../lib/search";

interface Side {
  res: SearchResponse | null;
  error: unknown;
}

/** A sensible "B" default: same as A but with a different strategy to compare. */
function defaultB(models: ModelsResponse | null): SearchConfig {
  const c = defaultConfig(models);
  const tr = models?.rerankers.text ?? [];
  const dino = modelOfFamily(models, "dinov2");
  const clip = modelOfFamily(models, "clip");
  return {
    ...c,
    textRerank: tr.includes("visual_prf") ? "visual_prf" : tr.includes("prf") ? "prf" : "none",
    imageModel: c.imageModel === dino && clip ? clip : dino ?? c.imageModel,
    fusion: "rrf",
  };
}

export function CompareView({ active }: { active: boolean }) {
  const { models, facets, requestSearch } = useApp();
  const [query, setQuery] = useState<QueryState>({ mode: "text", text: "", image: null });
  const [a, setA] = useState<SearchConfig>(() => defaultConfig(models));
  const [b, setB] = useState<SearchConfig>(() => defaultB(models));
  const [filters, setFilters] = useState<FilterState>(emptyFilters);
  const [showFilters, setShowFilters] = useState(false);
  const [sideA, setSideA] = useState<Side>({ res: null, error: null });
  const [sideB, setSideB] = useState<Side>({ res: null, error: null });
  const [loading, setLoading] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [selected, setSelected] = useState<SearchHit | null>(null);
  const ctrlRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!models) return;
    setA((c) => (c.textModel ? c : defaultConfig(models)));
    setB((c) => (c.textModel ? c : defaultB(models)));
  }, [models]);
  useEffect(() => () => ctrlRef.current?.abort(), []);

  const k = a.k;
  const run = (q = query) => {
    const p = queryProblem(q) ?? (q.mode === "hybrid" && q.text.trim() && q.image &&
      [a, b].some((c) => c.fusion === "early" && !earlyFusionAllowed(c))
      ? "Early fusion needs the same CLIP model for text and image." : null);
    setProblem(p);
    if (p) return;
    ctrlRef.current?.abort();
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    setLoading(true);
    const one = (c: SearchConfig, set: (s: Side) => void) =>
      runSearch(q, { ...c, k, minProb: a.minProb }, filters, ctrl.signal)
        .then((res) => set({ res, error: null }))
        .catch((error: unknown) => {
          if (!isAbortError(error) && !ctrl.signal.aborted) set({ res: null, error });
        });
    Promise.all([one(a, setSideA), one(b, setSideB)]).finally(() => {
      if (ctrlRef.current === ctrl) setLoading(false);
    });
  };

  const idsA = useMemo(() => sideA.res?.results.map((h) => h.image_id) ?? [], [sideA.res]);
  const idsB = useMemo(() => sideB.res?.results.map((h) => h.image_id) ?? [], [sideB.res]);
  const changes = useMemo(() => rankChanges(idsA, idsB), [idsA, idsB]);
  const inB = useMemo(() => new Set(idsB), [idsB]);
  const both = sideA.res && sideB.res;
  const overlap = both ? overlapAtK(idsA, idsB, k) : null;
  const nFilters = countActiveFilters(filters);

  const toSearch = (q: Partial<QueryState>, cfg: Partial<SearchConfig>) => {
    setSelected(null);
    requestSearch({ query: q, config: cfg, run: true });
    navigate("search");
  };

  return (
    <div className="view">
      <section className="panel">
        <QueryInput query={query} onChange={(q) => { setQuery(q); setProblem(null); }} onSubmit={(q) => run(q ?? query)}
          maxMb={models?.limits.max_upload_mb ?? 10} active={active && !selected} busy={loading} submitLabel="Compare" />
        {problem && <p className="field-error" role="alert">{problem}</p>}
        <div className="compare-configs">
          {([["A", a, setA], ["B", b, setB]] as const).map(([name, cfg, set]) => (
            <fieldset key={name} className={`subpanel config-${name}`}>
              <legend><span className={`side-badge side-${name}`}>{name}</span> Configuration</legend>
              <ConfigFields mode={query.mode} config={cfg} onChange={set} models={models} strategyOnly={name === "B"} />
            </fieldset>
          ))}
        </div>
        <div className="toolbar">
          <button type="button" className={`btn btn-ghost btn-sm${showFilters ? " on" : ""}`} aria-expanded={showFilters}
            onClick={() => setShowFilters((v) => !v)}>
            Shared filters{nFilters > 0 && <span className="count-badge">{nFilters}</span>}
          </button>
          <span className="hint">k and min P(match) from A apply to both.</span>
        </div>
        {showFilters && (
          <div className="subpanel">
            <FiltersPanel facets={facets} value={filters} onChange={setFilters} />
          </div>
        )}
      </section>

      {both && (
        <section className="compare-summary" aria-label="Comparison summary">
          <div className="stat">
            <span className="stat-label">overlap@{k}</span>
            <span className="stat-value">{((overlap ?? 0) * 100).toFixed(0)}%</span>
            <span className="hint">{Math.round((overlap ?? 0) * k)} of {k} shared</span>
          </div>
          <div className="stat">
            <span className="stat-label">latency A</span>
            <span className="stat-value">{fmtMs(sideA.res!.timings_ms.total)}</span>
          </div>
          <div className="stat">
            <span className="stat-label">latency B</span>
            <span className="stat-value">{fmtMs(sideB.res!.timings_ms.total)}</span>
          </div>
          <div className="stat">
            <span className="stat-label">new in B</span>
            <span className="stat-value">{[...changes.values()].filter((c) => c.kind === "new").length}</span>
          </div>
        </section>
      )}

      <div className="compare-cols">
        {([["A", sideA], ["B", sideB]] as const).map(([name, side]) => (
          <section key={name} className="compare-col" aria-label={`Results ${name}`}>
            <h2 className="col-title"><span className={`side-badge side-${name}`}>{name}</span>
              {side.res && <span className="muted">{fmtMs(side.res.timings_ms.total)}</span>}
            </h2>
            {side.error != null && <ErrorView error={side.error} compact onRetry={() => run()} />}
            {loading && !side.res && <SkeletonGrid count={6} compact />}
            {side.res && (
              <div className={loading ? "stale" : ""}>
                <ResultsHeader res={side.res} compact />
                {side.res.results.length === 0
                  ? <EmptyResults filtersActive={nFilters} />
                  : (
                    <ResultsGrid hits={side.res.results} onOpen={setSelected} compact
                      badgeFor={(h) => name === "B"
                        ? <ChangeBadge change={changes.get(h.image_id)} />
                        : (sideB.res && !inB.has(h.image_id) ? <span className="delta delta-out" title="Not in B's results">only A</span> : null)} />
                  )}
              </div>
            )}
            {!side.res && !loading && side.error == null && (
              <p className="hint empty-col">Run a query to see configuration {name}'s results.</p>
            )}
          </section>
        ))}
      </div>

      {selected && (
        <DetailModal hit={selected} onClose={() => setSelected(null)}
          onFindSimilar={(id, model) => toSearch(
            { mode: "image", image: { kind: "id", imageId: id, previewUrl: thumbnailUrl(id) } }, { imageModel: model })}
          onUseInHybrid={(id) => toSearch(
            { mode: "hybrid", text: query.text, image: { kind: "id", imageId: id, previewUrl: thumbnailUrl(id) } }, {})} />
      )}
    </div>
  );
}

function ChangeBadge({ change }: { change: RankChange | undefined }) {
  if (!change) return null;
  const title = change.kind === "new" ? "Not in A's results"
    : change.kind === "same" ? "Same rank as in A"
      : `${change.kind === "up" ? "Up" : "Down"} ${change.by} vs. A`;
  return <span className={`delta delta-${change.kind}`} title={title}>{rankChangeLabel(change)}</span>;
}
