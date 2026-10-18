# VisualSearch web UI

Vite + React 18 + TypeScript (strict), plain CSS, no UI/chart libraries.

## Run

```bash
npm install
npm run dev        # http://localhost:5173, proxies /api -> http://127.0.0.1:8000
npm run build      # tsc -b && vite build -> dist/
npm run preview    # serve dist/ (also proxies /api)
npm test           # vitest unit tests for pure helpers
npm run typecheck
```

Start the backend separately (`visualsearch serve`, default port 8000).

## Configuration

| Variable        | Default | Meaning |
|-----------------|---------|---------|
| `VITE_API_BASE` | `""`    | Origin prefix for API calls. Empty = same origin (dev proxy, or nginx proxying `/api` in production). Set e.g. `VITE_API_BASE=https://api.example.com` at build time to call another origin (the backend's CORS settings must allow it). |

## Views

Tabs live in the URL hash: `#/search`, `#/compare`, `#/experiments`, `#/index`, `#/system`.

- **Search**: text / image / hybrid queries, options (models, k, reranker, fusion, min P(match)), facet filters, results with per-signal scores, detail modal with "find similar".
- **Compare**: one query, two configurations side by side, overlap@k and rank changes.
- **Experiments**: renders `report` sections (tables + SVG charts) from `GET /api/v1/experiments`.
- **Index**: upload images into a collection and follow the background job.
- **System**: health, models, calibrations, latency metrics (auto-refresh every 5 s).

## Structure

```
src/
  api.ts             typed client, schema types, error-envelope parsing (ApiError)
  AppContext.tsx     /models + /facets bootstrap, cross-view "open search with ..." intents
  App.tsx            header, hash-routed tabs (views stay mounted once visited)
  styles.css         design tokens (light + dark) and all styles
  lib/               pure helpers (+ *.test.ts): filters, format, compare, scale, search, route, hooks
  components/        QueryInput, ImageInput, ConfigFields, FiltersPanel, ResultsGrid, ResultCard,
                     ResultsHeader, TimingBar, SignalScores, DetailModal, Modal, Chart, DataTable, ...
  views/             SearchView, CompareView, ExperimentsView, IndexView, SystemView
```
