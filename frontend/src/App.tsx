import { useEffect, useState, type ReactNode } from "react";
import { AppProvider, useApp } from "./AppContext";
import { ErrorView } from "./components/ErrorView";
import { useHashRoute, type Route } from "./lib/route";
import { CompareView } from "./views/CompareView";
import { ExperimentsView } from "./views/ExperimentsView";
import { IndexView } from "./views/IndexView";
import { SearchView } from "./views/SearchView";
import { SystemView } from "./views/SystemView";

const TABS: { route: Route; label: string }[] = [
  { route: "search", label: "Search" },
  { route: "compare", label: "Compare" },
  { route: "experiments", label: "Experiments" },
  { route: "index", label: "Index" },
  { route: "system", label: "System" },
];

export default function App() {
  return (
    <AppProvider>
      <Shell />
    </AppProvider>
  );
}

function Shell() {
  const route = useHashRoute();
  const { bootError, loading, reload, models } = useApp();
  // Views mount on first visit and then stay mounted (hidden) so their state survives tab switches.
  const [visited, setVisited] = useState<Set<Route>>(() => new Set([route]));
  useEffect(() => {
    setVisited((v) => (v.has(route) ? v : new Set(v).add(route)));
    const tab = TABS.find((t) => t.route === route);
    document.title = tab && route !== "search" ? `${tab.label} · VisualSearch` : "VisualSearch";
  }, [route]);

  const views: Record<Route, ReactNode> = {
    search: <SearchView active={route === "search"} />,
    compare: <CompareView active={route === "compare"} />,
    experiments: <ExperimentsView />,
    index: <IndexView active={route === "index"} />,
    system: <SystemView active={route === "system"} />,
  };
  const status = bootError ? "down" : loading && !models ? "pending" : "up";
  // A read-only server has no indexing routes.
  const tabs = models?.read_only ? TABS.filter((t) => t.route !== "index") : TABS;

  return (
    <>
      <a className="skip-link" href="#main">Skip to content</a>
      <header className="app-header">
        <div className="header-inner">
          <a className="brand" href="#/search">
            <svg width="24" height="24" viewBox="0 0 32 32" aria-hidden="true">
              <rect width="32" height="32" rx="7" fill="var(--accent)" />
              <circle cx="14" cy="14" r="7" fill="none" stroke="var(--on-accent)" strokeWidth="3" />
              <path d="M19 19l6 6" stroke="var(--on-accent)" strokeWidth="3" strokeLinecap="round" />
            </svg>
            <span>VisualSearch</span>
          </a>
          <nav className="tabs" aria-label="Main">
            {tabs.map((t) => (
              <a key={t.route} href={`#/${t.route}`} className={route === t.route ? "active" : ""}
                aria-current={route === t.route ? "page" : undefined}>
                {t.label}
              </a>
            ))}
          </nav>
          <span className={`backend-dot backend-${status}`} role="status"
            title={status === "up" ? "API connected" : status === "down" ? "API unreachable" : "Connecting…"}>
            <span className="visually-hidden">
              {status === "up" ? "API connected" : status === "down" ? "API unreachable" : "Connecting to API"}
            </span>
          </span>
        </div>
      </header>
      <main id="main" className="main">
        {bootError && route !== "system" && route !== "experiments" && (
          <ErrorView error={bootError} onRetry={reload} />
        )}
        {TABS.map(({ route: r }) => visited.has(r) && (
          <div key={r} hidden={route !== r}>{views[r]}</div>
        ))}
      </main>
    </>
  );
}
