import { useEffect, useState } from "react";

export const ROUTES = ["search", "compare", "experiments", "index", "system"] as const;
export type Route = (typeof ROUTES)[number];

export function parseHash(hash: string): Route {
  const m = /^#\/?([\w-]+)/.exec(hash);
  const r = m?.[1] as Route | undefined;
  return r && (ROUTES as readonly string[]).includes(r) ? r : "search";
}

export function navigate(route: Route): void {
  if (parseHash(window.location.hash) !== route || !window.location.hash) {
    window.location.hash = `#/${route}`;
  }
}

/** Current tab, kept in the URL hash (#/search, #/compare, ...). */
export function useHashRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));
  useEffect(() => {
    const on = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", on);
    if (!window.location.hash) window.history.replaceState(null, "", "#/search");
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}
