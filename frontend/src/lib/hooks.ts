import { useEffect, useRef, useState } from "react";

/** True while the browser tab is visible. */
export function usePageVisible(): boolean {
  const [visible, setVisible] = useState(() => document.visibilityState !== "hidden");
  useEffect(() => {
    const on = () => setVisible(document.visibilityState !== "hidden");
    document.addEventListener("visibilitychange", on);
    return () => document.removeEventListener("visibilitychange", on);
  }, []);
  return visible;
}

/**
 * Call `fn` immediately and then every `ms` while `enabled`. Each call gets an
 * AbortSignal that is aborted when the poller stops or the next tick starts.
 */
export function useInterval(fn: (signal: AbortSignal) => void | Promise<void>, ms: number,
  enabled: boolean): void {
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => {
    if (!enabled) return;
    let ctrl = new AbortController();
    let stopped = false;
    let timer: number | undefined;
    const tick = async () => {
      ctrl.abort();
      ctrl = new AbortController();
      try {
        await ref.current(ctrl.signal);
      } finally {
        if (!stopped) timer = window.setTimeout(tick, ms);
      }
    };
    void tick();
    return () => {
      stopped = true;
      ctrl.abort();
      window.clearTimeout(timer);
    };
  }, [ms, enabled]);
}

/** Close on Escape. */
export function useEscape(onEscape: () => void, enabled = true): void {
  const ref = useRef(onEscape);
  ref.current = onEscape;
  useEffect(() => {
    if (!enabled) return;
    const on = (e: KeyboardEvent) => {
      if (e.key === "Escape") ref.current();
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [enabled]);
}
