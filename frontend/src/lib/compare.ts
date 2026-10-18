/** Helpers for comparing two ranked result lists (Compare view). */

/** |top-k(A) ∩ top-k(B)| / k. Uses min(k, ...) lists; returns 0 when k <= 0. */
export function overlapAtK(a: number[], b: number[], k: number): number {
  if (k <= 0) return 0;
  const sa = new Set(a.slice(0, k));
  let n = 0;
  for (const id of new Set(b.slice(0, k))) if (sa.has(id)) n++;
  return n / k;
}

export type RankChange =
  | { kind: "new" }
  | { kind: "same" }
  | { kind: "up"; by: number }
  | { kind: "down"; by: number };

/**
 * For each id in B: how its position moved relative to A.
 * "up" means it ranks better (smaller index) in B than in A.
 */
export function rankChanges(a: number[], b: number[]): Map<number, RankChange> {
  const posA = new Map<number, number>();
  a.forEach((id, i) => {
    if (!posA.has(id)) posA.set(id, i);
  });
  const out = new Map<number, RankChange>();
  b.forEach((id, i) => {
    const pa = posA.get(id);
    if (pa === undefined) out.set(id, { kind: "new" });
    else if (pa === i) out.set(id, { kind: "same" });
    else if (pa > i) out.set(id, { kind: "up", by: pa - i });
    else out.set(id, { kind: "down", by: i - pa });
  });
  return out;
}

export function rankChangeLabel(c: RankChange): string {
  switch (c.kind) {
    case "new":
      return "new";
    case "same":
      return "=";
    case "up":
      return `↑${c.by}`;
    case "down":
      return `↓${c.by}`;
  }
}
