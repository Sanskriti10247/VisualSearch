import { describe, expect, it } from "vitest";
import { overlapAtK, rankChangeLabel, rankChanges } from "./compare";

describe("overlapAtK", () => {
  it("counts shared ids in the top-k of both lists", () => {
    expect(overlapAtK([1, 2, 3, 4], [4, 3, 9, 8], 4)).toBe(0.5);
    expect(overlapAtK([1, 2, 3], [1, 2, 3], 3)).toBe(1);
    expect(overlapAtK([1, 2, 3], [4, 5, 6], 3)).toBe(0);
  });
  it("only looks at the first k entries", () => {
    expect(overlapAtK([1, 2, 3, 4], [5, 6, 1, 2], 2)).toBe(0);
  });
  it("divides by k even when lists are shorter", () => {
    expect(overlapAtK([1, 2], [1, 2], 4)).toBe(0.5);
    expect(overlapAtK([], [], 0)).toBe(0);
  });
});

describe("rankChanges", () => {
  it("reports up / down / same / new relative to A", () => {
    const c = rankChanges([10, 20, 30, 40], [40, 20, 10, 99]);
    expect(c.get(40)).toEqual({ kind: "up", by: 3 });
    expect(c.get(20)).toEqual({ kind: "same" });
    expect(c.get(10)).toEqual({ kind: "down", by: 2 });
    expect(c.get(99)).toEqual({ kind: "new" });
    expect(rankChangeLabel(c.get(40)!)).toBe("↑3");
    expect(rankChangeLabel(c.get(10)!)).toBe("↓2");
    expect(rankChangeLabel(c.get(99)!)).toBe("new");
  });
});
