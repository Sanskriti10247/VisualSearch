import { describe, expect, it } from "vitest";
import { bestRows, fmtBytes, fmtCell, fmtMs, fmtNumber, isMetricColumn, topPercent } from "./format";

describe("topPercent", () => {
  it("converts a percentile into a top-x% label", () => {
    expect(topPercent(0.977)).toBe("top 2.3%");
    expect(topPercent(0.5)).toBe("top 50%");
    expect(topPercent(0.9995)).toBe("top 0.05%");
    expect(topPercent(1)).toBe("top <0.01%");
    expect(topPercent(null)).toBeNull();
  });
});

describe("metric formatting", () => {
  it("recognises metric column names", () => {
    for (const c of ["R@1", "R@10", "MRR", "mAP", "mAP@100", "P@5", "recall@5", "Text R@5", "nDCG@10"]) {
      expect(isMetricColumn(c)).toBe(true);
    }
    for (const c of ["Config", "latency_ms", "dim", "Model", "Index MB"]) {
      expect(isMetricColumn(c)).toBe(false);
    }
  });
  it("formats [0,1] metric values as percentages", () => {
    expect(fmtCell("R@1", 0.588)).toBe("58.8%");
    expect(fmtCell("MRR", 1)).toBe("100.0%");
    expect(fmtCell("latency_ms", 12.3456)).toBe("12.35");
    expect(fmtCell("Config", "CLIP B/32")).toBe("CLIP B/32");
    expect(fmtCell("x", null)).toBe("–");
  });
  it("rounds generic numbers sensibly", () => {
    expect(fmtNumber(512)).toBe("512");
    expect(fmtNumber(123456)).toBe("123,456");
    expect(fmtNumber(0.123456)).toBe("0.123");
    expect(fmtNumber(1234.5)).toBe("1,235");
  });
  it("finds the best rows of a column, including ties", () => {
    const rows = [["a", 0.5], ["b", 0.7], ["c", 0.7], ["d", null]];
    expect([...bestRows(rows, 1)]).toEqual([1, 2]);
  });
});

describe("units", () => {
  it("formats latency and bytes", () => {
    expect(fmtMs(0.5)).toBe("0.50 ms");
    expect(fmtMs(12.34)).toBe("12.3 ms");
    expect(fmtMs(1234)).toBe("1234 ms");
    expect(fmtMs(undefined)).toBe("–");
    expect(fmtBytes(2048)).toBe("2.0 KB");
    expect(fmtBytes(10)).toBe("10 B");
  });
});
