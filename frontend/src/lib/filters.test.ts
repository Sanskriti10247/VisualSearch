import { describe, expect, it } from "vitest";
import { countActiveFilters, emptyFilters, filtersFormValue, parseCsv, serializeFilters } from "./filters";

describe("filter serialization", () => {
  it("returns null when no filter is active", () => {
    expect(serializeFilters(emptyFilters())).toBeNull();
    expect(filtersFormValue(emptyFilters())).toBeUndefined();
    expect(countActiveFilters(emptyFilters())).toBe(0);
  });
  it("drops empty fields and keeps active ones", () => {
    const f = { ...emptyFilters(), collections: ["flickr30k"], orientations: ["portrait" as const], min_width: 640 };
    expect(serializeFilters(f)).toEqual({ collections: ["flickr30k"], orientations: ["portrait"], min_width: 640 });
    expect(countActiveFilters(f)).toBe(3);
    expect(JSON.parse(filtersFormValue(f)!)).toEqual(serializeFilters(f));
  });
  it("ignores invalid dimensions and dedupes values", () => {
    const f = { ...emptyFilters(), tags_any: ["a", " a ", ""], min_height: 0, min_width: Number.NaN };
    expect(serializeFilters(f)).toEqual({ tags_any: ["a"] });
    expect(countActiveFilters(f)).toBe(1);
  });
  it("parses comma-separated tags", () => {
    expect(parseCsv(" cat, dog ,,cat ")).toEqual(["cat", "dog"]);
  });
});
