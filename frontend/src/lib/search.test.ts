import { describe, expect, it } from "vitest";
import { emptyFilters } from "./filters";
import type { ModelInfo, ModelsResponse } from "../api";
import { buildSearchRequest, defaultConfig, modelOfFamily, queryProblem, type QueryState } from "./search";

const cfg = { ...defaultConfig(null), textModel: "clip-vit-b32", imageModel: "dinov2-small" };

describe("buildSearchRequest", () => {
  it("builds a JSON body for text search", () => {
    const q: QueryState = { mode: "text", text: "  two dogs  ", image: null };
    const f = { ...emptyFilters(), labels: ["dog"] };
    const r = buildSearchRequest(q, { ...cfg, minProb: 0.2 }, f);
    expect(r.endpoint).toBe("text");
    if (r.endpoint !== "text") return;
    expect(r.body).toEqual({ query: "two dogs", k: 24, rerank: "none", model: "clip-vit-b32",
      filters: { labels: ["dog"] }, min_prob: 0.2 });
  });
  it("builds multipart form fields for hybrid search with an indexed image", () => {
    const q: QueryState = { mode: "hybrid", text: "snow", image: { kind: "id", imageId: 7, previewUrl: "" } };
    const r = buildSearchRequest(q, cfg, { ...emptyFilters(), collections: ["c1"] });
    expect(r.endpoint).toBe("hybrid");
    if (r.endpoint === "text") return;
    expect(r.form.get("image_id")).toBe("7");
    expect(r.form.get("text")).toBe("snow");
    expect(r.form.get("fusion")).toBe("zscore");
    expect(r.form.get("text_weight")).toBe("0.5");
    expect(JSON.parse(String(r.form.get("filters")))).toEqual({ collections: ["c1"] });
    expect(r.form.get("file")).toBeNull();
  });
  it("omits filters for image search when none are active", () => {
    const q: QueryState = { mode: "image", text: "", image: { kind: "id", imageId: 3, previewUrl: "" } };
    const r = buildSearchRequest(q, cfg, emptyFilters());
    if (r.endpoint === "text") throw new Error("wrong endpoint");
    expect(r.form.get("filters")).toBeNull();
    expect(r.form.get("model")).toBe("dinov2-small");
  });
});

describe("queryProblem", () => {
  it("validates per mode", () => {
    expect(queryProblem({ mode: "text", text: " ", image: null })).not.toBeNull();
    expect(queryProblem({ mode: "hybrid", text: "x", image: null })).toBeNull();
    expect(queryProblem({ mode: "image", text: "x", image: null })).not.toBeNull();
  });
});

describe("modelOfFamily", () => {
  const model = (key: string, family: string, modalities: string[]) =>
    ({ key, family, modalities }) as ModelInfo;
  const models = (clipModalities: string[]) => ({
    default_text_model: "clip-vit-b32", default_image_model: "dinov2-small",
    models: [model("clip-vit-b32", "clip", clipModalities), model("dinov2-small", "dinov2", ["image"])],
  }) as ModelsResponse;
  it("picks image-capable models only", () => {
    expect(modelOfFamily(models(["image", "text"]), "clip")).toBe("clip-vit-b32");
    expect(modelOfFamily(models(["text"]), "clip")).toBeNull();  // slim server: text tower only
    expect(modelOfFamily(models(["text"]), "dinov2")).toBe("dinov2-small");
  });
});
