import { describe, expect, it } from "vitest";
import { linearScale, logTicks, niceTicks } from "./scale";
import { parseErrorEnvelope, validationIssues } from "../api";

describe("ticks", () => {
  it("produces nice linear ticks", () => {
    expect(niceTicks(0, 1, 5)).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
    expect(niceTicks(3, 97, 4)).toEqual([0, 20, 40, 60, 80, 100]);
  });
  it("produces log ticks", () => {
    expect(logTicks(1, 1000)).toEqual([1, 10, 100, 1000]);
    expect(logTicks(1, 50)).toEqual([1, 2, 5, 10, 20, 50]);
  });
  it("maps a linear domain to a range", () => {
    const s = linearScale(0, 10, 100, 200);
    expect(s(5)).toBe(150);
  });
});

describe("parseErrorEnvelope", () => {
  it("reads the API error envelope", () => {
    const e = parseErrorEnvelope(422, { error: { code: "validation_error", message: "Request validation failed",
      details: [{ loc: ["body", "query"], msg: "too short", type: "x" }] } });
    expect(e.code).toBe("validation_error");
    expect(e.status).toBe(422);
    expect(validationIssues(e.details)).toHaveLength(1);
  });
  it("falls back for non-envelope bodies", () => {
    expect(parseErrorEnvelope(404, { detail: "Not Found" }).message).toBe("Not Found");
    expect(parseErrorEnvelope(502, "<html>Bad gateway</html>").code).toBe("http_error");
    expect(parseErrorEnvelope(500, null, "Internal").message).toBe("Internal");
  });
});
