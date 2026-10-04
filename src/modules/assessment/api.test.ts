import { beforeEach, describe, expect, it, vi } from "vitest";

import { assessSession, assessmentDecisionSchema, AssessmentUnavailableError } from "./api";
import { assessmentSessionSchema } from "./types";
import fullSessionFixture from "../../../src-tauri/src/assessment/fixtures/full-session.json";
import { reviewSession, type LocalReviewOutcome } from "./review";
import sessionFixture from "../../../src-tauri/src/assessment/fixtures/session.json";
const bridge = vi.hoisted(() => ({ raw: undefined as unknown, error: undefined as unknown, calls: [] as unknown[][] }));
vi.mock("@tauri-apps/api/core", () => ({ invoke: async (...args: unknown[]) => {
  bridge.calls.push(args);
  if (bridge.error) throw bridge.error;
  return bridge.raw;
} }));
const decision = {
  model: "ggml-org/Clef-Flash-GGUF:Q4_K_M",
  route: { selected: "clinical_review", probability: 0.55, confidence: 0.41,
    probabilities: { routine: 0.03, caregiver_review: 0.1, clinical_review: 0.55, urgent_review: 0.32 } },
  safety: { concernProbability: 0.74 },
  urgency: { score: 1.73, mostLikely: "Same day", mostLikelyIndex: 2, confidence: 0.37,
    probabilities: { "0": 0.06, "1": 0.32, "2": 0.44, "3": 0.18 } },
  inputTokens: 398,
};
beforeEach(() => { bridge.raw = undefined; bridge.error = undefined; bridge.calls = []; });
describe("assessment IPC", () => {
  it("validates session and normalized output", async () => {
    bridge.raw = decision;
    const session = assessmentSessionSchema.parse(sessionFixture);
    expect(await assessSession(session)).toEqual(decision);
    expect(bridge.calls[0]).toEqual(["assess_session", { session }]);
  });
  it("rejects malformed distributions, inconsistent mode/legend and invalid scores", () => {
    expect(assessmentDecisionSchema.safeParse(decision).success).toBe(true);
    for (const invalid of [
      { ...decision, safety: { concernProbability: 2 } },
      { ...decision, urgency: { ...decision.urgency, mostLikely: "Soon" } },
      { ...decision, route: { ...decision.route, probability: 0.9 } },
      { ...decision, route: { ...decision.route, probabilities: { routine: 1 } } },
    ]) expect(assessmentDecisionSchema.safeParse(invalid).success).toBe(false);
  });
  it("preserves typed backend errors without crashing", async () => {
    bridge.error = { code: "startup_timeout", message: "Model unavailable" };
    const error = await assessSession(assessmentSessionSchema.parse(sessionFixture)).catch(e => e);
    expect(error).toBeInstanceOf(AssessmentUnavailableError);
    expect(error.code).toBe("startup_timeout");
    expect(error.message).toBe("Model unavailable");
  });
  it("wraps browser transport and invalid responses", async () => {
    bridge.raw = {};
    await expect(assessSession(assessmentSessionSchema.parse(sessionFixture))).rejects.toBeInstanceOf(AssessmentUnavailableError);
    bridge.error = new Error("no IPC");
    const error = await assessSession(assessmentSessionSchema.parse(sessionFixture)).catch(e => e);
    expect(error.code).toBe("unavailable");
  });
  it("rejects invalid input before invoking Rust", async () => {
    await expect(assessSession({ ...sessionFixture, participant: { age: 2, sex: "female" } } as never)).rejects.toThrow();
    expect(bridge.calls).toHaveLength(0);
  });
});


describe("nonblocking local review", () => {
  it("validates the full fixture used by the managed GPU smoke test", () => {
    expect(assessmentSessionSchema.safeParse(fullSessionFixture).success).toBe(true);
  });
  it("allows saving while inference is pending and ignores a stale result after reset", async () => {
    let resolve!: (value: unknown) => void;
    bridge.raw = new Promise(r => { resolve = r; });
    let current = true;
    const outcomes: LocalReviewOutcome[] = [];
    const review = reviewSession(assessmentSessionSchema.parse(sessionFixture), () => current, o => outcomes.push(o));
    // No model result is required for independently saving the original session.
    const saved = assessmentSessionSchema.parse(sessionFixture);
    expect(saved.id).toBe(sessionFixture.id);
    expect(outcomes).toHaveLength(0);
    current = false;
    resolve(decision);
    await review;
    expect(outcomes).toHaveLength(0);
  });
  it("surfaces typed model failure separately from the session", async () => {
    bridge.error = { code: "startup", message: "Missing sidecar" };
    const session = assessmentSessionSchema.parse(sessionFixture);
    const outcomes: LocalReviewOutcome[] = [];
    await reviewSession(session, () => true, o => outcomes.push(o));
    expect(outcomes).toEqual([{ state: "unavailable", error: { code: "startup", message: "Missing sidecar" } }]);
    expect(session).toEqual(sessionFixture);
  });
  it("delivers a successful validated decision for the current session", async () => {
    bridge.raw = decision;
    const outcomes: LocalReviewOutcome[] = [];
    await reviewSession(assessmentSessionSchema.parse(sessionFixture), () => true, o => outcomes.push(o));
    expect(outcomes).toEqual([{ state: "ready", decision }]);
  });
});
