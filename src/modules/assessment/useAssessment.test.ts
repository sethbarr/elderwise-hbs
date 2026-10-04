import { describe, expect, it } from "vitest";
import { sampleSession } from "../export/__fixtures__/sampleSession";
import {
  assessmentReducer,
  createInitialAssessmentState,
} from "./useAssessment";

describe("assessmentReducer", () => {
  it("starts in setup step with blank participant and empty results", () => {
    const state = createInitialAssessmentState("2026-10-04T12:00:00.000Z");
    expect(state.step).toBe("setup");
    expect(state.participant).toEqual({ age: null, sex: "unspecified" });
    expect(state.voiceTasks).toEqual([]);
    expect(state.voice).toBeNull();
    expect(state.vitals).toBeNull();
    expect(state.eye).toBeNull();
    expect(state.saveState).toBe("idle");
    expect(state.session).toBeNull();
  });

  it("advances to voice step on START with provided participant", () => {
    const initial = createInitialAssessmentState();
    const state = assessmentReducer(initial, {
      type: "START",
      participant: { age: 76, sex: "male" },
      startedAt: "2026-10-04T12:05:00.000Z",
    });

    expect(state.step).toBe("voice");
    expect(state.participant).toEqual({ age: 76, sex: "male" });
    expect(state.startedAt).toBe("2026-10-04T12:05:00.000Z");
  });

  it("allows updating participant demographics", () => {
    const initial = createInitialAssessmentState();
    const state = assessmentReducer(initial, {
      type: "SET_PARTICIPANT",
      participant: { age: 80, sex: "female" },
    });
    expect(state.participant).toEqual({ age: 80, sex: "female" });
  });

  it("records voice tasks and updates existing task if retried", () => {
    const initial = createInitialAssessmentState();
    const task1 = sampleSession.voice!.tasks[0]!;
    const state1 = assessmentReducer(initial, {
      type: "COMPLETE_VOICE_TASK",
      result: task1,
    });
    expect(state1.voiceTasks.length).toBe(1);
    expect(state1.voiceTasks[0]?.task).toBe(task1.task);

    const task1Updated = { ...task1, prompt: "Updated prompt" };
    const state2 = assessmentReducer(state1, {
      type: "COMPLETE_VOICE_TASK",
      result: task1Updated,
    });
    expect(state2.voiceTasks.length).toBe(1);
    expect(state2.voiceTasks[0]?.prompt).toBe("Updated prompt");
  });

  it("finishVoice aggregates voice tasks and advances to vitals step", () => {
    const initial = createInitialAssessmentState();
    const task1 = sampleSession.voice!.tasks[0]!;
    const withTask = assessmentReducer(initial, {
      type: "COMPLETE_VOICE_TASK",
      result: task1,
    });

    const finished = assessmentReducer(withTask, { type: "FINISH_VOICE" });
    expect(finished.step).toBe("vitals");
    expect(finished.voice).not.toBeNull();
    expect(finished.voice?.tasks.length).toBe(1);
  });

  it("finishVoice with zero tasks sets voice to null and advances to vitals step", () => {
    const initial = createInitialAssessmentState();
    const finished = assessmentReducer(initial, { type: "FINISH_VOICE" });
    expect(finished.step).toBe("vitals");
    expect(finished.voice).toBeNull();
  });

  it("skipVoice sets voice to null, empties voiceTasks, and advances to vitals", () => {
    const initial = createInitialAssessmentState();
    const task1 = sampleSession.voice!.tasks[0]!;
    const withTask = assessmentReducer(initial, {
      type: "COMPLETE_VOICE_TASK",
      result: task1,
    });

    const skipped = assessmentReducer(withTask, { type: "SKIP_VOICE" });
    expect(skipped.step).toBe("vitals");
    expect(skipped.voice).toBeNull();
    expect(skipped.voiceTasks).toEqual([]);
  });

  it("handles complete and skip transitions for vitals step", () => {
    const initial = { ...createInitialAssessmentState(), step: "vitals" as const };
    const completed = assessmentReducer(initial, {
      type: "COMPLETE_VITALS",
      result: sampleSession.vitals!,
    });
    expect(completed.step).toBe("eye");
    expect(completed.vitals).toEqual(sampleSession.vitals);

    const skipped = assessmentReducer(initial, { type: "SKIP_VITALS" });
    expect(skipped.step).toBe("eye");
    expect(skipped.vitals).toBeNull();
  });

  it("handles complete and skip transitions for eye tracking step", () => {
    const initial = { ...createInitialAssessmentState(), step: "eye" as const };
    const completed = assessmentReducer(initial, {
      type: "COMPLETE_EYE",
      result: sampleSession.eye!,
    });
    expect(completed.step).toBe("summary");
    expect(completed.eye).toEqual(sampleSession.eye);

    const skipped = assessmentReducer(initial, { type: "SKIP_EYE" });
    expect(skipped.step).toBe("summary");
    expect(skipped.eye).toBeNull();
  });

  it("retreats step with BACK action", () => {
    const initial = { ...createInitialAssessmentState(), step: "vitals" as const };
    const backed = assessmentReducer(initial, { type: "BACK" });
    expect(backed.step).toBe("voice");
  });

  it("manages saveState, error, session, and reset", () => {
    const initial = createInitialAssessmentState();
    const saving = assessmentReducer(initial, {
      type: "SET_SAVE_STATE",
      saveState: "saving",
    });
    expect(saving.saveState).toBe("saving");

    const withSession = assessmentReducer(saving, {
      type: "SET_SESSION",
      session: sampleSession,
      summarySource: "local",
    });
    expect(withSession.session).toEqual(sampleSession);
    expect(withSession.summarySource).toBe("local");
    expect(withSession.summaryText).toBe(sampleSession.summaryText);

    const resetState = assessmentReducer(withSession, {
      type: "RESET",
      startedAt: "2026-10-04T13:00:00.000Z",
    });
    expect(resetState.step).toBe("setup");
    expect(resetState.session).toBeNull();
    expect(resetState.startedAt).toBe("2026-10-04T13:00:00.000Z");
  });
});

describe("local decision state", () => {
  it("keeps model failure separate from the completed session and saving", () => {
    const state = assessmentReducer(createInitialAssessmentState(), {
      type: "SET_SESSION", session: sampleSession, summarySource: "local",
    });
    const failed = assessmentReducer(state, {
      type: "SET_DECISION", decision: null, decisionState: "unavailable", decisionError: "Model did not start",
    });
    expect(failed.session).toBe(sampleSession);
    expect(failed.error).toBeNull();
    expect(failed.decisionError).toBe("Model did not start");
    expect(assessmentReducer(failed, { type: "RESET" }).decisionState).toBe("idle");
  });
});
