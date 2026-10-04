import { useCallback, useEffect, useReducer, useRef } from "react";
import type { AssessmentDecision } from "./api";
import { reviewSession } from "./review";
import { toMessage } from "../../lib/errors";
import { fetchAiStatus, summarizeAssessment } from "../ai/api";
import { appendSession } from "../history/api";
import { aggregateVoiceResult } from "../voice/biomarkers";
import {
  buildSession,
  fallbackSummary,
  newSessionId,
  prevStep,
} from "./flow";
import type {
  AssessmentSession,
  AssessmentStep,
  EyeResult,
  Participant,
  VitalsResult,
  VoiceResult,
  VoiceTaskResult,
} from "./types";

export interface AssessmentState {
  readonly decision: AssessmentDecision | null;
  readonly decisionState: "idle" | "pending" | "ready" | "unavailable";
  readonly decisionError: string | null;
  readonly step: AssessmentStep;
  readonly participant: Participant;
  readonly voiceTasks: readonly VoiceTaskResult[];
  readonly voice: VoiceResult | null;
  readonly vitals: VitalsResult | null;
  readonly eye: EyeResult | null;
  readonly summaryText: string | null;
  readonly summarySource: "llm" | "local" | null;
  readonly saveState: "idle" | "saving" | "saved" | "error";
  readonly error: string | null;
  readonly startedAt: string;
  readonly session: AssessmentSession | null;
}

export type AssessmentAction =
  | { readonly type: "SET_DECISION"; readonly decision: AssessmentDecision | null; readonly decisionState: AssessmentState["decisionState"]; readonly decisionError?: string | null }
  | { readonly type: "START"; readonly participant: Participant; readonly startedAt?: string }
  | { readonly type: "SET_PARTICIPANT"; readonly participant: Participant }
  | { readonly type: "COMPLETE_VOICE_TASK"; readonly result: VoiceTaskResult }
  | { readonly type: "FINISH_VOICE" }
  | { readonly type: "SKIP_VOICE" }
  | { readonly type: "COMPLETE_VITALS"; readonly result: VitalsResult }
  | { readonly type: "SKIP_VITALS" }
  | { readonly type: "COMPLETE_EYE"; readonly result: EyeResult }
  | { readonly type: "SKIP_EYE" }
  | { readonly type: "BACK" }
  | { readonly type: "SET_SAVE_STATE"; readonly saveState: "idle" | "saving" | "saved" | "error"; readonly error?: string | null }
  | { readonly type: "SET_SESSION"; readonly session: AssessmentSession; readonly summarySource: "llm" | "local" }
  | { readonly type: "RESET"; readonly startedAt?: string };

export function createInitialAssessmentState(startedAt?: string): AssessmentState {
  return {
    decision: null,
    decisionState: "idle",
    decisionError: null,
    step: "setup",
    participant: { age: null, sex: "unspecified" },
    voiceTasks: [],
    voice: null,
    vitals: null,
    eye: null,
    summaryText: null,
    summarySource: null,
    saveState: "idle",
    error: null,
    startedAt: startedAt ?? new Date().toISOString(),
    session: null,
  };
}

export function assessmentReducer(state: AssessmentState, action: AssessmentAction): AssessmentState {
  switch (action.type) {
    case "SET_DECISION":
      return { ...state, decision: action.decision, decisionState: action.decisionState, decisionError: action.decisionError ?? null };
    case "START":
      return {
        ...state,
        step: "voice",
        participant: action.participant,
        startedAt: action.startedAt ?? new Date().toISOString(),
        error: null,
      };
    case "SET_PARTICIPANT":
      return { ...state, participant: action.participant };
    case "COMPLETE_VOICE_TASK": {
      const existingIdx = state.voiceTasks.findIndex((task) => task.task === action.result.task);
      const nextTasks = existingIdx >= 0
        ? state.voiceTasks.map((task, idx) => (idx === existingIdx ? action.result : task))
        : [...state.voiceTasks, action.result];
      return { ...state, voiceTasks: nextTasks };
    }
    case "FINISH_VOICE": {
      const voice = state.voiceTasks.length > 0 ? aggregateVoiceResult(state.voiceTasks) : null;
      return { ...state, voice, step: "vitals" };
    }
    case "SKIP_VOICE":
      return { ...state, voice: null, voiceTasks: [], step: "vitals" };
    case "COMPLETE_VITALS":
      return { ...state, vitals: action.result, step: "eye" };
    case "SKIP_VITALS":
      return { ...state, vitals: null, step: "eye" };
    case "COMPLETE_EYE":
      return { ...state, eye: action.result, step: "summary" };
    case "SKIP_EYE":
      return { ...state, eye: null, step: "summary" };
    case "BACK":
      return { ...state, step: prevStep(state.step) };
    case "SET_SAVE_STATE":
      return {
        ...state,
        saveState: action.saveState,
        error: action.error !== undefined ? action.error : state.error,
      };
    case "SET_SESSION":
      return {
        ...state,
        session: action.session,
        summaryText: action.session.summaryText,
        summarySource: action.summarySource,
      };
    case "RESET":
      return createInitialAssessmentState(action.startedAt);
    default: {
      const unreachable: never = action;
      return unreachable;
    }
  }
}

export function useAssessment() {
  const [state, dispatch] = useReducer(assessmentReducer, undefined, createInitialAssessmentState);
  const generation = useRef(0);
  const finalizing = useRef(false);
  useEffect(() => () => { generation.current += 1; }, []);
  const stateRef = useRef(state);
  stateRef.current = state;

  const start = useCallback((participant: Participant) => {
    dispatch({ type: "START", participant });
  }, []);

  const setParticipant = useCallback((participant: Participant) => {
    dispatch({ type: "SET_PARTICIPANT", participant });
  }, []);

  const completeVoiceTask = useCallback((result: VoiceTaskResult) => {
    dispatch({ type: "COMPLETE_VOICE_TASK", result });
  }, []);

  const finishVoice = useCallback(() => {
    dispatch({ type: "FINISH_VOICE" });
  }, []);

  const skipVoice = useCallback(() => {
    dispatch({ type: "SKIP_VOICE" });
  }, []);

  const completeVitals = useCallback((result: VitalsResult) => {
    dispatch({ type: "COMPLETE_VITALS", result });
  }, []);

  const skipVitals = useCallback(() => {
    dispatch({ type: "SKIP_VITALS" });
  }, []);

  const completeEye = useCallback((result: EyeResult) => {
    dispatch({ type: "COMPLETE_EYE", result });
  }, []);

  const skipEye = useCallback(() => {
    dispatch({ type: "SKIP_EYE" });
  }, []);

  const back = useCallback(() => {
    dispatch({ type: "BACK" });
  }, []);

  const reset = useCallback(() => {
    generation.current += 1;
    finalizing.current = false;
    dispatch({ type: "RESET" });
  }, []);

  const finalize = useCallback(async () => {
    const current = stateRef.current;
    if (finalizing.current || current.saveState === "saving" || current.saveState === "saved") {
      return;
    }
    finalizing.current = true;
    const run = generation.current;
    dispatch({ type: "SET_SAVE_STATE", saveState: "saving", error: null });

    const completedAt = new Date().toISOString();
    const preliminarySession = buildSession({
      id: newSessionId(),
      startedAt: current.startedAt || new Date().toISOString(),
      completedAt,
      participant: current.participant,
      voice: current.voice,
      vitals: current.vitals,
      eye: current.eye,
      summaryText: null,
    });

    // A history retry reuses the completed session, including its identity.
    if (current.session) {
      try {
        await appendSession(current.session);
        if (generation.current === run) dispatch({ type: "SET_SAVE_STATE", saveState: "saved", error: null });
      } catch (raised) {
        if (generation.current === run) dispatch({ type: "SET_SAVE_STATE", saveState: "error", error: toMessage(raised) });
      } finally { if (generation.current === run) finalizing.current = false; }
      return;
    }
    dispatch({ type: "SET_DECISION", decision: null, decisionState: "pending" });
    // Inference never blocks saving or PDF access. Ignore results after reset/unmount.
    void reviewSession(preliminarySession, () => generation.current === run, outcome => {
      dispatch(outcome.state === "ready"
        ? { type: "SET_DECISION", decision: outcome.decision, decisionState: "ready" }
        : { type: "SET_DECISION", decision: null, decisionState: "unavailable", decisionError: outcome.error.message });
    });

    let summaryText: string | null = null;
    let summarySource: "llm" | "local" = "local";

    try {
      const aiStatus = await fetchAiStatus();
      if (aiStatus.chat.configured) {
        try {
          summaryText = await summarizeAssessment(preliminarySession);
          summarySource = "llm";
        } catch {
          summaryText = fallbackSummary(preliminarySession);
          summarySource = "local";
        }
      } else {
        summaryText = fallbackSummary(preliminarySession);
        summarySource = "local";
      }
    } catch {
      summaryText = fallbackSummary(preliminarySession);
      summarySource = "local";
    }

    if (generation.current !== run) return;
    const finalSession: AssessmentSession = {
      ...preliminarySession,
      summaryText,
    };

    try {
      await appendSession(finalSession);
      if (generation.current !== run) return;
      dispatch({ type: "SET_SESSION", session: finalSession, summarySource });
      dispatch({ type: "SET_SAVE_STATE", saveState: "saved", error: null });
    } catch (raised) {
      const msg = toMessage(raised);
      if (generation.current !== run) return;
      dispatch({ type: "SET_SESSION", session: finalSession, summarySource });
      dispatch({ type: "SET_SAVE_STATE", saveState: "error", error: msg });
    } finally { if (generation.current === run) finalizing.current = false; }
  }, []);

  useEffect(() => {
    if (state.step === "summary" && state.saveState === "idle" && !state.session) {
      void finalize();
    }
  }, [state.step, state.saveState, state.session, finalize]);

  return {
    state,
    session: state.session,
    start,
    setParticipant,
    completeVoiceTask,
    finishVoice,
    skipVoice,
    completeVitals,
    skipVitals,
    completeEye,
    skipEye,
    back,
    finalize,
    reset,
  };
}
