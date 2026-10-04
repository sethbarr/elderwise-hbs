import { invoke } from "@tauri-apps/api/core";
import { z } from "zod";
import { toMessage } from "../../lib/errors";
import { assessmentSessionSchema, type AssessmentSession } from "./types";

const probability = z.number().finite().min(0).max(1);
const route = z.enum(["routine", "caregiver_review", "clinical_review", "urgent_review"]);
const urgencyLabels = ["Routine", "Soon", "Same day", "Immediate"] as const;
const routeProbabilities = z.object({ routine: probability, caregiver_review: probability, clinical_review: probability, urgent_review: probability });
const urgencyProbabilities = z.object({ "0": probability, "1": probability, "2": probability, "3": probability });
const sumsToOne = (p: Record<string, number>) => Math.abs(Object.values(p).reduce((a, b) => a + b, 0) - 1) < 0.00001;
/** Experimental model preferences; none of these values are calibrated medical risk. */
export const assessmentDecisionSchema = z.object({
  model: z.string().min(1).max(256),
  route: z.object({ selected: route, probability, confidence: probability, probabilities: routeProbabilities }).refine(
    r => sumsToOne(r.probabilities) && Math.abs(r.probability - r.probabilities[r.selected]) < 0.00001
      && Object.values(r.probabilities).every(p => p <= r.probability + 1e-9),
    "Invalid route distribution",
  ),
  safety: z.object({ concernProbability: probability }),
  urgency: z.object({
    score: z.number().finite().min(0).max(3),
    mostLikely: z.enum(urgencyLabels),
    mostLikelyIndex: z.number().int().min(0).max(3),
    confidence: probability,
    probabilities: urgencyProbabilities,
  }).refine(u => {
    const p = Object.values(u.probabilities);
    const mode = p.indexOf(Math.max(...p));
    return sumsToOne(u.probabilities) && mode === u.mostLikelyIndex && urgencyLabels[mode] === u.mostLikely;
  }, "Invalid urgency distribution or legend"),
  inputTokens: z.number().int().nonnegative(),
});
export type AssessmentDecision = z.infer<typeof assessmentDecisionSchema>;
const errorSchema = z.object({ code: z.string(), message: z.string() });
export class AssessmentUnavailableError extends Error {
  constructor(readonly code: string, message: string) { super(message); this.name = "AssessmentUnavailableError"; }
}
export async function assessSession(session: AssessmentSession): Promise<AssessmentDecision> {
  const validated = assessmentSessionSchema.parse(session);
  let raw: unknown;
  try { raw = await invoke("assess_session", { session: validated }); }
  catch (raised) {
    const typed = errorSchema.safeParse(raised);
    throw new AssessmentUnavailableError(typed.success ? typed.data.code : "unavailable", typed.success ? typed.data.message : toMessage(raised));
  }
  const parsed = assessmentDecisionSchema.safeParse(raw);
  if (!parsed.success) throw new AssessmentUnavailableError("invalid_response", "Local model returned an invalid decision");
  return parsed.data;
}
export const engineStatusSchema = z.object({
  state: z.enum(["idle", "starting", "ready", "unavailable", "stopped"]),
  error: errorSchema.nullable(),
});
export async function fetchAssessmentEngineStatus() {
  return engineStatusSchema.parse(await invoke("assessment_engine_status"));
}
