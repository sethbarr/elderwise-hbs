import { toMessage } from "../../lib/errors";
import { assessSession, AssessmentUnavailableError, type AssessmentDecision } from "./api";
import type { AssessmentSession } from "./types";

export type LocalReviewOutcome =
  | { state: "ready"; decision: AssessmentDecision }
  | { state: "unavailable"; error: { code: string; message: string } };

/** Caller starts this alongside saving; reset/unmount invalidates the generation. */
export async function reviewSession(
  session: AssessmentSession,
  isCurrent: () => boolean,
  onOutcome: (outcome: LocalReviewOutcome) => void,
): Promise<void> {
  let outcome: LocalReviewOutcome;
  try {
    outcome = { state: "ready", decision: await assessSession(session) };
  } catch (raised) {
    outcome = {
      state: "unavailable",
      error: {
        code: raised instanceof AssessmentUnavailableError ? raised.code : "invalid_input",
        message: toMessage(raised),
      },
    };
  }
  if (isCurrent()) onOutcome(outcome);
}
