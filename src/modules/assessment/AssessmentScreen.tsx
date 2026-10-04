import { useEffect } from "react";
import { type WakeLockHandle, requestScreenWakeLock } from "../../lib/wakeLock";
import { EyeStep } from "./steps/EyeStep";
import { SetupStep } from "./steps/SetupStep";
import { SummaryStep } from "./steps/SummaryStep";
import { VitalsStep } from "./steps/VitalsStep";
import { VoiceStep } from "./steps/VoiceStep";
import { ASSESSMENT_STEPS, type AssessmentStep } from "./types";
import { useAssessment } from "./useAssessment";

const STEP_LABELS: Record<AssessmentStep, string> = {
  setup: "1. Setup",
  voice: "2. Voice",
  vitals: "3. Vitals",
  eye: "4. Eye Tracking",
  summary: "5. Summary",
};

export function AssessmentScreen() {
  const {
    state,
    session,
    start,
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
  } = useAssessment();

  useEffect(() => {
    let handle: WakeLockHandle | null = null;
    const isCapture = state.step === "voice" || state.step === "vitals" || state.step === "eye";

    if (isCapture) {
      void requestScreenWakeLock().then((acquired) => {
        handle = acquired;
      });
    }

    return () => {
      if (handle) {
        void handle.release();
      }
    };
  }, [state.step]);

  const currentIdx = ASSESSMENT_STEPS.indexOf(state.step);

  const renderCurrentStep = () => {
    switch (state.step) {
      case "setup":
        return <SetupStep participant={state.participant} onStart={start} />;
      case "voice":
        return (
          <VoiceStep
            voiceTasks={state.voiceTasks}
            onCompleteTask={completeVoiceTask}
            onFinish={finishVoice}
            onSkip={skipVoice}
            onBack={back}
          />
        );
      case "vitals":
        return (
          <VitalsStep
            vitals={state.vitals}
            onComplete={completeVitals}
            onSkip={skipVitals}
            onBack={back}
          />
        );
      case "eye":
        return (
          <EyeStep
            eye={state.eye}
            onComplete={completeEye}
            onSkip={skipEye}
            onBack={back}
          />
        );
      case "summary":
        return (
          <SummaryStep
            session={session}
            decision={state.decision}
            decisionState={state.decisionState}
            decisionError={state.decisionError}
            summarySource={state.summarySource}
            saveState={state.saveState}
            error={state.error}
            onRetrySave={finalize}
            onReset={reset}
          />
        );
      default: {
        const unreachable: never = state.step;
        return unreachable;
      }
    }
  };

  return (
    <section className="screen">
      <h1>Guided Wellness Check-in</h1>
      <p className="lede">
        A fast, non-invasive check-in measuring voice acoustics, facial vitals (rPPG), and ocular saccades.
      </p>

      <div className="stepper" role="tablist" aria-label="Assessment steps">
        {ASSESSMENT_STEPS.map((step, idx) => {
          const isActive = idx === currentIdx;
          const isCompleted = idx < currentIdx;
          return (
            <div
              key={step}
              className={`stepper__step ${isActive ? "active" : ""} ${isCompleted ? "completed" : ""}`}
              role="tab"
              aria-selected={isActive}
            >
              <span className="stepper__step-index">
                {isCompleted ? "✓" : idx + 1}
              </span>
              <span>{STEP_LABELS[step]}</span>
            </div>
          );
        })}
      </div>

      {state.error && state.step !== "summary" && (
        <div className="panel" style={{ background: "var(--band-limited-soft)", border: "1px solid var(--band-limited)", marginBottom: "var(--space-4)" }}>
          <p className="error" style={{ margin: 0 }}>{state.error}</p>
        </div>
      )}

      {renderCurrentStep()}
    </section>
  );
}

