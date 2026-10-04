import { useState } from "react";
import { toMessage } from "../../../lib/errors";
import { bandMeta } from "../../../ui/bandColor";
import { BandScale } from "../../../ui/BandScale";
import { ScoreHero } from "../../../ui/ScoreHero";
import { savePdf } from "../../export/api";
import { buildAssessmentPdf } from "../../export/assessmentPdf";
import type { AssessmentDecision } from "../api";
import type { AssessmentSession } from "../types";

export interface SummaryStepProps {
  readonly decision?: AssessmentDecision | null;
  readonly decisionState?: "idle" | "pending" | "ready" | "unavailable";
  readonly decisionError?: string | null;
  readonly session: AssessmentSession | null;
  readonly summarySource: "llm" | "local" | null;
  readonly saveState: "idle" | "saving" | "saved" | "error";
  readonly error: string | null;
  readonly onRetrySave: () => void;
  readonly onReset: () => void;
}

const REVIEW_LABELS: Record<AssessmentDecision["route"]["selected"], string> = {
  routine: "routine wellness follow-up",
  caregiver_review: "review with a trusted person",
  clinical_review: "review with a qualified professional",
  urgent_review: "prioritized human review",
};

const SCALE_SEGMENTS = [
  { tone: "good" as const, label: "Steady", range: "Optimal range" },
  { tone: "moderate" as const, label: "Watch", range: "Mild variance" },
  { tone: "limited" as const, label: "Follow up", range: "Outside normal bounds" },
];

export function SummaryStep({
  session,
  decision,
  decisionState,
  decisionError,
  summarySource,
  saveState,
  error,
  onRetrySave,
  onReset,
}: SummaryStepProps) {
  const [exporting, setExporting] = useState(false);
  const [exportNotice, setExportNotice] = useState<{ text: string; isError?: boolean } | null>(null);

  if (!session) {
    return (
      <div className="panel" style={{ textAlign: "center", padding: "var(--space-8) var(--space-4)" }}>
        <h2>Compiling your check-in...</h2>
        <p className="hint">Analyzing measurements and building your wellness summary.</p>
      </div>
    );
  }

  const overall = bandMeta(session.overallBand);

  const handleExport = async () => {
    setExporting(true);
    setExportNotice(null);
    try {
      const bytes = await buildAssessmentPdf(session);
      const fileName = `Elderwise_Report_${session.startedAt.slice(0, 10)}.pdf`;
      const saved = await savePdf(fileName, bytes);
      setExportNotice({ text: saved ? "Report saved successfully." : "Export cancelled." });
    } catch (err) {
      setExportNotice({ text: `Export failed: ${toMessage(err)}`, isError: true });
    } finally {
      setExporting(false);
    }
  };

  const prosaccade = session.eye?.tasks.find((t) => t.task === "prosaccade");
  const fixation = session.eye?.tasks.find((t) => t.task === "fixation");

  return (
    <div className="summary-step">
      <div className="panel">
        <ScoreHero
          value={overall.label}
          caption="Overall wellness signal based on available sensor checks"
          tone={session.overallBand}
          bandLabel={overall.label}
          bandLabelSuffix="signal"
        />
        <BandScale tone={session.overallBand} segments={SCALE_SEGMENTS} />
      </div>

      {decisionState && decisionState !== "idle" && (
        <div className="panel" role="status" aria-live="polite">
          <h2>Experimental local review</h2>
          {decisionState === "pending" && <p className="hint">Preparing your local review. First use downloads the model and may take several minutes. Your summary and report remain available.</p>}
          {decisionState === "unavailable" && <>
            <p className="hint">Local review unavailable. Your check-in, history and PDF report remain available.</p>
            <details><summary>Review details</summary><p>{decisionError}</p><p>Check the model setup and restart Elderwise to retry.</p></details>
          </>}
          {decisionState === "ready" && decision && <>
            <p>Human-review category: {REVIEW_LABELS[decision.route.selected]}.</p>
            <p className="hint">These experimental model scores are not a diagnosis, medical risk estimate or recommendation for when to seek care. A person should review the measurements and their quality in context.</p>
          </>}
        </div>
      )}

      <div className="summary-modules-grid">
        <div className="panel summary-card">
          <div className="summary-card__header">
            <h3>Voice Acoustics</h3>
            <ModuleBadge band={session.voice?.band} />
          </div>
          {session.voice ? (
            <div className="summary-metrics-list">
              <MetricRow label="Pitch (F0)" val={session.voice.markers.f0MeanHz ? `${Math.round(session.voice.markers.f0MeanHz)} Hz` : "--"} />
              <MetricRow label="Jitter" val={session.voice.markers.jitterPct ? `${session.voice.markers.jitterPct.toFixed(2)}%` : "--"} />
              <MetricRow label="Shimmer" val={session.voice.markers.shimmerPct ? `${session.voice.markers.shimmerPct.toFixed(2)}%` : "--"} />
              <MetricRow label="Quality" val={session.voice.quality} />
            </div>
          ) : (
            <p className="hint">No voice recordings captured for this session.</p>
          )}
        </div>

        <div className="panel summary-card">
          <div className="summary-card__header">
            <h3>Facial Vitals</h3>
            <ModuleBadge band={session.vitals?.band} />
          </div>
          {session.vitals ? (
            <div className="summary-metrics-list">
              <MetricRow label="Heart rate" val={session.vitals.heartRateBpm ? `${Math.round(session.vitals.heartRateBpm)} bpm` : "--"} />
              <MetricRow label="Breathing rate" val={session.vitals.respiratoryRateBpm ? `${Math.round(session.vitals.respiratoryRateBpm)} breaths/min` : "--"} />
              <MetricRow label="HRV (RMSSD)" val={session.vitals.hrvRmssdMs ? `${Math.round(session.vitals.hrvRmssdMs)} ms` : "--"} />
              <MetricRow label="Quality" val={session.vitals.quality} />
            </div>
          ) : (
            <p className="hint">Facial vitals check was skipped for this session.</p>
          )}
        </div>

        <div className="panel summary-card">
          <div className="summary-card__header">
            <h3>Eye Movement</h3>
            <ModuleBadge band={session.eye?.band} />
          </div>
          {session.eye ? (
            <div className="summary-metrics-list">
              <MetricRow label="Saccade latency" val={prosaccade?.meanSaccadeLatencyMs ? `${Math.round(prosaccade.meanSaccadeLatencyMs)} ms` : "--"} />
              <MetricRow label="Fixation stability" val={fixation?.fixationStability ? fixation.fixationStability.toFixed(3) : "--"} />
              <MetricRow label="Tracking accuracy" val={prosaccade?.saccadeAccuracy ? `${Math.round(prosaccade.saccadeAccuracy * 100)}%` : "--"} />
              <MetricRow label="Quality" val={session.eye.quality} />
            </div>
          ) : (
            <p className="hint">Eye tracking tasks were skipped for this session.</p>
          )}
        </div>
      </div>

      <div className="panel">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "var(--space-2)" }}>
          <h2>Session Narrative</h2>
          <span className="badge ok">{summarySource === "llm" ? "AI summary" : "On-device summary"}</span>
        </div>
        <p style={{ fontSize: "var(--text-md)", lineHeight: 1.6, color: "var(--text)" }}>
          {session.summaryText ?? "Summary narrative is being generated..."}
        </p>

        <div className="summary-save-bar">
          {saveState === "saved" && <p className="hint" style={{ color: "var(--band-good)", margin: 0 }}>Session saved to history.</p>}
          {saveState === "saving" && <p className="hint" style={{ margin: 0 }}>Saving to history...</p>}
          {saveState === "error" && (
            <div style={{ display: "flex", alignItems: "center", gap: "var(--space-3)" }}>
              <p className="error" style={{ margin: 0 }}>Failed to save: {error ?? "Unknown error"}</p>
              <button type="button" className="secondary" onClick={onRetrySave} style={{ minHeight: "36px", padding: "4px 12px" }}>Retry save</button>
            </div>
          )}
        </div>

        {exportNotice && (
          <div className="panel" style={{ padding: "var(--space-3) var(--space-4)", marginTop: "var(--space-3)" }}>
            <p className={exportNotice.isError ? "error" : "hint"} style={{ margin: 0 }}>{exportNotice.text}</p>
          </div>
        )}

        <div className="answer-row" style={{ marginTop: "var(--space-5)" }}>
          <button type="button" className="primary" onClick={() => void handleExport()} disabled={exporting}>
            {exporting ? "Building report..." : "Export PDF"}
          </button>
          <button type="button" className="secondary" onClick={onReset}>Start another check-in</button>
        </div>

        <p className="hint" style={{ marginTop: "var(--space-4)", fontSize: "var(--text-xs)" }}>
          Elderwise provides wellness estimates only. It is not a medical device and does not diagnose any condition.
        </p>
      </div>
    </div>
  );
}

function MetricRow({ label, val }: { readonly label: string; readonly val: string }) {
  return (
    <div className="metric-row">
      <span>{label}</span>
      <strong>{val}</strong>
    </div>
  );
}

function ModuleBadge({ band }: { readonly band?: "good" | "moderate" | "limited" }) {
  if (!band) return <span className="badge neutral">Skipped</span>;
  const meta = bandMeta(band);
  return (
    <span className="badge" style={{ background: `var(${meta.softVarName})`, color: `var(${meta.varName})` }}>
      {meta.label}
    </span>
  );
}
