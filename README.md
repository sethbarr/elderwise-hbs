# Elderwise

Local-first macOS desktop app that runs a short, guided check-in for a single
older adult and reports aging-related biomarkers from three signals:

- **Voice** — sustained vowel, reading passage and free speech, analysed on
  device (F0 mean/SD, jitter, shimmer, HNR, speech and articulation rate,
  pause ratio, voiced ratio).
- **Vitals** — webcam remote photoplethysmography (heart rate, HRV RMSSD/SDNN,
  respiratory rate, signal-to-noise), analysed on device.
- **Eye movement** — fixation, prosaccade and smooth-pursuit tasks tracked with
  on-device face/iris landmarks (fixation stability, saccade count/latency/peak
  velocity/accuracy, pursuit gain, blink rate).

Results are banded (Steady / Watch / Follow up), saved to a local JSONL history,
trended over time, and exportable as a PDF.

> Elderwise provides wellness estimates only. It is not a medical device and
> does not diagnose any condition.

## Cloud boundary

Only two kinds of data leave the device, and only when you configure a key:

| Data | Destination | Purpose |
| --- | --- | --- |
| Short WAV clips from the voice tasks | ElevenLabs Scribe (default) or OpenAI `gpt-4o-mini-transcribe` | Transcription |
| Instruction text | ElevenLabs TTS (default) or macOS system voice (no key needed) | Reading instructions aloud |
| De-identified results JSON for one session | OpenAI (default `gpt-5.5`) | Plain-English summary paragraph |

Everything else — voice acoustics, rPPG, gaze metrics, banding, history, PDF —
runs inside the app. With no keys configured the check-in still works:
transcripts are empty, instructions use the system voice, and the summary is
generated on device.

API keys are stored per-user in the macOS keychain (service
`com.elderwise.app`) via Settings, or read from `OPENAI_API_KEY` /
`ELEVENLABS_API_KEY` environment variables. Keys never cross the IPC boundary
to the webview.

## Getting started

Requirements: macOS 10.15+, [Bun](https://bun.sh) 1.4.x, Rust stable, Xcode
command line tools.

```sh
bun install
bun run models:fetch      # downloads MediaPipe face_landmarker.task + WASM into public/ (git-ignored)
bun run tauri dev         # desktop app
```

Other scripts:

```sh
bun run dev               # webview only (camera/mic work, Tauri IPC does not)
bun run test              # vitest
bunx tsc --noEmit         # typecheck
bun run build             # vite production bundle
cd src-tauri && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test
```

Grant microphone and camera access when macOS prompts. The usage strings live
in `src-tauri/Info.plist`.

### Local `.app` / DMG build

`tauri.conf.json` pins the release signing identity used by CI
(`Developer ID Application: AJENTIK AI PTE. LTD.`). On a machine without that
certificate, override it with ad-hoc signing:

```sh
PATH="/usr/bin:$PATH" APPLE_SIGNING_IDENTITY=- bun run tauri build
open src-tauri/target/release/bundle/macos/elderwise.app
```

The `PATH` prefix matters if an asdf/pyenv Python `xattr` shim shadows Apple's
`/usr/bin/xattr`; the bundler calls `xattr -cr` and fails with
`failed to run xattr` otherwise.

Ad-hoc signed builds change identity on every rebuild, so macOS may re-prompt
for Keychain access to saved API keys. For development, exporting
`OPENAI_API_KEY` / `ELEVENLABS_API_KEY` in the shell that runs
`bun run tauri dev` avoids the Keychain entirely.

## Architecture

```
src/
  modules/assessment/   shared Zod contract (types.ts) + guided flow setup → voice → vitals → eye → summary
  modules/voice/        WAV capture, aging biomarkers (pitch.ts, rhythm.ts, biomarkers.ts), task specs, TTS/ASR glue
  modules/vitals/       rPPG pipeline (POS, bandpass, FFT HR, IBI → HRV, respiratory band), face ROI, capture hook + panel
  modules/eye/          gaze proxy from iris landmarks, I-VT saccades, task schedules, capture hook + canvas
  modules/ai/           settings/key management UI and typed `invoke` wrappers
  modules/history/      JSONL history client, trends, History screen
  modules/export/       jsPDF report model + document builder, native save dialog
  lib/faceLandmarker.ts shared MediaPipe Face Landmarker loader (singleton, GPU/VIDEO mode)
  ui/                   design tokens, band colours, ScoreHero, BandScale, icons
src-tauri/src/
  ai/                   OpenAI chat (Responses API, streaming), ElevenLabs TTS + STT, keychain store, settings file
  history/              append-only JSONL under ~/Library/Application Support/com.elderwise.app/
  export/               PDF write + dialog mode
```

Design rules carried over: Zod validation at every IPC boundary, keys only in
Rust, local-first storage, large touch targets and ≥16px text for older users.

## What was reused from pulsewise

Elderwise was scaffolded from [pulsewise](https://github.com/ajentik/pulsewise)
(HealthWise v0.10.0, a cardiovascular pre-screening app). Reuse map:

| Area | Decision |
| --- | --- |
| Tauri v2 + React 19 + Vite + Bun + vitest + Zod + jsPDF shell | **Kept** as-is; lets the Apple signing / notarization secrets and release pipeline work unchanged |
| `src-tauri/src/ai/` (keychain store, settings file, error codes, OpenAI Responses streaming transport, ElevenLabs TTS) | **Kept**, pruned to two providers (OpenAI, ElevenLabs); added ElevenLabs Scribe STT; prompts rewritten for an aging-biomarker narrative |
| `src-tauri/src/history/`, `src-tauri/src/export/` | **Kept** unchanged (JSONL store, PDF write) |
| `src/modules/voice/recordWav.ts`, `speech.ts` | **Kept**; `biomarkers.ts` extended from capture-quality metrics to aging markers |
| `src/modules/export/PdfDoc.ts`, `pdfTokens.ts`, `api.ts` | **Kept**; report model rewritten for the new session shape |
| `src/ui/` tokens, band colours, ScoreHero, BandScale, error boundary, icons | **Kept**, relabelled |
| `src/modules/history/` client + trends | **Kept**, schema swapped to `AssessmentSession` |
| CI: `ci.yml`, `fast-feedback.yml`, `release.yml`, `signing-preflight.yml`, `deploy-site.yml` | **Kept**; renamed, `site:check` dropped, Cloudflare Worker renamed `elderwise` |
| Repo secrets (Apple cert/notary, Cloudflare) | **Reused** unchanged |
| MERaLiON / exo / local whisper / kokoro providers | **Dropped** (cloud is OpenAI + ElevenLabs only) |
| Polar BLE, AccessLink, ECG algorithm, chair-stand, DASI, CVD question bank, LLM interviewer | **Dropped** (not relevant to aging voice/vitals/eye) |
| i18n (zh-CN/ms/ta), Remotion demo videos, `site-src` generator, iPadOS workflow | **Dropped** for simplicity (English, macOS only, minimal static site) |
| New: `src/modules/vitals/`, `src/modules/eye/`, `src/lib/faceLandmarker.ts`, `scripts/fetch-models.sh` | **Added** — all run locally via MediaPipe tasks-vision WASM |

## Releasing

Push a tag matching the version in `package.json`, `src-tauri/tauri.conf.json`
and `src-tauri/Cargo.toml` (e.g. `git tag v0.1.0 && git push origin v0.1.0`).
`release.yml` builds signed, notarized Apple Silicon and Intel DMGs when the
Apple secrets are present (see `docs/signing-setup.md`), and copies them into
`site/downloads/` for the Cloudflare Worker landing page.

## Known limitations

- rPPG and gaze metrics have been validated against synthetic signals, not yet
  against reference devices or a real cohort. Treat numbers as relative trends.
- Smooth-pursuit gain uses percentile auto-scaling and currently over-reads
  (~1.5 for ideal pursuit).
- Face/iris tracking quality depends on lighting and camera; the UI reports
  coverage and quality per module.
- One local user; no accounts or sync.

### Local Clef assessment review

The guided check-in can run an experimental human-review decision locally through a Tauri-managed Clef-Flash Q4 llama.cpp sidecar. Model acquisition needs a first-run Hugging Face download; subsequent inference uses the local cache. History, PDF and fallback summaries remain available if the engine fails. See [setup, provisioning, policy and verification](docs/clef-assessment.md).
