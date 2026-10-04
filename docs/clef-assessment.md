# Local assessment decision engine

The guided check-in's existing Zod `AssessmentSession` feeds `assess_session`.
Rust projects only age/sex, voice aggregate markers/quality/band, five vitals
metrics/quality/band, compact eye task metrics/tracking coverage/quality/band,
and overallBand. Null/skipped modules remain unknown. Raw media, transcripts,
identity/timestamps and summaryText never reach the model. Fixed Rust-owned
choice/noul/score questions are sent together to `/v1/systemone`.

Rust validates answer types, bounded values, all distribution keys and sums,
winning route and urgency legend; it normalizes small distribution rounding
errors. Urgency mode is the most probable legend index, not rounded expected
score. The Zod IPC result retains model, distributions, confidences, safety
score, urgency score/mode/index and inputTokens. `assessment_engine_status`
returns idle/starting/ready/unavailable/stopped with a typed error.

## Ownership and policy boundary

Tauri starts the engine in a blocking worker during setup, so the window does
not wait for model acquisition. The single-instance plugin focuses the existing
window rather than starting another engine. A mutex serializes startup and
requests. The native engine owns one child, binds an ephemeral IPv4 loopback
port, disables HTTP proxy/redirects, and authenticates requests using an
in-memory random API key supplied via the child's environment. Health and an
authenticated properties request verify readiness. The server runs with Q4,
99 GPU layers, 2048 context/batch/ubatch, flash attention and one slot. Startup
allows 20 minutes for first-run acquisition; requests allow 60 seconds. Failed
startup/response becomes a typed error. No frontend shell permission is added.

On app exit, shutdown cancels startup, sends SIGTERM on Unix, waits up to two
seconds, then kills/reaps if needed. RAII also cleans up on state destruction.
A forcibly killed/crashed parent cannot run normal cleanup; orphan recovery
is not implemented. Reopening uses a fresh port/key and never attaches to an
arbitrary manually running server. Child output is discarded to avoid logging
private prompts; errors distinguish missing binary, early exit, startup
timeout, transport and malformed answer. Early-exit details point to download,
disk/RAM/VRAM and platform compatibility rather than exposing raw server logs.

No approved clinical rule set exists here. There are no invented medical
thresholds or automated actions. Categories are experimental human-review
routing only; safety/confidence/probability values are **uncalibrated model
preferences**, not medical risk. The score legend (Routine/Soon/Same day/
Immediate) ranks model review priority; it is not an instruction about when to
seek care. The summary shows only the human-review category and explanatory
copy, not risk percentages or diagnostic advice. No emergency dispatch exists.

Inference starts alongside summary finalization and never blocks saving. Results
and failures are isolated from summary/history errors and ignored after reset
or unmount. The existing deterministic fallback, history append and PDF export
continue independently. History retry reuses the completed session identity.
The decision is held in the current assessment state and is **not persisted or
included in PDF/history**; the canonical session schema is unchanged. Cloud
chat/voice integrations remain independent of this engine.

## Development

For the already built WSL CUDA server:

```bash
export ELDERWISE_LLAMA_SERVER="$PWD/.clef-local/llama.cpp/build/bin/llama-server"
bun run tauri dev
```

The native override is accepted only in debug builds. The app launches the
server itself; do not manually launch another server. Ordinary browser previews
have no Tauri IPC and show local-review unavailable while retaining the summary.

## Model acquisition and cache

`-hf ggml-org/Clef-Flash-GGUF:Q4_K_M` downloads about 6.5 GB on first launch.
This requires internet access to Hugging Face, enough disk space and adequate
RAM/VRAM. llama.cpp handles resumable acquisition and reuses its local HF cache;
inference uses the local file after acquisition. At the pinned revision the
cache precedence is LLAMA_CACHE, HF_HUB_CACHE, HUGGINGFACE_HUB_CACHE, HF_HOME/hub,
XDG_CACHE_HOME/huggingface/hub, then ~/.cache/huggingface/hub (also on macOS).
Native deployment can set LLAMA_CACHE for an alternate writable cache. The app
never sends biomarker state to an external inference service. A first launch
offline without the cache fails gracefully; offline use requires prior model
acquisition. The model repository is not revision pinned, so a future model
update may require cache/protocol revalidation. No model, cache or binary is
tracked in Git.

## Reproducible sidecar provisioning

`scripts/provision-clef.sh` fetches the tested llama.cpp revision
`46847e61582097979f539595d893d83d8e1d1af1`, builds `llama-server` and copies it to
`src-tauri/binaries/llama-server-<target-triple>`. It uses static llama/ggml/MTMD
and static OpenSSL libraries with HTTPS enabled for HF acquisition; Metal is
embedded on macOS. Generated trees/binaries live in ignored directories.
The model uses the tested System One API; do not substitute chat completions.

```bash
# On a native Apple Silicon or Intel Mac, respectively:
brew install cmake openssl@3 ripgrep
bash scripts/provision-clef.sh aarch64-apple-darwin
# bash scripts/provision-clef.sh x86_64-apple-darwin
bun run tauri build --target aarch64-apple-darwin --config src-tauri/tauri.clef.conf.json
```

Use a matching native host architecture (including Homebrew/OpenSSL); cross
building an Intel binary with ARM static OpenSSL is unsupported. The release
matrix now uses macos-15 and macos-15-intel separately and passes the overlay
that declares `bundle.externalBin`. The normal config deliberately omits the
sidecar so unit tests/frontend builds do not require generated executables.
A release must use the overlay; a build without it degrades to engine unavailable.
Tauri embeds and signs externalBin with the app. The script checks `otool -L`
for non-system dependencies before packaging. The existing certificate and
notarization workflow is retained. The JavaScript Tauri API/dialog/opener
dependencies are constrained to the locked Rust crates' minor versions, fixing
the pre-existing CLI version-mismatch gate without a broad native dependency
upgrade.

Linux native provisioning requires CMake, a C++ compiler and static OpenSSL
development libraries. CUDA builds can set `ELDERWISE_BUILD_CUDA=ON`; runtime
CUDA libraries must exist on that development machine. Linux binaries never
ship in a macOS DMG. Windows provisioning is deliberately unsupported for this
macOS-focused release pipeline.

## Verification

Unit tests require neither a GPU nor model download. An explicitly ignored
Rust integration test exercises the same process, request, transport and parser
used by the Tauri command:

```bash
ELDERWISE_LLAMA_SERVER="$PWD/.clef-local/llama.cpp/build/bin/llama-server" \
  cargo test --manifest-path src-tauri/Cargo.toml managed_clef_smoke -- --ignored --nocapture
```

It launches the child, verifies health/authentication, checks duplicate readiness
calls reuse the PID, submits a fixture validated by the canonical frontend
session schema, prints the normalized decision, shuts down and checks the port
is closed. Unit tests also cover missing binary, startup timeout, cancellation,
full/null input projection, real response parsing, malformed distributions and
legend, and IPC validation/failure isolation.

Validation performed in WSL: all requested install/typecheck/test/build/Rust
fmt/check/test/Clippy commands passed (245 frontend tests, 55 Rust unit tests,
one opt-in GPU test). The full-session managed CUDA smoke returned 1,010 input
tokens and a normalized routine route, safety score and urgency distribution.
The native Linux provisioning script also built successfully; `ldd` showed no
non-system llama/ggml/OpenSSL shared-library dependencies. A Tauri debug
application build with the sidecar overlay passed.

The actual app was launched on WSLg: its child reached `/health` = ok, a second
launch retained the original single app/server pair, and an ordinary window
close terminated both processes. GTK fell back to software rendering because
WSLg's GL context was unavailable. Full capture-to-summary GUI interaction and
real webview-to-command IPC were not automated; the API/flow helpers and the
command's Rust engine path were tested separately.

Still verify on both native macOS architectures: build/linkage,
embedded binary discovery, Metal load/inference, first-run HTTPS acquisition,
subsequent offline startup, single-instance behavior, window exit cleanup,
Developer ID signing/notarization/Gatekeeper and the actual minimum macOS
version supported by the pinned sidecar/static OpenSSL. The existing 10.15
bundle deployment setting is not evidence that those dependencies run there.

Implementation references: [Tauri external binaries](https://v2.tauri.app/develop/sidecar/),
[Tauri single-instance plugin](https://v2.tauri.app/plugin/single-instance/),
[GitHub native runner labels](https://docs.github.com/en/actions/reference/runners/github-hosted-runners),
[pinned llama.cpp server API](https://github.com/ggml-org/llama.cpp/blob/46847e61582097979f539595d893d83d8e1d1af1/tools/server/README.md).
