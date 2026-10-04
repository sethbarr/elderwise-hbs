# Age models (research prototypes)

Estimates a speaker's age from a short recording. Offline Python only — not yet wired into the app. **Not usable on app recordings yet:** it does not
transfer to read speech on consumer mics (see Common Voice below).

## Pipeline

1. First 10 s of audio → 16 kHz mono
2. `microsoft/wavlm-base-plus` (frozen, run in MLX via `ml/wavlm_mlx.py`) → mean + std pool of all 13 hidden layers (19,968 dims).
   Matches the PyTorch/transformers embeddings (cosine ≥ 0.9999998 on all 2,729 clips)
3. StandardScaler → PCA(256) → SVR(C=30), samples weighted so each age decade counts equally

## Data (not committed — see `.gitignore`)

| Set | Use | Clips | Labels |
| --- | --- | --- | --- |
| [`Exgc/voxceleb_age`](https://huggingface.co/datasets/Exgc/voxceleb_age) audio + [voxceleb_enrichment_age_gender](https://github.com/hechmik/voxceleb_enrichment_age_gender) `age-{train,test}.txt` | train / test | 2,729 (1 per speaker; 1,630 / 1,099) | exact age (upload year − birth year) |
| [`saeedzou/common-voice-17-en-age-gender-sampled`](https://huggingface.co/datasets/saeedzou/common-voice-17-en-age-gender-sampled) test + validation | out-of-domain check | 4,917 | decade band, gender |

Expected layout: `data/voxceleb-age/{age-train.txt,age-test.txt,voxceleb_age/<id>/<video>/*.m4a}`, `data/cv17-age/*.parquet`,
`data/fairface/*.parquet` ([`HuggingFaceM4/FairFace`](https://huggingface.co/datasets/HuggingFaceM4/FairFace) `1.25/`).

## Results (VoxCeleb held-out, n = 1,099)

MAE 7.6 y, r 0.76. Error by decade: 20s 7.7, 30s 6.9, 40s 6.0, 50s 6.8, 60s 9.9, 70s 12.9.
Predictions regress toward ~40: speakers 60+ are underestimated by ~9–12 y. Ridge, SVR, PCA size,
decade balancing and linear calibration all land at ~7.6 MAE, so the limit is data, not the regressor.

`ml/age_model.joblib` is refit on train + test after reporting, so never score it on the VoxCeleb test
split (it has seen those clips). Fit on `split == "train"` for any evaluation.

## Out-of-domain: Common Voice 17 (read speech, consumer mics)

4,917 clips, 2,749 speakers, decade bands only. Mean predicted age per band:

| Band | n | Midpoint | VoxCeleb-trained (shipped) | CV-trained (5-fold, grouped by speaker) |
| --- | --- | --- | --- | --- |
| teens | 700 | 15 | 48.7 | 27.1 |
| 20s | 2,374 | 25 | 49.2 | 29.5 |
| 30s | 983 | 35 | 49.9 | 34.6 |
| 40s | 442 | 45 | 51.7 | 38.4 |
| 50s | 239 | 55 | 52.5 | 42.5 |
| 60s | 108 | 65 | 55.2 | 50.0 |
| 70s | 59 | 75 | 54.1 | 49.6 |
| 80s | 11 | 85 | 57.9 | 41.9 |
| MAE vs midpoint | | | 20.9 y (r 0.27) | 8.6 y (r 0.54) |

- The VoxCeleb model predicts ~50 for everyone: it learned the recording conditions, not just the voice
- WavLM features do carry age in read speech (CV-trained ordering is right up to the 60s), but 60+ are
  still pulled to ~50. Only 39 CV speakers are 70+, so the limit is again elderly training data
- No model here can tell 65 from 80, which is the range Elderwise cares about. **Next step is data:**
  read speech from older adults with true ages (e.g. consented app users), not a different regressor

## Pretrained alternative considered (not used)

`audeering/wav2vec2-large-robust-24-ft-age-gender` was benchmarked on the same clips: VoxCeleb test
MAE 6.9 y (70s: 8.1 vs our 12.9) and on CV the best 60+ ordering (60s 58.9, 70s 63.3). Rejected:

- **License CC-BY-NC-SA 4.0** — no commercial use
- It was trained on VoxCeleb2 and Common Voice, so these scores are optimistic (likely speaker overlap)
- It still underestimates 70+ by ~10 y, so it does not remove the need for elderly data

## How much speech to record

Training clips are 3.9–78 s (median 5.6 s; 15% reach the 10 s cap). Test clips that are at least
10 s long (n = 163), cut to each length, train-only model:

| Speech used | 2 s | 3 s | 5 s | 7 s | 10 s |
| --- | --- | --- | --- | --- | --- |
| MAE (y) | 9.2 | 8.0 | 7.5 | 7.3 | 7.1 |

- Need at least 5 s of speech; below 3 s, error is 1–2 y worse
- Aim for 10 s of continuous speech. Only the first 10 s are used, so trim leading silence first
- The app's reading passage (40 s) and free speech (30 s) tasks are long enough; the sustained vowel is
  not connected speech and is not what the model was trained on
- Averaging several 10 s windows from a longer recording is untested

## Face geometry (research only — not for the app)

Tested whether frontal landmark ratios from the MediaPipe Face Landmarker the app already ships
(`public/models/face_landmarker.task`, Apache 2.0) can estimate age. Data: FairFace 1.25-padding crops
(CC BY 4.0; ages are **annotator-perceived** decade bands, not true ages). FaceAge (no licence; trained on
IMDB-WIKI, academic-only) and IMDB-WIKI itself were ruled out on licence.

`ml/face_features.py` levels head roll and computes 7 ratios (indices port 1:1 to `src/lib/faceLandmarker.ts`)
plus all 478 normalised landmarks; `ml/face_eval.py` fits on FairFace train and scores validation.
Strict = |yaw|, |pitch| < 15° and neutral expression (4,679 faces; 34 test faces 70+).

Feature means by band (strict, 20s → 70+), direction expected from the literature:

| Feature | 20s | 40s | 60s | 70+ | Spearman | Expected | Holds? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Philtrum / face height | 0.089 | 0.096 | 0.100 | 0.100 | +0.30 | up | yes |
| Upper vermilion / mouth width | 0.132 | 0.120 | 0.108 | 0.101 | −0.33 | down | yes |
| Lower vermilion / mouth width | 0.205 | 0.186 | 0.167 | 0.155 | −0.38 | down | yes |
| Palpebral aperture | 0.311 | 0.286 | 0.285 | 0.293 | −0.13 | down | to 50s, then rises |
| Canthal tilt (°) | 3.76 | 2.43 | 2.49 | 2.86 | −0.15 | down | to 50s, then rises |
| Lower-face height | 0.412 | 0.418 | 0.414 | 0.406 | +0.01 | up | no |
| Mouth-corner drop | 0.000 | 0.001 | 0.000 | −0.002 | −0.01 | up | no |

Age prediction (validation, Spearman vs band; mean prediction for 60s / 70+):

| Model | Strict | Loose (< 25°, any expression) | 60s / 70+ (strict) |
| --- | --- | --- | --- |
| 7 table ratios → gradient boosting | 0.37 | 0.38 | 54 / 56 |
| All 478 landmarks → SVR (geometry ceiling) | 0.46 | 0.48 | 59 / 64 |
| Voice, CV-trained (for comparison) | — | 0.54 | 50 / 50 |

**Verdict: not good enough to ship.** Weaker than voice, and like voice it compresses 60+ so it cannot
tell 65 from 80. Real webcam frames and true ages would likely score lower than posed photos with
perceived ages. A useful face age needs age-labelled older faces tied to health outcomes (FaceAge's
real contribution), not better geometry. No dataset here has face and voice from the same person, so
a face + voice fusion cannot be evaluated yet.

## Usage

```sh
# Voice (MLX)
python3.12 -m venv .venv-ml && .venv-ml/bin/pip install -r ml/requirements.txt
.venv-ml/bin/python ml/embed.py voxceleb      # ~5 min, ~1 GB peak on Apple Silicon → data/emb-voxceleb.npz
.venv-ml/bin/python ml/embed.py cv17          # optional out-of-domain check
.venv-ml/bin/python ml/train.py               # prints metrics, writes ml/age_model.joblib
.venv-ml/bin/python ml/predict.py clip.wav    # → "~62 years (±8)"

# Face geometry: separate env, mediapipe 0.10.21 needs numpy < 2 (1.0.1 aborts on macOS: Metal service unavailable)
python3.12 -m venv .venv-face && .venv-face/bin/python -m pip install -r ml/requirements-face.txt
.venv-face/bin/python ml/face_features.py     # ~8 min CPU → data/face-fairface.npz (needs data/fairface/*.parquet)
.venv-face/bin/python ml/face_eval.py strict  # or loose
```

Wellness estimate only; not a medical device.
