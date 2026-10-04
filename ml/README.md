# Voice age model (research prototype)

Estimates a speaker's age from a short recording. Offline Python only — not yet wired into the app.

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

Expected layout: `data/voxceleb-age/{age-train.txt,age-test.txt,voxceleb_age/<id>/<video>/*.m4a}`, `data/cv17-age/*.parquet`.

## Results (VoxCeleb held-out, n = 1,099)

MAE 7.6 y, r 0.76. Error by decade: 20s 7.7, 30s 6.9, 40s 6.0, 50s 6.8, 60s 9.9, 70s 12.9.
Predictions regress toward ~40: speakers 60+ are underestimated by ~9–12 y. Ridge, SVR, PCA size,
decade balancing and linear calibration all land at ~7.6 MAE, so the limit is data, not the regressor.

`ml/age_model.joblib` is refit on train + test after reporting, so never score it on the VoxCeleb test
split (it has seen those clips). Fit on `split == "train"` for any evaluation.

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

## Usage

```sh
python3.12 -m venv .venv-ml && .venv-ml/bin/pip install -r ml/requirements.txt
.venv-ml/bin/python ml/embed.py voxceleb      # ~5 min, ~1 GB peak on Apple Silicon → data/emb-voxceleb.npz
.venv-ml/bin/python ml/embed.py cv17          # optional out-of-domain check
.venv-ml/bin/python ml/train.py               # prints metrics, writes ml/age_model.joblib
.venv-ml/bin/python ml/predict.py clip.wav    # → "~62 years (±8)"
```

Wellness estimate only; not a medical device.
