"""Estimate a speaker's age from a recording.

Usage: python ml/predict.py clip.wav [more.m4a ...]
Needs ml/age_model.joblib from ml/train.py. Any format ffmpeg/PyAV can read; first 10 s used.
"""
import sys
from pathlib import Path

import joblib
import torch
from transformers import AutoFeatureExtractor, WavLMModel

sys.path.insert(0, str(Path(__file__).parent))
from embed import MODEL, SR, decode  # noqa: E402

MAE_YEARS = 7.6  # VoxCeleb held-out error; older voices run ~10 y low


def main(paths):
    reg = joblib.load(Path(__file__).parent / "age_model.joblib")
    fe = AutoFeatureExtractor.from_pretrained(MODEL)
    model = WavLMModel.from_pretrained(MODEL).eval()
    for p in paths:
        wav = decode(p)
        inp = fe(wav, sampling_rate=SR, return_tensors="pt").input_values
        with torch.no_grad():
            h = torch.stack(model(inp, output_hidden_states=True).hidden_states)[:, 0]
        x = torch.cat([h.mean(1), h.std(1)], -1).flatten().numpy()[None].astype("float16").astype("float32")
        print(f"{p}: ~{reg.predict(x)[0]:.0f} years (±{MAE_YEARS:.0f})")


if __name__ == "__main__":
    main(sys.argv[1:])
