"""Estimate a speaker's age from a recording (MLX on Apple Silicon).

Usage: python ml/predict.py clip.wav [more.m4a ...]
Needs ml/age_model.joblib from ml/train.py. Any format PyAV can read; first 10 s used.
"""
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).parent))
from embed import decode  # noqa: E402
from wavlm_mlx import WavLM  # noqa: E402

MAE_YEARS = 7.6  # VoxCeleb held-out error; older voices run ~10 y low


def main(paths):
    reg = joblib.load(Path(__file__).parent / "age_model.joblib")
    model = WavLM()
    for p in paths:
        x = model.embed(decode(p))[None].astype("float32")
        print(f"{p}: ~{reg.predict(x)[0]:.0f} years (±{MAE_YEARS:.0f})")


if __name__ == "__main__":
    main(sys.argv[1:])
