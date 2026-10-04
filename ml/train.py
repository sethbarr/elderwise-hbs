"""Fit an age regressor on VoxCeleb WavLM embeddings; report MAE and CV17 sanity check.

Usage: python ml/train.py   (after ml/embed.py voxceleb [and cv17])
Writes ml/age_model.joblib.
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BANDS = ["teens", "twenties", "thirties", "fourties", "fifties", "sixties", "seventies", "eighties", "nineties"]


def report(name, y, p):
    print(f"{name}: MAE {np.mean(np.abs(y - p)):.1f} y, r {np.corrcoef(y, p)[0, 1]:.2f}, n {len(y)}")
    for lo in range(10, 90, 10):
        m = (y >= lo) & (y < lo + 10)
        if m.sum():
            print(f"  {lo}s  n={m.sum():4d}  true {y[m].mean():5.1f}  pred {p[m].mean():5.1f}  MAE {np.abs(y[m] - p[m]).mean():5.1f}")


def main():
    d = np.load(DATA / "emb-voxceleb.npz")
    X, y, split = d["X"].astype(np.float32), d["age"].astype(float), d["split"]
    tr, te = split == "train", split == "test"
    # Weight each decade equally so the rare 60+ speakers are not drowned out by 20-40s.
    dec = np.clip(y // 10, 1, 8).astype(int)
    w = 1 / np.bincount(dec)[dec]
    w = w / w.mean()
    model = make_pipeline(StandardScaler(), PCA(256, random_state=0), SVR(C=30))
    model.fit(X[tr], y[tr], svr__sample_weight=w[tr])
    report("VoxCeleb test", y[te], model.predict(X[te]))

    model.fit(X, y, svr__sample_weight=w)  # refit on everything for the shipped model
    joblib.dump(model, ROOT / "ml/age_model.joblib")

    cv = DATA / "emb-cv17.npz"
    if cv.exists():
        c = np.load(cv)
        p = model.predict(c["X"].astype(np.float32))
        print("CV17 (out of domain), mean predicted age per band:")
        for b in BANDS:
            m = c["band"] == b
            if m.sum():
                print(f"  {b:10s} n={m.sum():4d}  pred {p[m].mean():5.1f} ± {p[m].std():4.1f}")


if __name__ == "__main__":
    main()
