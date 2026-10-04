"""Frontal face-geometry age features from MediaPipe Face Landmarker (478 points).

Usage: python ml/face_features.py [per_band_cap]
Reads FairFace (CC BY 4.0, data/fairface/*.parquet), writes data/face-fairface.npz with
F (n, 7) named ratios, L (n, 478*2) roll-normalised landmarks, yaw, pitch, smile, jaw, band, split.
Every detected face is kept; filter on pose/expression at evaluation time.
Feature definitions are meant to port 1:1 to src/lib/faceLandmarker.ts.
"""
import glob, io, math, sys
from pathlib import Path

import mediapipe as mp
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODEL = ROOT / "public/models/face_landmarker.task"
BANDS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
ADULT = range(3, 9)  # 20-29 .. 70+

# Canonical MediaPipe mesh indices.
TOP, CHIN, SUBNASALE = 10, 152, 2
LIP_TOP, LIP_IN_UP, LIP_IN_LO, LIP_BOT = 0, 13, 14, 17
MOUTH_R, MOUTH_L = 61, 291
R_OUT, R_IN, R_UP, R_LO = 33, 133, 159, 145
L_OUT, L_IN, L_UP, L_LO = 263, 362, 386, 374
NAMES = ["philtrum", "upper_vermilion", "lower_vermilion", "palpebral_aperture",
         "canthal_tilt_deg", "lower_face_height", "mouth_corner_drop"]


def features(p: np.ndarray) -> np.ndarray:
    """p: (478, 2) pixel coords, already rotated so the outer eye corners are level."""
    d = lambda a, b: float(np.linalg.norm(p[a] - p[b]))  # noqa: E731
    face_h, mouth_w = d(TOP, CHIN), d(MOUTH_R, MOUTH_L)
    aperture = np.mean([d(R_UP, R_LO) / d(R_OUT, R_IN), d(L_UP, L_LO) / d(L_OUT, L_IN)])
    # Image y points down, so outer corner above inner corner gives a positive (upward) tilt.
    tilt = np.mean([math.degrees(math.atan2(p[R_IN, 1] - p[R_OUT, 1], p[R_IN, 0] - p[R_OUT, 0])),
                    math.degrees(math.atan2(p[L_IN, 1] - p[L_OUT, 1], p[L_OUT, 0] - p[L_IN, 0]))])
    stomion_y = (p[LIP_IN_UP, 1] + p[LIP_IN_LO, 1]) / 2
    corner_drop = ((p[MOUTH_R, 1] + p[MOUTH_L, 1]) / 2 - stomion_y) / face_h  # > 0: corners below lip line
    return np.array([d(SUBNASALE, LIP_TOP) / face_h, d(LIP_TOP, LIP_IN_UP) / mouth_w,
                     d(LIP_IN_LO, LIP_BOT) / mouth_w, aperture, tilt,
                     d(SUBNASALE, CHIN) / face_h, corner_drop], dtype=np.float32)


def level(p: np.ndarray) -> np.ndarray:
    """Rotate about the eye midpoint so the outer eye corners are horizontal (removes head roll)."""
    v = p[L_OUT] - p[R_OUT]
    a = -math.atan2(v[1], v[0])
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    c = (p[L_OUT] + p[R_OUT]) / 2
    return (p - c) @ rot.T


def pose_deg(m: np.ndarray) -> tuple[float, float]:
    """Yaw and pitch from MediaPipe's facial transformation matrix."""
    r = m[:3, :3] / np.linalg.norm(m[:3, 0])
    return math.degrees(math.asin(-r[2, 0])), math.degrees(math.atan2(r[2, 1], r[2, 2]))


def main(cap: int):
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL), delegate=mp.tasks.BaseOptions.Delegate.CPU),
        output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1)
    lm = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
    rng = np.random.default_rng(0)
    seen = {b: 0 for b in ADULT}
    F, L, band, split, Q = [], [], [], [], []
    no_face = 0
    for f in sorted(glob.glob(str(DATA / "fairface/*.parquet"))):
        sp = Path(f).stem.split("-")[0]
        for b in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["image", "age"]):
            for r in b.to_pylist():
                if r["age"] not in ADULT or (sp == "train" and seen[r["age"]] >= cap):
                    continue
                if sp == "train" and rng.random() > 0.5 and r["age"] < 7:  # spread the cap across shards
                    continue
                img = np.asarray(Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"))
                res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
                if not res.face_landmarks:
                    no_face += 1; continue
                yaw, pitch = pose_deg(np.array(res.facial_transformation_matrixes[0]))
                bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
                Q.append((yaw, pitch, max(bs["mouthSmileLeft"], bs["mouthSmileRight"]), bs["jawOpen"]))
                h, w = img.shape[:2]
                p = level(np.array([[q.x * w, q.y * h] for q in res.face_landmarks[0]]))
                F.append(features(p))
                L.append((p / np.linalg.norm(p[TOP] - p[CHIN])).flatten().astype(np.float32))
                band.append(r["age"]); split.append(sp)
                if sp == "train":
                    seen[r["age"]] += 1
                if len(F) % 1000 == 0:
                    print(len(F), "no_face", no_face, flush=True)
    yaw, pitch, smile, jaw = np.array(Q, dtype=np.float32).T
    np.savez_compressed(DATA / "face-fairface.npz", F=np.stack(F), L=np.stack(L), band=np.array(band),
                        split=np.array(split), names=np.array(NAMES), yaw=yaw, pitch=pitch, smile=smile, jaw=jaw)
    print("saved", len(F), "no_face", no_face)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
