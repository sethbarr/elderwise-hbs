"""Turn voice clips into WavLM embeddings (MLX) for age regression.

Usage: python ml/embed.py voxceleb|cv17
Writes data/emb-<name>.npz with X (n, layers*768), plus labels.
"""
import csv, glob, io, sys
from pathlib import Path

import av
import mlx.core as mx
import numpy as np

from wavlm_mlx import WavLM

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SR = 16000
MAX_SEC = 10


def decode(src) -> np.ndarray:
    """Decode any audio (path or bytes) to 16 kHz mono float32, capped at MAX_SEC."""
    with av.open(io.BytesIO(src) if isinstance(src, bytes) else str(src)) as c:
        rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
        out = []
        for frame in c.decode(audio=0):
            for f in rs.resample(frame):
                out.append(f.to_ndarray().reshape(-1))
            if sum(len(x) for x in out) >= SR * MAX_SEC:
                break
    return np.concatenate(out)[: SR * MAX_SEC]


def items(name):
    if name == "voxceleb":
        lab = {}
        for split in ("train", "test"):
            for r in csv.DictReader(open(DATA / f"voxceleb-age/age-{split}.txt")):
                lab[(r["VoxCeleb_ID"], r["video_id"])] = (float(r["speaker_age"]), split, r["Name"])
        for f in sorted(glob.glob(str(DATA / "voxceleb-age/voxceleb_age/*/*/*.m4a"))):
            spk, vid = f.split("/")[-3:-1]
            if (spk, vid) in lab:
                age, split, nm = lab[(spk, vid)]
                yield f, dict(age=age, split=split, speaker=spk)
    else:
        import pyarrow.parquet as pq
        for f in sorted(glob.glob(str(DATA / "cv17-age/*.parquet"))):
            t = pq.read_table(f, columns=["audio", "age", "gender", "client_id"]).to_pylist()
            for r in t:
                yield r["audio"]["bytes"], dict(band=r["age"], gender=r["gender"], speaker=r["client_id"],
                                                split=Path(f).stem.split("-")[0])


def main(name, limit=None):
    # Clip lengths vary, so MLX's buffer cache never reuses and grows until the Mac panics. Cap it hard.
    mx.set_memory_limit(4 << 30)
    mx.set_cache_limit(256 << 20)
    model = WavLM()
    X, meta = [], []
    for i, (src, m) in enumerate(items(name)):
        try:
            wav = decode(src)
        except Exception as e:
            print("skip", e); continue
        if len(wav) < SR:  # under 1 s of audio
            continue
        X.append(model.embed(wav))  # mean + std pool of all 13 layers; age cues sit in middle layers
        meta.append(m)
        mx.clear_cache()
        if i % 200 == 0:
            print(name, i, f"peak {mx.get_peak_memory() / 2**30:.2f} GB", flush=True)
        if limit and len(X) >= limit:
            break
    keys = meta[0].keys()
    out = DATA / (f"emb-{name}.npz" if not limit else f"emb-{name}-test.npz")
    np.savez_compressed(out, X=np.stack(X),
                        **{k: np.array([m[k] for m in meta]) for k in keys})
    print("saved", len(X))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
