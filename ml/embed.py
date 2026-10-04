"""Turn voice clips into WavLM embeddings for age regression.

Usage: python ml/embed.py voxceleb|cv17
Writes data/emb-<name>.npz with X (n, layers*768), plus labels.
"""
import csv, glob, io, sys
from pathlib import Path

import av
import numpy as np
import torch
from transformers import AutoFeatureExtractor, WavLMModel

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODEL = "microsoft/wavlm-base-plus"
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


def main(name):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(MODEL)
    model = WavLMModel.from_pretrained(MODEL).to(dev).eval()
    X, meta = [], []
    for i, (src, m) in enumerate(items(name)):
        try:
            wav = decode(src)
        except Exception as e:
            print("skip", e); continue
        if len(wav) < SR:  # under 1 s of audio
            continue
        inp = fe(wav, sampling_rate=SR, return_tensors="pt").input_values.to(dev)
        with torch.no_grad():
            hs = model(inp, output_hidden_states=True).hidden_states  # 13 x (1, T, 768)
        # Mean + std pool every layer; age cues sit in middle layers.
        h = torch.stack(hs)[:, 0]
        X.append(torch.cat([h.mean(1), h.std(1)], -1).flatten().cpu().numpy().astype(np.float16))
        meta.append(m)
        if i % 200 == 0:
            print(name, i, flush=True)
    keys = meta[0].keys()
    np.savez_compressed(DATA / f"emb-{name}.npz", X=np.stack(X),
                        **{k: np.array([m[k] for m in meta]) for k in keys})
    print("saved", len(X))


if __name__ == "__main__":
    main(sys.argv[1])
