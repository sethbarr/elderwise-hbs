"""WavLM base+ encoder in MLX (inference only), weight-compatible with microsoft/wavlm-base-plus.

Mirrors transformers' WavLMModel (post-norm encoder, gated relative position bias) so
embeddings match the PyTorch version that the age model was trained on.
"""
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download

REPO = "microsoft/wavlm-base-plus"
REVISION = "98fd61b9c652129c839c0a25a05987d8f59256a4"  # safetensors conversion of main
CONV = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2  # (dim, kernel, stride)
D, HEADS, LAYERS, FFN = 768, 12, 12, 3072
POS_K, POS_GROUPS = 128, 16
BUCKETS, MAX_DIST = 320, 800
EPS = 1e-5


def _layer_norm(x, p):
    return mx.fast.layer_norm(x, p["weight"], p["bias"], EPS)


def _linear(x, p):
    return x @ p["weight"].T + p["bias"]


def _rel_buckets(t: int) -> np.ndarray:
    rel = np.arange(t)[None, :] - np.arange(t)[:, None]
    half = BUCKETS // 2
    out = (rel > 0).astype(np.int64) * half
    rel = np.abs(rel)
    exact = half // 2
    with np.errstate(divide="ignore"):
        large = exact + (np.log(rel / exact) / math.log(MAX_DIST / exact) * (half - exact))
    large = np.minimum(np.nan_to_num(large, neginf=0).astype(np.int64), half - 1)
    return out + np.where(rel < exact, rel, large)


class WavLM:
    def __init__(self, path: str | None = None):
        path = path or hf_hub_download(REPO, "model.safetensors", revision=REVISION)
        w = mx.load(path)
        # Fold weight norm (dim=2) into one conv kernel, then move every conv to MLX layout (out, K, in).
        g, v = w.pop("encoder.pos_conv_embed.conv.weight_g"), w.pop("encoder.pos_conv_embed.conv.weight_v")
        w["encoder.pos_conv_embed.conv.weight"] = g * v / mx.sqrt((v * v).sum(axis=(0, 1), keepdims=True))
        for k in list(w):
            if k.endswith("conv.weight"):
                w[k] = w[k].transpose(0, 2, 1)
        self.w = w
        mx.eval(self.w)

    def _p(self, prefix):
        n = len(prefix) + 1
        return {k[n:]: v for k, v in self.w.items() if k.startswith(prefix + ".")}

    def _attention(self, x, p, bias):
        b, t, _ = x.shape
        hd = D // HEADS
        split = lambda y: y.reshape(b, t, HEADS, hd).transpose(0, 2, 1, 3)  # noqa: E731
        q, k, v = (split(_linear(x, p[n])) for n in ("q_proj", "k_proj", "v_proj"))
        # Per-token gate on the shared relative position bias.
        g = mx.sigmoid((split(x) @ p["gru_rel_pos_linear"]["weight"].T + p["gru_rel_pos_linear"]["bias"])
                       .reshape(b, HEADS, t, 2, 4).sum(-1))
        gate = g[..., :1] * (g[..., 1:] * p["gru_rel_pos_const"] - 1.0) + 2.0
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5, mask=gate * bias)
        return _linear(out.transpose(0, 2, 1, 3).reshape(b, t, D), p["out_proj"])

    def hidden_states(self, wav: np.ndarray) -> list[mx.array]:
        """16 kHz mono float waveform -> 13 hidden states of shape (1, T, 768)."""
        x = mx.array(wav, dtype=mx.float32)[None, :, None]
        for i, (_, k, s) in enumerate(CONV):
            p = self._p(f"feature_extractor.conv_layers.{i}")
            x = mx.conv1d(x, p["conv.weight"], stride=s)
            if i == 0:  # GroupNorm with one group per channel = normalise each channel over time
                mu, var = x.mean(1, keepdims=True), x.var(1, keepdims=True)
                x = (x - mu) * mx.rsqrt(var + EPS) * p["layer_norm.weight"] + p["layer_norm.bias"]
            x = nn.gelu(x)
        x = _linear(_layer_norm(x, self._p("feature_projection.layer_norm")), self._p("feature_projection.projection"))

        pc = self._p("encoder.pos_conv_embed.conv")
        pos = mx.conv1d(x, pc["weight"], padding=POS_K // 2, groups=POS_GROUPS) + pc["bias"]
        x = _layer_norm(x + nn.gelu(pos[:, :-1]), self._p("encoder.layer_norm"))

        t = x.shape[1]
        emb = self.w["encoder.layers.0.attention.rel_attn_embed.weight"]
        bias = emb[mx.array(_rel_buckets(t))].transpose(2, 0, 1)[None]  # (1, H, T, T)

        states = [x]
        for i in range(LAYERS):
            pre = f"encoder.layers.{i}"
            att = {n: self._p(f"{pre}.attention.{n}") for n in ("q_proj", "k_proj", "v_proj", "out_proj", "gru_rel_pos_linear")}
            att["gru_rel_pos_const"] = self.w[f"{pre}.attention.gru_rel_pos_const"]
            x = _layer_norm(x + self._attention(x, att, bias), self._p(f"{pre}.layer_norm"))
            ff = nn.gelu(_linear(x, self._p(f"{pre}.feed_forward.intermediate_dense")))
            x = _layer_norm(x + _linear(ff, self._p(f"{pre}.feed_forward.output_dense")), self._p(f"{pre}.final_layer_norm"))
            states.append(x)
        return states

    def embed(self, wav: np.ndarray) -> np.ndarray:
        """Mean + std pool of every hidden state -> (13 * 1536,) float16, same layout as ml/embed.py."""
        h = mx.stack(self.hidden_states(wav))[:, 0]  # (13, T, 768)
        v = mx.concatenate([h.mean(1), mx.std(h, axis=1, ddof=1)], -1).flatten()
        return np.array(v.astype(mx.float16))
