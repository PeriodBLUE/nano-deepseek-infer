"""DeepSeek-V2 MLA: expanded KV versus latent cache with weight absorption.

Equations: https://arxiv.org/abs/2405.04434 section 2.1.1.
This lab isolates MLA from the V4.1 runtime studied elsewhere in this repo.
"""
import numpy as np
from .model import softmax


def rope(x, positions):
    d = x.shape[-1]
    if d % 2:
        raise ValueError("RoPE width must be even")
    angle = np.asarray(positions)[..., None] * 10000.**(-np.arange(0, d, 2) / d)
    while angle.ndim < x.ndim:
        angle = np.expand_dims(angle, -2)
    out = np.empty_like(x)
    out[..., 0::2] = x[..., 0::2] * np.cos(angle) - x[..., 1::2] * np.sin(angle)
    out[..., 1::2] = x[..., 0::2] * np.sin(angle) + x[..., 1::2] * np.cos(angle)
    return out


class MLA:
    def __init__(self, seed=7, width=24, heads=4, content=6, value=6, latent=8, rotary=4):
        rng = np.random.default_rng(seed)
        w = lambda *shape: rng.normal(0, shape[0]**-.5, shape)
        self.heads, self.content, self.value = heads, content, value
        self.latent, self.rotary = latent, rotary
        self.down = w(width, latent)
        self.q = w(width, heads, content)
        self.uk, self.uv = w(latent, heads, content), w(latent, heads, value)
        self.qr, self.kr = w(width, heads, rotary), w(width, rotary)
        self.out = w(heads * value, width)

    def cache(self):
        return dict(latent=np.empty((0, self.latent)), rope=np.empty((0, self.rotary)))

    def forward(self, x, cache, absorbed=True):
        start = len(cache["latent"])
        pos = np.arange(start, start + len(x))
        c = x @ self.down
        qr = rope(np.einsum("td,dhr->thr", x, self.qr), pos)
        kr = rope(x @ self.kr, pos)
        cache["latent"] = np.concatenate([cache["latent"], c])
        cache["rope"] = np.concatenate([cache["rope"], kr])
        c = cache["latent"]
        qc = np.einsum("td,dhk->thk", x, self.q)
        if absorbed:
            qa = np.einsum("thk,lhk->thl", qc, self.uk)
            scores = np.einsum("thl,sl->hts", qa, c)
        else:
            keys = np.einsum("sl,lhk->shk", c, self.uk)
            scores = np.einsum("thk,shk->hts", qc, keys)
        scores += np.einsum("thr,sr->hts", qr, cache["rope"])
        scores /= (self.content + self.rotary)**.5
        visible = np.arange(len(c))[None, :] <= pos[:, None]
        probs = softmax(np.where(visible[None], scores, -np.inf))
        if absorbed:
            latent_values = np.einsum("hts,sl->thl", probs, c)
            values = np.einsum("thl,lhv->thv", latent_values, self.uv)
        else:
            expanded = np.einsum("sl,lhv->shv", c, self.uv)
            values = np.einsum("hts,shv->thv", probs, expanded)
        return values.reshape(len(x), -1) @ self.out


def mla_lab():
    model = MLA()
    x = np.random.default_rng(2).normal(size=(17, 24))
    expanded = model.forward(x, model.cache(), absorbed=False)
    cache = model.cache()
    incremental = np.concatenate([model.forward(x[:9], cache),
                                  *[model.forward(x[i:i+1], cache) for i in range(9, len(x))]])
    error = float(np.max(np.abs(expanded - incremental)))
    assert error < 1e-12
    elements_full = len(x) * (model.heads * (model.content + model.value) + model.rotary)
    return dict(max_abs_error=error, tokens=len(x), latent_cache_bytes=sum(v.nbytes for v in cache.values()),
                expanded_cache_bytes=elements_full * 8,
                dimensions=dict(heads=model.heads, content=model.content, value=model.value,
                                latent=model.latent, rotary=model.rotary),
                equation="Q_C K_C^T = (Q_C W_UK^T) C^T; softmax(S) V = (softmax(S) C) W_UV")
