"""Teacher forcing and analytic backprop for the SAME attention + MoE model.

Routing IDs are piecewise constant; gradients flow through selected gate scores.
Every parameter is optimized. No autograd or training framework is hidden here.
"""
import json
from pathlib import Path

import numpy as np

from .model import TinyModel, VOCAB, softmax


def parameters(model):
    result = {k: getattr(model, k) for k in ("embedding", "wq", "wk", "wv", "wo", "gate", "head")}
    for i, pair in enumerate(model.experts):
        for j, value in enumerate(pair):
            result[f"expert_{i}_{j}"] = value
    return result


def loss_and_grad(model, ids, targets):
    """Return mean next-character cross entropy, gradients, and [B,T,V] logits."""
    ids, targets = np.asarray(ids), np.asarray(targets)
    if ids.ndim != 2 or ids.shape != targets.shape or not ids.size:
        raise ValueError("nonempty matching [batch, time] token arrays required")
    if ids.dtype.kind not in "iu" or targets.dtype.kind not in "iu":
        raise ValueError("integer tokens required")
    if min(ids.min(), targets.min()) < 0 or max(ids.max(), targets.max()) >= len(VOCAB):
        raise ValueError("token outside vocabulary")
    b, t = ids.shape
    d = model.width
    pos = np.arange(t)
    x = model.embedding[ids] + .1 * np.sin(pos[:, None] * 10000.**(-np.arange(d) / d))
    q, k, v = x @ model.wq, x @ model.wk, x @ model.wv
    visible = pos[None, :] <= pos[:, None]
    if model.window is not None:
        visible &= pos[None, :] > pos[:, None] - model.window
    a = softmax(np.where(visible, q @ k.swapaxes(-1, -2) / d**.5, -np.inf))
    c = a @ v
    h = (x + c @ model.wo).reshape(-1, d)
    scores = h @ model.gate
    routes = np.argsort(-scores, axis=-1, kind="stable")[:, :2]
    weights = softmax(np.take_along_axis(scores, routes, axis=1))
    routed = np.zeros_like(h)
    saved = []
    for i, (w1, w2) in enumerate(model.experts):
        rows, branches = np.where(routes == i)
        hidden = np.tanh(h[rows] @ w1)
        values = hidden @ w2
        routed[rows] += weights[rows, branches, None] * values
        saved.append((rows, branches, hidden, values))
    z = h + routed
    logits = z @ model.head
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = softmax(logits)
    y = targets.ravel()
    loss = np.mean(np.log(np.exp(shifted).sum(axis=-1)) - shifted[np.arange(len(y)), y])
    dl = probs.copy()
    dl[np.arange(len(y)), y] -= 1
    dl /= len(y)
    grads = {name: np.zeros_like(value) for name, value in parameters(model).items()}
    grads["head"] = z.T @ dl
    dz = dl @ model.head.T
    dh = dz.copy()
    dw = np.zeros_like(weights)
    for i, ((w1, w2), (rows, branches, hidden, values)) in enumerate(zip(model.experts, saved)):
        dv = dz[rows] * weights[rows, branches, None]
        dw[rows, branches] = (dz[rows] * values).sum(axis=-1)
        grads[f"expert_{i}_1"] = hidden.T @ dv
        da = (dv @ w2.T) * (1 - hidden**2)
        grads[f"expert_{i}_0"] = h[rows].T @ da
        dh[rows] += da @ w1.T
    ds = weights * (dw - (weights * dw).sum(axis=-1, keepdims=True))
    dscores = np.zeros_like(scores)
    np.put_along_axis(dscores, routes, ds, axis=1)
    grads["gate"] = h.T @ dscores
    dh += dscores @ model.gate.T
    dh = dh.reshape(b, t, d)
    grads["wo"] = c.reshape(-1, d).T @ dh.reshape(-1, d)
    dc = dh @ model.wo.T
    da = dc @ v.swapaxes(-1, -2)
    dv = a.swapaxes(-1, -2) @ dc
    ds = a * (da - (da * a).sum(axis=-1, keepdims=True)) / d**.5
    dq, dk = ds @ k, ds.swapaxes(-1, -2) @ q
    dx = dh.copy()
    for name, derivative in (("wq", dq), ("wk", dk), ("wv", dv)):
        grads[name] = x.reshape(-1, d).T @ derivative.reshape(-1, d)
        dx += derivative @ getattr(model, name).T
    np.add.at(grads["embedding"], ids.ravel(), dx.reshape(-1, d))
    return float(loss), grads, logits.reshape(b, t, -1)


class Adam:
    def __init__(self, model, lr=.003, clip=1.):
        if not np.isfinite(lr) or lr <= 0 or clip <= 0:
            raise ValueError("positive learning rate and clipping threshold required")
        self.params, self.lr, self.clip = parameters(model), lr, clip
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.step = 0

    def update(self, grads):
        norm = float(np.sqrt(sum(np.sum(g * g) for g in grads.values())))
        if not np.isfinite(norm):
            raise FloatingPointError("nonfinite gradient")
        self.step += 1
        for name, param in self.params.items():
            g = grads[name] * min(1., self.clip / (norm + 1e-12))
            self.m[name] = .9 * self.m[name] + .1 * g
            self.v[name] = .999 * self.v[name] + .001 * g * g
            m = self.m[name] / (1 - .9**self.step)
            v = self.v[name] / (1 - .999**self.step)
            param -= self.lr * m / (np.sqrt(v) + 1e-8)
        return norm


def save_checkpoint(model, path, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = dict(format=1, width=model.width, window=model.window, vocab=VOCAB,
                  metadata=metadata or {})
    with path.open("wb") as file:
        np.savez_compressed(file, config=json.dumps(config), **parameters(model))


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        config = json.loads(str(data["config"]))
        if config["format"] != 1 or config["vocab"] != VOCAB:
            raise ValueError("unsupported checkpoint format or vocabulary")
        width = config["width"]
        if type(width) is not int or not 1 <= width <= 4096:
            raise ValueError("invalid checkpoint width")
        model = TinyModel(width=width, window=config["window"])
        expected = parameters(model)
        if set(data.files) != set(expected) | {"config"}:
            raise ValueError("checkpoint tensor names do not match model")
        for name, value in expected.items():
            tensor = data[name]
            if tensor.dtype.kind != "f" or tensor.shape != value.shape or not np.isfinite(tensor).all():
                raise ValueError(f"invalid checkpoint tensor: {name}")
            value[:] = tensor
    return model
