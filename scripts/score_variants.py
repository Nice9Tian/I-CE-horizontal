# -*- coding: utf-8 -*-
"""Compare ways of treating a game's coverage in the routed score, on one cached draw (plan section 6).

    python scripts/score_variants.py --run runs/b64_k0 --step 0 --cache-dir C:/ice_horizontal_cache --data-dir C:/ice_horizontal_cache/data

For each variant: test-fold Name@1 / Stripped@1 and the validation rank score.  Variants: raw (no coverage, the prior
is a hub), shrink (omega = n/(n+lam)), norm (shrink renormalised), hard (min(1, n)), sat (1 - exp(-n)), hard_norm.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ice_h.data import CACHE_DIR, DATA_DIR, Fixtures, load_draw  # noqa: E402
from ice_h.evaluate import encode_topics, rank_score, ranks  # noqa: E402
from ice_h.runs import load_model  # noqa: E402


def scores(Q, G, weight_fn, renorm, dev, chunk=128):
    Ga = torch.as_tensor(G["anchors"], device=dev).float()
    Gw = weight_fn(torch.as_tensor(G["cov"], device=dev).float())                 # (NG, K) game-side topic weights
    out = torch.empty(Q["anchors"].shape[0], Ga.shape[0], device=dev)
    for i in range(0, out.shape[0], chunk):
        q = torch.as_tensor(Q["anchors"][i:i + chunk], device=dev).float()
        h = torch.as_tensor(Q["hist"][i:i + chunk], device=dev).float()
        s = torch.einsum("qkd,gkd->qgk", q, Ga)
        w = h[:, None, :] * Gw[None]
        out[i:i + chunk] = (s * w).sum(-1) / (w.sum(-1).clamp(min=1e-6) if renorm else 1.0)
    return out.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--step", default="0"); ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--cap", type=int, default=512); ap.add_argument("--cache-dir", default=str(CACHE_DIR)); ap.add_argument("--data-dir", default=str(DATA_DIR))
    a = ap.parse_args()
    dev = torch.device("cuda")
    model, step, targs = load_model(Path(a.run), a.step, dev)
    lam = model.lam
    F = Fixtures(Path(a.data_dir))
    X, L, _, _ = load_draw(a.cap, a.draw, Path(a.cache_dir))
    G = encode_topics(model, X, L, dev)
    S, _ = F.queries_std()
    Q = encode_topics(model, S, F.QL, dev, batch=128)
    test, val = F.fold_games(targs["fold"])
    n_of = lambda cov: lam * cov / (1.0 - cov).clamp(min=1e-6)                     # mass back from the coverage
    variants = {
        "raw": (lambda c: torch.ones_like(c), False),
        "shrink": (lambda c: c, False),
        "norm": (lambda c: c, True),
        "hard": (lambda c: n_of(c).clamp(max=1.0), False),
        "hard_norm": (lambda c: n_of(c).clamp(max=1.0), True),
        "sat": (lambda c: 1.0 - torch.exp(-n_of(c)), False),
        "sat_norm": (lambda c: 1.0 - torch.exp(-n_of(c)), True),
        "present3": (lambda c: (n_of(c) >= 3.0).float(), False),
        "present3_norm": (lambda c: (n_of(c) >= 3.0).float(), True),
    }
    cov = torch.as_tensor(G["cov"]); n = lam * cov / (1 - cov).clamp(min=1e-6)
    print(f"step {step}: topics per pack with n>=1: {float((n >= 1).float().sum(1).mean()):.1f}, n>=3: {float((n >= 3).float().sum(1).mean()):.1f}, "
          f"query topics with mass>=1: {float((torch.as_tensor(Q['cov']) >= 0.5).float().sum(1).mean()):.1f}")
    print(f"{'variant':14s} {'Name@1':>7s} {'Strip@1':>8s} {'Strip@5':>8s} {'val rank':>9s}")
    for name, (fn, renorm) in variants.items():
        res = []
        for var in ("neutral", "noname"):
            q = F.query_rows(test, var)
            sim = scores({k: v[q] for k, v in Q.items()}, G, fn, renorm, dev)
            r = ranks(sim, F.gA[q]); res += [float((r == 1).mean()), float((r <= 5).mean())]
        qv = np.concatenate([F.query_rows(val, "neutral"), F.query_rows(val, "noname")])
        vs = rank_score(scores({k: v[qv] for k, v in Q.items()}, G, fn, renorm, dev), F.gA[qv])
        print(f"{name:14s} {res[0]:7.3f} {res[2]:8.3f} {res[3]:8.3f} {vs:9.2f}")


if __name__ == "__main__":
    main()
