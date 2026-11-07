# -*- coding: utf-8 -*-
"""Encoding packs and queries, set-to-set retrieval scores, the tag probes.

Retrieval follows the paper's protocol (k10_eval.score): for one anchor draw the candidates
are all 2,020 games; a query is a held-out test game's rewritten wiki article (variant
``neutral`` = Name, ``noname`` = Stripped); hit@k is averaged over the draws.  With K anchors
per game the query is also a set of K anchors, and a (query, game) score is one of

    slot     (1/K) sum_k cos(q_k, g_k)                 slots aligned by the shared latents
    mean     cos(mean_k q_k, mean_k g_k)                the single-vector reading
    chamfer  (1/K) sum_k max_k' cos(q_k, g_k')          alignment-free (the only one for random groups)

Checkpoint selection (w9_selection.rank_score): sum of exp(-rank) over the validation fold's
neutral and noname queries, ranks 1-based against all games; ties prefer the earlier step.
Tag probe (k10_eval.score / text_variant_eval.train_anchor_ridge): ridge fit on training-pool
anchor features, alpha and threshold picked on validation games' anchor features, micro-F1 on
the test games' neutral query features.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import PUBLIC_REPO, Fixtures, rown

MODES = ("slot", "mean", "chamfer")


def l2(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


# ---------------------------------------------------------------------------- encoding
@torch.no_grad()
def encode_sets(model, X, lens, device, batch: int = 64, lens_cap: int | None = None) -> np.ndarray:
    """X fp16 (N, L, 1024) standardised (memmap OK), lens (N,) -> anchors float32 (N, K, 1024)"""
    model.eval()
    N = X.shape[0]
    out = []
    for i in range(0, N, batch):
        L = int(lens[i:i + batch].max())
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch, :L])).to(device)
        mb = torch.arange(L, device=device)[None, :] >= torch.as_tensor(lens[i:i + batch], device=device)[:, None]
        out.append(model(xb, mb)["anchors"].float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def encode_queries(model, F_: Fixtures, device, batch: int = 128) -> np.ndarray:
    S, mask = F_.queries_std()
    return encode_sets(model, S, F_.QL, device, batch=batch)


def frozen_mean(X, lens, batch: int = 32) -> np.ndarray:
    """the frozen-embedder baseline: mean of the standardised sentences (not unit-normalised)"""
    out = np.zeros((X.shape[0], X.shape[2]), np.float32)
    for i in range(0, X.shape[0], batch):
        xb = np.asarray(X[i:i + batch], dtype=np.float32)
        L = lens[i:i + batch]
        w = (np.arange(xb.shape[1])[None, :] < L[:, None]).astype(np.float32)[..., None]
        out[i:i + batch] = (xb * w).sum(1) / np.maximum(w.sum(1), 1)
    return out


def random_group_anchors(X, lens, K: int, seed: int = 0, batch: int = 64) -> np.ndarray:
    """baseline B3: each set's sentences randomly partitioned into K groups, group means as unit anchors"""
    rng = np.random.default_rng(seed)
    N, _, D = X.shape
    out = np.zeros((N, K, D), np.float32)
    for i in range(0, N, batch):
        xb = np.asarray(X[i:i + batch], dtype=np.float32)
        for b in range(xb.shape[0]):
            n = int(lens[i + b])
            lab = rng.permutation(n) % K if n >= K else rng.integers(0, K, n)
            for k in range(K):
                sel = lab == k
                out[i + b, k] = xb[b, :n][sel].mean(0) if sel.any() else xb[b, :n].mean(0)
    return l2(out)


# ---------------------------------------------------------------------------- scores
def set_scores(Q: np.ndarray, G: np.ndarray, mode: str, device=None, chunk: int = 256) -> np.ndarray:
    """Q (nq, K, D), G (NG, K, D) -> similarity (nq, NG) float32 for one scoring mode"""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Qt = F.normalize(torch.as_tensor(np.asarray(Q, np.float32), device=dev), dim=-1)
    Gt = F.normalize(torch.as_tensor(np.asarray(G, np.float32), device=dev), dim=-1)
    nq, K, D = Qt.shape
    NG = Gt.shape[0]
    if mode == "mean":
        return (F.normalize(Qt.mean(1), dim=-1) @ F.normalize(Gt.mean(1), dim=-1).T).cpu().numpy()
    out = torch.empty(nq, NG, device=dev)
    for i in range(0, nq, chunk):
        q = Qt[i:i + chunk]
        if mode == "slot":
            out[i:i + chunk] = torch.einsum("qkd,gkd->qg", q, Gt) / K
        elif mode == "chamfer":
            s = torch.einsum("qkd,gjd->qgkj", q, Gt)                 # (chunk, NG, K, K)
            out[i:i + chunk] = s.max(-1).values.mean(-1)
        else:
            raise ValueError(mode)
    return out.cpu().numpy()


def ranks(sim: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """1-based rank of each query's own game; equal scores share the better rank (k10_eval)"""
    own = sim[np.arange(len(tgt)), tgt]
    return (sim > own[:, None]).sum(1) + 1


def rank_score(sim: np.ndarray, tgt: np.ndarray) -> float:
    return float(np.exp(-ranks(sim, tgt).astype(np.float64)).sum())


def retrieval_metrics(Qa: np.ndarray, Ga: np.ndarray, F_: Fixtures, fold: int, modes=MODES, device=None) -> dict:
    """test-fold hit@1/hit@5 per variant and mode, validation rank score per mode; Qa (3256, K, D), Ga (NG, K, D)"""
    test, val = F_.fold_games(fold)
    out = {}
    for mode in modes:
        for var in ("neutral", "noname"):
            q = F_.query_rows(test, var)
            sim = set_scores(Qa[q], Ga, mode, device)
            r = ranks(sim, F_.gA[q])
            out[f"{mode}_h1_{var}"] = float((r == 1).mean())
            out[f"{mode}_h5_{var}"] = float((r <= 5).mean())
            out[f"{mode}_mrr_{var}"] = float((1.0 / r).mean())
        qv = np.concatenate([F_.query_rows(val, "neutral"), F_.query_rows(val, "noname")])
        out[f"{mode}_val_rank_score"] = rank_score(set_scores(Qa[qv], Ga, mode, device), F_.gA[qv])
    return out


def single_vector_metrics(Qv: np.ndarray, Gv: np.ndarray, F_: Fixtures, fold: int) -> dict:
    """the paper's reading for one vector per game / per query (used for the frozen baseline)"""
    return retrieval_metrics(Qv[:, None, :], Gv[:, None, :], F_, fold, modes=("slot",))


# ---------------------------------------------------------------------------- features and tag probes
def features(anchors: np.ndarray, kind: str) -> np.ndarray:
    """'mean': unit mean of the K anchors (D); 'concat': the K unit anchors flattened (K*D)"""
    a = l2(anchors)
    if kind == "mean":
        return l2(a.mean(1))
    if kind == "concat":
        return a.reshape(a.shape[0], -1)
    raise ValueError(kind)


def _ridge_tools():
    sys.path.insert(0, str(Path(PUBLIC_REPO) / "contrast_experiment" / "w9"))
    from VICReg_review.text_variant_eval import micro_prf, train_anchor_ridge   # noqa: E402
    return micro_prf, train_anchor_ridge


CONCAT_ALPHAS = (0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0)


def tag_probe(F_: Fixtures, fold: int, Gfeat: np.ndarray, Qfeat_by_row: dict, alphas=None) -> dict:
    """ridge on training-pool anchor features -> micro-F1 on the test games' neutral query features.

    Gfeat (NG, d); Qfeat_by_row maps a wiki_eval row index to its (d,) feature (neutral test rows needed).
    alphas: the paper's grid (0.01 ... 1000) by default; the 65,536-dimensional F3 feature gets CONCAT_ALPHAS, the
    same grid extended upward (after column standardisation the trace of the Gram matrix scales with d, so the
    paper's upper bound 1000 under-regularises at d = 65,536); the choice stays on the validation games."""
    from types import SimpleNamespace
    micro_prf, train_anchor_ridge = _ridge_tools()
    import VICReg_review.text_variant_eval as tve
    test, val = F_.fold_games(fold)
    split = {"train": [n for n in F_.names if n not in test | val], "val": sorted(val), "test": sorted(test)}
    targs = SimpleNamespace(tag_text_train_frac=0.7, tag_text_val_frac=0.15, tag_text_split_seed=42, seed=42, tag_text_threshold_steps=33)
    old = tve.DEFAULT_ALPHAS
    if alphas is not None:
        tve.DEFAULT_ALPHAS = tuple(alphas)
    try:
        sc, rg, alpha, th, vm = train_anchor_ridge(targs, Gfeat.astype(np.float32), F_.y, F_.n2i, split)
    finally:
        tve.DEFAULT_ALPHAS = old
    ti = F_.query_rows(test, "neutral")
    Xq = np.stack([Qfeat_by_row[int(i)] for i in ti]).astype(np.float32)
    lab = np.stack([F_.y[F_.n2i[F_.art[i]]] for i in ti])
    m = micro_prf(lab, rg.predict(sc.transform(Xq)), th)
    return {"micro_f1": float(m["micro_f1"]), "precision": float(m["precision"]), "recall": float(m["recall"]),
            "alpha": float(alpha), "threshold": float(th), "val_f1": float(vm["micro_f1"])}


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """AP of one tag; nan when the tag has no positive among the scored games"""
    y = np.asarray(y) > 0
    if not y.any():
        return float("nan")
    order = np.argsort(-s, kind="stable")
    hits = y[order]
    prec = np.cumsum(hits) / (np.arange(len(hits)) + 1)
    return float((prec * hits).sum() / hits.sum())


def zeroshot_tags(anchors: np.ndarray, tag_emb: np.ndarray, F_: Fixtures, fold: int, reduce: str = "max") -> dict:
    """score(n, t) = max_k cos(a_k^(n), e_t) on the test games; macro-mAP over tags with positives"""
    test, _ = F_.fold_games(fold)
    gi = np.array(sorted(F_.n2i[n] for n in test))
    a = l2(anchors[gi])                                     # (nt, K, D)
    e = l2(tag_emb)                                         # (T, D)
    s = np.einsum("nkd,td->ntk", a, e)                      # (nt, T, K)
    score = s.max(-1) if reduce == "max" else s.mean(-1)
    arg = s.argmax(-1)
    y = F_.y[gi]
    aps = np.array([average_precision(y[:, t], score[:, t]) for t in range(y.shape[1])])
    freq = F_.y.sum(0)
    order = np.argsort(freq)
    bands = {"low": order[: len(order) // 3], "mid": order[len(order) // 3: 2 * len(order) // 3], "high": order[2 * len(order) // 3:]}
    out = {"macro_map": float(np.nanmean(aps)), "ap": {F_.tag_names[t]: (None if np.isnan(aps[t]) else float(aps[t])) for t in range(len(aps))},
           "slot_hist": {F_.tag_names[t]: np.bincount(arg[y[:, t] > 0, t], minlength=anchors.shape[1]).tolist() for t in range(len(aps))}}
    for name, ids in bands.items():
        out[f"map_{name}"] = float(np.nanmean(aps[ids]))
    return out


def zeroshot_single(vec: np.ndarray, tag_emb: np.ndarray, F_: Fixtures, fold: int) -> dict:
    """the frozen-mean baseline for the zero-shot reading: cos(mean, e_t)"""
    return zeroshot_tags(vec[:, None, :], tag_emb, F_, fold)


# ============================================================================ design B: topic encodings, scores, features
TMODES = ("routed", "routed_norm", "routed_raw", "mean", "hist")


@torch.no_grad()
def encode_topics(model, X, lens, device, batch: int = 64) -> dict:
    """X fp16 (N, L, D) standardised (memmap OK), lens (N,) -> {anchors fp16 (N, K, D), hist (N, K), cov (N, K), mean (N, D)}"""
    model.eval()
    N = X.shape[0]
    acc = {"anchors": [], "hist": [], "cov": [], "mean": []}
    for i in range(0, N, batch):
        L = int(lens[i:i + batch].max())
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch, :L])).to(device)
        mb = torch.arange(L, device=device)[None, :] >= torch.as_tensor(lens[i:i + batch], device=device)[:, None]
        o = model(xb, mb)
        acc["anchors"].append(o["anchors"].half().cpu()); acc["hist"].append(o["hist"].float().cpu())
        acc["cov"].append(o["cov"].float().cpu()); acc["mean"].append(o["mean"].float().cpu())
    return {k: torch.cat(v).numpy() for k, v in acc.items()}


def topic_subset(D: dict, idx) -> dict:
    return {k: v[idx] for k, v in D.items()}


def topic_scores(Q: dict, G: dict, mode: str, device=None, chunk: int = 128) -> np.ndarray:
    """similarity (nq, NG) of query views Q against game views G (dicts from encode_topics) for one scoring mode:

        routed       sum_k h^q_k omega^g_k cos(a^q_k, a^g_k)            the query's topics, shrunk by the game's coverage:
                                                                        a topic the game lacks contributes 0 (main score)
        routed_norm  sum_k h^q_k omega^g_k cos(.) / sum_k h^q_k omega^g_k   renormalised: lacking a topic is not penalised
        routed_raw   sum_k h^q_k cos(a^q_k, a^g_k)                      no coverage: a lacking game offers the topic prior,
                                                                        whose unit vector is a hub (reference only)
        mean         cos(abar^q, abar^g)                                the K = 1 reading
        hist         sum_k sqrt(h^q_k h^g_k)                            topic distribution only (Bhattacharyya)
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "mean":
        q = F.normalize(torch.as_tensor(np.asarray(Q["mean"], np.float32), device=dev), dim=-1)
        g = F.normalize(torch.as_tensor(np.asarray(G["mean"], np.float32), device=dev), dim=-1)
        return (q @ g.T).cpu().numpy()
    if mode == "hist":
        q = torch.as_tensor(np.asarray(Q["hist"], np.float32), device=dev).clamp(min=0).sqrt()
        g = torch.as_tensor(np.asarray(G["hist"], np.float32), device=dev).clamp(min=0).sqrt()
        return (q @ g.T).cpu().numpy()
    if mode not in ("routed", "routed_norm", "routed_raw"):
        raise ValueError(mode)
    Ga = torch.as_tensor(np.asarray(G["anchors"]), device=dev).float()            # (NG, K, D)
    Gc = torch.as_tensor(np.asarray(G["cov"], np.float32), device=dev)             # (NG, K)
    Qa, Qh = Q["anchors"], Q["hist"]
    nq = Qa.shape[0]
    out = torch.empty(nq, Ga.shape[0], device=dev)
    for i in range(0, nq, chunk):
        q = torch.as_tensor(np.asarray(Qa[i:i + chunk]), device=dev).float()
        h = torch.as_tensor(np.asarray(Qh[i:i + chunk], np.float32), device=dev)
        s = torch.einsum("qkd,gkd->qgk", q, Ga)                                    # (chunk, NG, K)
        if mode == "routed_raw":
            out[i:i + chunk] = (s * h[:, None, :]).sum(-1)
        else:
            w = h[:, None, :] * Gc[None]
            out[i:i + chunk] = (s * w).sum(-1)
            if mode == "routed_norm":
                out[i:i + chunk] /= w.sum(-1).clamp(min=1e-6)
    return out.cpu().numpy()


def topic_retrieval_metrics(Q: dict, G: dict, F_: Fixtures, fold: int, modes=TMODES, device=None) -> dict:
    """test-fold hit@1 / hit@5 / MRR per variant and mode, validation rank score per mode (selection: routed)"""
    test, val = F_.fold_games(fold)
    out = {}
    for mode in modes:
        for var in ("neutral", "noname"):
            q = F_.query_rows(test, var)
            sim = topic_scores(topic_subset(Q, q), G, mode, device)
            r = ranks(sim, F_.gA[q])
            out[f"{mode}_h1_{var}"] = float((r == 1).mean())
            out[f"{mode}_h5_{var}"] = float((r <= 5).mean())
            out[f"{mode}_mrr_{var}"] = float((1.0 / r).mean())
        qv = np.concatenate([F_.query_rows(val, "neutral"), F_.query_rows(val, "noname")])
        out[f"{mode}_val_rank_score"] = rank_score(topic_scores(topic_subset(Q, qv), G, mode, device), F_.gA[qv])
    return out


def topic_features(G: dict, kind: str, rows=None) -> np.ndarray:
    """plan section 7: 'mean' (D) unit mean anchor; 'hist' (K) topic distribution; 'mean_hist' (D + K);
    'concat' (K D) the topic anchors weighted by sqrt(h_k) and flattened: a unit vector whatever the view's length
    (packs and queries share one scale), whose inner product is sum_k sqrt(h^q_k h^g_k) cos(a^q_k, a^g_k)"""
    idx = slice(None) if rows is None else np.asarray(rows)
    if kind == "mean":
        return l2(G["mean"][idx])
    if kind == "hist":
        return np.asarray(G["hist"][idx], np.float32)
    if kind == "mean_hist":
        return np.concatenate([l2(G["mean"][idx]), np.asarray(G["hist"][idx], np.float32)], 1)
    if kind == "concat":
        a = l2(np.asarray(G["anchors"][idx], np.float32))
        w = np.sqrt(np.clip(np.asarray(G["hist"][idx], np.float32), 0, None))
        return (a * w[..., None]).reshape(a.shape[0], -1)
    raise ValueError(kind)


def zeroshot_topics(G: dict, tag_emb: np.ndarray, tag_hist: np.ndarray, F_: Fixtures, fold: int) -> dict:
    """plan section 8: routed score(n, t) = sum_k h^t_k omega^(n)_k cos(a_k^(n), e_t) with h^t the tag phrase's own
    topic distribution and omega the game's coverage (a topic the game lacks contributes 0); the design-A max_k
    reading alongside.  Macro-mAP over tags with positives on the test games."""
    test, _ = F_.fold_games(fold)
    gi = np.array(sorted(F_.n2i[n] for n in test))
    a = l2(np.asarray(G["anchors"][gi], np.float32))                               # (nt, K, D)
    cov = np.asarray(G["cov"][gi], np.float32)                                      # (nt, K)
    e = l2(tag_emb)                                                                 # (T, D)
    s = np.einsum("nkd,td->ntk", a, e)                                              # (nt, T, K)
    scores = {"routed": (s * np.asarray(tag_hist, np.float32)[None] * cov[:, None, :]).sum(-1), "max": s.max(-1)}
    y = F_.y[gi]
    freq = F_.y.sum(0)
    order = np.argsort(freq)
    bands = {"low": order[: len(order) // 3], "mid": order[len(order) // 3: 2 * len(order) // 3], "high": order[2 * len(order) // 3:]}
    out = {}
    for name, sc in scores.items():
        aps = np.array([average_precision(y[:, t], sc[:, t]) for t in range(y.shape[1])])
        out[f"macro_map_{name}"] = float(np.nanmean(aps))
        out[f"ap_{name}"] = {F_.tag_names[t]: (None if np.isnan(aps[t]) else float(aps[t])) for t in range(len(aps))}
        for b, ids in bands.items():
            out[f"map_{b}_{name}"] = float(np.nanmean(aps[ids]))
    out["tag_topics"] = {F_.tag_names[t]: [int(k) for k in np.argsort(-tag_hist[t])[:3]] for t in range(len(F_.tag_names))}
    return out
