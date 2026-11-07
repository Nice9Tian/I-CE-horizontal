# -*- coding: utf-8 -*-
"""CPU unit tests for design B: the running-dual membership, the codebook model, the routed losses, the topic scores.

    C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ice_h.codebook import TopicCodebook, balanced_duals, spherical_kmeans  # noqa: E402
from ice_h.evaluate import ranks, topic_features, topic_scores, zeroshot_topics  # noqa: E402
from ice_h.losses import routed_ice_loss, routed_inv_loss, topic_monitors  # noqa: E402

torch.manual_seed(0)


def _model(K=16, d=32, **kw):
    m = TopicCodebook(K=K, d_in=d, c=2.0, eps=0.05, lam=1.0, beta=0.1, **kw)
    X = torch.randn(600, d) + 0.5 * torch.randn(1, d)                              # off-centre data, like Qwen vectors
    m.set_center(X.mean(0))
    m.init_codebook(X - X.mean(0), mode="kmeans", seed=0, iters=10)
    return m


def test_balanced_duals_equalise_columns_and_fix_gauge():
    X = torch.randn(500, 8) * 3 + torch.tensor([3.0, 2, 1, 0, 0, -1, -2, -3])      # skewed topic logits
    g = balanced_duals(X, c=2.0, g0=torch.zeros(8), iters=10, newton=8)
    assert abs(float(g.sum())) < 1e-4, "gauge: the column duals sum to zero"
    from ice_h.model import _shift_for_sum
    f = _shift_for_sum(X + g, torch.full((500, 1), 2.0), dim=1, newton=8)
    M = torch.sigmoid(X + f + g)
    assert torch.allclose(M.sum(1), torch.full((500,), 2.0), atol=2e-3), "rows exact"
    col = M.sum(0)
    assert float(((col - 125.0).abs() / 125.0).max()) < 0.05, "columns within 5 percent of N c / K"
    g0 = balanced_duals(X, 2.0, torch.zeros(8), 10, 8)
    assert float(torch.sigmoid(X).sum(0).std()) > float(col.std()) * 3, "the duals removed most of the imbalance"
    assert torch.allclose(g0, g)


def test_membership_rows_exact_capped_and_sentence_local():
    m = _model(K=64, d=32)
    x = torch.randn(3, 30, 32)
    lens = torch.tensor([30, 12, 1])
    mask = torch.arange(30)[None, :] >= lens[:, None]
    o = m(x, mask)
    M = o["M"]
    for b in range(3):
        n = int(lens[b])
        assert torch.allclose(M[b, :n].sum(-1), torch.full((n,), 2.0), atol=2e-3), "every real sentence carries mass c"
        if n < 30:
            assert float(M[b, n:].abs().max()) == 0.0
    assert float(M.max()) <= 1.0 + 1e-6
    # the same sentence gets the same topics whatever view it sits in
    x2 = torch.cat([x[0, :5], torch.randn(20, 32)])[None]
    o2 = m(x2, torch.zeros(1, 25, dtype=torch.bool))
    assert torch.allclose(o2["M"][0, :5], M[0, :5], atol=1e-5), "membership depends on the sentence alone"
    assert torch.allclose(o["hist"].sum(-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(o["anchors"].norm(dim=-1), torch.ones(3, 64), atol=1e-4)
    assert torch.allclose(o["mean"].norm(dim=-1), torch.ones(3), atol=1e-4)


def test_aggregate_prior_for_absent_topics_and_coverage():
    m = _model(K=8, d=16)
    x = torch.randn(1, 6, 16)
    mask = torch.zeros(1, 6, dtype=torch.bool)
    xc = m.map_space(x, mask)
    M = torch.zeros(1, 6, 8); M[0, :, 0] = 1.0; M[0, :, 1] = 1.0                   # every sentence in topics 0 and 1 only
    o = m.aggregate(xc, mask, M)
    prior = F.normalize(m.proto_mean[2], dim=-1)
    assert torch.allclose(o["anchors"][0, 2], prior, atol=1e-5), "an absent topic's anchor is the prior (W = I at init)"
    data_mean = F.normalize((xc[0].sum(0) + m.lam * m.proto_mean[0]) / (6 + m.lam), dim=-1)
    assert torch.allclose(o["anchors"][0, 0], data_mean, atol=1e-5)
    assert torch.allclose(o["cov"][0, :2], torch.full((2,), 6 / 7.0), atol=1e-5) and float(o["cov"][0, 2:].max()) == 0.0
    assert torch.allclose(o["hist"][0, :2], torch.full((2,), 0.5), atol=1e-5)


def test_running_duals_track_the_stream():
    torch.manual_seed(1)
    m = _model(K=8, d=16)
    m.g_bar.zero_()                                                                # start unbalanced
    errs = []
    for t in range(60):
        x = torch.randn(4, 40, 16) + 0.5 * torch.randn(1, 1, 16) * 0 + torch.tensor([1.0] + [0.0] * 15) * 2   # a fixed skew
        mask = torch.zeros(4, 40, dtype=torch.bool)
        xc = m.map_space(x, mask)
        S = m.affinity(xc, mask)
        M = m.membership(S, mask)
        run = m.update_running([mask], [xc], [M])
        errs.append(run["batch_err_max"])
    assert np.mean(errs[-10:]) < np.mean(errs[:5]) and np.mean(errs[-10:]) < 0.5, (errs[:5], errs[-10:])
    assert abs(float(m.g_bar.sum())) < 1e-3
    assert run["balance_err_max"] < 0.5 and torch.allclose(m.share_ema.sum(), torch.tensor(1.0), atol=1e-4)


def test_implicit_gradient_conserves_row_mass():
    m = _model(K=8, d=16)
    x = torch.randn(2, 12, 16); mask = torch.arange(12)[None, :] >= torch.tensor([12, 7])[:, None]
    xc = m.map_space(x, mask)
    S = m.affinity(xc, mask).detach().requires_grad_(True)
    M = m.membership(S, mask)
    M.sum(-1)[~mask].sum().backward()                                              # d(sum_k M_ik)/dS = 0 when the row dual follows
    assert float(S.grad.abs().max()) < 1e-4, "autograd sees the conservation sum_k M_ik = c"
    m.implicit_grad = False
    S2 = S.detach().requires_grad_(True)
    M2 = m.membership(S2, mask)
    assert torch.allclose(M2, M.detach(), atol=1e-6), "the values do not depend on the gradient mode"
    M2.sum(-1)[~mask].sum().backward()
    assert float(S2.grad.abs().max()) > 1e-3, "without it a uniform shift of the logits looks like it adds mass"
    m.implicit_grad = True
    S3 = S.detach().requires_grad_(True)
    (m.membership(S3, mask)[0, 0, 0]).backward()
    assert torch.isfinite(S3.grad).all() and float(S3.grad[0, 0].abs().sum()) > 0


def test_batch_mode_duals_balance_exactly():
    m = _model(K=8, d=16)
    x = torch.randn(4, 50, 16); mask = torch.zeros(4, 50, dtype=torch.bool)
    xc = m.map_space(x, mask); S = m.affinity(xc, mask)
    g = m.batch_duals([S], [mask])
    M = m.membership(S, mask, g)
    col = M.sum((0, 1)); target = 200 * 2.0 / 8
    assert float(((col - target).abs() / target).max()) < 0.05


def _fake_views(B=3, Ns=9, Nw=20, K=6, D=16, seed=0):
    torch.manual_seed(seed)
    zs = torch.randn(B, Ns, D); ms = torch.arange(Ns)[None, :] >= torch.tensor([9, 6, 4])[:, None]
    M_s = torch.rand(B, Ns, K); M_s = 2.0 * M_s / M_s.sum(-1, keepdim=True); M_s = M_s.masked_fill(ms[..., None], 0.0)
    a_w = F.normalize(torch.randn(B, K, D), dim=-1).requires_grad_()
    a_s = F.normalize(torch.randn(B, K, D), dim=-1).requires_grad_()
    cov_w = torch.rand(B, K)
    return zs, ms, M_s, a_w, cov_w, a_s


def test_routed_loss_finite_backward_and_bruteforce():
    zs, ms, M_s, a_w, cov_w, a_s = _fake_views()
    cov_s = torch.rand(3, 6)
    loss, st = routed_ice_loss(zs, ms, M_s, a_w, cov_w, a_s, cov_s, tau=0.1, margin=0.05, sigma=0.0)
    assert torch.isfinite(loss) and 0 <= st["pos_prob"] <= 1
    loss.backward()
    assert torch.isfinite(a_w.grad).all() and float(a_w.grad.abs().sum()) > 0
    assert torch.isfinite(a_s.grad).all() and float(a_s.grad.abs().sum()) > 0, "strong anchors get gradient as the other games' negatives"
    # brute force: per (i, k) term with explicit candidate sets, every negative weighted by its own coverage
    B, Ns, D = zs.shape; K = a_w.shape[1]
    zn = F.normalize(zs, dim=-1)
    for neg_cov in (True, False):
        num = 0.0; den = 0.0
        for b in range(B):
            for i in range(Ns):
                if ms[b, i]:
                    continue
                pos_k = torch.nonzero(M_s[b, i] > 0.5).flatten().tolist() or [int(M_s[b, i].argmax())]
                negs, ws = [], []
                for bb in range(B):
                    for k in range(K):
                        if bb != b:
                            negs += [a_w[bb, k], a_s[bb, k]]; ws += [float(cov_w[bb, k]), float(cov_s[bb, k])]
                        elif k not in pos_k:
                            negs.append(a_w[bb, k]); ws.append(float(cov_w[bb, k]))
                negs = torch.stack(negs); ws = torch.tensor(ws) if neg_cov else torch.ones(len(ws))
                lse = torch.logsumexp((zn[b, i] @ negs.T) / 0.1 + ws.clamp(min=1e-6).log(), 0)
                for k in pos_k:
                    lp = ((zn[b, i] @ a_w[b, k]) - 0.05) / 0.1
                    num = num + float(cov_w[b, k]) * float(torch.logaddexp(lp, lse) - lp)
                    den += float(cov_w[b, k])
        loss2, _ = routed_ice_loss(zs, ms, M_s, a_w.detach(), cov_w, a_s.detach(), cov_s, tau=0.1, margin=0.05, sigma=0.0, neg_cov=neg_cov)
        assert abs(float(loss2) - num / den) < 1e-4, (neg_cov, float(loss2), num / den)
    # a negative that is only the prior (coverage 0) drops out of the denominator
    cov0 = cov_w.clone(); cov0[1] = 0.0
    l_with, _ = routed_ice_loss(zs, ms, M_s, a_w.detach(), cov0, None, tau=0.1, margin=0.0, sigma=0.0)
    a_far = a_w.detach().clone(); a_far[1] = F.normalize(zn[0, :1].expand(6, -1) + 0.01 * torch.randn(6, 16), dim=-1)   # game 1's anchors sit on game 0's element
    l_far, _ = routed_ice_loss(zs, ms, M_s, a_far, cov0, None, tau=0.1, margin=0.0, sigma=0.0)
    assert abs(float(l_with) - float(l_far)) < 1e-4, "zero-coverage anchors do not act as negatives"


def test_routed_loss_prefers_own_pack_anchors():
    zs, ms, M_s, a_w, cov_w, a_s = _fake_views()
    zn = F.normalize(zs, dim=-1)
    good = F.normalize(torch.einsum("bnk,bnd->bkd", M_s, zn) + 1e-3 * torch.randn(3, 6, 16), dim=-1)   # pack anchors at the elements' topic means
    l_good, _ = routed_ice_loss(zs, ms, M_s, good, torch.ones(3, 6), None, tau=0.1, margin=0.0, sigma=0.0)
    l_rand, _ = routed_ice_loss(zs, ms, M_s, a_w.detach(), torch.ones(3, 6), None, tau=0.1, margin=0.0, sigma=0.0)
    assert float(l_good) < float(l_rand)
    l_zero, _ = routed_ice_loss(zs, ms, M_s, good, torch.zeros(3, 6), None, tau=0.1, margin=0.0, sigma=0.0)
    assert float(l_zero) == 0.0, "no covered positive, no loss"


def test_routed_inv_loss_weights_and_zero():
    a = F.normalize(torch.randn(2, 5, 8), dim=-1)
    h = torch.rand(2, 5); cov = torch.rand(2, 5)
    assert float(routed_inv_loss(a, a.clone(), h, cov)) < 1e-6
    b = a.clone(); b[:, 0] = -a[:, 0]
    w = h * cov
    expect = float((2.0 * w[:, 0]).sum() / w.sum())
    assert abs(float(routed_inv_loss(b, a, h, cov)) - expect) < 1e-5


def test_topic_scores_modes_and_self_retrieval():
    m = _model(K=8, d=16)
    x = torch.randn(6, 15, 16); mask = torch.zeros(6, 15, dtype=torch.bool)
    with torch.no_grad():
        o = m(x, mask)
    G = {"anchors": o["anchors"].half().numpy(), "hist": o["hist"].numpy(), "cov": o["cov"].numpy(), "mean": o["mean"].numpy()}
    with torch.no_grad():
        o2 = m(torch.cat([x[:3], torch.randn(3, 15, 16)]), mask)                 # three identical views, three new ones
    Q = {"anchors": o2["anchors"][:3].half().numpy(), "hist": o2["hist"][:3].numpy(), "cov": o2["cov"][:3].numpy(), "mean": o2["mean"][:3].numpy()}
    for mode in ("routed", "routed_norm", "routed_raw", "mean", "hist"):
        sim = topic_scores(Q, G, mode, device="cpu")
        assert sim.shape == (3, 6)
        assert (ranks(sim, np.arange(3)) == 1).all(), mode
    # a game lacking a topic contributes nothing there under the shrunk score, the prior's hub cosine under the raw one
    G0 = {k: v.copy() for k, v in G.items()}
    G0["cov"][:] = 0.0
    assert float(np.abs(topic_scores(Q, G0, "routed", device="cpu")).max()) < 1e-6
    assert float(np.abs(topic_scores(Q, G0, "routed_raw", device="cpu")).max()) > 0.1
    assert topic_features(G, "mean").shape == (6, 16) and topic_features(G, "hist").shape == (6, 8)
    F3 = topic_features(G, "concat")
    assert F3.shape == (6, 128) and np.allclose(np.linalg.norm(F3, axis=1), 1.0, atol=1e-3), "F3 is a unit vector whatever the view length"
    assert topic_features(G, "mean_hist").shape == (6, 24)


def test_kmeans_init_statistics_and_monitors():
    torch.manual_seed(2)
    centers = F.normalize(torch.randn(5, 16), dim=-1)
    X = torch.cat([c + 0.3 * torch.randn(200, 16) for c in centers])
    C, lab = spherical_kmeans(X, 5, iters=20, seed=0)
    assert C.shape == (5, 16) and torch.allclose(C.norm(dim=-1), torch.ones(5), atol=1e-5)
    assert len(torch.unique(lab)) == 5
    m = TopicCodebook(K=5, d_in=16, c=2.0, eps=0.05)
    st = m.init_codebook(X, mode="kmeans", seed=0, iters=20)
    assert st["cluster_min"] > 100 and st["within_cos"] > 0.5                      # noise norm 1.2 vs centre norm 1: cos about 0.64
    assert abs(float(m.g_bar.sum())) < 1e-4
    x = X[:40][None]; mask = torch.zeros(1, 40, dtype=torch.bool)
    with torch.no_grad():
        ow = m(x, mask); os_ = m(X[200:230][None], torch.zeros(1, 30, dtype=torch.bool))
    mon = topic_monitors(ow, os_, mask, torch.zeros(1, 30, dtype=torch.bool), np.zeros((1, 30), np.int8), m, 0.1)
    assert set(mon) >= {"eff_topics_w", "topics_per_elem_w", "xgame_same_topic_cos", "xtopic_same_game_cos", "proto_drift", "proto_dup_max", "stability"}
    assert abs(mon["proto_drift"] - 1.0) < 1e-5 and abs(mon["stability"] - 0.9) < 1e-6
    rnd = TopicCodebook(K=5, d_in=16, c=2.0, eps=0.05)
    st2 = rnd.init_codebook(X, mode="random", seed=0)
    assert st2["mode"] == "random" and torch.allclose(rnd.protos.norm(dim=-1), torch.ones(5), atol=1e-5)


class _FakeF:
    def __init__(self, n_games=12, n_tags=4, seed=0):
        rng = np.random.default_rng(seed)
        self.names = [f"g{i}" for i in range(n_games)]
        self.n2i = {n: i for i, n in enumerate(self.names)}
        self.y = (rng.random((n_games, n_tags)) < 0.4).astype(np.int8); self.y[0] = 1
        self.tag_names = [f"t{j}" for j in range(n_tags)]
        self._test = set(self.names[:6])

    def fold_games(self, k):
        return self._test, set(self.names[6:9])


def test_zeroshot_topics_shapes():
    rng = np.random.default_rng(0)
    G = {"anchors": rng.normal(size=(12, 8, 16)).astype(np.float16), "hist": np.full((12, 8), 1 / 8, np.float32),
         "cov": np.ones((12, 8), np.float32), "mean": rng.normal(size=(12, 16)).astype(np.float32)}
    tag_emb = rng.normal(size=(4, 16)).astype(np.float32)
    tag_hist = rng.random((4, 8)).astype(np.float32); tag_hist /= tag_hist.sum(1, keepdims=True)
    out = zeroshot_topics(G, tag_emb, tag_hist, _FakeF(), 0)
    assert 0 <= out["macro_map_routed"] <= 1 and 0 <= out["macro_map_max"] <= 1
    assert set(out["tag_topics"]) == {"t0", "t1", "t2", "t3"} and len(out["tag_topics"]["t0"]) == 3
