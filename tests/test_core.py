# -*- coding: utf-8 -*-
"""CPU unit tests: Sinkhorn marginals, loss behaviour, the sampler on a synthetic pool.

    C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ice_h.data import DOC_ID_SP, PoolSampler  # noqa: E402
from ice_h.evaluate import ranks, set_scores  # noqa: E402
from ice_h.losses import ice_loss, inv_loss, monitors, soft_jaccard  # noqa: E402
from ice_h.model import LatentArray, fd_sinkhorn, log_sinkhorn  # noqa: E402

torch.manual_seed(0)


def _check_marginals(M, lens, c, K, col_atol, col_rtol=0.0):
    N = M.shape[1]
    for b in range(M.shape[0]):
        n = int(lens[b])
        assert torch.allclose(M[b, :n].sum(1), torch.full((n,), c), atol=1e-3), "rows must sum to c (last update)"
        assert torch.allclose(M[b].sum(0), torch.full((K,), n * c / K), atol=col_atol, rtol=col_rtol), "columns must sum to N c / K"
        if n < N:
            assert float(M[b, n:].abs().max()) == 0.0, "padding carries no mass"
    assert torch.isfinite(M).all()


def test_exp_sinkhorn_marginals_with_padding():
    B, N, K, c = 3, 40, 6, 2.0
    S = torch.randn(B, N, K)
    lens = torch.tensor([40, 25, 7])
    mask = torch.arange(N)[None, :] >= lens[:, None]
    _check_marginals(log_sinkhorn(S, mask, c=c, eps=0.3, iters=100), lens, c, K, col_atol=2e-2)


def test_exp_sinkhorn_is_c_times_unit_plan():
    # without a cap the entropic plan with row mass c is exactly c times the unit-row plan: no overlap is created
    S = torch.randn(1, 30, 6) * 3
    mask = torch.zeros(1, 30, dtype=torch.bool)
    M2 = log_sinkhorn(S, mask, 2.0, 0.1, 200)
    M1 = log_sinkhorn(S, mask, 1.0, 0.1, 200)
    assert torch.allclose(M2, 2 * M1, atol=1e-3)


def test_fd_sinkhorn_marginals_cap_and_overlap():
    B, N, K, c = 3, 40, 6, 2.0
    lens = torch.tensor([40, 25, 7])
    mask = torch.arange(N)[None, :] >= lens[:, None]
    S = torch.randn(B, N, K)                                   # logit std 20 at eps 0.05: the regime the model starts in
    M = fd_sinkhorn(S, mask, c=c, eps=0.05, iters=10)
    _check_marginals(M, lens, c, K, col_atol=0.0, col_rtol=2e-2)
    S = torch.randn(B, N, K) * 3                               # logit std 60: near-hard; rows exact, columns approximate
    M = fd_sinkhorn(S, mask, c=c, eps=0.05, iters=10)
    _check_marginals(M, lens, c, K, col_atol=0.0, col_rtol=0.15)
    assert float(M.max()) <= 1.0 + 1e-6, "entries are capped at 1"
    valid = ~mask
    assert float(((M > 0.5).sum(-1)[valid] >= 2).float().mean()) > 0.9, "with c = 2 almost every element sits in two groups"
    S.requires_grad_(True)
    fd_sinkhorn(S, mask, c, 0.05, 5)[:, :, 0].sum().backward()
    assert torch.isfinite(S.grad).all()


def test_sinkhorn_unbalanced_relaxes_columns():
    S = torch.randn(1, 30, 6) * 3
    mask = torch.zeros(1, 30, dtype=torch.bool)
    Mb = log_sinkhorn(S, mask, 2.0, 0.05, 100)
    Mu = log_sinkhorn(S, mask, 2.0, 0.05, 100, rho=0.5)
    assert torch.allclose(Mu[0].sum(1), torch.full((30,), 2.0), atol=2e-2), "rows stay exact"
    assert Mu[0].sum(0).std() > Mb[0].sum(0).std() + 1e-3, "columns become unequal"


def test_sinkhorn_is_differentiable():
    S = torch.randn(2, 20, 4, requires_grad=True)
    mask = torch.zeros(2, 20, dtype=torch.bool)
    M = log_sinkhorn(S, mask, 2.0, 0.1, 5)
    (M[:, :, 0] ** 2).sum().backward()
    assert S.grad is not None and torch.isfinite(S.grad).all() and float(S.grad.abs().sum()) > 0


def test_model_forward_shapes_and_unit_anchors():
    m = LatentArray(K=5, d_in=32, d_model=16, heads=4, rounds=3, sinkhorn_iters=10)
    x = torch.randn(2, 12, 32)
    mask = torch.arange(12)[None, :] >= torch.tensor([12, 7])[:, None]
    o = m(x, mask)
    assert o["anchors"].shape == (2, 5, 32) and o["M"].shape == (2, 12, 5)
    assert torch.allclose(o["anchors"].norm(dim=-1), torch.ones(2, 5), atol=1e-4)
    assert float(o["M"][1, 7:].abs().max()) == 0.0
    o["anchors"].sum().backward()
    assert all(p.grad is not None for n, p in m.named_parameters() if "sa" not in n)


def test_soft_jaccard_identity():
    M = torch.rand(2, 10, 4)
    J = soft_jaccard(M)
    assert torch.allclose(torch.diagonal(J, dim1=1, dim2=2), torch.ones(2, 4), atol=1e-5)
    assert torch.allclose(J, J.transpose(1, 2), atol=1e-6)
    assert float(J.min()) >= 0 and float(J.max()) <= 1 + 1e-6


def _fake_batch(B=3, Ns=9, Nw=20, K=4, D=16):
    zs = torch.randn(B, Ns, D); zw = torch.randn(B, Nw, D)
    ms = torch.arange(Ns)[None, :] >= torch.tensor([9, 6, 4])[:, None]
    mw = torch.zeros(B, Nw, dtype=torch.bool)
    a_s = torch.nn.functional.normalize(torch.randn(B, K, D), dim=-1).requires_grad_()
    a_w = torch.nn.functional.normalize(torch.randn(B, K, D), dim=-1).requires_grad_()
    M_s = fd_sinkhorn(torch.randn(B, Ns, K), ms, 2.0, 0.1, 10)
    M_w = fd_sinkhorn(torch.randn(B, Nw, K), mw, 2.0, 0.1, 10)
    return zs, ms, a_s, M_s, a_w, M_w, zw, mw


def test_ice_loss_finite_and_backward():
    zs, ms, a_s, M_s, a_w, M_w, zw, mw = _fake_batch()
    loss, st = ice_loss(zs, ms, a_s, M_s, a_w, M_w, zw, mw, tau=0.07, margin=0.15, alpha_in=0.8, alpha_x=0.8, sigma=0.05)
    assert torch.isfinite(loss) and 0 <= st["pos_prob"] <= 1
    loss.backward()
    assert torch.isfinite(a_s.grad).all() and torch.isfinite(a_w.grad).all()
    assert float(a_s.grad.abs().sum()) > 0 and float(a_w.grad.abs().sum()) > 0, "weak anchors get gradient as negatives"


def test_ice_loss_prefers_anchors_at_their_members():
    zs, ms, a_s, M_s, a_w, M_w, zw, mw = _fake_batch()
    # anchors placed at the membership-weighted means of their own elements should score lower loss than random
    zn = torch.nn.functional.normalize(zs, dim=-1)
    good = torch.nn.functional.normalize(torch.einsum("bnk,bnd->bkd", M_s, zn), dim=-1)
    l_good, _ = ice_loss(zs, ms, good, M_s, a_w.detach(), M_w, zw, mw, tau=0.07, margin=0.0, alpha_in=0.8, alpha_x=0.8, sigma=0.0)
    l_rand, _ = ice_loss(zs, ms, a_s.detach(), M_s, a_w.detach(), M_w, zw, mw, tau=0.07, margin=0.0, alpha_in=0.8, alpha_x=0.8, sigma=0.0)
    assert float(l_good) < float(l_rand)


def test_no_decay_weights_are_one_and_monitors_run():
    zs, ms, a_s, M_s, a_w, M_w, zw, mw = _fake_batch()
    l1, _ = ice_loss(zs, ms, a_s, M_s, a_w, M_w, zw, mw, tau=0.07, margin=0.1, alpha_in=0.0, alpha_x=0.0, sigma=0.0)
    l2_, _ = ice_loss(zs, ms, a_s, M_s, a_w, M_w, zw, mw, tau=0.07, margin=0.1, alpha_in=0.8, alpha_x=0.8, sigma=0.0, decay=False)
    assert torch.allclose(l1, l2_, atol=1e-5)
    B, Nw, K = M_w.shape
    tw = np.zeros((B, Nw), np.int8); tw[:, :3] = 1
    ds = np.random.default_rng(0).integers(0, 5, size=(B, ms.shape[1]))
    mon = monitors(M_w, mw, tw, M_s, ms, ds, a_s.detach(), a_w.detach())
    assert set(mon) >= {"usage_min", "groups_per_elem_w", "docs_per_group_s", "intra_cos", "stability", "xgame_nn_median", "desc_mix"}


def test_inv_loss_zero_when_equal():
    a = torch.nn.functional.normalize(torch.randn(2, 3, 8), dim=-1)
    assert float(inv_loss(a, a.clone())) < 1e-6
    assert float(inv_loss(a, -a)) > 1.9


def test_set_scores_and_ranks():
    Q = torch.nn.functional.normalize(torch.randn(5, 3, 8), dim=-1).numpy()
    G = np.concatenate([Q, np.random.default_rng(0).normal(size=(7, 3, 8)).astype(np.float32)])
    for mode in ("slot", "mean", "chamfer"):
        sim = set_scores(Q, G, mode, device="cpu")
        assert sim.shape == (5, 12)
        assert (ranks(sim, np.arange(5)) == 1).all(), mode


class _FakeF:
    """a Fixtures stand-in with two games: game 0 has a store page, game 1 has none"""
    NG = 2

    def __init__(self):
        self._sp = {0: np.random.default_rng(1).normal(size=(10, 1024)).astype(np.float16)}

    def doc(self, g):
        return self._sp.get(g)

    def wiki_doc(self, g):
        return None


def _fake_pool(rng, n_rev=(30, 25)):
    NG, R = 2, 2048
    pool = rng.normal(size=(NG, R, 1024)).astype(np.float16)
    rid = np.full((NG, R), -1, np.int32); plen = np.zeros(NG, np.int32)
    for g in range(NG):
        lens = rng.integers(3, 40, size=n_rev[g]); row = 0
        for j, L in enumerate(lens):
            L = min(int(L), R - row)
            rid[g, row:row + L] = j; row += L
            if row >= R:
                break
        plen[g] = row
    return pool, rid, plen


def test_sampler_views_are_disjoint_and_within_budget():
    rng = np.random.default_rng(0)
    pool, rid, plen = _fake_pool(rng)
    F = _FakeF()
    S = PoolSampler(F, np.array([0, 1]), cap=512, ns_min=64, ns_max=160, p_doc=0.0, seed=0, pool=pool, rid=rid, plen=plen)
    for _ in range(20):
        b = S.sample(gids=[0, 1])
        for i, g in enumerate((0, 1)):
            lw, ls = int(b["lw"][i]), int(b["ls"][i])
            assert lw <= 512 and ls >= 64 or ls == plen[g] - lw, "weak under the cap, strong at least ns_min"
            dw = set(b["dw"][i, :lw].tolist()) - {DOC_ID_SP}
            ds = set(b["ds"][i, :ls].tolist())
            assert not (dw & ds), "no review appears in both views"
            if g == 0:
                assert (b["tw"][i, :10] == 1).all() and b["dw"][i, 0] == DOC_ID_SP, "store page is the prefix of the pack"
            assert (b["ts"][i, :ls] == 0).all()
            # whole reviews: every review id in a view appears with all its sentences
            for d in ds:
                assert int((b["ds"][i, :ls] == d).sum()) == int((rid[g, :plen[g]] == d).sum())


def test_sampler_doc_view_drops_shared_prefix():
    rng = np.random.default_rng(1)
    pool, rid, plen = _fake_pool(rng)
    S = PoolSampler(_FakeF(), np.array([0]), cap=512, ns_min=64, ns_max=160, p_doc=1.0, seed=0, pool=pool, rid=rid, plen=plen)
    b = S.sample(gids=[0])
    assert int(b["ls"][0]) == 10 and (b["ts"][0, :10] == 1).all(), "strong view is the store page"
    assert (b["tw"][0, : int(b["lw"][0])] == 0).all(), "the weak pack then has no description sentence"
