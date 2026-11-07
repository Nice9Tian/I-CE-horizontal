# -*- coding: utf-8 -*-
"""Tests of the whole-gallery anchors (plan 4.5): the bank's update rule, anchors through W, the fixed-pack filler,
the fixed packs themselves, the sampler switches, and the bank loss against a brute force."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ice_h.bank import AnchorBank, fill_bank, fill_bank_from_packs, log_weights, routed_bank_loss  # noqa: E402
from ice_h.data import DOC_ID_SP, PoolSampler, fixed_packs  # noqa: E402
from ice_h.losses import routed_anchor_loss  # noqa: E402

from test_codebook import _model  # noqa: E402
from test_core import _FakeF, _fake_pool  # noqa: E402

torch.manual_seed(0)


def _bank(G, K, D, m=None, beta=0.5):
    bank = AnchorBank(G, K, D, beta=beta)
    for g in range(G):
        bank.update(torch.tensor([g]), torch.rand(1, 7, K), torch.randn(1, 7, D))
    return bank


def test_bank_update_first_visit_then_ema():
    bank = AnchorBank(5, 4, 8, beta=0.5)
    M = torch.rand(2, 6, 4); x = torch.randn(2, 6, 8)
    rows = torch.tensor([3, 1])
    bank.update(rows, M, x)
    S = torch.einsum("bnk,bnd->bkd", M, x)
    assert torch.allclose(bank.S[3], S[0]) and torch.allclose(bank.n[1], M[1].sum(0)) and bank.seen[3] and bank.seen[1] and not bank.seen[0]
    M2 = torch.rand(1, 6, 4); x2 = torch.randn(1, 6, 8)
    bank.update(torch.tensor([3]), M2, x2)
    S2 = torch.einsum("bnk,bnd->bkd", M2, x2)[0]
    assert torch.allclose(bank.S[3], 0.5 * S[0] + 0.5 * S2)
    m = _model(K=4, d=8)
    A, cov = bank.anchors(m)
    assert A.shape == (5, 4, 8) and torch.allclose(A.norm(dim=-1), torch.ones(5, 4), atol=1e-4)
    assert torch.allclose(A[0], F.normalize(m.proto_mean, dim=-1), atol=1e-5), "an unseen game's anchors are the priors"
    assert float(cov[0].max()) == 0.0 and float(cov[3].min()) > 0
    st = bank.stats(lam=4.0)
    assert abs(st["bank_seen_frac"] - 0.4) < 1e-6 and 0.0 <= st["bank_topics_covered"] <= 4


def test_bank_loss_matches_brute_force_and_has_gradient_through_W():
    G, K, D, B, Ns = 6, 3, 10, 2, 5
    m = _model(K=K, d=D)
    bank = _bank(G, K, D)
    A, cov = bank.anchors(m)
    assert A.requires_grad, "the bank anchors carry the gradient of W"
    zs = torch.randn(B, Ns, D); ms = torch.arange(Ns)[None, :] >= torch.tensor([5, 3])[:, None]
    M_s = torch.rand(B, Ns, K); M_s = 2.0 * M_s / M_s.sum(-1, keepdim=True)
    rows = torch.tensor([4, 1])
    loss, st = routed_bank_loss(zs, ms, M_s, rows, A, cov, tau=0.1, margin=0.05, sigma=0.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.dW.weight.grad is not None and torch.isfinite(m.dW.weight.grad).all() and float(m.dW.weight.grad.abs().sum()) > 0
    assert m.protos.grad is None and m.gain.grad is None, "the bank gives the assignment no gradient"
    Ad, covd = A.detach(), cov.detach()
    zn = F.normalize(zs, dim=-1)
    for own_neg in (True, False):
        num = den = 0.0
        for b in range(B):
            for i in range(Ns):
                if ms[b, i]:
                    continue
                pos_k = torch.nonzero(M_s[b, i] > 0.5).flatten().tolist() or [int(M_s[b, i].argmax())]
                for k in pos_k:
                    negs, ws = [], []
                    for g in range(G):                                             # same topic, every other game
                        if g != int(rows[b]):
                            negs.append(Ad[g, k]); ws.append(float(covd[g, k]))
                    for bb in range(B):                                            # the batch games' other topics
                        for j in range(K):
                            if j == k:
                                continue
                            if int(rows[bb]) == int(rows[b]) and (not own_neg or j in pos_k):
                                continue
                            negs.append(Ad[int(rows[bb]), j]); ws.append(float(covd[int(rows[bb]), j]))
                    negs = torch.stack(negs)
                    lse = torch.logsumexp((zn[b, i] @ negs.T) / 0.1 + torch.tensor(ws).clamp(min=1e-8).log(), 0)
                    lp = (float(zn[b, i] @ Ad[int(rows[b]), k]) - 0.05) / 0.1
                    w = float(covd[int(rows[b]), k])
                    num += w * float(torch.logaddexp(torch.tensor(lp), lse) - lp); den += w
        loss2, _ = routed_bank_loss(zs, ms, M_s, rows, Ad, covd, tau=0.1, margin=0.05, sigma=0.0, own_neg=own_neg)
        assert abs(float(loss2) - num / den) < 1e-4, (own_neg, float(loss2), num / den)
    # padding never matters
    zs2, Ms2 = zs.clone(), M_s.clone(); zs2[ms] = -123.0; Ms2[ms] = 0.99
    l1, _ = routed_bank_loss(zs, ms, M_s, rows, Ad, covd, tau=0.05, margin=0.05, sigma=0.0)
    l2, _ = routed_bank_loss(zs2, ms, Ms2, rows, Ad, covd, tau=0.05, margin=0.05, sigma=0.0)
    assert abs(float(l1) - float(l2)) < 1e-6
    # B = 1 with own_neg False: only the bank's other games remain, still finite
    lb, _ = routed_bank_loss(zs[:1], ms[:1], M_s[:1], rows[:1], Ad, covd, tau=0.05, margin=0.05, sigma=0.0, own_neg=False)
    assert torch.isfinite(lb)


def test_bank_loss_zero_coverage_excluded_and_query_gradient():
    G, K, D, B, Ns = 6, 3, 10, 2, 5
    m = _model(K=K, d=D)
    bank = _bank(G, K, D)
    A, cov = bank.anchors(m); A, cov = A.detach(), cov.detach()
    zs = torch.randn(B, Ns, D); ms = torch.zeros(B, Ns, dtype=torch.bool)
    M_s = torch.rand(B, Ns, K); M_s = 2.0 * M_s / M_s.sum(-1, keepdim=True)
    rows = torch.tensor([4, 1])
    cov0 = cov.clone(); cov0[0] = 0.0                                               # game 0 covers nothing: a pure prior
    A_hit = A.clone(); A_hit[0] = F.normalize(zs[0, :1], dim=-1).expand(K, -1)       # ... parked right on an element
    l_ref, _ = routed_bank_loss(zs, ms, M_s, rows, A, cov0, tau=0.03, margin=0.05, sigma=0.0)
    l_hit, _ = routed_bank_loss(zs, ms, M_s, rows, A_hit, cov0, tau=0.03, margin=0.05, sigma=0.0)
    assert abs(float(l_ref) - float(l_hit)) < 1e-5, "a zero-coverage anchor is not a negative, even at tau 0.03"
    assert torch.isinf(log_weights(torch.tensor([0.0]))).all() and abs(float(log_weights(torch.tensor([0.5]))) - float(np.log(0.5))) < 1e-6
    # the view-level reading: the strong view's topic anchors as the elements keep their gradient
    a_q = F.normalize(torch.randn(B, K, D), dim=-1).requires_grad_()
    eye = torch.eye(K)[None].expand(B, -1, -1)
    h = torch.rand(B, K); h = h / h.sum(-1, keepdim=True)
    la, _ = routed_bank_loss(a_q, torch.zeros(B, K, dtype=torch.bool), eye, rows, A, cov, tau=0.05, margin=0.05, sigma=0.0, elem_w=h)
    la.backward()
    assert a_q.grad is not None and float(a_q.grad.abs().sum()) > 0, "query-side gradient flows (plan 4.5, view level)"


def test_fill_bank_from_packs_chunking_padding_recompute_and_gradient_scope():
    G, cap, K, D = 11, 24, 6, 16
    m = _model(K=K, d=D, film=True, assign_map=True)
    rng = np.random.default_rng(0)
    L = rng.integers(4, cap + 1, size=G)
    X = torch.randn(G, cap, D)
    for g in range(G):
        X[g, L[g]:] = 999.0                                                         # garbage in padding must never enter S or n
    Xh = X.half()
    b_chunk, b_one = AnchorBank(G, K, D), AnchorBank(G, K, D)
    fill_bank_from_packs(b_chunk, m, Xh, L, B=4)
    fill_bank_from_packs(b_one, m, Xh.numpy(), L, B=1)                             # numpy input, one game at a time
    assert torch.allclose(b_chunk.S, b_one.S, atol=1e-4) and torch.allclose(b_chunk.n, b_one.n, atol=1e-4)
    X_clean = Xh.clone()
    for g in range(G):
        X_clean[g, L[g]:] = 0
    b_clean = AnchorBank(G, K, D)
    fill_bank_from_packs(b_clean, m, X_clean, L, B=4)
    assert torch.allclose(b_chunk.S, b_clean.S, atol=1e-4) and torch.allclose(b_chunk.n, b_clean.n, atol=1e-4)
    m.gain.data.fill_(5.5)                                                          # a second call replaces, never blends
    b_fresh = AnchorBank(G, K, D)
    fill_bank_from_packs(b_chunk, m, X_clean, L, B=4); fill_bank_from_packs(b_fresh, m, X_clean, L, B=4)
    assert torch.equal(b_chunk.S, b_fresh.S) and torch.equal(b_chunk.n, b_fresh.n)
    A, _ = b_clean.anchors(m)
    A.sum().backward()
    assert m.dW.weight.grad is not None and float(m.dW.weight.grad.abs().sum()) > 0
    assert m.film_g.grad is not None and m.film_b.grad is not None
    assert m.protos.grad is None and m.gain.grad is None and m.dU.weight.grad is None, "gradient only to W and FiLM"


def test_fixed_packs_rule_bounds_and_exclusion_sets():
    rng = np.random.default_rng(1)
    pool = rng.normal(size=(2, 80, 1024)).astype(np.float16)
    sp0 = rng.normal(size=(5, 1024)).astype(np.float16)
    fake_F = SimpleNamespace(doc=lambda g: sp0 if g == 0 else None)
    segs = [(np.array([0, 8, 58, 64], np.int64), np.array([8, 50, 6, 4], np.int64), np.array([10, 11, 12, 13], np.int64)),
            (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64))]
    fake_sampler = SimpleNamespace(segs=segs, POOL=pool)
    X, L, used = fixed_packs(fake_F, fake_sampler, [0, 1], cap=20, seed_offset=11)
    assert X.dtype == np.float16 and X.shape == (2, 20, 1024) and int(L[0]) <= 20 and int(L[1]) == 0
    assert np.array_equal(X[0, :5], sp0) and (X[0, L[0]:] == 0).all() and (X[1] == 0).all()
    assert set(used[0].tolist()) <= {10, 11, 12, 13} and 11 not in used[0], "the 50-sentence review cannot fit under cap 20"
    assert int(L[0]) == 5 + sum(int(segs[0][1][list(segs[0][2]).index(r)]) for r in used[0])
    X2, _, _ = fixed_packs(fake_F, fake_sampler, [0], cap=3, seed_offset=11)
    assert int(X2.shape[1]) == 3 and np.array_equal(X2[0, :3], sp0[:3]), "a prefix longer than the cap is cut"
    Xb, _, _ = fixed_packs(fake_F, fake_sampler, [0, 1], cap=20, seed_offset=11)
    assert np.array_equal(X, Xb), "deterministic"


def test_sampler_pack_only_exclusion_and_doc_source():
    rng = np.random.default_rng(0)
    pool, rid, plen = _fake_pool(rng)
    S = PoolSampler(_FakeF(), np.array([0, 1]), cap=512, ns_min=64, ns_max=160, p_doc=1.0, seed=0, pool=pool, rid=rid, plen=plen)
    S.allow_sp_doc = False                                                          # game 0 has a store page but no wiki: review strong view
    b = S.sample(gids=[0, 1])
    assert (b["ts"][0, : int(b["ls"][0])] == 0).all() and (b["tw"][0, :10] == 1).all(), "no store-page doc view; the pack keeps its prefix"
    pk = S.sample_packs([0, 1])
    assert pk["xw"].shape[0] == 2 and (pk["xw"][0, :10] == S.F.doc(0)).all() and int(pk["lw"][0]) <= 512, "pack-only sampling always has the prefix"
    S.exclude_rids = {0: set(range(0, 15))}
    S.build_weak = False
    b2 = S.sample(gids=[0])
    assert int(b2["lw"][0]) == 0 and b2["xw"].shape[1] == 0, "gallery mode builds no weak pack"
    ds = set(b2["ds"][0, : int(b2["ls"][0])].tolist())
    assert ds and not (ds & set(range(0, 15))), "the strong view avoids the fixed pack's reviews"


def test_fill_bank_uses_pack_only_sampling():
    rng = np.random.default_rng(0)
    pool, rid, plen = _fake_pool(rng)
    S = PoolSampler(_FakeF(), np.array([0, 1]), cap=512, ns_min=64, ns_max=160, p_doc=1.0, seed=0, pool=pool, rid=rid, plen=plen)
    m = _model(K=4, d=1024)
    bank = AnchorBank(2, 4, 1024)
    row_of_game = torch.tensor([0, 1])
    fill_bank(bank, m, S, [0, 1], row_of_game, B=2)
    assert bank.seen.all() and float(bank.n.sum()) > 0
    pk = S.sample_packs([0])
    assert int(pk["lw"][0]) >= 10, "game 0's bank pack includes the store-page prefix even though p_doc = 1"


def test_routed_anchor_loss_gradients_both_sides():
    B, K, D = 3, 5, 12
    a_s = F.normalize(torch.randn(B, K, D), dim=-1).requires_grad_()
    a_w = F.normalize(torch.randn(B, K, D), dim=-1).requires_grad_()
    cov_s, cov_w = torch.rand(B, K), torch.rand(B, K)
    h_s = torch.rand(B, K); h_s = h_s / h_s.sum(-1, keepdim=True)
    for own_neg in (True, False):
        for use_s in (True, False):
            loss, st = routed_anchor_loss(a_s, cov_s, h_s, a_w, cov_w, tau=0.07, margin=0.05, own_neg=own_neg, use_strong_neg=use_s)
            loss.backward()
            assert torch.isfinite(a_s.grad).all() and torch.isfinite(a_w.grad).all() and 0.0 <= st["apos_prob"] <= 1.0
            a_s.grad = None; a_w.grad = None
