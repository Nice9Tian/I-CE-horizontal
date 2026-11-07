# -*- coding: utf-8 -*-
"""Tests of the EMA anchor bank (plan 4.6): update rule, anchors through W, and the bank loss against a brute force."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ice_h.bank import AnchorBank, routed_bank_loss  # noqa: E402

from test_codebook import _model  # noqa: E402

torch.manual_seed(0)


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


def test_bank_loss_matches_brute_force_and_has_gradient_through_W():
    G, K, D, B, Ns = 6, 3, 10, 2, 5
    m = _model(K=K, d=D)
    bank = AnchorBank(G, K, D, beta=0.5)
    for g in range(G):
        bank.update(torch.tensor([g]), torch.rand(1, 7, K), torch.randn(1, 7, D))
    A, cov = bank.anchors(m)
    assert A.requires_grad, "the bank anchors carry the gradient of W"
    zs = torch.randn(B, Ns, D); ms = torch.arange(Ns)[None, :] >= torch.tensor([5, 3])[:, None]
    M_s = torch.rand(B, Ns, K); M_s = 2.0 * M_s / M_s.sum(-1, keepdim=True)
    rows = torch.tensor([4, 1])
    loss, st = routed_bank_loss(zs, ms, M_s, rows, A, cov, rows, tau=0.1, margin=0.05, sigma=0.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.dW.weight.grad is not None and torch.isfinite(m.dW.weight.grad).all() and float(m.dW.weight.grad.abs().sum()) > 0
    # brute force
    Ad, covd = A.detach(), cov.detach()
    zn = F.normalize(zs, dim=-1)
    num = den = 0.0
    for b in range(B):
        for i in range(Ns):
            if ms[b, i]:
                continue
            pos_k = torch.nonzero(M_s[b, i] > 0.5).flatten().tolist() or [int(M_s[b, i].argmax())]
            for k in pos_k:
                negs, ws = [], []
                for g in range(G):                                                 # same topic, every other game
                    if g != int(rows[b]):
                        negs.append(Ad[g, k]); ws.append(float(covd[g, k]))
                for bb in range(B):                                                # the batch games' other topics
                    for j in range(K):
                        if j == k:
                            continue
                        if int(rows[bb]) == int(rows[b]) and j in pos_k:
                            continue
                        negs.append(Ad[int(rows[bb]), j]); ws.append(float(covd[int(rows[bb]), j]))
                negs = torch.stack(negs)
                lse = torch.logsumexp((zn[b, i] @ negs.T) / 0.1 + torch.tensor(ws).clamp(min=1e-6).log(), 0)
                lp = (float(zn[b, i] @ Ad[int(rows[b]), k]) - 0.05) / 0.1
                w = float(covd[int(rows[b]), k])
                num += w * float(torch.logaddexp(torch.tensor(lp), lse) - lp); den += w
    loss2, _ = routed_bank_loss(zs, ms, M_s, rows, Ad, covd, rows, tau=0.1, margin=0.05, sigma=0.0)
    assert abs(float(loss2) - num / den) < 1e-4, (float(loss2), num / den)
    # anchors as the elements (the view-level reading): one-hot memberships, uncovered topics masked
    a_s = F.normalize(torch.randn(B, K, D), dim=-1)
    eye = torch.eye(K)[None].expand(B, -1, -1)
    l_a, _ = routed_bank_loss(a_s, torch.zeros(B, K, dtype=torch.bool), eye, rows, Ad, covd, rows, tau=0.1, margin=0.0, sigma=0.0)
    assert torch.isfinite(l_a)
