# -*- coding: utf-8 -*-
"""Tests added after the design-B code review: the implicit Jacobian against finite differences, the unbalanced KKT
solution, the all-excluded negative row, the gain projection, checkpoint round trips, and the scoring-mode names."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ice_h.codebook import TopicCodebook, balanced_duals  # noqa: E402
from ice_h.evaluate import TMODES  # noqa: E402
from ice_h.losses import routed_ice_loss  # noqa: E402
from ice_h.model import _shift_for_sum  # noqa: E402
from ice_h.runs import build_model, load_state  # noqa: E402

from test_codebook import _fake_views, _model  # noqa: E402

torch.manual_seed(0)


def test_implicit_gradient_matches_finite_difference_jacobian():
    m = _model(K=6, d=12)
    x = torch.randn(1, 3, 12)
    mask = torch.tensor([[False, False, True]])
    S0 = m.affinity(m.map_space(x, mask), mask).detach()
    J_auto = torch.zeros(6, 6)
    for k in range(6):
        S = S0.clone().requires_grad_(True)
        m.membership(S, mask)[0, 0, k].backward()
        J_auto[k] = S.grad[0, 0]
        assert float(S.grad[0, 2].abs().max()) == 0.0, "padded row carries zero gradient"
    delta = 1e-3
    J_num = torch.zeros(6, 6)
    with torch.no_grad():
        for j in range(6):
            Sp, Sm = S0.clone(), S0.clone()
            Sp[0, 0, j] += delta
            Sm[0, 0, j] -= delta
            J_num[:, j] = (m.membership(Sp, mask)[0, 0] - m.membership(Sm, mask)[0, 0]) / (2 * delta)
    assert torch.allclose(J_auto, J_num, atol=2e-3), (J_auto - J_num).abs().max()


def test_unbalanced_duals_satisfy_kkt():
    X = torch.randn(800, 8) * 2 + torch.tensor([3.0, 2, 1, 0, 0, -1, -2, -3])
    kappa = 0.05 / 0.5
    g = balanced_duals(X, c=2.0, g0=torch.zeros(8), iters=15, newton=8, kappa=kappa)
    f = _shift_for_sum(X + g[None], torch.full((800, 1), 2.0), dim=1, newton=8)
    col = torch.sigmoid(X + f + g[None]).sum(0)
    t_col = 800 * 2.0 / 8
    tgt = t_col * torch.exp(-kappa * g)
    assert float(((col - tgt).abs() / t_col).max()) < 1e-2, (col, tgt)             # the KKT condition holds to the solver's tolerance
    assert float((col / t_col).std()) > 0.05, "unbalanced columns are allowed to differ"
    gb = balanced_duals(X, c=2.0, g0=torch.zeros(8), iters=15, newton=8)
    assert abs(float(gb.sum())) < 1e-4, "the balanced duals keep the zero-mean gauge"


def test_routed_loss_no_own_neg_and_empty_negatives_safe():
    zs, ms, M_s, a_w, cov_w, a_s = _fake_views(B=3, Ns=6, K=5, D=12)
    loss, _ = routed_ice_loss(zs, ms, M_s, a_w, cov_w, None, tau=0.1, margin=0.05, sigma=0.0, own_neg=False)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(a_w.grad).all()
    a1 = a_w[:1].detach().clone().requires_grad_(True)                             # B = 1, own_neg False, no strong anchors: nothing left
    loss0, _ = routed_ice_loss(zs[:1], ms[:1], M_s[:1], a1, cov_w[:1], None, tau=0.1, margin=0.05, sigma=0.0, own_neg=False)
    loss0.backward()
    assert float(loss0) == 0.0 and torch.isfinite(a1.grad).all()


def test_gain_projection_keeps_the_gradient_alive():
    m = TopicCodebook(K=4, d_in=8, gain_init=2.05, gain_min=2.0, gain_max=8.0)
    opt = torch.optim.AdamW([m.gain], lr=0.05)
    x = torch.randn(1, 5, 8); mask = torch.zeros(1, 5, dtype=torch.bool)
    with torch.no_grad():
        sign = 1.0 if float(m.affinity(x, mask).sum()) > 0 else -1.0               # so that descending the loss lowers the gain
    for _ in range(10):                                                            # push the gain down past its bound
        opt.zero_grad(); (sign * m.affinity(x, mask).sum()).backward(); opt.step()
        with torch.no_grad():
            m.gain.clamp_(m.gain_min, m.gain_max)                                  # the projection train.py applies after every step
    assert abs(float(m.gain) - 2.0) < 1e-6
    opt.zero_grad(); (-sign * m.affinity(x, mask).sum()).backward()
    assert float(m.gain.grad.abs()) > 0, "at the bound the gain still receives gradient and can move back"


def test_load_state_roundtrip_both_designs():
    m = _model(K=8, d=16, assign_map=True, film=True)
    m2 = TopicCodebook(K=8, d_in=16, c=2.0, eps=0.05, lam=1.0, beta=0.1, assign_map=True, film=True)
    load_state(m2, m.state_dict(), "B")
    for k in ("mu", "protos0", "g_bar", "proto_mean", "share_ema"):
        assert torch.equal(getattr(m, k), getattr(m2, k))
    assert torch.equal(m2.protos, m.protos) and float(m2.gain) == float(m.gain)
    args_a = {"design": "A", "K": 4, "d_model": 32, "heads": 4, "rounds": 1, "dropout": 0.0, "c": 2.0, "eps": 0.05, "sinkhorn_iters": 5, "rho": 0.0,
              "anchor_mode": "meanmap", "center": True}
    ma = build_model(args_a, "cpu")
    sd_legacy = {k: v.clone() for k, v in ma.state_dict().items() if k != "dW.weight"}
    sd_legacy["W.weight"] = torch.eye(1024) + 0.01 * torch.randn(1024, 1024)
    load_state(ma, sd_legacy, "A")
    assert torch.allclose(ma.dW.weight, sd_legacy["W.weight"] - torch.eye(1024), atol=1e-6)


def test_kernel_ridge_matches_sklearn():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from ice_h.evaluate import kernel_ridge_scores
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(60, 300)) * rng.uniform(0.5, 3, size=300); ytr = (rng.random((60, 4)) < 0.4).astype(np.float32)
    Xva = rng.normal(size=(15, 300)) * rng.uniform(0.5, 3, size=300)
    sc = StandardScaler().fit(Xtr); Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)
    alphas = (0.1, 10.0, 1000.0)
    ours = kernel_ridge_scores(Xtr_s, ytr, [Xva_s], alphas)
    for a, (sva,) in zip(alphas, ours):
        ref = Ridge(alpha=a).fit(Xtr_s, ytr).predict(Xva_s)
        assert np.allclose(sva, ref, atol=1e-5), (a, np.abs(sva - ref).max())


def test_topic_retrieval_keys_match_tmodes():
    assert TMODES == ("routed", "routed_norm", "routed_raw", "mean", "hist")       # guards against stale names such as routed_cov
    src = Path(__file__).resolve().parents[1]
    for f in ("scripts/eval_run.py", "scripts/compare.py", "scripts/select_ckpt.py"):
        assert "routed_cov" not in (src / f).read_text(encoding="utf-8"), f
