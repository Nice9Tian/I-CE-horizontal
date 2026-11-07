# -*- coding: utf-8 -*-
"""Design B: the shared topic codebook (plan sections 3 and 4.2).

K global unit prototypes p_k score every sentence x_i = z_i - mu:  S_ik = gain * cos(U x_i, p_k).  The membership

    M_ik = sigmoid(S_ik / eps + f_i + g_k),      sum_k M_ik = c,      0 <= M_ik <= 1

is solved PER SENTENCE for its row dual f_i (Fermi-Dirac: entries capped at 1, so c > 1 spreads a sentence over at
least c topics); the first-order dependence of f_i on the logits is put back by the implicit function theorem.  The
column duals g are not solved per view: ``g_bar`` is a running estimate, moved after every step by a stochastic
step toward equal topic mass over the training stream (no gradient; --balance batch solves the batch exactly
instead).  At evaluation g_bar is frozen, so a sentence's topics depend on the sentence alone, never on the view or
the batch.

A view (pack, strong view, wiki query, tag phrase) is summarised by

    m_k = sum_i M_ik          topic mass            h_k = m_k / sum_k m_k      topic distribution
    mbar_k = (sum_i M_ik x_i + lam pbar_k) / (m_k + lam)     smoothed topic mean (pbar_k: running mean of topic k)
    a_k = normalize(W mbar_k)                               topic anchor,  W = I + dW
    abar = normalize(W xbar)                                mean anchor (the K = 1 reading)
    cov_k = m_k / (m_k + lam)                               coverage: how much of a_k is data rather than prior

W acts on the anchor side only; the optional U = I + dU on the assignment side only (plan: one space).  Shapes:
x (B, N, 1024) float, mask (B, N) bool with True = padding.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import NEG, _shift_for_sum


@torch.no_grad()
def balanced_duals(X: torch.Tensor, c: float, g0: torch.Tensor, iters: int = 10, newton: int = 8,
                   kappa: float | None = None) -> torch.Tensor:
    """Column duals g (K,) that equalise the K topic masses of the pooled logits X (N, K) (X = S / eps): with the row
    duals f solved for row mass c, every column of sigmoid(X + f + g) sums to N c / K (balanced), or satisfies the
    unbalanced KKT condition sum_i sigma(X_ik + f_i + g_k) = t_k exp(-kappa g_k) with kappa = eps / rho when kappa is
    given.  Alternating block solves, warm started at g0, columns last; the gauge is fixed by sum_k g_k = 0 (the row
    duals absorb the shift)."""
    N, K = X.shape
    X = X.float()
    t_row = torch.full((N, 1), float(c), device=X.device)
    t_col = torch.full((1, K), N * c / K, device=X.device)
    g = g0.detach().float().reshape(1, K).clone()
    for _ in range(iters):
        f = _shift_for_sum(X + g, t_row, dim=1, newton=newton)                    # (N, 1)
        if kappa is None:
            g = _shift_for_sum(X + f, t_col, dim=0, newton=newton)                # (1, K) exact column solve
        else:                                                                      # unbalanced: KKT Newton on the columns, warm from g
            for _ in range(newton):
                p_ = torch.sigmoid(X + f + g)
                tgt = t_col * torch.exp((-kappa * g).clamp(-50.0, 50.0))
                h = p_.sum(0, keepdim=True) - tgt
                dh = (p_ * (1 - p_)).sum(0, keepdim=True) + kappa * tgt
                g = g - (h / dh.clamp(min=1e-4)).clamp(-5.0, 5.0)
    if kappa is None:                                                              # the shift freedom exists only when balanced
        g = g - g.mean()
    return g[0]


@torch.no_grad()
def spherical_kmeans(X: torch.Tensor, K: int, iters: int = 25, seed: int = 0):
    """cosine k-means on the rows of X (n, d): unit centroids (K, d) and hard labels (n,); empty clusters are reseeded"""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    Xn = F.normalize(X.float(), dim=-1)
    n = Xn.shape[0]
    C = Xn[torch.randperm(n, generator=gen)[:K].to(X.device)].clone()
    for _ in range(iters):
        lab = (Xn @ C.T).argmax(-1)
        Cn = torch.zeros_like(C).index_add_(0, lab, Xn)
        cnt = torch.bincount(lab, minlength=K)
        empty = cnt == 0
        if empty.any():
            Cn[empty] = Xn[torch.randperm(n, generator=gen)[: int(empty.sum())].to(X.device)]
        C = F.normalize(Cn, dim=-1)
    lab = (Xn @ C.T).argmax(-1)
    return C, lab


class TopicCodebook(nn.Module):
    """K shared topic prototypes -> per-sentence Fermi-Dirac membership -> topic distribution + topic anchors per view"""

    def __init__(self, K: int = 64, d_in: int = 1024, c: float = 2.0, eps: float = 0.05, lam: float = 1.0,
                 beta: float = 0.05, rho: float | None = None, assign_map: bool = False, film: bool = False,
                 iters: int = 10, newton: int = 8, gain_init: float = 3.0, gain_min: float = 2.0, gain_max: float = 8.0,
                 dual_lr: float = 0.2, implicit_grad: bool = True, center: bool = True):
        super().__init__()
        assert c < K, "the per-sentence mass must be below the number of topics"
        self.K, self.d_in, self.c, self.eps, self.lam, self.beta = K, d_in, c, eps, lam, beta
        self.rho = rho if rho and rho > 0 else None
        self.iters, self.newton, self.center = iters, newton, center
        self.gain_min, self.gain_max, self.dual_lr, self.implicit_grad = gain_min, gain_max, dual_lr, implicit_grad
        self.register_buffer("mu", torch.zeros(d_in))
        self.protos = nn.Parameter(F.normalize(torch.randn(K, d_in), dim=-1))
        self.register_buffer("protos0", self.protos.detach().clone())            # the initial codebook (drift monitor)
        self.register_buffer("g_bar", torch.zeros(K))                             # running column duals (sum zero when balanced)
        self.register_buffer("proto_mean", torch.zeros(K, d_in))                 # running topic means (the prior)
        self.register_buffer("share_ema", torch.full((K,), 1.0 / K))             # EMA of the topics' mass shares (balance monitor)
        self.gain = nn.Parameter(torch.tensor(float(gain_init)))
        self.dW = nn.Linear(d_in, d_in, bias=False)
        nn.init.zeros_(self.dW.weight)
        self.dU = None
        if assign_map:
            self.dU = nn.Linear(d_in, d_in, bias=False)
            nn.init.zeros_(self.dU.weight)
        self.film_g = self.film_b = None
        if film:
            self.film_g = nn.Parameter(torch.zeros(K, d_in))
            self.film_b = nn.Parameter(torch.zeros(K, d_in))
        self.last_balance_err = (float("nan"), float("nan"))

    # ---- spaces
    def set_center(self, mu):
        self.mu.copy_(torch.as_tensor(mu, dtype=self.mu.dtype, device=self.mu.device))

    def W(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.dW(z)

    def U(self, z: torch.Tensor) -> torch.Tensor:
        return z if self.dU is None else z + self.dU(z)

    def map_space(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """raw standardised Qwen vectors -> the centred space the loss, the anchors and the readouts live in"""
        z = z.float() - self.mu if self.center else z.float()
        return z.masked_fill(mask[..., None], 0.0) if mask is not None else z

    # ---- the three stages
    def affinity(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x centred (B, N, D) -> S (B, N, K) = gain cos(U x, p); padded rows 0"""
        q = F.normalize(self.U(x), dim=-1)
        p = F.normalize(self.protos, dim=-1)
        S = (q @ p.T) * self.gain.clamp(self.gain_min, self.gain_max)
        return S.masked_fill(mask[..., None], 0.0)

    def membership(self, S: torch.Tensor, mask: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        """M (B, N, K): rows of real sentences sum to c exactly, entries in [0, 1]; g defaults to the running duals.

        The row dual f_i is solved without gradient; with ``implicit_grad`` its exact first-order dependence on the
        logits is added back (implicit function theorem on sum_k sigmoid(X_ik + f_i + g_k) = c:
        df_i/dX_ij = -w_ij / sum_k w_ik with w = M (1 - M)), so autograd sees the conservation sum_k M_ik = c and a
        uniform shift of a sentence's logits (e.g. through gain) carries no gradient."""
        g = self.g_bar if g is None else g
        X = (S.float() / self.eps).masked_fill(mask[..., None], NEG)
        with torch.no_grad():
            t_row = torch.full(X.shape[:2] + (1,), float(self.c), device=X.device)
            f0 = _shift_for_sum(X.detach() + g, t_row, dim=2, newton=self.newton).masked_fill(mask[..., None], 0.0)
        f = f0
        if self.implicit_grad and X.requires_grad:
            with torch.no_grad():
                M0 = torch.sigmoid(X + f0 + g).masked_fill(mask[..., None], 0.0)
                w = M0 * (1.0 - M0)
                w = w / w.sum(-1, keepdim=True).clamp(min=1e-6)
            f = f0 - (w * (X - X.detach())).sum(-1, keepdim=True)                  # value f0, gradient of the implicit solution
        M = torch.sigmoid(X + f + g)
        return M.masked_fill(mask[..., None], 0.0)

    def aggregate(self, x: torch.Tensor, mask: torch.Tensor, M: torch.Tensor) -> dict:
        """x centred and zero at padding; M (B, N, K) -> anchors, hist, mass, cov, mean"""
        mass = M.sum(1)                                                            # (B, K)
        hist = mass / mass.sum(1, keepdim=True).clamp(min=1e-6)
        num = torch.einsum("bnk,bnd->bkd", M, x) + self.lam * self.proto_mean[None]
        mbar = num / (mass[..., None] + self.lam)
        if self.film_g is not None:
            mbar = mbar * (1.0 + self.film_g[None]) + self.film_b[None]
        anchors = F.normalize(self.W(mbar), dim=-1)
        n = (~mask).sum(1).clamp(min=1).float()
        xbar = x.sum(1) / n[:, None]
        mean = F.normalize(self.W(xbar), dim=-1)
        cov = mass / (mass + self.lam)
        return {"anchors": anchors, "hist": hist, "mass": mass, "cov": cov, "mean": mean}

    def forward(self, x: torch.Tensor, mask: torch.Tensor, g: torch.Tensor | None = None) -> dict:
        x = self.map_space(x, mask)
        S = self.affinity(x, mask)
        M = self.membership(S, mask, g)
        out = self.aggregate(x, mask, M)
        out.update(M=M, S=S)
        return out

    # ---- the running estimates (no gradient)
    @torch.no_grad()
    def batch_duals(self, S_list, mask_list) -> torch.Tensor:
        """the column duals that equalise the topics over all real sentences of these views (warm start g_bar)"""
        X = torch.cat([(S.float() / self.eps)[~m] for S, m in zip(S_list, mask_list)], 0)   # (N_tot, K)
        return balanced_duals(X, self.c, self.g_bar, self.iters, self.newton, self.kappa)

    @property
    def kappa(self) -> float | None:
        return None if self.rho is None else self.eps / self.rho

    @torch.no_grad()
    def update_running(self, mask_list, x_list, M_list, g_star: torch.Tensor | None = None, update_duals: bool = True) -> dict:
        """The running estimates after one step, from the memberships the batch actually used.

        g_bar: stochastic-approximation step on the population balance condition E[mass_k] = N c / K,
               g_k <- g_k - dual_lr (mass_k - target_k) / target (unbiased: the fixed point is exact balance in
               expectation; solving the batch exactly and averaging the duals would be biased for rare topics);
               target_k = N c / K, or N c / K exp(-kappa g_k) when unbalanced.  Balanced duals are kept at mean zero.
               With ``g_star`` (the batch-exact duals of --balance batch) the EMA toward them is used instead.
        share_ema: EMA of the topics' mass shares (the balance monitor: a single batch's shares carry binomial noise
               of about 15 percent per topic, the EMA does not).
        proto_mean: EMA toward this batch's prior-free topic means, for the topics that carried mass.
        Returns the monitor values."""
        Mp = torch.cat([M[~m] for M, m in zip(M_list, mask_list)], 0)                   # (N_tot, K)
        xp = torch.cat([x[~m] for x, m in zip(x_list, mask_list)], 0)                   # (N_tot, D)
        col = Mp.sum(0)
        target = Mp.shape[0] * self.c / self.K
        if not update_duals:                                                             # frozen assignment: g_bar stays, the sentence-to-topic function is fixed
            pass
        elif g_star is not None:
            self.g_bar.mul_(1.0 - self.beta).add_(self.beta * g_star)
        elif self.kappa is None:
            self.g_bar.sub_(self.dual_lr * (col - target) / target)
            self.g_bar.sub_(self.g_bar.mean())
        else:
            tgt = target * torch.exp((-self.kappa * self.g_bar).clamp(-50.0, 50.0))
            self.g_bar.sub_(self.dual_lr * (col - tgt) / target)
        share = col / col.sum().clamp(min=1e-6)
        self.share_ema.mul_(1.0 - self.beta).add_(self.beta * share)
        rel = (col - target).abs() / target
        rel_ema = (self.share_ema * self.K - 1.0).abs()
        self.last_balance_err = (float(rel_ema.max()), float(rel_ema.mean()))
        pm = (Mp.T @ xp) / col[:, None].clamp(min=1e-6)
        upd = col > 1.0                                                                 # topics with at least one sentence's mass
        self.proto_mean[upd] = (1.0 - self.beta) * self.proto_mean[upd] + self.beta * pm[upd]
        return {"balance_err_max": self.last_balance_err[0], "balance_err_mean": self.last_balance_err[1],
                "batch_err_max": float(rel.max()), "usage_min_xK": float(self.share_ema.min() * self.K),
                "dead_topics": int((self.share_ema < 0.1 / self.K).sum()), "g_bar_std": float(self.g_bar.std())}

    @torch.no_grad()
    def init_codebook(self, X: torch.Tensor, mode: str = "kmeans", seed: int = 0, iters: int = 25) -> dict:
        """X (n, D) centred training sentences.  Prototypes = spherical k-means centroids (or random unit vectors),
        proto_mean = cluster means, g_bar = the balanced column duals on this sample.  Returns init statistics."""
        X = X.float().to(self.protos.device)
        if mode == "kmeans":
            C, lab = spherical_kmeans(X, self.K, iters=iters, seed=seed)
        else:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            C = F.normalize(torch.randn(self.K, self.d_in, generator=gen), dim=-1).to(X.device)
            lab = (F.normalize(X, dim=-1) @ C.T).argmax(-1)
        self.protos.data.copy_(C)
        self.protos0.copy_(C)
        cnt = torch.bincount(lab, minlength=self.K).float()
        pm = torch.zeros(self.K, self.d_in, device=X.device).index_add_(0, lab, X) / cnt.clamp(min=1)[:, None]
        if (cnt == 0).any():
            pm[cnt == 0] = C[cnt == 0] * X.norm(dim=-1).mean()
        self.proto_mean.copy_(pm)
        mask = torch.zeros(1, X.shape[0], dtype=torch.bool, device=X.device)
        S = self.affinity(X[None], mask)[0]
        self.g_bar.copy_(balanced_duals(S / self.eps, self.c, torch.zeros(self.K, device=X.device), self.iters, self.newton, self.kappa))
        self.share_ema.fill_(1.0 / self.K)
        M = self.membership(S[None], mask)[0]
        within = float((F.normalize(X, dim=-1) * C[lab]).sum(-1).mean())
        gram = C @ C.T
        off = gram[~torch.eye(self.K, dtype=torch.bool, device=X.device)]
        h = M.sum(0) / M.sum()
        return {"mode": mode, "n": int(X.shape[0]), "cluster_min": int(cnt.min()), "cluster_max": int(cnt.max()),
                "within_cos": within, "proto_cos_mean": float(off.mean()), "proto_cos_max": float(off.max()),
                "mass_share_min_xK": float(h.min() * self.K), "topics_per_sentence": float((M > 0.5).sum(-1).float().mean()),
                "top_mass_mean": float(M.max(-1).values.mean()), "s_std": float(S.std()), "g_bar_std": float(self.g_bar.std())}

    # ---- monitors
    @torch.no_grad()
    def proto_stats(self) -> dict:
        p = F.normalize(self.protos, dim=-1)
        gram = p @ p.T
        off = gram[~torch.eye(self.K, dtype=torch.bool, device=p.device)]
        gain = float(self.gain.clamp(self.gain_min, self.gain_max))
        return {"proto_drift": float((p * F.normalize(self.protos0, dim=-1)).sum(-1).mean()),
                "proto_dup_max": float(off.max()), "gain": gain,
                "gain_at_bound": int(gain <= self.gain_min + 1e-6 or gain >= self.gain_max - 1e-6)}

    def assignment_learns(self) -> bool:
        """does any parameter of the sentence-to-topic function currently receive gradient"""
        return bool(self.protos.requires_grad or self.gain.requires_grad or (self.dU is not None and self.dU.weight.requires_grad))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
