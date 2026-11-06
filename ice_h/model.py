# -*- coding: utf-8 -*-
"""The latent array: K shared latents read a set of frozen sentence embeddings through
cross-attention (no attention among elements, O(N*K)); the last round's affinities go
through a box-constrained (Fermi-Dirac) Sinkhorn projection that defines the membership
matrix M in [0, 1], which is also the last round's attention weight.  Anchors are unit vectors in the (row-standardised)
Qwen space: a_k = normalize(W_a g_k).

Shapes: x (B, N, 1024) float, mask (B, N) bool with True = padding.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e9   # "minus infinity" in the log domain; finite so that neg - neg stays 0, exp(neg) is 0


def log_sinkhorn(S: torch.Tensor, mask: torch.Tensor, c: float, eps: float, iters: int, rho: float | None = None):
    """Entropic optimal transport between elements (supply c each) and K groups (capacity N c / K each).

    S     (B, N, K) affinities; the plan is argmin <Q, -S> + eps KL(Q || 1) under the marginals
    mask  (B, N) True = padding; padded rows carry no mass
    rho   None = balanced columns; finite = the column marginal is a KL penalty of weight rho
          (unbalanced Sinkhorn: the column scaling is damped by rho / (rho + eps))
    Returns M (B, N, K), differentiable through the iterations.  Rows of real elements sum to c;
    columns sum to N_b c / K in the balanced case.
    """
    B, N, K = S.shape
    S = S.float()
    valid = ~mask
    n_b = valid.sum(1).clamp(min=1).float()                                   # (B,)
    logK = (S / eps).masked_fill(mask[..., None], NEG)                        # (B, N, K)
    log_a = torch.full((B, N), math.log(c), device=S.device).masked_fill(mask, NEG)
    log_b = torch.log(n_b * c / K)[:, None]                                   # (B, 1)
    damp = 1.0 if rho is None else rho / (rho + eps)
    f = torch.zeros(B, N, device=S.device)
    for _ in range(iters):
        g = damp * (log_b - torch.logsumexp(logK + f[:, :, None], dim=1))     # (B, K) column scaling
        f = log_a - torch.logsumexp(logK + g[:, None, :], dim=2)              # (B, N) row scaling, applied last:
    M = torch.exp(logK + f[:, :, None] + g[:, None, :])                       # every real element carries mass c exactly
    return M.masked_fill(mask[..., None], 0.0)


@torch.no_grad()
def transport_errors(M: torch.Tensor, mask: torch.Tensor, c: float) -> tuple[float, float]:
    """largest |row sum - c| over real elements and largest relative column error, for monitoring convergence"""
    valid = ~mask
    n_b = valid.sum(1).clamp(min=1).float()
    K = M.shape[-1]
    row_err = (M.sum(-1) - c).abs().masked_fill(mask, 0).max()
    col_err = ((M.sum(1) - (n_b * c / K)[:, None]).abs() / (n_b * c / K)[:, None]).max()
    return float(row_err), float(col_err)


def _shift_for_sum(X: torch.Tensor, target: torch.Tensor, dim: int, newton: int) -> torch.Tensor:
    """the shift s (broadcast along ``dim``) with sum_dim sigmoid(X + s) = target, for a monotone 1-D problem per slice.

    Start from order statistics: with t = floor(target), s0 = -(x_(t) + x_(t+1)) / 2 puts exactly the t largest
    entries above zero, so the saturated sum is already t; a few damped Newton steps then fix the fractional part.
    Padded entries (X = NEG) sort last and never count.  Runs without gradient (the duals are treated as constants).
    """
    xs = X.sort(dim=dim, descending=True).values
    n = X.shape[dim]
    t = target.floor().long().clamp(min=1, max=n - 1)                      # shape of target (broadcast of the slice index)
    shape = list(X.shape); shape[dim] = 1
    t = t.expand(shape) if t.dim() == X.dim() else t.reshape(shape)
    hi = xs.gather(dim, t - 1); lo = xs.gather(dim, t)
    s = -(hi + lo) / 2
    for _ in range(newton):
        p_ = torch.sigmoid(X + s)
        h = p_.sum(dim, keepdim=True) - target.reshape(shape) if target.dim() != X.dim() else p_.sum(dim, keepdim=True) - target
        dh = (p_ * (1 - p_)).sum(dim, keepdim=True).clamp(min=1e-4)
        s = s - (h / dh).clamp(-5.0, 5.0)
    return s


def fd_sinkhorn(S: torch.Tensor, mask: torch.Tensor, c: float, eps: float, iters: int, rho: float | None = None,
                newton: int = 8):
    """Box-constrained entropic transport: M_ik = sigmoid(S_ik / eps + f_i + g_k) in [0, 1].

    The regulariser is the Fermi-Dirac entropy sum[Q log Q + (1 - Q) log(1 - Q)], so every entry is capped at 1
    by construction; rows (real elements) sum to c, columns to N_b c / K.  With c = 2 an element therefore
    spreads its mass over at least two groups, which the plain entropic plan cannot do: without the cap its
    optimum is exactly c times the unit-row plan (Q* = c P*), and a post-hoc min(1, .) clamp breaks the
    marginals.  Alternating block solves of the dual (columns, then rows last so rows are exact); the dual
    variables are computed without gradient, so the gradient reaches S only through the sigmoid.
    """
    B, N, K = S.shape
    assert c < K, "the per-element mass must be below the number of groups"
    X = (S.float() / eps).masked_fill(mask[..., None], NEG)                   # padded rows -> sigmoid 0
    valid = ~mask
    n_b = valid.sum(1).clamp(min=1).float()
    t_col = (n_b * c / K)[:, None, None]                                      # (B, 1, 1)
    t_row = torch.full((B, N, 1), float(c), device=S.device)
    damp = 1.0 if rho is None else rho / (rho + eps)
    with torch.no_grad():
        Xd = X.detach()
        f = torch.zeros(B, N, 1, device=S.device)
        g = torch.zeros(B, 1, K, device=S.device)
        for _ in range(iters):
            g = damp * _shift_for_sum(Xd + f, t_col, dim=1, newton=newton)      # columns
            f = _shift_for_sum(Xd + g, t_row, dim=2, newton=newton).masked_fill(mask[..., None], 0.0)   # rows, last
    M = torch.sigmoid(X + f + g)
    return M.masked_fill(mask[..., None], 0.0)


class CrossAttnRound(nn.Module):
    """pre-norm multi-head cross-attention from the latents to the elements, then a per-latent MLP"""

    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        assert d_model % heads == 0
        self.h = heads
        self.ln_q = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.ln_m = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d_model, d_model))
        self.drop = nn.Dropout(dropout)
        self.p = dropout

    def forward(self, g, k, v, mask):
        B, K, d = g.shape
        N = k.shape[1]
        dh = d // self.h
        q = self.q(self.ln_q(g)).view(B, K, self.h, dh).transpose(1, 2)        # (B, h, K, dh)
        kk = k.view(B, N, self.h, dh).transpose(1, 2)
        vv = v.view(B, N, self.h, dh).transpose(1, 2)
        attend = (~mask)[:, None, None, :]                                       # True = may attend
        a = F.scaled_dot_product_attention(q, kk, vv, attn_mask=attend, dropout_p=self.p if self.training else 0.0)
        g = g + self.drop(self.o(a.transpose(1, 2).reshape(B, K, d)))
        g = g + self.drop(self.mlp(self.ln_m(g)))
        return g


class SinkhornRound(nn.Module):
    """the last round: single-head affinities -> Sinkhorn membership M -> M as the attention weight"""

    def __init__(self, d_model: int, dropout: float, c: float, eps: float, iters: int, rho: float | None, ot: str = "fermi"):
        super().__init__()
        assert ot in ("fermi", "exp")
        self.ot = ot
        self.ln_q = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.ln_m = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d_model, d_model))
        self.drop = nn.Dropout(dropout)
        self.c, self.eps, self.iters, self.rho = c, eps, iters, rho
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, g, k, v, mask):
        q = self.q(self.ln_q(g))                                                  # (B, K, d)
        S = torch.einsum("bkd,bnd->bnk", q, k) * self.scale                       # (B, N, K) affinities
        if self.ot == "fermi":
            M = fd_sinkhorn(S, mask, self.c, self.eps, self.iters, self.rho)
            self.last_errors = transport_errors(M, mask, self.c)
        else:                                                                     # legacy: plain entropic plan + clamp
            M_raw = log_sinkhorn(S, mask, self.c, self.eps, self.iters, self.rho)
            self.last_errors = transport_errors(M_raw, mask, self.c)
            M = M_raw.clamp(max=1.0)
        A = M / M.sum(1, keepdim=True).clamp(min=1e-6)                            # column-normalised weights
        agg = torch.einsum("bnk,bnd->bkd", A.to(v.dtype), v)
        g = g + self.drop(self.o(agg))
        g = g + self.drop(self.mlp(self.ln_m(g)))
        return g, S, M


class LatentArray(nn.Module):
    """K latents -> T rounds of cross-attention (the last one through Sinkhorn) -> K unit anchors in Qwen space"""

    def __init__(self, K: int = 6, d_in: int = 1024, d_model: int = 256, heads: int = 8, rounds: int = 3,
                 dropout: float = 0.1, c: float = 2.0, eps: float = 0.05, sinkhorn_iters: int = 10,
                 rho: float = 0.0, self_attn: bool = False, ot: str = "fermi"):
        super().__init__()
        assert rounds >= 1
        self.K, self.d_in, self.d_model = K, d_in, d_model
        self.g0 = nn.Parameter(torch.randn(K, d_model) * 0.02)
        self.k = nn.Linear(d_in, d_model)
        self.v = nn.Linear(d_in, d_model)
        self.rounds = nn.ModuleList([CrossAttnRound(d_model, heads, dropout) for _ in range(rounds - 1)])
        self.final = SinkhornRound(d_model, dropout, c, eps, sinkhorn_iters, rho if rho and rho > 0 else None, ot=ot)
        self.sa = None
        if self_attn:   # ablation: attention among the latents after every round
            self.sa = nn.ModuleList([nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True) for _ in range(rounds)])
            self.sa_ln = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(rounds)])
        self.ln_a = nn.LayerNorm(d_model)
        self.anchor = nn.Linear(d_model, d_in)

    def _self(self, i, g):
        if self.sa is None:
            return g
        h = self.sa_ln[i](g)
        return g + self.sa[i](h, h, h, need_weights=False)[0]

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict:
        B = x.shape[0]
        x = x.float()
        k, v = self.k(x), self.v(x)
        g = self.g0[None].expand(B, -1, -1)
        for i, r in enumerate(self.rounds):
            g = self._self(i, r(g, k, v, mask))
        g, S, M = self.final(g, k, v, mask)
        g = self._self(len(self.rounds), g)
        a = F.normalize(self.anchor(self.ln_a(g)), dim=-1)
        return {"anchors": a, "M": M, "S": S, "g": g, "ot_row_err": self.final.last_errors[0], "ot_col_err": self.final.last_errors[1]}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
