# -*- coding: utf-8 -*-
"""Design B's EMA anchor bank (plan section 4.6): the data-side topic statistics of EVERY training game, so that the
loss sees all 1,694 games' anchors at the element's topic as negatives instead of the 63 other games of the batch
(the largest asymmetry against the original I-CE, whose anchors are the whole gallery).

The bank stores, per training game g and topic k, EMAs over that game's successive 512-sentence packs of

    S_gk = sum_i M_ik x_i        (the topic's sentence sum, centred space, fp32, NOT through W)
    n_gk = sum_i M_ik            (the topic's mass)

so the bank anchor a_gk = normalize(W (S_gk + lam pbar_k) / (n_gk + lam)) is always computed with the CURRENT W:
only the gradient from the data side (M, x) is cut, W keeps its gradient, and the bank never goes stale when W
moves.  A pack only updates its own game's rows (no gradient), so the pack forward runs without gradient.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class AnchorBank:
    def __init__(self, n_games: int, K: int, d_in: int, beta: float = 0.05, device=None):
        self.S = torch.zeros(n_games, K, d_in, device=device)
        self.n = torch.zeros(n_games, K, device=device)
        self.seen = torch.zeros(n_games, dtype=torch.bool, device=device)
        self.beta = beta

    @torch.no_grad()
    def update(self, rows: torch.Tensor, M: torch.Tensor, x: torch.Tensor):
        """rows (B,) bank rows of the batch's games; M (B, N, K) memberships and x (B, N, D) centred sentences
        (zero at padding) of their packs.  First visit takes the pack as is, later visits move by beta."""
        S_b = torch.einsum("bnk,bnd->bkd", M.float(), x.float())
        n_b = M.float().sum(1)
        new = ~self.seen[rows]
        self.S[rows] = torch.where(new[:, None, None], S_b, (1.0 - self.beta) * self.S[rows] + self.beta * S_b)
        self.n[rows] = torch.where(new[:, None], n_b, (1.0 - self.beta) * self.n[rows] + self.beta * n_b)
        self.seen[rows] = True

    def anchors(self, model, rows: torch.Tensor | None = None):
        """unit anchors (G or len(rows), K, D) through the model's current W (gradient to W only) and their
        coverage omega = n / (n + lam)"""
        S = self.S if rows is None else self.S[rows]
        n = self.n if rows is None else self.n[rows]
        mbar = (S + model.lam * model.proto_mean[None]) / (n[..., None] + model.lam)
        if model.film_g is not None:
            mbar = mbar * (1.0 + model.film_g[None]) + model.film_b[None]
        return F.normalize(model.W(mbar), dim=-1), n / (n + model.lam)

    def state_dict(self) -> dict:
        return {"S": self.S, "n": self.n, "seen": self.seen, "beta": self.beta}

    @torch.no_grad()
    def stats(self) -> dict:
        cov = self.n / (self.n + 1.0)
        return {"bank_seen_frac": float(self.seen.float().mean()), "bank_cov_mean": float(cov[self.seen].mean()) if self.seen.any() else 0.0,
                "bank_topics_covered": float((cov[self.seen] > 0.5).float().sum(-1).mean()) if self.seen.any() else 0.0}


@torch.no_grad()
def fill_bank(bank: AnchorBank, model, sampler, train_games, row_of_game: torch.Tensor, B: int = 64, log=None):
    """one pack of every training game into the bank (its first visit), before training starts"""
    dev = bank.S.device
    games = [int(g) for g in train_games]
    for i in range(0, len(games), B):
        gids = games[i:i + B]
        batch = sampler.sample(gids=gids)
        xw = torch.from_numpy(batch["xw"]).to(dev)
        lw = torch.as_tensor(batch["lw"], device=dev)
        mw = torch.arange(xw.shape[1], device=dev)[None, :] >= lw[:, None]
        xw_c = model.map_space(xw, mw)
        M = model.membership(model.affinity(xw_c, mw), mw)
        bank.update(row_of_game[torch.as_tensor(gids, device=dev)], M, xw_c)
        if log is not None and (i // B) % 10 == 0:
            log({"event": "bank_fill", "games": min(i + B, len(games))})


def routed_bank_loss(zs, mask_s, M_s, rows, A, cov_A, batch_rows, *, tau: float, margin: float, sigma: float, own_neg: bool = True):
    """Design B's I-CE against the bank (plan 4.6).  zs (B, Ns, D): strong-view elements in the anchors' space;
    M_s (B, Ns, K) their memberships (positive sets P_i = {k : M > 0.5}, argmax fallback, stop-gradient);
    rows (B,): the batch games' bank rows; A (G, K, D) the bank anchors through the current W; cov_A (G, K) their
    coverage; batch_rows (B,) = rows.  For element i of game b in topic k the positive is A[rows[b], k]; the negatives
    are (i) the other G - 1 games' bank anchors at topic k, weighted by their coverage, and (ii) the batch games'
    bank anchors at the other topics (the own game's other topics too when own_neg), weighted by coverage.
    Returns (loss, stats)."""
    B, Ns, D = zs.shape
    G, K, _ = A.shape
    dev = zs.device
    with torch.no_grad():
        zn = F.normalize(zs.float(), dim=-1)
        zhat = F.normalize(zn + sigma * torch.randn_like(zn), dim=-1) if sigma > 0 else zn
        valid = ~mask_s
        Ms = M_s.detach()
        P = Ms > 0.5
        none = (~P.any(-1)) & valid
        if none.any():
            P = P | (F.one_hot(Ms.argmax(-1), K).bool() & none[..., None])
        P = P & valid[..., None]                                                    # (B, Ns, K)
        logw_A = torch.log(cov_A.detach().float().clamp(min=1e-6))                  # (G, K)
        own_row = rows[:, None].expand(B, Ns)                                        # (B, Ns)
        bg = rows.repeat_interleave(K)                                              # (B K,) game row of every batch anchor column
        bt = torch.arange(K, device=dev).repeat(B)                                  # (B K,) topic of every batch anchor column
    A_batch = A[batch_rows].reshape(B * K, D)                                       # (B K, D) with gradient
    sims_batch = torch.einsum("bnd,jd->bnj", zhat, A_batch)                         # (B, Ns, B K)
    logw_batch = logw_A[batch_rows].reshape(-1)                                     # (B K,)
    total = torch.zeros((), device=dev)
    wsum = torch.zeros((), device=dev)
    pos_cos_acc = torch.zeros((), device=dev)
    prob_acc = torch.zeros((), device=dev)
    for k in range(K):
        sel = P[:, :, k]                                                            # (B, Ns) elements in topic k
        if not sel.any():
            continue
        X = zhat[sel]                                                               # (n_k, D)
        own = own_row[sel]                                                          # (n_k,) own bank rows
        sims_same = X @ A[:, k].T                                                   # (n_k, G) same topic, every training game
        pos = sims_same.gather(1, own[:, None])[:, 0]                               # (n_k,)
        neg_same = (sims_same / tau + logw_A[:, k][None]).scatter(1, own[:, None], float("-inf"))
        lse_same = torch.logsumexp(neg_same, dim=-1)
        nb = (sims_batch[sel] / tau + logw_batch[None])                             # (n_k, B K) other topics of the batch games
        excl = (bt == k)[None, :].expand_as(nb).clone()                             # topic k columns are already in the bank set
        same_game = bg[None, :] == own[:, None]                                     # (n_k, B K)
        if own_neg:
            Pk = P[sel]                                                             # (n_k, K) the element's own positive topics
            excl |= same_game & Pk[:, bt]                                           # own game's other positive topics drop out
        else:
            excl |= same_game
        lse_other = torch.logsumexp(nb.masked_fill(excl, float("-inf")), dim=-1)
        lse = torch.logaddexp(lse_same, lse_other)
        logit_pos = (pos - margin) / tau
        nll = torch.logaddexp(logit_pos, lse) - logit_pos
        w = cov_A[own, k].detach().float()
        total = total + (nll * w).sum()
        wsum = wsum + w.sum()
        with torch.no_grad():
            pos_cos_acc = pos_cos_acc + (pos * w).sum()
            prob_acc = prob_acc + (torch.exp(-nll) * w).sum()
    wsum = wsum.clamp(min=1e-6)
    loss = total / wsum
    with torch.no_grad():
        stats = {"pos_prob": float(prob_acc / wsum), "pos_cos": float(pos_cos_acc / wsum),
                 "n_pos_per_elem": float(P.sum(-1)[valid].float().mean()) if valid.any() else 0.0}
    return loss, stats
