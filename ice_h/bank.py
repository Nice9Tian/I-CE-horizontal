# -*- coding: utf-8 -*-
"""Design B's whole-gallery anchors (plan section 4.5): the data-side topic statistics of EVERY training game, so that
the loss sees all 1,694 games' anchors at the element's topic instead of the 63 other games of the batch (the largest
asymmetry against the original I-CE, whose anchors are the whole gallery).

The bank stores, per training game g and topic k,

    S_gk = sum_i M_ik x_i        (the topic's sentence sum, centred space, fp32, NOT through W)
    n_gk = sum_i M_ik            (the topic's mass)

either of one FIXED 512-sentence pack per game (--gallery full, the original I-CE's gallery) or as EMAs over the
game's successive sampled packs (--bank ema).  The bank anchor a_gk = normalize(W (S_gk + lam pbar_k) / (n_gk + lam))
is always computed with the CURRENT W: only the gradient from the data side (M, x) is cut, W keeps its gradient, and
the bank never goes stale when W moves.  The assignment (prototypes, gain, U) must be frozen while a bank is used
with the sentence-level loss: the bank gives the assignment no gradient (plan 4.5).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def log_weights(cov: torch.Tensor) -> torch.Tensor:
    """log coverage weights of candidate anchors; a zero-coverage anchor (pure prior) gets exactly -inf, so it leaves
    the denominator instead of entering it with weight 1e-6 (at tau = 0.03 that would still be a 0.4-cosine discount)"""
    c = cov.detach().float()
    return torch.where(c > 1e-8, c.clamp(min=1e-8).log(), torch.full_like(c, float("-inf")))


class AnchorBank:
    def __init__(self, n_games: int, K: int, d_in: int, beta: float = 0.1, device=None):
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
        """unit anchors (G or len(rows), K, D) through the model's current W (gradient to W and FiLM only) and
        their coverage omega = n / (n + lam)"""
        S = self.S if rows is None else self.S[rows]
        n = self.n if rows is None else self.n[rows]
        mbar = (S + model.lam * model.proto_mean[None]) / (n[..., None] + model.lam)
        if model.film_g is not None:
            mbar = mbar * (1.0 + model.film_g[None]) + model.film_b[None]
        return F.normalize(model.W(mbar), dim=-1), n / (n + model.lam)

    def state_dict(self) -> dict:
        return {"S": self.S, "n": self.n, "seen": self.seen, "beta": self.beta}

    @torch.no_grad()
    def stats(self, lam: float = 1.0) -> dict:
        cov = self.n / (self.n + lam)
        seen = self.seen
        has = (self.n > 1.0) & seen[:, None]                                        # topics with mass, of games seen
        return {"bank_seen_frac": float(seen.float().mean()),
                "bank_cov_mean": float(cov[has].mean()) if has.any() else 0.0,      # over topics with mass, like cov_mean_w
                "bank_topics_covered": float((cov[seen] > 0.5).float().sum(-1).mean()) if seen.any() else 0.0}


@torch.no_grad()
def fill_bank(bank: AnchorBank, model, sampler, train_games, row_of_game: torch.Tensor, B: int = 64, log=None, reset: bool = True):
    """one freshly sampled pack of every training game into the bank (--bank ema: the cold start, and --bank-refill).
    The packs come from sampler.sample_packs: the store-page prefix is always included (the training sampler drops
    it only when the strong view IS the store page, which the bank must not inherit)."""
    dev = bank.S.device
    if reset:
        bank.seen.zero_()
    games = [int(g) for g in train_games]
    for i in range(0, len(games), B):
        gids = games[i:i + B]
        batch = sampler.sample_packs(gids)
        xw = torch.from_numpy(batch["xw"]).to(dev)
        lw = torch.as_tensor(batch["lw"], device=dev)
        mw = torch.arange(xw.shape[1], device=dev)[None, :] >= lw[:, None]
        xw_c = model.map_space(xw, mw)
        M = model.membership(model.affinity(xw_c, mw), mw)
        bank.update(row_of_game[torch.as_tensor(gids, device=dev)], M, xw_c)
        if log is not None and (i // B) % 10 == 0:
            log({"event": "bank_fill", "games": min(i + B, len(games))})


@torch.no_grad()
def fill_bank_from_packs(bank: AnchorBank, model, X, L, B: int = 64):
    """the bank rows from FIXED packs (--gallery full): X fp16 (G, cap, D) numpy or tensor (CPU is fine, chunks are
    streamed to the device), L (G,).  Every row is replaced (first-visit semantics), so calling it again recomputes
    the gallery under the current assignment."""
    dev = bank.S.device
    G = X.shape[0]
    bank.seen.zero_()
    Lt = torch.as_tensor(L, device=dev)
    for i in range(0, G, B):
        xb = X[i:i + B]
        xb = xb if torch.is_tensor(xb) else torch.from_numpy(xb)
        n = int(Lt[i:i + B].max())
        xb = xb[:, :n].to(dev)
        mb = torch.arange(n, device=dev)[None, :] >= Lt[i:i + B][:, None]
        xc = model.map_space(xb, mb)
        M = model.membership(model.affinity(xc, mb), mb)
        bank.update(torch.arange(i, min(i + B, G), device=dev), M, xc)


def routed_bank_loss(zs, mask_s, M_s, rows, A, cov_A, *, tau: float, margin: float, sigma: float, own_neg: bool = True, elem_w=None):
    """Design B's I-CE against the whole gallery (plan 4.5).  zs (B, Ns, D): the strong view's elements in the
    anchors' space (sentences, or its topic anchors for the view-level reading: the gradient of zs is kept);
    mask_s True = padding; M_s (B, Ns, K) memberships (positive sets P_i = {k : M > 0.5}, argmax fallback,
    stop-gradient); rows (B,): the batch games' bank rows; A (G, K, D) the bank anchors through the current W;
    cov_A (G, K) their coverage; elem_w (B, Ns) optional extra pair weights (the strong view's topic shares for the
    view-level reading).  For element i of game b in topic k the positive is A[rows[b], k]; the negatives are (i) the
    other G - 1 games' bank anchors at topic k and (ii) the batch games' bank anchors at the other topics (the own
    game's other topics too when own_neg), every negative weighted by its coverage (zero coverage = excluded).  Pair
    weight = own coverage at k (x elem_w).  The same-topic part is computed per topic over contiguous slices of the
    (element, topic) pairs sorted by topic: one host sync per step.  Returns (loss, stats)."""
    B, Ns, D = zs.shape
    G, K, _ = A.shape
    dev = zs.device
    zn = F.normalize(zs.float(), dim=-1)                                            # gradient kept when zs has one
    if sigma > 0:
        with torch.no_grad():
            noise = sigma * torch.randn_like(zn)
        zhat = F.normalize(zn + noise, dim=-1)
    else:
        zhat = zn
    with torch.no_grad():
        valid = ~mask_s
        Ms = M_s.detach()
        P = Ms > 0.5
        none = (~P.any(-1)) & valid
        if none.any():
            P = P | (F.one_hot(Ms.argmax(-1), K).bool() & none[..., None])
        P = P & valid[..., None]                                                    # (B, Ns, K)
        logw_A = log_weights(cov_A)                                                 # (G, K)
        own_row = rows[:, None].expand(B, Ns)[valid]                                # (Nv,) own bank row of every valid element
        Pv = P[valid]                                                               # (Nv, K)
        pairs = Pv.nonzero()                                                        # (n_pairs, 2): valid element, topic
        pairs = pairs[pairs[:, 1].argsort(stable=True)]
        e_idx, k_idx = pairs[:, 0], pairs[:, 1]
        counts = torch.bincount(k_idx, minlength=K).tolist()                        # the one host sync
        bg = rows.repeat_interleave(K)                                              # (B K,) game row of every batch anchor column
        bt = torch.arange(K, device=dev).repeat(B)                                  # (B K,) topic of every batch anchor column
        own_p = own_row[e_idx]                                                      # (n_pairs,)
        excl = (bt[None, :] == k_idx[:, None])                                      # topic k columns live in the bank set already
        same_game = bg[None, :] == own_p[:, None]
        excl |= same_game & (Pv[e_idx][:, bt] if own_neg else torch.ones_like(same_game))   # own game's positive topics (or all)
        w = cov_A.detach().float()[own_p, k_idx]
        if elem_w is not None:
            w = w * elem_w.detach().float()[valid][e_idx]
    Zv = zhat[valid]                                                                # (Nv, D)
    X_p = Zv[e_idx]                                                                 # (n_pairs, D)
    A_batch = A[rows].reshape(B * K, D)                                             # (B K, D) with gradient
    nb = X_p @ A_batch.T / tau + logw_A[rows].reshape(-1)[None, :]                 # (n_pairs, B K)
    lse_other = torch.logsumexp(nb.masked_fill(excl, float("-inf")), dim=-1)        # (n_pairs,)
    pos_parts, lse_parts = [], []
    off = 0
    for k, c in enumerate(counts):
        if c == 0:
            continue
        sl = slice(off, off + c)
        off += c
        ss = X_p[sl] @ A[:, k].T                                                    # (c, G): topic k, every training game
        own_k = own_p[sl, None]
        pos_parts.append(ss.gather(1, own_k)[:, 0])
        lse_parts.append(torch.logsumexp((ss / tau + logw_A[:, k][None, :]).scatter(1, own_k, float("-inf")), dim=-1))
    if not pos_parts:                                                               # no valid element at all
        return zhat.sum() * 0.0, {"pos_prob": 0.0, "pos_cos": 0.0, "n_pos_per_elem": 0.0}
    pos = torch.cat(pos_parts)
    lse = torch.logaddexp(torch.cat(lse_parts), lse_other)
    logit_pos = (pos - margin) / tau
    nll = torch.logaddexp(logit_pos, lse) - logit_pos
    wsum = w.sum().clamp(min=1e-6)
    loss = (nll * w).sum() / wsum
    with torch.no_grad():
        stats = {"pos_prob": float((torch.exp(-nll) * w).sum() / wsum), "pos_cos": float((pos * w).sum() / wsum),
                 "n_pos_per_elem": float(P.sum(-1)[valid].float().mean()) if valid.any() else 0.0}
    return loss, stats
