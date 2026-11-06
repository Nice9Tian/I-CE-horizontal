# -*- coding: utf-8 -*-
"""Losses and training monitors.

L_ICE  every strong-view element i against the anchors: positives P_i = {k : M_ik > 0.5}
       (strong view's own M, stop-gradient; argmax when empty), negatives = the same game's
       other strong anchors (groups i is not in) + every anchor (strong and weak) of every
       other game in the batch.  Negatives are down-weighted by w_kj = 1 - alpha r_kj, with
       r = soft Jaccard of the memberships inside a game and r = max(0, cos) of the group means
       in the frozen space across games; all weights are stop-gradient.
L_inv  slot-wise 1 - cos(a_k^s, sg(a_k^w)).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def soft_jaccard(M: torch.Tensor) -> torch.Tensor:
    """pairwise soft Jaccard between the columns of M (B, N, K) -> (B, K, K)"""
    a = M[:, :, :, None]
    b = M[:, :, None, :]
    inter = torch.minimum(a, b).sum(1)
    union = torch.maximum(a, b).sum(1)
    return inter / union.clamp(min=1e-6)


def group_means(M: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """normalize(M^T z): (B, N, K), (B, N, D) -> (B, K, D)"""
    return F.normalize(torch.einsum("bnk,bnd->bkd", M, z), dim=-1)


def ice_loss(zs, mask_s, a_s, M_s, a_w, M_w, zw, mask_w, *, tau: float, margin: float, alpha_in: float,
             alpha_x: float, sigma: float, decay: bool = True):
    """Returns (loss, stats).  zs/zw: elements (B, N, D) already brought into the anchors' space
    (model.map_space: centred when the model centres); masks True = padding; a_s/a_w anchors (B, K, D) unit;
    M_s/M_w memberships (B, N, K).  The decay references (group means) are computed in that same space."""
    B, Ns, D = zs.shape
    K = a_s.shape[1]
    dev = zs.device
    with torch.no_grad():
        zn = F.normalize(zs.float(), dim=-1)
        zhat = F.normalize(zn + sigma * torch.randn_like(zn), dim=-1) if sigma > 0 else zn
        valid = ~mask_s                                                           # (B, Ns)
        Ms = M_s.detach()
        P = Ms > 0.5
        none = (~P.any(-1)) & valid
        if none.any():
            P = P | (F.one_hot(Ms.argmax(-1), K).bool() & none[..., None])
        P = P & valid[..., None]                                                  # (B, Ns, K)
        # candidate layout: j in [0, BK) strong anchor of game j // K, slot j % K; j in [BK, 2BK) weak
        J = 2 * B * K
        cand_game = (torch.arange(J, device=dev) // K) % B                        # (J,)
        is_weak = torch.arange(J, device=dev) >= B * K
        same = cand_game[None, :] == torch.arange(B, device=dev)[:, None]         # (B, J)
        if decay:
            w_in = 1.0 - alpha_in * soft_jaccard(Ms)                              # (B, K, K)
            mbar_s = group_means(Ms, zn)                                          # (B, K, D)
            mbar_w = group_means(M_w.detach(), F.normalize(zw.float(), dim=-1))
            allbar = torch.cat([mbar_s.reshape(B * K, D), mbar_w.reshape(B * K, D)], 0)
            w_x = 1.0 - alpha_x * torch.einsum("bkd,jd->bkj", mbar_s, allbar).clamp(min=0)   # (B, K, J)
        else:
            w_in = torch.ones(B, K, K, device=dev)
            w_x = torch.ones(B, K, J, device=dev)
        logw = torch.log(w_x.clamp(min=1e-6))
        logw_in = torch.log(w_in.clamp(min=1e-6))
        P_ext = torch.zeros(B, Ns, J, dtype=torch.bool, device=dev)
        for b in range(B):
            logw[b, :, b * K:(b + 1) * K] = logw_in[b]
            P_ext[b, :, b * K:(b + 1) * K] = P[b]
        # exclusions per (b, i, j): same game's weak anchors; same game's strong groups i belongs to
        excl = same[:, None, :] & (is_weak[None, None, :] | P_ext)              # (B, Ns, J)
    anchors = torch.cat([a_s.reshape(B * K, D), a_w.reshape(B * K, D)], 0)        # (J, D), with gradient
    sims = torch.einsum("bnd,jd->bnj", zhat, anchors)                             # (B, Ns, J)
    pos = torch.stack([sims[b, :, b * K:(b + 1) * K] for b in range(B)])          # (B, Ns, K)
    logit_pos = (pos - margin) / tau
    neg = sims[:, :, None, :] / tau + logw[:, None, :, :]                         # (B, Ns, K, J)
    neg = neg.masked_fill(excl[:, :, None, :], float("-inf"))
    lse_neg = torch.logsumexp(neg, dim=-1)                                        # (B, Ns, K)
    nll = torch.logaddexp(logit_pos, lse_neg) - logit_pos                         # -log p_ik
    n_pos = P.sum().clamp(min=1)
    loss = (nll * P).sum() / n_pos
    with torch.no_grad():
        stats = {
            "pos_prob": float((torch.exp(-nll) * P).sum() / n_pos),
            "n_pos_per_elem": float(P.sum(-1)[valid].float().mean()) if valid.any() else 0.0,
            "pos_cos": float((pos * P).sum() / n_pos),
        }
    return loss, stats


def inv_loss(a_s: torch.Tensor, a_w: torch.Tensor) -> torch.Tensor:
    """slot-wise 1 - cos(a^s, sg(a^w)), mean over batch and slots"""
    return (1.0 - (a_s * a_w.detach()).sum(-1)).mean()


@torch.no_grad()
def monitors(M_w, mask_w, tw, M_s, mask_s, ds, a_s, a_w, ts=None) -> dict:
    """the health signals of plan section 10, on one batch"""
    B, Nw, K = M_w.shape
    out = {}
    valid_w, valid_s = ~mask_w, ~mask_s
    mass = M_w.sum(1)                                                              # (B, K)
    share = mass / mass.sum(1, keepdim=True).clamp(min=1e-6)
    out["usage_min"] = float(share.mean(0).min())
    out["usage"] = [round(float(x), 4) for x in share.mean(0)]
    hw, hs = M_w > 0.5, M_s > 0.5
    hard_share = (hw & valid_w[..., None]).float().sum(1) / valid_w.float().sum(1, keepdim=True).clamp(min=1)   # (B, K) hard members per group
    out["hard_usage"] = [round(float(x), 4) for x in hard_share.mean(0)]
    out["hard_usage_min"] = float(hard_share.mean(0).min())
    out["groups_per_elem_w"] = float(hw.sum(-1)[valid_w].float().mean())
    out["groups_per_elem_s"] = float(hs.sum(-1)[valid_s].float().mean()) if valid_s.any() else 0.0
    # distinct source documents per strong-view group
    hs_np, ds_np = hs.cpu().numpy(), np.asarray(ds)
    review_view = np.ones(B, bool) if ts is None else (np.asarray(ts)[:, 0] == 0)   # document views have one source by construction
    docs, empty = [], 0
    for b in np.flatnonzero(review_view):
        for k in range(K):
            sel = hs_np[b, :, k]
            if sel.any():
                docs.append(len(np.unique(ds_np[b][sel])))
            else:
                empty += 1
    docs = np.asarray(docs) if docs else np.zeros(1)
    out["docs_per_group_s"] = float(docs.mean())
    out["groups_single_doc_frac"] = float((docs <= 1).mean())
    out["groups_empty_frac"] = float(empty / max(1, len(docs) + empty))
    out["review_strong_frac"] = float(review_view.mean())
    # share of weak groups containing a description sentence
    tw_t = torch.as_tensor(np.asarray(tw), device=M_w.device) == 1                 # (B, Nw)
    out["desc_mix"] = float((hw & tw_t[..., None]).any(1).float().mean())
    # anchor geometry
    gram = torch.einsum("bkd,bjd->bkj", a_w, a_w)
    off = gram[:, ~torch.eye(K, dtype=torch.bool, device=gram.device)]
    out["intra_cos"] = float(off.mean())
    out["stability"] = float((a_s * a_w).sum(-1).mean())
    flat = a_w.reshape(B * K, -1)
    sim = flat @ flat.T
    game = torch.arange(B, device=sim.device).repeat_interleave(K)
    sim = sim.masked_fill(game[:, None] == game[None, :], -2.0)
    nn_ = sim.max(1).values
    out["xgame_nn_median"] = float(nn_.median())
    out["xgame_nn_frac_gt09"] = float((nn_ > 0.9).float().mean())
    out["xgame_nn_frac_lt05"] = float((nn_ < 0.5).float().mean())
    return out
