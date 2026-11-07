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
    out["dup_pair_frac"] = float((off > 0.95).float().mean())                   # near-duplicate anchors inside a game
    out["eff_anchors"] = float((1.0 / (gram > 0.95).float().sum(-1)).sum(-1).mean())   # distinct anchors per game (duplicates merged)
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


# ============================================================================ design B: topic-routed losses (plan 4.1, 4.3)
def routed_ice_loss(zs, mask_s, M_s, a_w, cov_w, a_s=None, cov_s=None, *, tau: float, margin: float, sigma: float,
                    own_neg: bool = True, neg_cov: bool = True):
    """Design B's I-CE.  zs: strong-view elements (B, Ns, D) in the anchors' space (centred); M_s their memberships
    (used stop-gradient for the positive sets P_i = {k: M > 0.5}, argmax fallback); a_w (B, K, D): the same games'
    PACK anchors = the positives at the element's topics and the other games' negatives; cov_w (B, K): the packs'
    topic coverage n / (n + lam) = the positive pair weights omega; a_s / cov_s (B, K, D) / (B, K) or None: the
    strong-view anchors and their coverage, extra negatives for the OTHER games (never the own game's).
    neg_cov: every negative is weighted by its own coverage (exp(s/tau) * omega_j), so an anchor that is only the
    topic prior (an uncovered topic) drops out of the denominator instead of pushing W off the topic centres.
    own_neg: the own pack's anchors at the topics the element is not in count as negatives.  The negative sum is
    the same for every positive topic of an element, so it is one logsumexp over the (elements x candidates)
    matrix.  Returns (loss, stats)."""
    B, Ns, D = zs.shape
    K = a_w.shape[1]
    dev = zs.device
    use_s = a_s is not None
    J = (2 if use_s else 1) * B * K
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
        omega = cov_w.detach().float()[:, None, :] * P.float()                      # (B, Ns, K)
        game = torch.arange(B, device=dev)
        cand_game = (torch.arange(J, device=dev) // K) % B                          # (J,) game of candidate j
        same = cand_game[None, :] == game[:, None]                                  # (B, J)
        is_weak = torch.arange(J, device=dev) < B * K
        excl = (same & ~is_weak[None, :])[:, None, :].expand(B, Ns, J).clone()      # the own strong anchors never count
        if own_neg:
            for b in range(B):
                excl[b, :, b * K:(b + 1) * K] = P[b]                                # own pack: only the positive topics drop out
        else:
            excl |= (same & is_weak[None, :])[:, None, :]                           # own pack entirely out of the negatives
        if neg_cov:
            cov_j = cov_w.detach().float().reshape(-1)
            if use_s:
                cov_j = torch.cat([cov_j, (cov_s if cov_s is not None else torch.ones(B, K, device=dev)).detach().float().reshape(-1)])
            logw = torch.log(cov_j.clamp(min=1e-6))                                 # (J,)
        else:
            logw = torch.zeros(J, device=dev)
    cands = a_w.reshape(B * K, D) if not use_s else torch.cat([a_w.reshape(B * K, D), a_s.reshape(B * K, D)], 0)
    sims = torch.einsum("bnd,jd->bnj", zhat, cands)                                 # (B, Ns, J)
    pos = torch.stack([sims[b, :, b * K:(b + 1) * K] for b in range(B)])            # (B, Ns, K): the own pack's anchors
    lse_neg = torch.logsumexp((sims / tau + logw).masked_fill(excl, float("-inf")), dim=-1)   # (B, Ns)
    logit_pos = (pos - margin) / tau
    nll = torch.logaddexp(logit_pos, lse_neg[..., None]) - logit_pos                # -log p_ik
    wsum = omega.sum().clamp(min=1e-6)
    loss = (nll * omega).sum() / wsum
    with torch.no_grad():
        stats = {"pos_prob": float((torch.exp(-nll) * omega).sum() / wsum),
                 "pos_cos": float((pos * omega).sum() / wsum),
                 "n_pos_per_elem": float(P.sum(-1)[valid].float().mean()) if valid.any() else 0.0,
                 "pos_cov": float((cov_w.detach()[:, None, :] * P).sum() / P.sum().clamp(min=1))}
    return loss, stats


def routed_anchor_loss(a_s, cov_s, h_s, a_w, cov_w, *, tau: float, margin: float, own_neg: bool = True, use_strong_neg: bool = True):
    """The view-level routed I-CE (the original I-CE's structure lifted to topics): the strong view's topic anchor
    a^{s,b}_k is the query, its positive is the same game's pack anchor a^{w,b}_k, the negatives are the other games'
    pack anchors (all topics) and strong anchors, each weighted by its coverage, plus the own pack's other topics
    (own_neg).  Pair weights h^s_bk omega^w_bk (stop-gradient), as in the invariance term.  Returns (loss, stats)."""
    B, K, D = a_s.shape
    dev = a_s.device
    J = (2 if use_strong_neg else 1) * B * K
    with torch.no_grad():
        w = (h_s * cov_w).float()                                                      # (B, K) pair weights
        cov_j = cov_w.float().reshape(-1)
        if use_strong_neg:
            cov_j = torch.cat([cov_j, cov_s.float().reshape(-1)])
        logw = torch.log(cov_j.clamp(min=1e-6))                                        # (J,)
        game_q = torch.arange(B, device=dev).repeat_interleave(K)                      # (BK,) game of each query anchor
        cand_game = (torch.arange(J, device=dev) // K) % B
        is_weak = torch.arange(J, device=dev) < B * K
        same = cand_game[None, :] == game_q[:, None]                                    # (BK, J)
        excl = same & ~is_weak[None, :]                                                # own strong anchors never count
        own_pos = torch.zeros(B * K, J, dtype=torch.bool, device=dev)
        own_pos[torch.arange(B * K, device=dev), torch.arange(B * K, device=dev)] = True   # own pack anchor at the same topic
        excl |= own_pos
        if not own_neg:
            excl |= same & is_weak[None, :]
    cands = a_w.reshape(B * K, D) if not use_strong_neg else torch.cat([a_w.reshape(B * K, D), a_s.reshape(B * K, D)], 0)
    q = a_s.reshape(B * K, D)
    sims = q @ cands.T                                                                  # (BK, J)
    pos = (q * a_w.reshape(B * K, D)).sum(-1)                                           # (BK,)
    lse_neg = torch.logsumexp((sims / tau + logw).masked_fill(excl, float("-inf")), dim=-1)
    logit_pos = (pos - margin) / tau
    nll = torch.logaddexp(logit_pos, lse_neg) - logit_pos
    wf = w.reshape(-1)
    wsum = wf.sum().clamp(min=1e-6)
    loss = (nll * wf).sum() / wsum
    with torch.no_grad():
        stats = {"apos_prob": float((torch.exp(-nll) * wf).sum() / wsum), "apos_cos": float((pos * wf).sum() / wsum)}
    return loss, stats


def routed_inv_loss(a_s, a_w, h_s, cov_w):
    """plan 4.3: 1 - cos(a^s_k, sg(a^w_k)) over the topics both views talk about, weights h^s_k cov^w_k (stop-gradient)"""
    w = (h_s * cov_w).detach().float()
    return (((1.0 - (a_s * a_w.detach()).sum(-1)) * w).sum() / w.sum().clamp(min=1e-6))


def _eff_topics(h: torch.Tensor) -> torch.Tensor:
    """exp(entropy) of each row of a distribution matrix"""
    p = h.clamp(min=1e-9)
    return torch.exp(-(p * p.log()).sum(-1))


@torch.no_grad()
def topic_monitors(ow: dict, os_: dict, mask_w, mask_s, ts, model, l_inv: float) -> dict:
    """plan section 10 for design B, on one batch.  ow / os_: the aggregate dicts (with M) of the pack and strong views;
    ts (B, Ns) element types of the strong view (0 = review) or None; model: the TopicCodebook (prototype stats)."""
    a = ow["anchors"]
    B, K, _ = a.shape
    dev = a.device
    out = {"eff_topics_w": float(_eff_topics(ow["hist"]).mean())}
    e_s = _eff_topics(os_["hist"])
    nonempty = (~mask_s).any(1)                                                     # an empty strong view is neither kind
    if ts is None:
        review, doc = nonempty, torch.zeros_like(nonempty)
    else:
        t0 = torch.as_tensor(np.asarray(ts)[:, 0] if np.asarray(ts).shape[1] else np.full(B, -1), device=dev)
        review, doc = nonempty & (t0 == 0), nonempty & (t0 > 0)
    out["eff_topics_s_review"] = float(e_s[review].mean()) if review.any() else float("nan")
    out["eff_topics_s_doc"] = float(e_s[doc].mean()) if doc.any() else float("nan")
    out["topics_per_elem_w"] = float((ow["M"] > 0.5).sum(-1)[~mask_w].float().mean())
    out["topics_per_elem_s"] = float((os_["M"] > 0.5).sum(-1)[~mask_s].float().mean()) if (~mask_s).any() else 0.0
    has_mass = ow["mass"] > 1.0
    out["cov_mean_w"] = float(ow["cov"][has_mass].mean()) if has_mass.any() else 0.0   # coverage over the topics the pack has mass in
    out["topics_covered_w"] = float((ow["cov"] > 0.5).float().sum(-1).mean())
    out["stability"] = 1.0 - float(l_inv)
    cov = ow["cov"] > 0.5                                                           # (B, K)
    g_k = torch.einsum("bkd,ckd->kbc", a, a)                                        # (K, B, B) same topic, two games
    pair = cov.T[:, :, None] & cov.T[:, None, :] & ~torch.eye(B, dtype=torch.bool, device=dev)[None]
    out["xgame_same_topic_cos"] = float(g_k[pair].mean()) if pair.any() else float("nan")
    g_b = torch.einsum("bkd,bjd->bkj", a, a)                                        # (B, K, K) same game, two topics
    pair2 = cov[:, :, None] & cov[:, None, :] & ~torch.eye(K, dtype=torch.bool, device=dev)[None]
    out["xtopic_same_game_cos"] = float(g_b[pair2].mean()) if pair2.any() else float("nan")
    out.update(model.proto_stats())
    out.update(balance_err_max=model.last_balance_err[0], balance_err_mean=model.last_balance_err[1])
    return out
