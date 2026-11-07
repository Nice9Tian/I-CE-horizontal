# -*- coding: utf-8 -*-
"""Stage A training on one fold at the 512-sentence anchor budget: design B (the shared topic codebook, default) or
design A (the per-game latent array).

    python scripts/train.py --name b64_frozen --K 64 --steps 20000                       # B-frozen: k-means prototypes fixed
    python scripts/train.py --name b64_learn  --K 64 --steps 20000 --freeze-protos 1000  # B-learn: prototypes learn after warmup
    python scripts/train.py --name b64_random --K 64 --steps 0 --protos random --gain-init 14 --gain-max 14   # B3: random codebook at a matched logit spread
    python scripts/train.py --design A --name k6 --K 6 --steps 20000                      # design A

Every --eval-every steps (and at step 0 for design B): encode the cached draw 0 (512-sentence packs of all 2,020
games) and the wiki queries, compute the validation rank score (checkpoint selection: routed scoring for B, slot for
A) and the test-fold hit@1 of every scoring mode (report only), save the checkpoint.  Logs are JSON lines in
runs/<name>/log.jsonl and eval.jsonl; the selection in runs/<name>/selection.json.
"""
import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ice_h.codebook import TopicCodebook  # noqa: E402
from ice_h.data import CACHE_DIR, DATA_DIR, Fixtures, PoolSampler, Prefetcher, load_draw, pool_global_mean, sample_pool_rows  # noqa: E402
from ice_h.evaluate import encode_queries, encode_sets, encode_topics, retrieval_metrics, topic_retrieval_metrics  # noqa: E402
from ice_h.losses import ice_loss, inv_loss, monitors, routed_anchor_loss, routed_ice_loss, routed_inv_loss, topic_monitors  # noqa: E402
from ice_h.model import LatentArray  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--design", default="B", choices=["A", "B"], help="B = shared topic codebook (default), A = per-game latent array")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--out-root", default=str(Path(__file__).resolve().parents[1] / "runs"))
    # shared model settings
    ap.add_argument("--K", type=int, default=None, help="topics (B, default 64) or latents (A, default 6)")
    ap.add_argument("--c", type=float, default=2.0)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--sinkhorn-iters", type=int, default=10, help="outer rounds of the box-constrained solver (8 Newton steps per block)")
    ap.add_argument("--rho", type=float, default=0.0, help="0 = balanced columns; finite = unbalanced (KL penalty)")
    ap.add_argument("--center", dest="center", action="store_true", default=True, help="subtract the training pool's global mean vector from every input (default)")
    ap.add_argument("--no-center", dest="center", action="store_false")
    ap.add_argument("--pool-in-ram", action="store_true", help="read the pool into RAM instead of memory-mapping it")
    ap.add_argument("--weak-no-grad", action="store_true", help="ablation: encode the weak view without gradient")
    # design B
    ap.add_argument("--lam", type=float, default=1.0, help="prior smoothing mass of the topic means")
    ap.add_argument("--beta", type=float, default=0.05, help="EMA rate of the running topic means and mass shares")
    ap.add_argument("--dual-lr", type=float, default=0.2, help="step of the stochastic update of the running column duals")
    ap.add_argument("--gain-init", type=float, default=3.0)
    ap.add_argument("--gain-min", type=float, default=2.0)
    ap.add_argument("--gain-max", type=float, default=8.0)
    ap.add_argument("--freeze-gain", action="store_true", help="keep the affinity gain fixed even when the prototypes learn")
    ap.add_argument("--learn-gain", action="store_true", help="learn the gain even while the prototypes stay frozen")
    ap.add_argument("--no-implicit-grad", action="store_true", help="ablation: stop the gradient of the row duals (no conservation)")
    ap.add_argument("--protos", default="kmeans", choices=["kmeans", "random"], help="codebook initialisation")
    ap.add_argument("--kmeans-games", type=int, default=400)
    ap.add_argument("--kmeans-per-game", type=int, default=500)
    ap.add_argument("--kmeans-iters", type=int, default=25)
    ap.add_argument("--freeze-protos", type=int, default=-1, help="steps the prototypes stay frozen; -1 = always (B-frozen), 0 = never")
    ap.add_argument("--proto-lr-mult", type=float, default=0.1)
    ap.add_argument("--assign-map", action="store_true", help="learn U = I + dU on the assignment side")
    ap.add_argument("--film", action="store_true", help="per-topic FiLM on the topic means")
    ap.add_argument("--balance", default="running", choices=["running", "batch"], help="column duals: running EMA (default) or exact per batch")
    ap.add_argument("--no-own-neg", action="store_true", help="the own pack's other-topic anchors are not negatives")
    ap.add_argument("--no-strong-neg", action="store_true", help="the other games' strong-view anchors are not negatives")
    ap.add_argument("--no-neg-cov", action="store_true", help="ablation: negatives not weighted by their coverage")
    ap.add_argument("--loss-level", default="sentence", choices=["sentence", "anchor", "both"],
                    help="routed I-CE on the strong view's sentences (plan 4.1), on its topic anchors (view level, the original I-CE's structure), or both")
    # design A
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--self-attn", action="store_true", help="A: attention among latents")
    ap.add_argument("--ot", default="fermi", choices=["fermi", "exp"], help="A: membership solver")
    ap.add_argument("--anchor-mode", default="meanmap", choices=["decode", "mean", "meanmap", "hybrid"], help="A: anchor definition")
    ap.add_argument("--alpha-in", type=float, default=0.8, help="A: within-game decay")
    ap.add_argument("--alpha-x", type=float, default=0.8, help="A: cross-game decay")
    ap.add_argument("--no-decay", action="store_true", help="A: no negative decay")
    # views
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--ns-min", type=int, default=64)
    ap.add_argument("--ns-max", type=int, default=160)
    ap.add_argument("--p-doc", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=64)
    # loss
    ap.add_argument("--tau", type=float, default=None, help="B: 0.03; A: 0.07 (design A's original default)")
    ap.add_argument("--margin", type=float, default=None, help="B: 0.05; A: 0.15 (design A's original default)")
    ap.add_argument("--lam-inv", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=0.01, help="noise std per dim on the unit sphere; 0.01 -> cos about 0.95 before/after")
    # optimisation
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--monitor-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--no-eval", action="store_true", help="skip the draw-0 evaluation (checkpoints are still saved; select later with scripts/select_ckpt.py)")
    args = ap.parse_args()
    if args.K is None:
        args.K = 64 if args.design == "B" else 6
    if args.tau is None:
        args.tau = 0.03 if args.design == "B" else 0.07
    if args.margin is None:
        args.margin = 0.05 if args.design == "B" else 0.15
    return args


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def to_dev(batch, dev):
    xw = torch.from_numpy(batch["xw"]).to(dev, non_blocking=True)
    xs = torch.from_numpy(batch["xs"]).to(dev, non_blocking=True)
    lw = torch.as_tensor(batch["lw"], device=dev); ls = torch.as_tensor(batch["ls"], device=dev)
    mw = torch.arange(xw.shape[1], device=dev)[None, :] >= lw[:, None]
    ms = torch.arange(xs.shape[1], device=dev)[None, :] >= ls[:, None]
    return xw, mw, xs, ms


def step_A(model, batch, args, dev):
    xw, mw, xs, ms = to_dev(batch, dev)
    if args.weak_no_grad:
        with torch.no_grad():
            ow = model(xw, mw)
    else:
        ow = model(xw, mw)
    os_ = model(xs, ms)
    xs_c, xw_c = model.map_space(xs, ms), model.map_space(xw, mw)
    l_ice, st = ice_loss(xs_c, ms, os_["anchors"], os_["M"], ow["anchors"], ow["M"], xw_c, mw, tau=args.tau, margin=args.margin,
                         alpha_in=args.alpha_in, alpha_x=args.alpha_x, sigma=args.sigma, decay=not args.no_decay)
    l_inv = inv_loss(os_["anchors"], ow["anchors"])
    return l_ice, l_inv, st, {"ow": ow, "os": os_, "mw": mw, "ms": ms, "xs": xs}


def step_B(model, batch, args, dev):
    xw, mw, xs, ms = to_dev(batch, dev)
    xw_c, xs_c = model.map_space(xw, mw), model.map_space(xs, ms)
    ctx = torch.no_grad() if args.weak_no_grad else contextlib.nullcontext()
    with ctx:
        S_w = model.affinity(xw_c, mw)
    S_s = model.affinity(xs_c, ms)
    g = model.batch_duals([S_w, S_s], [mw, ms]) if args.balance == "batch" else None
    with ctx:
        M_w = model.membership(S_w, mw, g)
        ow = model.aggregate(xw_c, mw, M_w)
    M_s = model.membership(S_s, ms, g)
    os_ = model.aggregate(xs_c, ms, M_s)
    st = {}
    l_ice = torch.zeros((), device=dev)
    if args.loss_level in ("sentence", "both"):
        l_ice, st = routed_ice_loss(xs_c, ms, M_s, ow["anchors"], ow["cov"], None if args.no_strong_neg else os_["anchors"],
                                    None if args.no_strong_neg else os_["cov"], tau=args.tau, margin=args.margin, sigma=args.sigma,
                                    own_neg=not args.no_own_neg, neg_cov=not args.no_neg_cov)
    if args.loss_level in ("anchor", "both"):
        l_a, st_a = routed_anchor_loss(os_["anchors"], os_["cov"], os_["hist"], ow["anchors"], ow["cov"], tau=args.tau, margin=args.margin,
                                       own_neg=not args.no_own_neg, use_strong_neg=not args.no_strong_neg)
        l_ice = l_ice + l_a
        st.update(st_a)
    l_inv = routed_inv_loss(os_["anchors"], ow["anchors"], os_["hist"], ow["cov"])
    ow["M"], os_["M"] = M_w, M_s
    return l_ice, l_inv, st, {"S": [S_w, S_s], "mask": [mw, ms], "x": [xw_c, xs_c], "M": [M_w, M_s], "g": g,
                              "ow": ow, "os": os_, "mw": mw, "ms": ms, "xs": xs, "S_s": S_s}


def main():
    args = parse()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda")
    out = Path(args.out_root) / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1), encoding="utf-8")
    log_f = open(out / "log.jsonl", "a", encoding="utf-8")
    eval_f = open(out / "eval.jsonl", "a", encoding="utf-8")

    def log(rec):
        rec["time"] = time.time()
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        print(json.dumps(rec), flush=True)

    t0 = time.time()
    F = Fixtures(Path(args.data_dir))
    idx = F.fold_indices(args.fold)
    log({"event": "fold", "fold": args.fold, "train": len(idx["train"]), "val": len(idx["val"]), "test": len(idx["test"])})
    sampler = PoolSampler(F, idx["train"], cap=args.cap, ns_min=args.ns_min, ns_max=args.ns_max, p_doc=args.p_doc,
                          seed=args.seed, data_dir=Path(args.data_dir), mmap=not args.pool_in_ram)
    plen = np.load(Path(args.data_dir) / "wscan_pool_rev_len.npy")
    log({"event": "pool_loaded", "sec": round(time.time() - t0, 1)})

    # ---- model
    if args.design == "A":
        model = LatentArray(K=args.K, d_model=args.d_model, heads=args.heads, rounds=args.rounds, dropout=args.dropout,
                            c=args.c, eps=args.eps, sinkhorn_iters=args.sinkhorn_iters, rho=args.rho, self_attn=args.self_attn, ot=args.ot,
                            anchor_mode=args.anchor_mode, center=args.center).to(dev)
    else:
        model = TopicCodebook(K=args.K, c=args.c, eps=args.eps, lam=args.lam, beta=args.beta, rho=args.rho or None,
                              assign_map=args.assign_map, film=args.film, iters=args.sinkhorn_iters, gain_init=args.gain_init,
                              gain_min=args.gain_min, gain_max=args.gain_max, dual_lr=args.dual_lr,
                              implicit_grad=not args.no_implicit_grad, center=args.center).to(dev)
    if args.center:
        mu = pool_global_mean(sampler.POOL, plen, idx["train"])
        model.set_center(mu); np.save(out / "center_mu.npy", mu)
        log({"event": "center", "mu_norm": float(np.linalg.norm(mu))})
    if args.design == "B":
        Xs = sample_pool_rows(sampler.POOL, plen, idx["train"], args.kmeans_games, args.kmeans_per_game, seed=args.seed)
        Xs_t = torch.from_numpy(Xs).to(dev)
        if args.center:
            Xs_t = Xs_t - model.mu
        init = model.init_codebook(Xs_t, mode=args.protos, seed=args.seed, iters=args.kmeans_iters)
        del Xs_t, Xs
        np.save(out / "protos_init.npy", model.protos.detach().cpu().numpy())
        log({"event": "codebook_init", **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in init.items()}, "sec": round(time.time() - t0, 1)})
    log({"event": "model", "design": args.design, "params": model.n_params(), "K": args.K})

    # ---- optimiser
    if args.design == "A":
        decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "log_temp" not in n]
        no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 or "log_temp" in n]
        groups = [{"params": decay, "weight_decay": args.wd, "mult": 1.0}, {"params": no_decay, "weight_decay": 0.0, "mult": 1.0}]
    else:
        named = dict(model.named_parameters())
        decay = [p for n, p in named.items() if n in ("dW.weight", "dU.weight")]
        others = [p for n, p in named.items() if n not in ("dW.weight", "dU.weight", "protos")]
        groups = [{"params": decay, "weight_decay": args.wd, "mult": 1.0}, {"params": others, "weight_decay": 0.0, "mult": 1.0},
                  {"params": [named["protos"]], "weight_decay": 0.0, "mult": args.proto_lr_mult}]
        model.protos.requires_grad_(args.freeze_protos == 0)
        # the gain follows the prototypes' schedule (B-frozen: both fixed; B-learn: both learn after the warm-up)
        gain_learns = args.learn_gain or (args.freeze_protos != -1 and not args.freeze_gain)
        gain_now = args.learn_gain or (gain_learns and args.freeze_protos == 0)
        model.gain.requires_grad_(gain_now)
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.98))

    # ---- evaluation assets: draw 0 at the training budget (memory-mapped) and the standardised queries
    X0 = L0 = SQ = None
    if not args.no_eval:
        X0, L0, _, _ = load_draw(args.cap, 0, Path(args.cache_dir))
        SQ, _ = F.queries_std()
        log({"event": "draw0_loaded", "shape": list(X0.shape), "sec": round(time.time() - t0, 1)})
    selection = {"best_step": None, "best_score": -1.0, "scores": {}}
    sel_key = "routed_norm_val_rank_score" if args.design == "B" else "slot_val_rank_score"   # B: the coverage-renormalised routed score (plan 6)

    def save_ckpt(step):
        torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, out / f"ckpt_step{step}.pt")

    def evaluate(step):
        te = time.time()
        if args.design == "A":
            m = retrieval_metrics(encode_queries(model, F, dev), encode_sets(model, X0, L0, dev), F, args.fold, device=dev)
        else:
            m = topic_retrieval_metrics(encode_topics(model, SQ, F.QL, dev, batch=128), encode_topics(model, X0, L0, dev), F, args.fold, device=dev)
        rec = {"step": step, **{k: round(v, 4) for k, v in m.items()}, "sec": round(time.time() - te, 1)}
        eval_f.write(json.dumps(rec) + "\n"); eval_f.flush()
        print("EVAL", json.dumps(rec), flush=True)
        score = m[sel_key]
        selection["scores"][str(step)] = score
        if score > selection["best_score"]:
            selection.update(best_step=step, best_score=score)
        (out / "selection.json").write_text(json.dumps(selection, indent=1), encoding="utf-8")
        model.train()

    if args.design == "B":                                                         # step 0 = the untrained codebook (B0 / B3)
        save_ckpt(0)
        if not args.no_eval:
            evaluate(0)

    pref = Prefetcher(sampler, args.batch) if args.steps > 0 else None
    model.train()
    ema = {}
    ts = time.time()
    step_fn = step_A if args.design == "A" else step_B
    for step in range(1, args.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step - 1, args) * gp.get("mult", 1.0)
        if args.design == "B" and args.freeze_protos > 0 and step == args.freeze_protos + 1:
            model.protos.requires_grad_(True)
            if gain_learns:
                model.gain.requires_grad_(True)
            log({"event": "protos_unfrozen", "step": step, "gain_learns": gain_learns})
        batch = pref.next()
        l_ice, l_inv, st, aux = step_fn(model, batch, args, dev)
        loss = l_ice + args.lam_inv * l_inv
        opt.zero_grad(set_to_none=True)
        if not loss.requires_grad:                                                 # every learnable path switched off by the ablation flags
            raise SystemExit("the loss has no gradient path: check --weak-no-grad / --no-strong-neg / --lam-inv 0 together")
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip)
        opt.step()
        run = {}
        if args.design == "B":
            with torch.no_grad():
                model.gain.clamp_(model.gain_min, model.gain_max)                 # projection: a clamped parameter would lose its gradient at the bound
                if model.protos.requires_grad:
                    model.protos.copy_(torch.nn.functional.normalize(model.protos, dim=-1))   # stay on the sphere: constant effective lr
            run = model.update_running(aux["mask"], aux["x"], aux["M"], g_star=aux["g"])
        for k, v in {"loss": float(loss), "ice": float(l_ice), "inv": float(l_inv), "gn": float(gn), **st}.items():
            ema[k] = v if k not in ema else 0.9 * ema[k] + 0.1 * v
        if step % args.log_every == 0:
            log({"step": step, **{k: round(v, 4) for k, v in ema.items()}, "lr": lr_at(step - 1, args),
                 "ms_per_step": round(1000 * (time.time() - ts) / args.log_every, 1), "ls_max": int(aux["xs"].shape[1])})
            ts = time.time()
        if step % args.monitor_every == 0:
            if args.design == "A":
                ow, os_, mw, ms = aux["ow"], aux["os"], aux["mw"], aux["ms"]
                mon = monitors(ow["M"].detach(), mw, batch["tw"], os_["M"].detach(), ms, batch["ds"], os_["anchors"].detach(), ow["anchors"].detach(), ts=batch["ts"])
                mon.update(ot_row_err_w=ow["ot_row_err"], ot_col_err_w=ow["ot_col_err"], ot_row_err_s=os_["ot_row_err"], ot_col_err_s=os_["ot_col_err"],
                           s_std=float(os_["S"][~ms].std()))
            else:
                mon = topic_monitors(aux["ow"], aux["os"], aux["mw"], aux["ms"], batch["ts"], model, float(l_inv))
                mon.update(run)
                mon["s_std"] = float(aux["S_s"][~aux["ms"]].std())
            log({"step": step, "monitor": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in mon.items()}})
        do_eval = not args.no_eval and (step % args.eval_every == 0 or step == args.steps)
        if step % args.save_every == 0 or step == args.steps or do_eval:
            save_ckpt(step)
        if do_eval:
            evaluate(step)
    if pref is not None:
        pref.close()
    if selection["best_step"] is None and args.design == "B":
        selection.update(best_step=0, best_score=float("nan"))
        (out / "selection.json").write_text(json.dumps(selection, indent=1), encoding="utf-8")
    log({"event": "done", "steps": args.steps, "hours": round((time.time() - t0) / 3600, 2), "selection": selection})


if __name__ == "__main__":
    main()
