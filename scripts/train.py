# -*- coding: utf-8 -*-
"""Stage A: train the latent array on one fold at the 512-sentence anchor budget.

    python scripts/train.py --name k6_fold0 --fold 0 --K 6 --steps 20000

Every --eval-every steps: encode the cached draw 0 (512-sentence packs of all 2,020 games) and
the wiki queries, compute the validation rank score (checkpoint selection, slot scoring) and the
test-fold hit@1 of every scoring mode (report only), save the checkpoint.  Logs are JSON lines
in runs/<name>/log.jsonl and eval.jsonl; the selection in runs/<name>/selection.json.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ice_h.data import CACHE_DIR, DATA_DIR, Fixtures, PoolSampler, Prefetcher, load_draw, pool_global_mean  # noqa: E402
from ice_h.evaluate import encode_queries, encode_sets, retrieval_metrics  # noqa: E402
from ice_h.losses import ice_loss, inv_loss, monitors  # noqa: E402
from ice_h.model import LatentArray  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--out-root", default=str(Path(__file__).resolve().parents[1] / "runs"))
    # model
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--c", type=float, default=2.0)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--sinkhorn-iters", type=int, default=10, help="outer rounds of the box-constrained solver (6 Newton steps per block)")
    ap.add_argument("--rho", type=float, default=0.0, help="0 = balanced columns")
    ap.add_argument("--self-attn", action="store_true", help="ablation: attention among latents")
    ap.add_argument("--ot", default="fermi", choices=["fermi", "exp"], help="membership solver: box-constrained (fermi) or plain entropic + clamp (exp)")
    ap.add_argument("--anchor-mode", default="decode", choices=["decode", "mean", "meanmap", "hybrid"], help="free decode, soft group mean, learned linear map of the mean, or mean + decode")
    ap.add_argument("--center", action="store_true", help="subtract the training pool's global mean vector from every input")
    ap.add_argument("--pool-in-ram", action="store_true", help="read the pool into RAM instead of memory-mapping it")
    # views
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--ns-min", type=int, default=64)
    ap.add_argument("--ns-max", type=int, default=160)
    ap.add_argument("--p-doc", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=32)
    # loss
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--alpha-in", type=float, default=0.8)
    ap.add_argument("--alpha-x", type=float, default=0.8)
    ap.add_argument("--lam-inv", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=0.01, help="noise std per dim on the unit sphere; 0.01 -> cos about 0.95 before/after")
    ap.add_argument("--no-decay", action="store_true", help="ablation: no negative decay")
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
    return ap.parse_args()


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
    log({"event": "pool_loaded", "sec": round(time.time() - t0, 1)})
    pref = Prefetcher(sampler, args.batch)

    model = LatentArray(K=args.K, d_model=args.d_model, heads=args.heads, rounds=args.rounds, dropout=args.dropout,
                        c=args.c, eps=args.eps, sinkhorn_iters=args.sinkhorn_iters, rho=args.rho, self_attn=args.self_attn, ot=args.ot,
                        anchor_mode=args.anchor_mode, center=args.center).to(dev)
    if args.center:
        mu = pool_global_mean(sampler.POOL, np.load(Path(args.data_dir) / "wscan_pool_rev_len.npy"), idx["train"])
        model.set_center(mu); np.save(out / "center_mu.npy", mu)
        log({"event": "center", "mu_norm": float(np.linalg.norm(mu))})
    log({"event": "model", "params": model.n_params(), "anchor_mode": args.anchor_mode})
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))

    # evaluation assets: draw 0 at the training budget, loaded once into RAM
    X0 = L0 = None
    if not args.no_eval:
        X0, L0, _, _ = load_draw(args.cap, 0, Path(args.cache_dir))        # memory-mapped (NVMe page cache, shared across runs)
        log({"event": "draw0_loaded", "shape": list(X0.shape), "sec": round(time.time() - t0, 1)})
    selection = {"best_step": None, "best_score": -1.0, "scores": {}}

    def save_ckpt(step):
        torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, out / f"ckpt_step{step}.pt")

    def evaluate(step):
        te = time.time()
        G = encode_sets(model, X0, L0, dev)
        Q = encode_queries(model, F, dev)
        m = retrieval_metrics(Q, G, F, args.fold, device=dev)
        rec = {"step": step, **{k: round(v, 4) for k, v in m.items()}, "sec": round(time.time() - te, 1)}
        eval_f.write(json.dumps(rec) + "\n"); eval_f.flush()
        print("EVAL", json.dumps(rec), flush=True)
        score = m["slot_val_rank_score"]
        selection["scores"][str(step)] = score
        if score > selection["best_score"]:
            selection.update(best_step=step, best_score=score)
        (out / "selection.json").write_text(json.dumps(selection, indent=1), encoding="utf-8")
        model.train()

    model.train()
    ema = {}
    ts = time.time()
    for step in range(1, args.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step - 1, args)
        batch = pref.next()
        xw, mw, xs, ms = to_dev(batch, dev)
        ow = model(xw, mw)
        os_ = model(xs, ms)
        l_ice, st = ice_loss(xs, ms, os_["anchors"], os_["M"], ow["anchors"], ow["M"], xw, mw, tau=args.tau,
                             margin=args.margin, alpha_in=args.alpha_in, alpha_x=args.alpha_x, sigma=args.sigma,
                             decay=not args.no_decay)
        l_inv = inv_loss(os_["anchors"], ow["anchors"])
        loss = l_ice + args.lam_inv * l_inv
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        for k, v in {"loss": float(loss), "ice": float(l_ice), "inv": float(l_inv), "gn": float(gn), **st}.items():
            ema[k] = v if k not in ema else 0.9 * ema[k] + 0.1 * v
        if step % args.log_every == 0:
            log({"step": step, **{k: round(v, 4) for k, v in ema.items()}, "lr": lr_at(step - 1, args),
                 "ms_per_step": round(1000 * (time.time() - ts) / args.log_every, 1), "ls_max": int(xs.shape[1])})
            ts = time.time()
        if step % args.monitor_every == 0:
            mon = monitors(ow["M"].detach(), mw, batch["tw"], os_["M"].detach(), ms, batch["ds"], os_["anchors"].detach(), ow["anchors"].detach(), ts=batch["ts"])
            mon.update(ot_row_err_w=ow["ot_row_err"], ot_col_err_w=ow["ot_col_err"], ot_row_err_s=os_["ot_row_err"], ot_col_err_s=os_["ot_col_err"],
                       s_std=float(os_["S"][~ms].std()))
            log({"step": step, "monitor": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in mon.items()}})
        if step % args.save_every == 0 or step == args.steps:
            save_ckpt(step)
        if not args.no_eval and (step % args.eval_every == 0 or step == args.steps):
            evaluate(step)
    pref.close()
    log({"event": "done", "steps": args.steps, "hours": round((time.time() - t0) / 3600, 2), "selection": selection})


if __name__ == "__main__":
    main()
