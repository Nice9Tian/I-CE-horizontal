# -*- coding: utf-8 -*-
"""Where does a design-B training step spend its time?  Builds the sampler and model like train.py, runs a few steps
and prints the wall time of every stage (CUDA synchronised).

    python scripts/profile_step.py --K 64 --batch 64 --steps 5 --data-dir C:/ice_horizontal_cache/data
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ice_h.codebook import TopicCodebook  # noqa: E402
from ice_h.data import DATA_DIR, Fixtures, PoolSampler, pool_global_mean, sample_pool_rows  # noqa: E402
from ice_h.losses import routed_ice_loss, routed_inv_loss  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    a = ap.parse_args()
    dev = torch.device("cuda")
    F = Fixtures(Path(a.data_dir))
    idx = F.fold_indices(0)
    sampler = PoolSampler(F, idx["train"], seed=0, data_dir=Path(a.data_dir))
    plen = np.load(Path(a.data_dir) / "wscan_pool_rev_len.npy")
    model = TopicCodebook(K=a.K).to(dev)
    model.set_center(pool_global_mean(sampler.POOL, plen, idx["train"]))
    Xs = torch.from_numpy(sample_pool_rows(sampler.POOL, plen, idx["train"], 100, 500)).to(dev) - model.mu
    model.init_codebook(Xs, iters=5)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    T = {}

    def tick(name, t0):
        torch.cuda.synchronize(); T[name] = T.get(name, 0.0) + time.time() - t0; return time.time()

    for _ in range(a.steps + 1):
        t = time.time()
        batch = sampler.sample(B=a.batch); t = tick("sample(cpu)", t)
        xw = torch.from_numpy(batch["xw"]).to(dev); xs = torch.from_numpy(batch["xs"]).to(dev)
        lw = torch.as_tensor(batch["lw"], device=dev); ls = torch.as_tensor(batch["ls"], device=dev)
        mw = torch.arange(xw.shape[1], device=dev)[None, :] >= lw[:, None]; ms = torch.arange(xs.shape[1], device=dev)[None, :] >= ls[:, None]
        t = tick("to_dev", t)
        xw_c, xs_c = model.map_space(xw, mw), model.map_space(xs, ms)
        S_w = model.affinity(xw_c, mw); S_s = model.affinity(xs_c, ms); t = tick("affinity", t)
        M_w = model.membership(S_w, mw); M_s = model.membership(S_s, ms); t = tick("membership", t)
        ow = model.aggregate(xw_c, mw, M_w); os_ = model.aggregate(xs_c, ms, M_s); t = tick("aggregate", t)
        l_ice, st = routed_ice_loss(xs_c, ms, M_s, ow["anchors"], ow["cov"], os_["anchors"], os_["cov"], tau=0.03, margin=0.05, sigma=0.01)
        l_inv = routed_inv_loss(os_["anchors"], ow["anchors"], os_["hist"], ow["cov"]); t = tick("loss", t)
        loss = l_ice + l_inv
        opt.zero_grad(); loss.backward(); t = tick("backward", t)
        opt.step(); t = tick("opt", t)
        model.update_running([mw, ms], [xw_c, xs_c], [M_w, M_s]); t = tick("update_running", t)
        if _ == 0:
            T.clear()                                                              # warm-up step not counted
    tot = sum(T.values())
    for k, v in T.items():
        print(f"{k:16s} {1000 * v / a.steps:8.1f} ms/step  {100 * v / tot:5.1f}%")
    print(f"{'total':16s} {1000 * tot / a.steps:8.1f} ms/step   (xw {tuple(xw.shape)}, xs {tuple(xs.shape)})")
    print("max GPU memory MB", round(torch.cuda.max_memory_allocated() / 2**20))


if __name__ == "__main__":
    main()
