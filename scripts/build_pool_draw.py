# -*- coding: utf-8 -*-
"""A 512-sentence pack per game built from the in-RAM review pool (NVMe), for checkpoint selection while
the full-pool draw cache is still being built on the slow disk.

Same pack rule as k10_eval (store-page prefix first, whole reviews in a seeded random order, a review
entering if it still fits, stop at the cap), but the review universe is the game's 2,048-sentence pool
instead of all its reviews, and the permutation seed is DRAW_SEED + g (draw index 0).  Written to
<cache>/draws_cap<cap>/draw0.npy etc. with progress.json marking source = "pool", so load_draw() reads it
unchanged; eval_run must point at the full-pool cache instead.

    python scripts/build_pool_draw.py --data-dir C:/ice_horizontal_cache/data --cache-dir C:/ice_horizontal_cache
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ice_h.data import CACHE_DIR, DATA_DIR, DRAW_SEED, Fixtures  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    a = ap.parse_args()
    d = Path(a.data_dir)
    out = Path(a.cache_dir) / f"draws_cap{a.cap}"
    out.mkdir(parents=True, exist_ok=True)
    F = Fixtures(d, load_wiki_docs=False, load_queries=False)
    POOL = np.load(d / "wscan_pool_rev.npy", mmap_mode="r")
    RID = np.load(d / "wscan_pool_rev_rid.npy"); PL = np.load(d / "wscan_pool_rev_len.npy")
    NG, cap = F.NG, a.cap
    X = np.lib.format.open_memmap(out / "draw0.npy", mode="w+", dtype=np.float16, shape=(NG, cap, 1024))
    L = np.lib.format.open_memmap(out / "len0.npy", mode="w+", dtype=np.int64, shape=(NG,))
    R = np.lib.format.open_memmap(out / "rows0.npy", mode="w+", dtype=np.int64, shape=(NG, cap)); R[:] = -1
    DL = np.lib.format.open_memmap(out / "doclen.npy", mode="w+", dtype=np.int64, shape=(NG,))
    t0 = time.time()
    for g in range(NG):
        r = np.asarray(RID[g, : int(PL[g])])
        brk = np.flatnonzero(np.diff(r) != 0) + 1
        starts = np.concatenate([[0], brk]); ends = np.concatenate([brk, [len(r)]]); lens = ends - starts
        doc = F.doc(g); dl = 0 if doc is None else len(doc)
        row, segs = dl, []
        for j in np.random.default_rng(DRAW_SEED + g).permutation(len(lens)):
            if row + int(lens[j]) <= cap:
                segs.append(j); row += int(lens[j])
                if row == cap:
                    break
        blk = np.asarray(POOL[g])
        x = X[g]; x[:] = 0
        if doc is not None:
            x[:dl] = doc
        pos = dl
        for j in segs:
            x[pos:pos + lens[j]] = blk[starts[j]:ends[j]]; pos += int(lens[j])
        assert pos == row
        L[g] = row; DL[g] = dl
        if (g + 1) % 500 == 0:
            print(f"  {g + 1}/{NG}  {time.time() - t0:.0f}s", flush=True)
    for m in (X, L, R, DL):
        m.flush()
    (out / "progress.json").write_text(json.dumps({"done": NG, "cap": cap, "K": 1, "NG": NG, "source": "pool", "seed": DRAW_SEED}))
    print("median pack", int(np.median(L)), "min", int(L.min()), "->", out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
