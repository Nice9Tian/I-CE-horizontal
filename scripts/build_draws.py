# -*- coding: utf-8 -*-
"""Materialise the paper's evaluation anchor draws at one budget (k10_eval.encode's rule).

    python scripts/build_draws.py --cap 512 --k 10 [--games N] [--out D:\\ice_horizontal_cache]

Writes <out>/draws_cap<cap>/draw<k>.npy (NG, cap, 1024) fp16 standardised, len<k>.npy, rows<k>.npy
(full-pool row of every sentence, -1 for the doc prefix / padding) and doclen.npy.  Resumable.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ice_h.data import CACHE_DIR, DATA_DIR, build_draw_cache  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--games", type=int, default=0, help="only the first N games (smoke test)")
    ap.add_argument("--out", default=str(CACHE_DIR))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    a = ap.parse_args()
    out = build_draw_cache(cap=a.cap, K=a.k, out_dir=Path(a.out), data_dir=Path(a.data_dir), games=a.games or None,
                           log=lambda s: print(s, flush=True))
    print("done:", out)
