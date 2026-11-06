# -*- coding: utf-8 -*-
"""B2: the paper's I-CE tower (one 128-d vector per game) scored on the SAME cached draws and
queries as our runs, with the paper's reading (cosine retrieval, ridge tag probe on the anchors).

    python scripts/baseline_ice.py --fold 0 --cap 512 --draws 10
        [--ckpt D:\\runpod_data\\w9_cv_out\\ckpt_w9cv_wcle_i2ce_icetf_fold0_fp_ep400.pt] [--out ...]

The default checkpoint per fold is the one the paper's rank rule selected (rank_ckpt_plan.json).
The tower class is imported from the public repository's worker (SetPoolN: 4 latent queries,
128-d, 4 heads, mean-pooled, two-layer MLP head, L2-normalised output).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ice_h.data import CACHE_DIR, DATA_DIR, PUBLIC_REPO, Fixtures, load_draw  # noqa: E402
from ice_h.evaluate import frozen_mean, l2, single_vector_metrics, tag_probe, zeroshot_single  # noqa: E402

W9 = Path(PUBLIC_REPO) / "contrast_experiment" / "w9"
sys.path.insert(0, str(W9)); sys.path.insert(0, str(W9 / "Pod"))
from Pod.w9_cv_worker import SetPoolN  # noqa: E402

CKPT_DIR = Path(r"D:\runpod_data\w9_cv_out")                                             # the published towers (pub512)
DOCFIX_DIR = Path(r"D:\Runpod-2\results_docfix512_20261107\podA\out\docfix512")        # retrained with the standardised prefix


def default_ckpt(fold: int, arm: str = "i2ce_icetf", family: str = "docfix512") -> Path:
    """docfix512: the retrained tower at the epoch its zsbest file selected; pub512: the paper's rank plan"""
    if family == "pub512":
        plan = json.loads((W9 / "eval" / "rank_ckpt_plan.json").read_text(encoding="utf-8"))
        return CKPT_DIR / plan[f"{arm}_fold{fold}"]["ckpt"]
    base = f"w9cv_wcle_{arm}_fold{fold}_d512_fp"
    ep = json.loads((DOCFIX_DIR / f"zsbest_{base}.json").read_text(encoding="utf-8"))["best_ep"]
    return DOCFIX_DIR / f"ckpt_{base}_ep{ep}.pt"


def load_tower(path, dev):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    sd = {k: v for k, v in sd.items() if k == "q0" or k.startswith(("attn.", "head."))}
    m = SetPoolN(4).to(dev).eval(); m.load_state_dict(sd)
    return m


@torch.no_grad()
def encode(tower, X, lens, dev, batch=64):
    out = []
    for i in range(0, X.shape[0], batch):
        L = int(lens[i:i + batch].max())
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch, :L])).to(dev).float()
        mb = torch.arange(L, device=dev)[None, :] >= torch.as_tensor(lens[i:i + batch], device=dev)[:, None]
        out.append(tower(xb, mb).float().cpu().numpy())
    return np.concatenate(out)


def agg(rows):
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))} for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--arm", default="i2ce_icetf", help="i2ce_icetf (I-CE) or ce_cetf (CE)")
    ap.add_argument("--family", default="docfix512", choices=["docfix512", "pub512"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--tag-emb", default=str(Path(CACHE_DIR) / "tag_emb.npz"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    ckpt = Path(a.ckpt) if a.ckpt else default_ckpt(a.fold, a.arm, a.family)
    tower = load_tower(ckpt, dev)
    F = Fixtures(Path(a.data_dir))
    S, _ = F.queries_std()
    Q = encode(tower, S, F.QL, dev)                                   # (3256, 128)
    Qf = frozen_mean(S, F.QL)
    tag_emb = np.load(a.tag_emb, allow_pickle=True)["phrase_std"] if Path(a.tag_emb).exists() else None
    rows, rows_f = [], []
    t0 = time.time()
    for d in range(a.draws):
        X, L, _, _ = load_draw(a.cap, d, Path(a.cache_dir))              # memory-mapped
        G = encode(tower, X, L, dev)
        r = {k.replace("slot_", ""): v for k, v in single_vector_metrics(Q, G, F, a.fold).items()}
        r["tag_f1"] = tag_probe(F, a.fold, G, {i: Q[i] for i in range(len(Q))})["micro_f1"]
        rows.append(r)
        Gf = frozen_mean(X, L)
        rf = {k.replace("slot_", ""): v for k, v in single_vector_metrics(Qf, Gf, F, a.fold).items()}
        rf["tag_f1"] = tag_probe(F, a.fold, Gf, {i: Qf[i] for i in range(len(Qf))})["micro_f1"]      # k10_eval: raw means
        if tag_emb is not None:
            rf["zs_map"] = zeroshot_single(Gf, tag_emb, F, a.fold)["macro_map"]
        rows_f.append(rf)
        print(f"draw {d}: {a.arm} name {r['h1_neutral']:.3f} stripped {r['h1_noname']:.3f} tag {r['tag_f1']:.3f} | frozen name {rf['h1_neutral']:.3f} "
              f"stripped {rf['h1_noname']:.3f} tag {rf['tag_f1']:.3f}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    res = {"arm": a.arm, "family": a.family, "ckpt": str(ckpt), "fold": a.fold, "cap": a.cap, "draws": a.draws, "tower": agg(rows), "frozen": agg(rows_f),
           "per_draw": rows, "per_draw_frozen": rows_f}
    out = Path(a.out) if a.out else Path(a.cache_dir) / f"baseline_{a.family}_{a.arm}_fold{a.fold}_cap{a.cap}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for name, src in ((f"B2 {a.family} {a.arm}", res["tower"]), ("B1 frozen", res["frozen"])):
        print(f"{name:16s} Name@1 {src['h1_neutral']['mean']:.3f}±{src['h1_neutral']['std']:.3f}  Stripped@1 {src['h1_noname']['mean']:.3f}±{src['h1_noname']['std']:.3f}"
              f"  Stripped@5 {src['h5_noname']['mean']:.3f}  tag F1 {src['tag_f1']['mean']:.3f}±{src['tag_f1']['std']:.3f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
