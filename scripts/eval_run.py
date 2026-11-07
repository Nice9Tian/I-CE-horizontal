# -*- coding: utf-8 -*-
"""Ten-draw evaluation of a trained run (and of the frozen-mean and random-group baselines on the
same draws), at one anchor budget.

    python scripts/eval_run.py --run runs/k6_fold0 [--step best] --cap 512 --draws 10 --out runs/k6_fold0/eval_cap512.json

Per draw: retrieval hit@1 / hit@5 (slot, mean, chamfer) on the test fold, the ridge tag probe on
mean and concat features, the zero-shot tag mAP against the tag-name embeddings; then mean +- std
over the draws.  The frozen baseline uses the paper's reading (mean sentence vector, cosine).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ice_h.data import CACHE_DIR, DATA_DIR, Fixtures, load_draw  # noqa: E402
from ice_h.evaluate import (MODES, encode_queries, encode_sets, features, frozen_mean, l2, random_group_anchors,  # noqa: E402
                            retrieval_metrics, single_vector_metrics, tag_probe, zeroshot_single, zeroshot_tags)
from ice_h.model import LatentArray  # noqa: E402


def load_model(run: Path, step, dev):
    if step in (None, "best"):
        step = json.loads((run / "selection.json").read_text())["best_step"]
    ck = torch.load(run / f"ckpt_step{step}.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    m = LatentArray(K=a["K"], d_model=a["d_model"], heads=a["heads"], rounds=a["rounds"], dropout=a["dropout"], c=a["c"],
                    eps=a["eps"], sinkhorn_iters=a["sinkhorn_iters"], rho=a["rho"], self_attn=a.get("self_attn", False), ot=a.get("ot", "fermi"),
                    anchor_mode=a.get("anchor_mode", "decode"), center=a.get("center", False)).to(dev)
    sd = dict(ck["model"])
    if "W.weight" in sd:                                                       # checkpoints from before W = I + dW
        w = sd.pop("W.weight"); sd["dW.weight"] = w - torch.eye(w.shape[0])
    sd = {k: v for k, v in sd.items() if k in m.state_dict()}                 # drop layers this mode no longer has
    m.load_state_dict(sd, strict=False); m.eval()
    return m, int(step), a


def agg(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))} for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", default="best")
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--tag-emb", default=str(Path(CACHE_DIR) / "tag_emb.npz"))
    ap.add_argument("--tag-key", default="phrase_std")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--no-tags", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda")
    run = Path(args.run)
    model, step, targs = load_model(run, args.step, dev)
    fold = args.fold if args.fold is not None else targs["fold"]
    K = targs["K"]
    F = Fixtures(Path(args.data_dir))
    tag_emb = np.load(args.tag_emb, allow_pickle=True)[args.tag_key] if Path(args.tag_emb).exists() else None
    tag_emb_model = None
    if tag_emb is not None:
        with torch.no_grad():
            tag_emb_model = model.map_space(torch.as_tensor(tag_emb, device=dev)).cpu().numpy()   # the space the anchors live in
    t0 = time.time()
    Q = encode_queries(model, F, dev)                                             # (3256, K, D)
    Qmean, Qcat = features(Q, "mean"), features(Q, "concat")
    S, _ = F.queries_std()
    Qfroz = frozen_mean(S, F.QL)                                                  # (3256, D)
    rows_model, rows_froz, rows_rand = [], [], []
    for d in range(args.draws):
        X, L, _, _ = load_draw(args.cap, d, Path(args.cache_dir))          # memory-mapped; batches are copied as needed
        G = encode_sets(model, X, L, dev)                                         # (NG, K, D)
        r = retrieval_metrics(Q, G, F, fold, device=dev)
        if not args.no_tags:
            r["tag_f1_mean"] = tag_probe(F, fold, features(G, "mean"), {i: Qmean[i] for i in range(len(Qmean))})["micro_f1"]
            r["tag_f1_concat"] = tag_probe(F, fold, features(G, "concat"), {i: Qcat[i] for i in range(len(Qcat))})["micro_f1"]
            if tag_emb is not None:
                z = zeroshot_tags(G, tag_emb_model, F, fold)
                r.update(zs_map=z["macro_map"], zs_map_low=z["map_low"], zs_map_mid=z["map_mid"], zs_map_high=z["map_high"])
                r["_zs_detail"] = {"ap": z["ap"], "slot_hist": z["slot_hist"]}
        rows_model.append(r)
        if not args.no_baselines:
            Gf = frozen_mean(X, L)
            rf = single_vector_metrics(Qfroz, Gf, F, fold)
            rf = {k.replace("slot_", ""): v for k, v in rf.items()}
            if not args.no_tags:
                rf["tag_f1_mean"] = tag_probe(F, fold, Gf, {i: Qfroz[i] for i in range(len(Qfroz))})["micro_f1"]   # k10_eval: raw means
                if tag_emb is not None:
                    rf["zs_map"] = zeroshot_single(Gf, tag_emb, F, fold)["macro_map"]
            rows_froz.append(rf)
            Gr = random_group_anchors(X, L, K, seed=d)
            Qr = random_group_anchors(S, F.QL, K, seed=1000 + d)
            rr = retrieval_metrics(Qr, Gr, F, fold, modes=("mean", "chamfer"), device=dev)
            if not args.no_tags:
                Qrc = features(Qr, "concat")
                rr["tag_f1_concat"] = tag_probe(F, fold, features(Gr, "concat"), {i: Qrc[i] for i in range(len(Qrc))})["micro_f1"]
                if tag_emb is not None:
                    rr["zs_map"] = zeroshot_tags(Gr, tag_emb, F, fold)["macro_map"]
            rows_rand.append(rr)
        print(f"draw {d}: slot h1 name {r['slot_h1_neutral']:.3f} stripped {r['slot_h1_noname']:.3f} | mean {r['mean_h1_noname']:.3f} "
              f"chamfer {r['chamfer_h1_noname']:.3f}" + (f" | tag {r['tag_f1_mean']:.3f}/{r['tag_f1_concat']:.3f}" if not args.no_tags else "")
              + (f" | zs {r['zs_map']:.3f}" if 'zs_map' in r else "") + f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    res = {"run": str(run), "step": step, "fold": fold, "cap": args.cap, "draws": args.draws, "K": K,
           "model": agg(rows_model), "per_draw": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows_model]}
    if rows_model and "_zs_detail" in rows_model[0]:
        res["zs_detail_draw0"] = rows_model[0]["_zs_detail"]
    if rows_froz:
        res["frozen"] = agg(rows_froz); res["random"] = agg(rows_rand)
    out = Path(args.out) if args.out else run / f"eval_cap{args.cap}_step{step}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\n== fold %d, cap %d, %d draws, step %d ==" % (fold, args.cap, args.draws, step))
    cols = [("Name@1", "slot_h1_neutral"), ("Stripped@1", "slot_h1_noname"), ("Stripped@5", "slot_h5_noname"),
            ("tagF1 mean", "tag_f1_mean"), ("tagF1 concat", "tag_f1_concat"), ("zs mAP", "zs_map")]
    for label, src in (("model/slot", res["model"]), ("model/mean", {k.replace("mean_", "slot_", 1): v for k, v in res["model"].items() if k.startswith("mean_")}),
                       ("model/chamfer", {k.replace("chamfer_", "slot_", 1): v for k, v in res["model"].items() if k.startswith("chamfer_")}),
                       ("frozen", {("slot_" + k if k.startswith("h") else k): v for k, v in res.get("frozen", {}).items()}),
                       ("random/chamfer", {k.replace("chamfer_", "slot_", 1): v for k, v in res.get("random", {}).items()})):
        cells = []
        for name, key in cols:
            v = src.get(key)
            cells.append(f"{name} {v['mean']:.3f}±{v['std']:.3f}" if v else f"{name} —")
        print(f"{label:16s} " + "  ".join(cells))
    print("wrote", out)


if __name__ == "__main__":
    main()
