# -*- coding: utf-8 -*-
"""Ten-draw evaluation of a trained run (and of the frozen-mean baselines on the same draws) at one anchor budget.

    python scripts/eval_run.py --run runs/b64_frozen [--step best] --cap 512 --draws 10 --cache-dir D:/ice_horizontal_cache

Design B (topic codebook), per draw: retrieval hit@1 / hit@5 for the five scorings (routed, routed_norm, routed_raw, mean, hist)
on the test fold; the ridge tag probe on the mean (F1), topic-distribution (F2), mean+hist and, for K <= 64, the
coverage-weighted topic concatenation (F3); the zero-shot tag mAP (routed and max readings); the frozen baseline B1
(raw mean, the paper's reading) and B1' (centred mean).  Design A keeps its previous evaluation (slot / mean /
chamfer, mean / concat probes, max zero-shot, random groups).  Then mean +- std over the draws.
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
from ice_h.evaluate import (CONCAT_ALPHAS, MODES, TMODES, encode_queries, encode_sets, encode_topics, features, frozen_mean, l2,  # noqa: E402
                            random_group_anchors, retrieval_metrics, single_vector_metrics, tag_probe, topic_features,
                            topic_retrieval_metrics, zeroshot_single, zeroshot_tags, zeroshot_topics)
from ice_h.runs import load_model  # noqa: E402


def agg(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))} for k in keys}


def fmt(src: dict, key: str) -> str:
    v = src.get(key)
    return f"{v['mean']:.3f}±{v['std']:.3f}" if v else "—"


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
    ap.add_argument("--no-concat", action="store_true", help="B: skip the K*D topic-concatenation probe")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda")
    run = Path(args.run)
    model, step, targs = load_model(run, args.step, dev)
    design = targs.get("design", "A")
    fold = args.fold if args.fold is not None else targs["fold"]
    K = targs["K"]
    F = Fixtures(Path(args.data_dir))
    tag_emb = np.load(args.tag_emb, allow_pickle=True)[args.tag_key] if Path(args.tag_emb).exists() else None
    t0 = time.time()
    S, _ = F.queries_std()
    Qfroz = frozen_mean(S, F.QL)                                                  # (3256, D) raw means (B1)
    mu = model.mu.cpu().numpy() if getattr(model, "center", False) else np.zeros(S.shape[-1], np.float32)
    test, _val = F.fold_games(fold)
    test_rows = F.query_rows(test, "neutral")
    rows_model, rows_froz, rows_rand = [], [], []
    if design == "B":
        with torch.no_grad():
            Q = encode_topics(model, S, F.QL, dev, batch=128)
            tag_emb_model = tag_hist = None
            if tag_emb is not None:
                te = torch.as_tensor(tag_emb, device=dev)
                tag_emb_model = model.map_space(te).cpu().numpy()                 # the centred space the anchors live in
                tag_hist = model(te[:, None, :], torch.zeros(len(te), 1, dtype=torch.bool, device=dev))["hist"].cpu().numpy()
        kinds = ["mean", "hist", "mean_hist"] + (["concat"] if K <= 64 and not args.no_concat else [])
        Qfeat = {kind: topic_features(Q, kind, rows=test_rows) for kind in kinds}
        for d in range(args.draws):
            X, L, _, _ = load_draw(args.cap, d, Path(args.cache_dir))
            G = encode_topics(model, X, L, dev)
            r = topic_retrieval_metrics(Q, G, F, fold, device=dev)
            if not args.no_tags:
                for kind in kinds:
                    r[f"tag_f1_{kind}"] = tag_probe(F, fold, topic_features(G, kind), {int(i): Qfeat[kind][j] for j, i in enumerate(test_rows)},
                                                    alphas=CONCAT_ALPHAS if kind == "concat" else None)["micro_f1"]
                if tag_emb is not None:
                    z = zeroshot_topics(G, tag_emb_model, tag_hist, F, fold)
                    r.update(zs_map_routed=z["macro_map_routed"], zs_map_max=z["macro_map_max"], zs_map_low=z["map_low_routed"],
                             zs_map_mid=z["map_mid_routed"], zs_map_high=z["map_high_routed"])
                    r["_zs_detail"] = {"ap_routed": z["ap_routed"], "ap_max": z["ap_max"], "tag_topics": z["tag_topics"]}
            rows_model.append(r)
            if not args.no_baselines:
                Gf = frozen_mean(X, L)
                rf = {k.replace("slot_", ""): v for k, v in single_vector_metrics(Qfroz, Gf, F, fold).items()}
                rfc = {k.replace("slot_", "") + "_c": v for k, v in single_vector_metrics(Qfroz - mu, Gf - mu, F, fold).items()}   # B1': centred
                rf.update(rfc)
                if not args.no_tags:
                    rf["tag_f1_mean"] = tag_probe(F, fold, Gf, {i: Qfroz[i] for i in range(len(Qfroz))})["micro_f1"]
                    if tag_emb is not None:
                        rf["zs_map"] = zeroshot_single(Gf, tag_emb, F, fold)["macro_map"]
                        rf["zs_map_c"] = zeroshot_single(Gf - mu, tag_emb - mu, F, fold)["macro_map"]
                rows_froz.append(rf)
            print(f"draw {d}: routed h1 name {r['routed_h1_neutral']:.3f} stripped {r['routed_h1_noname']:.3f} | norm {r['routed_norm_h1_noname']:.3f} "
                  f"raw {r['routed_raw_h1_noname']:.3f} mean {r['mean_h1_noname']:.3f} hist {r['hist_h1_noname']:.3f}"
                  + (f" | tag {r.get('tag_f1_mean', float('nan')):.3f}/{r.get('tag_f1_hist', float('nan')):.3f}/{r.get('tag_f1_mean_hist', float('nan')):.3f}"
                     + (f"/{r['tag_f1_concat']:.3f}" if 'tag_f1_concat' in r else "") if not args.no_tags else "")
                  + (f" | zs {r['zs_map_routed']:.3f}/{r['zs_map_max']:.3f}" if 'zs_map_routed' in r else "") + f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    else:
        tag_emb_model = None
        if tag_emb is not None:
            with torch.no_grad():
                tag_emb_model = model.map_space(torch.as_tensor(tag_emb, device=dev)).cpu().numpy()
        Q = encode_queries(model, F, dev)
        Qmean, Qcat = features(Q, "mean"), features(Q, "concat")
        for d in range(args.draws):
            X, L, _, _ = load_draw(args.cap, d, Path(args.cache_dir))
            G = encode_sets(model, X, L, dev)
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
                rf = {k.replace("slot_", ""): v for k, v in single_vector_metrics(Qfroz, Gf, F, fold).items()}
                if not args.no_tags:
                    rf["tag_f1_mean"] = tag_probe(F, fold, Gf, {i: Qfroz[i] for i in range(len(Qfroz))})["micro_f1"]
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

    res = {"run": str(run), "design": design, "step": step, "fold": fold, "cap": args.cap, "draws": args.draws, "K": K,
           "model": agg(rows_model), "per_draw": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows_model]}
    if rows_model and "_zs_detail" in rows_model[0]:
        res["zs_detail_draw0"] = rows_model[0]["_zs_detail"]
    if rows_froz:
        res["frozen"] = agg(rows_froz)
    if rows_rand:
        res["random"] = agg(rows_rand)
    out = Path(args.out) if args.out else run / f"eval_cap{args.cap}_step{step}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\n== %s fold %d, cap %d, %d draws, step %d, K %d ==" % (design, fold, args.cap, args.draws, step, K))
    m = res["model"]
    if design == "B":
        for mode in TMODES:
            print(f"model/{mode:11s} Name@1 {fmt(m, mode + '_h1_neutral')}  Name@5 {fmt(m, mode + '_h5_neutral')}  Stripped@1 {fmt(m, mode + '_h1_noname')}  Stripped@5 {fmt(m, mode + '_h5_noname')}")
        print("tag F1: " + "  ".join(f"{k} {fmt(m, 'tag_f1_' + k)}" for k in ("mean", "hist", "mean_hist", "concat")))
        print(f"zero-shot mAP: routed {fmt(m, 'zs_map_routed')}  max {fmt(m, 'zs_map_max')}")
        if "frozen" in res:
            fz = res["frozen"]
            print(f"B1 raw mean       Name@1 {fmt(fz, 'h1_neutral')}  Stripped@1 {fmt(fz, 'h1_noname')}  Stripped@5 {fmt(fz, 'h5_noname')}  tag {fmt(fz, 'tag_f1_mean')}  zs {fmt(fz, 'zs_map')}")
            print(f"B1' centred mean  Name@1 {fmt(fz, 'h1_neutral_c')}  Stripped@1 {fmt(fz, 'h1_noname_c')}  Stripped@5 {fmt(fz, 'h5_noname_c')}  zs {fmt(fz, 'zs_map_c')}")
    else:
        cols = [("Name@1", "slot_h1_neutral"), ("Stripped@1", "slot_h1_noname"), ("Stripped@5", "slot_h5_noname"),
                ("tagF1 mean", "tag_f1_mean"), ("tagF1 concat", "tag_f1_concat"), ("zs mAP", "zs_map")]
        for label, src in (("model/slot", m), ("model/mean", {k.replace("mean_", "slot_", 1): v for k, v in m.items() if k.startswith("mean_")}),
                           ("model/chamfer", {k.replace("chamfer_", "slot_", 1): v for k, v in m.items() if k.startswith("chamfer_")}),
                           ("frozen", {("slot_" + k if k.startswith("h") else k): v for k, v in res.get("frozen", {}).items()}),
                           ("random/chamfer", {k.replace("chamfer_", "slot_", 1): v for k, v in res.get("random", {}).items()})):
            print(f"{label:16s} " + "  ".join(f"{name} {fmt(src, key)}" for name, key in cols))
    print("wrote", out)


if __name__ == "__main__":
    main()
