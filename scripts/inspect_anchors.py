# -*- coding: utf-8 -*-
"""Human inspection of the anchors (plan section 10): for a few games, the sentences of the evaluation pack
closest to each anchor, with their text.  Needs the full-pool draw cache (rows{k}.npy gives the full-pool
row of every pack sentence) and the row-aligned text store.

    python scripts/inspect_anchors.py --run runs/k6_fold0 [--step best] --games 5 [--draw 0] [--top 5]
        [--cache-dir D:/ice_horizontal_cache] [--text-h5 "D:/studable query latent/game_review_data/build_new_gamedata/text_h5.h5"]

Also reports, per game, the within-game anchor cosines and how many distinct reviews feed each anchor's
top sentences (a group fed by one review is the shortcut the plan warns about).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ice_h.data import CACHE_DIR, DATA_DIR, Fixtures, FullPool, load_draw  # noqa: E402
from ice_h.evaluate import l2  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_run import load_model  # noqa: E402

TEXT_H5 = r"D:\studable query latent\game_review_data\build_new_gamedata\text_h5.h5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", default="best")
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--game-ids", default=None, help="comma-separated game indices instead of random test-fold games")
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--text-h5", default=TEXT_H5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import h5py
    dev = torch.device("cuda")
    model, step, targs = load_model(Path(a.run), a.step, dev)
    F = Fixtures(Path(a.data_dir), load_wiki_docs=False, load_queries=False)
    X, L, rows, doclen = load_draw(a.cap, a.draw, Path(a.cache_dir))
    test, _ = F.fold_games(targs["fold"])
    if a.game_ids:
        gids = [int(x) for x in a.game_ids.split(",")]
    else:
        rng = np.random.default_rng(a.seed)
        gids = sorted(rng.choice(sorted(F.n2i[n] for n in test), a.games, replace=False).tolist())
    P = FullPool(Path(a.data_dir)) if False else None   # review boundaries come from the meta file below
    meta = np.load(Path(a.data_dir) / "full_pool_meta.npz", allow_pickle=True)
    ro = meta["review_offsets"].astype(np.int64)
    h = h5py.File(a.text_h5, "r")
    texts = h["texts"]
    report = []
    with torch.no_grad():
        for g in gids:
            n = int(L[g]); xb = torch.from_numpy(np.ascontiguousarray(X[g:g + 1, :n])).to(dev)
            o = model(xb, torch.zeros(1, n, dtype=torch.bool, device=dev))
            A = o["anchors"][0].float().cpu().numpy()                     # (K, D)
            M = o["M"][0].float().cpu().numpy()                           # (n, K)
            Z = l2(np.asarray(X[g, :n], dtype=np.float32))                # (n, D)
            sims = Z @ A.T                                                # (n, K)
            gram = A @ A.T
            entry = {"game": F.names[g], "gidx": int(g), "pack": n, "doclen": int(doclen[g]),
                     "anchor_cos": np.round(gram[np.triu_indices(len(A), 1)], 3).tolist(), "anchors": []}
            print(f"\n=== {F.names[g]} (gidx {g}, {n} sentences, doc prefix {int(doclen[g])}) | anchor cos {entry['anchor_cos']}")
            for k in range(len(A)):
                top = np.argsort(-sims[:, k])[: a.top]
                items = []
                revs = set()
                for i in top:
                    r = int(rows[g, i])
                    if r < 0:
                        txt, rid = "[store-page sentence]", "doc"
                    else:
                        txt = texts[r].decode("utf-8", "replace") if isinstance(texts[r], bytes) else str(texts[r])
                        rid = int(np.searchsorted(ro, r, side="right") - 1); revs.add(rid)
                    items.append({"cos": round(float(sims[i, k]), 3), "M": round(float(M[i, k]), 2), "review": rid, "text": txt[:160]})
                mass = float(M[:, k].sum()); members = int((M[:, k] > 0.5).sum())
                print(f"  -- anchor {k}: mass {mass:.0f}, members {members}, top-{a.top} from {len(revs)} reviews")
                for it in items:
                    print(f"     {it['cos']:+.3f} M={it['M']:.2f} r{it['review']}: {it['text']}")
                entry["anchors"].append({"k": k, "mass": mass, "members": members, "distinct_reviews_top": len(revs), "top": items})
            report.append(entry)
    out = Path(a.out) if a.out else Path(a.run) / f"inspect_step{step}_draw{a.draw}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
