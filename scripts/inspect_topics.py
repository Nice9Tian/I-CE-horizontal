# -*- coding: utf-8 -*-
"""Human inspection of the topics (plan section 10, design B): for a few test games of one evaluation draw, every
topic's highest-membership sentences across those games, and every game's heaviest topics with sample sentences.
Needs the full-pool draw cache (rows{k}.npy gives the full-pool row of every pack sentence) and the row-aligned text store.

    python scripts/inspect_topics.py --run runs/b64_frozen [--step best] --games 5 [--draw 0] [--top 8]
        [--cache-dir D:/ice_horizontal_cache] [--text-h5 "D:/studable query latent/game_review_data/build_new_gamedata/text_h5.h5"]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ice_h.data import CACHE_DIR, DATA_DIR, Fixtures, load_draw  # noqa: E402
from ice_h.runs import load_model  # noqa: E402

TEXT_H5 = r"D:\studable query latent\game_review_data\build_new_gamedata\text_h5.h5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", default="best")
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--game-ids", default=None, help="comma-separated game indices instead of random test-fold games")
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--top", type=int, default=8, help="sentences per topic (across the games)")
    ap.add_argument("--per-game-topics", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--text-h5", default=TEXT_H5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import h5py
    dev = torch.device("cuda")
    model, step, targs = load_model(Path(a.run), a.step, dev)
    assert targs.get("design") == "B", "inspect_topics is for design B; use inspect_anchors.py for design A"
    F = Fixtures(Path(a.data_dir), load_wiki_docs=False, load_queries=False)
    X, L, rows, doclen = load_draw(a.cap, a.draw, Path(a.cache_dir))
    prog = json.loads((Path(a.cache_dir) / f"draws_cap{a.cap}" / "progress.json").read_text())
    assert prog.get("source") != "pool", "the pool-built selection pack has no full-pool row ids; point --cache-dir at the full-pool cache"
    test, _ = F.fold_games(targs["fold"])
    if a.game_ids:
        gids = [int(x) for x in a.game_ids.split(",")]
    else:
        rng = np.random.default_rng(a.seed)
        gids = sorted(rng.choice(sorted(F.n2i[n] for n in test), a.games, replace=False).tolist())
    meta = np.load(Path(a.data_dir) / "full_pool_meta.npz", allow_pickle=True)
    ro = meta["review_offsets"].astype(np.int64)
    h = h5py.File(a.text_h5, "r")
    texts = h["texts"]
    K = model.K

    def text_of(g, i):
        r = int(rows[g, i])
        if r < 0:
            return "[store-page sentence]", "doc"
        t = texts[r]
        t = t.decode("utf-8", "replace") if isinstance(t, bytes) else str(t)
        return t[:160], int(np.searchsorted(ro, r, side="right") - 1)

    per_game = {}
    pool = []                                                   # (M, game, i, k)
    with torch.no_grad():
        for g in gids:
            n = int(L[g]); xb = torch.from_numpy(np.ascontiguousarray(X[g:g + 1, :n])).to(dev)
            o = model(xb, torch.zeros(1, n, dtype=torch.bool, device=dev))
            M = o["M"][0].float().cpu().numpy(); hist = o["hist"][0].float().cpu().numpy(); cov = o["cov"][0].float().cpu().numpy()
            per_game[g] = {"M": M, "hist": hist, "cov": cov, "n": n}
            for k in range(K):
                for i in np.flatnonzero(M[:, k] > 0.2):
                    pool.append((float(M[i, k]), g, int(i), k))
    report = {"run": a.run, "step": step, "draw": a.draw, "games": [F.names[g] for g in gids], "topics": [], "per_game": []}
    print(f"== {len(gids)} test games, K = {K}, draw {a.draw}, step {step}")
    mass = np.zeros(K)
    for g in gids:
        mass += per_game[g]["M"].sum(0)
    share = mass / mass.sum()
    for k in np.argsort(-share):
        items = sorted([p for p in pool if p[3] == k], key=lambda p: -p[0])[: a.top]
        ent = {"k": int(k), "share": float(share[k]), "games_covering": int(sum(per_game[g]["cov"][k] > 0.5 for g in gids)), "top": []}
        print(f"\n-- topic {k}: share {share[k]:.3f} over these games, covered by {ent['games_covering']}/{len(gids)}")
        for Mv, g, i, _ in items:
            txt, rid = text_of(g, i)
            ent["top"].append({"M": round(Mv, 2), "game": F.names[g], "review": rid, "text": txt})
            print(f"   M={Mv:.2f} [{F.names[g][:28]:28s}] r{rid}: {txt}")
        report["topics"].append(ent)
    for g in gids:
        pg = per_game[g]
        order = np.argsort(-pg["hist"])[: a.per_game_topics]
        eff = float(np.exp(-(pg["hist"][pg["hist"] > 0] * np.log(pg["hist"][pg["hist"] > 0])).sum()))
        ent = {"game": F.names[g], "pack": pg["n"], "doclen": int(doclen[g]), "eff_topics": eff, "topics_covered": int((pg["cov"] > 0.5).sum()), "top_topics": []}
        print(f"\n=== {F.names[g]} ({pg['n']} sentences, doc prefix {int(doclen[g])}): effective topics {eff:.1f}, covered {ent['topics_covered']}")
        for k in order:
            idx = np.argsort(-pg["M"][:, k])[:3]
            sents = []
            for i in idx:
                txt, rid = text_of(g, int(i)); sents.append({"M": round(float(pg["M"][i, k]), 2), "review": rid, "text": txt})
            ent["top_topics"].append({"k": int(k), "h": round(float(pg["hist"][k]), 3), "sentences": sents})
            print(f"  -- topic {k} (h {pg['hist'][k]:.3f})")
            for s_ in sents:
                print(f"     M={s_['M']:.2f} r{s_['review']}: {s_['text']}")
        report["per_game"].append(ent)
    out = Path(a.out) if a.out else Path(a.run) / f"inspect_topics_step{step}_draw{a.draw}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
