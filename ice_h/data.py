# -*- coding: utf-8 -*-
"""Data for the Steam text-version overlapping-group I-CE experiment.

Everything is read from the original I-CE data folder (read-only), by default
``D:\\runpod_data\\fusion_cache_w9``:

    games.npz                     the 2,020 games, the index basis
    wiki_eval_split.json          the 814-game evaluation universe (train/val/test appids)
    sp_raw_views.npz              store-page sentences for 1,811 games (doc prefix of a pack)
    wiki_clean_views.npz          wiki article sentences for 814 games (training doc view)
    wiki_eval.npz                 the LLM-rewritten wiki queries, 814 games x 4 variants
    tag_labels.npz                23 coarse tags per game, plus the tag names
    wscan_pool_rev.npy/_rid/_len  the 2,048-sentence review pool: the TRAINING source
    full_pool_fp16.npy + meta     all 73 M review sentences: the EVALUATION draw source

All sentence vectors are row-standardised (``rown``: per-row mean 0, std 1) before
they enter the model, exactly as in the paper's workers and in k10_eval.py.  The
review pool on disk is already standardised; store-page, wiki and full-pool rows
are standardised here when loaded.

Fold protocol (the paper's five-fold, k10_eval.Fixtures.fold_games): the 814-game
universe is permuted once with seed 20260711 and split into five folds; fold k is
the test set, fold (k+1) % 5 the validation set, every other game (1,694 or 1,695,
including the 1,206 games without a wiki article) is the training pool.
"""
from __future__ import annotations

import json
import os
import threading
import queue
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.environ.get("ICEH_DATA_DIR", r"D:\runpod_data\fusion_cache_w9"))
CACHE_DIR = Path(os.environ.get("ICEH_CACHE_DIR", r"D:\ice_horizontal_cache"))
PUBLIC_REPO = Path(os.environ.get("ICEH_PUBLIC_REPO", r"C:\release\I2CE-GameRepresention-by-SteamReviews"))

SPLIT_SEED, NF = 20260711, 5          # the paper's split seed and fold count
DRAW_SEED = SPLIT_SEED + 5            # k10_eval: default_rng(SPLIT_SEED + 5 + 1000 k + g)
GAP, SPAN = 2048, 65536               # k10_eval.read_rows: merge reads closer than GAP, at most SPAN rows per read
DOC_ID_SP, DOC_ID_WIKI = 100_000, 100_001   # doc ids of description sentences (reviews use their pool id)
TYPE_REVIEW, TYPE_DESC, TYPE_WIKI = 0, 1, 2   # review sentence, store-page sentence, wiki sentence


def rown(x, eps=1e-6):
    """row standardisation, numpy version (float32 out); ddof = 1 like torch.std in the paper's scripts"""
    x = np.asarray(x, dtype=np.float32)
    m = x.mean(-1, keepdims=True)
    s = x.std(-1, keepdims=True, ddof=1)
    return (x - m) / (s + eps)


def rown_t(x, eps=1e-6):
    """row standardisation, torch version (float32 out)"""
    x = x.float()
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + eps)


# ---------------------------------------------------------------------------- fixtures
class Fixtures:
    """games, folds, documents, queries and tags; mirrors k10_eval.Fixtures without torch"""

    def __init__(self, data_dir: Path = DATA_DIR, load_wiki_docs: bool = True, load_queries: bool = True):
        self.dir = Path(data_dir)
        self.names = [str(x) for x in np.load(self.dir / "games.npz", allow_pickle=True)["names"]]
        self.NG = len(self.names)
        self.n2i = {n: i for i, n in enumerate(self.names)}
        self.appid2name = {n.split("_")[0]: n for n in self.names}
        self.split = json.loads((self.dir / "wiki_eval_split.json").read_text(encoding="utf-8"))
        self.universe = sorted(set(self.split["test"]) | set(self.split["val"]) | set(self.split["train"]))
        self.folds = np.array_split(np.random.default_rng(SPLIT_SEED).permutation(len(self.universe)), NF)
        # store-page sentences (the doc prefix of every pack), standardised once
        ST = np.load(self.dir / "sp_raw_views.npz", allow_pickle=True)
        self.sp_row = {int(ST["gidx"][i]): i for i in range(len(ST["gidx"]))}
        self.S_sp = ST["S"]                        # fp16 (1811, 136, 1024), raw
        self.L_sp = ST["S_len"].astype(np.int64)
        self._sp_cache: dict[int, np.ndarray] = {}
        # wiki article sentences (training-time document view of training games)
        self.wiki_row, self.S_wiki, self.L_wiki = {}, None, None
        if load_wiki_docs:
            W = np.load(self.dir / "wiki_clean_views.npz", allow_pickle=True)
            self.wiki_row = {int(W["gidx"][i]): i for i in range(len(W["gidx"]))}
            self.S_wiki, self.L_wiki = W["S"], W["S_len"].astype(np.int64)
        self._wiki_cache: dict[int, np.ndarray] = {}
        # the evaluation queries
        self.SA = self.mA_len = self.variants = self.art = self.gA = None
        if load_queries:
            A = np.load(self.dir / "wiki_eval.npz", allow_pickle=True)
            self.SA = A["S"]                        # fp16 (3256, 83, 1024), raw; standardised on use
            self.QL = A["S_len"].astype(np.int64)
            self.variants = [str(x) for x in A["variants"]]
            self.art = [str(x) for x in A["names"]]
            self.gA = np.asarray(A["gidx"]).astype(np.int64)
        T = np.load(self.dir / "tag_labels.npz", allow_pickle=True)
        self.y = T["y"].astype(np.int8)
        self.tag_names = [str(x) for x in T["tag_names"]]
        assert [str(x) for x in T["names"]] == self.names, "tag_labels.npz is not in games.npz order"

    # ---- folds
    def fold_games(self, k: int) -> tuple[set, set]:
        test = {self.appid2name[self.universe[i]] for i in self.folds[k]}
        val = {self.appid2name[self.universe[i]] for i in self.folds[(k + 1) % NF]}
        return test, val

    def fold_indices(self, k: int) -> dict:
        test, val = self.fold_games(k)
        held = test | val
        return {
            "test": np.array(sorted(self.n2i[n] for n in test), dtype=np.int64),
            "val": np.array(sorted(self.n2i[n] for n in val), dtype=np.int64),
            "train": np.array([i for i, n in enumerate(self.names) if n not in held], dtype=np.int64),
        }

    def query_rows(self, games: set, variant: str) -> np.ndarray:
        return np.array([i for i in range(len(self.art)) if self.art[i] in games and self.variants[i] == variant], dtype=np.int64)

    # ---- documents
    def doc(self, g: int):
        """store-page sentences of game g, standardised fp16 (L, 1024), or None"""
        if g not in self.sp_row:
            return None
        if g not in self._sp_cache:
            i = self.sp_row[g]
            self._sp_cache[g] = rown(self.S_sp[i, : self.L_sp[i]]).astype(np.float16)
        return self._sp_cache[g]

    def wiki_doc(self, g: int):
        """wiki article sentences of game g, standardised fp16, or None"""
        if g not in self.wiki_row:
            return None
        if g not in self._wiki_cache:
            i = self.wiki_row[g]
            self._wiki_cache[g] = rown(self.S_wiki[i, : self.L_wiki[i]]).astype(np.float16)
        return self._wiki_cache[g]

    def queries_std(self):
        """all 3,256 queries standardised: fp16 (3256, 83, 1024) and the mask (True = padding)"""
        S = rown(self.SA).astype(np.float16)
        mask = np.arange(S.shape[1])[None, :] >= self.QL[:, None]
        S[mask] = 0
        return S, mask


# ---------------------------------------------------------------------------- the training sampler
class PoolSampler:
    """Two disjoint views per game per step, drawn from the in-RAM 2,048-sentence review pool.

    weak view  = store-page prefix (all of it) + whole reviews, in a fresh random order,
                 a review entering if it still fits under ``cap`` (the k10_eval pack rule)
    strong view = with probability p_doc the document view (wiki article if the game has
                 one, else the store page, in which case the weak pack drops its prefix so
                 the two views share no sentence); otherwise whole reviews NOT used by the
                 weak view, accumulated to at least ns_min sentences (at most ns_max)

    Review sentences carry their pool review id as doc id; description sentences carry
    DOC_ID_SP / DOC_ID_WIKI.  Types: TYPE_REVIEW / TYPE_DESC (store page) / TYPE_WIKI.
    """

    def __init__(self, F: Fixtures, train_games: np.ndarray, cap: int = 512, ns_min: int = 64, ns_max: int = 160,
                 p_doc: float = 0.25, seed: int = 0, data_dir: Path = DATA_DIR, pool=None, rid=None, plen=None,
                 mmap: bool = True):
        self.F, self.cap, self.ns_min, self.ns_max, self.p_doc = F, cap, ns_min, ns_max, p_doc
        self.train_games = np.asarray(train_games, dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        d = Path(data_dir)
        # memory-mapped by default: on an NVMe mirror the OS page cache serves the rows and several
        # processes share one copy; mmap=False reads the 8.5 GB into this process
        self.POOL = pool if pool is not None else np.load(d / "wscan_pool_rev.npy", mmap_mode="r" if mmap else None)
        RID = rid if rid is not None else np.load(d / "wscan_pool_rev_rid.npy")
        PL = plen if plen is not None else np.load(d / "wscan_pool_rev_len.npy")
        assert self.POOL.shape[0] == F.NG
        # review segments per game: (start, length, review id), contiguous runs of one id
        self.segs = []
        for g in range(F.NG):
            r = np.asarray(RID[g, : int(PL[g])])
            if len(r) == 0:
                self.segs.append((np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)))
                continue
            brk = np.flatnonzero(np.diff(r) != 0) + 1
            starts = np.concatenate([[0], brk]).astype(np.int64)
            ends = np.concatenate([brk, [len(r)]]).astype(np.int64)
            self.segs.append((starts, ends - starts, r[starts].astype(np.int64)))

    # ---- one game
    def _views(self, g: int):
        starts, lens, rids = self.segs[g]
        order = self.rng.permutation(len(lens))
        use_doc_strong = self.rng.random() < self.p_doc
        wiki = self.F.wiki_doc(g) if use_doc_strong else None
        sp = self.F.doc(g)
        strong_doc = None
        if use_doc_strong:
            if wiki is not None:
                strong_doc, strong_doc_id = wiki, DOC_ID_WIKI
            elif sp is not None:
                strong_doc, strong_doc_id = sp, DOC_ID_SP
                sp = None                       # the weak pack drops the prefix: no shared sentence
        # weak pack: prefix + whole reviews under the cap
        row = 0 if sp is None else len(sp)
        taken = np.zeros(len(lens), bool)
        w_segs = []
        for j in order:
            L = int(lens[j])
            if row + L <= self.cap:
                w_segs.append(j); taken[j] = True; row += L
                if row == self.cap:
                    break
        w_rows = np.concatenate([np.arange(starts[j], starts[j] + lens[j]) for j in w_segs]) if w_segs else np.zeros(0, np.int64)
        w_doc = np.concatenate([np.repeat(rids[j], lens[j]) for j in w_segs]) if w_segs else np.zeros(0, np.int64)
        xw = self.POOL[g][w_rows]                                       # (n, 1024) fp16 copy
        tw = np.zeros(len(w_rows), np.int8)
        if sp is not None:
            xw = np.concatenate([sp, xw]); w_doc = np.concatenate([np.full(len(sp), DOC_ID_SP), w_doc])
            tw = np.concatenate([np.ones(len(sp), np.int8), tw])
        # strong view
        if strong_doc is not None:
            strong_doc = strong_doc[: self.ns_max]                              # long articles: the first ns_max sentences
            xs, ds = strong_doc, np.full(len(strong_doc), strong_doc_id, np.int64)
            ts = np.full(len(strong_doc), TYPE_WIKI if strong_doc_id == DOC_ID_WIKI else TYPE_DESC, np.int8)
        else:
            rest = [j for j in order if not taken[j]]
            fit = [j for j in rest if lens[j] <= self.ns_max]
            tot, s_segs = 0, []
            for j in fit:
                L = int(lens[j])
                if tot and tot + L > self.ns_max:
                    continue
                s_segs.append(j); tot += L
                if tot >= self.ns_min:
                    break
            if not s_segs and rest:                                             # only over-long reviews remain: the shortest, cut to ns_max
                j = min(rest, key=lambda j: lens[j])
                s_rows = np.arange(starts[j], starts[j] + min(int(lens[j]), self.ns_max)); s_segs = [j]
                xs = self.POOL[g][s_rows]; ds = np.full(len(s_rows), rids[j], np.int64); ts = np.zeros(len(s_rows), np.int8)
                return xw, w_doc.astype(np.int64), tw, xs, ds, ts
            s_rows = np.concatenate([np.arange(starts[j], starts[j] + lens[j]) for j in s_segs]) if s_segs else np.zeros(0, np.int64)
            xs = self.POOL[g][s_rows]
            ds = np.concatenate([np.repeat(rids[j], lens[j]) for j in s_segs]) if s_segs else np.zeros(0, np.int64)
            ts = np.zeros(len(s_rows), np.int8)
        return xw, w_doc.astype(np.int64), tw, xs, ds, ts

    # ---- one batch
    def sample(self, gids=None, B: int = 32) -> dict:
        if gids is None:
            gids = self.rng.choice(self.train_games, B, replace=False)
        gids = np.asarray(gids, dtype=np.int64)
        views = [self._views(int(g)) for g in gids]
        Lw = max(v[0].shape[0] for v in views); Ls = max(v[3].shape[0] for v in views)
        B = len(gids)
        out = {
            "gids": gids,
            "xw": np.zeros((B, Lw, 1024), np.float16), "lw": np.zeros(B, np.int64),
            "dw": np.full((B, Lw), -1, np.int64), "tw": np.full((B, Lw), -1, np.int8),
            "xs": np.zeros((B, Ls, 1024), np.float16), "ls": np.zeros(B, np.int64),
            "ds": np.full((B, Ls), -1, np.int64), "ts": np.full((B, Ls), -1, np.int8),
        }
        for b, (xw, dw, tw, xs, ds, ts) in enumerate(views):
            n, m = len(xw), len(xs)
            out["xw"][b, :n] = xw; out["lw"][b] = n; out["dw"][b, :n] = dw; out["tw"][b, :n] = tw
            out["xs"][b, :m] = xs; out["ls"][b] = m; out["ds"][b, :m] = ds; out["ts"][b, :m] = ts
        return out


def pool_global_mean(POOL, plen, games, max_games: int = 400, seed: int = 0) -> np.ndarray:
    """mean standardised sentence vector over (a sample of) the training games' pool rows: the centring vector"""
    rng = np.random.default_rng(seed)
    sel = rng.choice(np.asarray(games), min(max_games, len(games)), replace=False)
    acc = np.zeros(POOL.shape[2], np.float64); n = 0
    for g in sel:
        x = np.asarray(POOL[int(g), : int(plen[int(g)])], dtype=np.float32)
        acc += x.sum(0); n += x.shape[0]
    return (acc / max(n, 1)).astype(np.float32)


class Prefetcher:
    """runs sampler.sample() in a thread, two batches ahead"""

    def __init__(self, sampler: PoolSampler, B: int, depth: int = 2):
        self.q: "queue.Queue[dict]" = queue.Queue(maxsize=depth)
        self.stop = threading.Event()
        self.sampler, self.B = sampler, B
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start()

    def _run(self):
        while not self.stop.is_set():
            batch = self.sampler.sample(B=self.B)
            while not self.stop.is_set():
                try:
                    self.q.put(batch, timeout=0.5); break
                except queue.Full:
                    continue

    def next(self) -> dict:
        return self.q.get()

    def close(self):
        self.stop.set()


# ---------------------------------------------------------------------------- evaluation draws (the paper's rule)
def read_rows(V, uni: np.ndarray) -> np.ndarray:
    """rows ``uni`` (sorted, unique) of the full pool; rows closer than one storage chunk share a read (k10_eval)"""
    out = np.empty((len(uni), V.shape[1]), np.float16)
    brk = np.flatnonzero(np.diff(uni) > GAP) + 1
    for i0, j0 in zip(np.concatenate([[0], brk]), np.concatenate([brk, [len(uni)]])):
        while i0 < j0:
            a = int(uni[i0]); j1 = int(np.searchsorted(uni[i0:j0], a + SPAN)) + i0
            b = int(uni[j1 - 1]) + 1
            blk = np.array(V[a:b])                   # ONE sequential read of the span; indexing the memmap
            out[i0:j1] = blk[uni[i0:j1] - a]         # directly would page-fault row by row (fatal on a USB disk)
            i0 = j1
    return out


class FullPool:
    """the 73 M-sentence store and its review offsets, memory-mapped"""

    def __init__(self, data_dir: Path = DATA_DIR):
        d = Path(data_dir)
        meta = np.load(d / "full_pool_meta.npz", allow_pickle=True)
        self.game_names = [str(x) for x in meta["game_names"]]
        self.gro = meta["game_review_offsets"].astype(np.int64)
        self.ro = meta["review_offsets"].astype(np.int64)
        self.V = np.load(d / "full_pool_fp16.npy", mmap_mode="r")
        self.e2i = {n: i for i, n in enumerate(self.game_names)}

    def reviews_of(self, name: str):
        """(start rows, lengths) of every review of the game with this name"""
        fi = self.e2i[name]; r0, r1 = int(self.gro[fi]), int(self.gro[fi + 1])
        rs = self.ro[r0:r1]; rl = self.ro[r0 + 1:r1 + 1] - self.ro[r0:r1]
        return rs, rl


def draw_plans(rs: np.ndarray, rl: np.ndarray, g: int, cap: int, K: int, doc_len: int):
    """k10_eval.encode's K independent packs of game g: [(segments [(start, L)], total rows)]"""
    plans = []
    for k in range(K):
        rk = np.random.default_rng(DRAW_SEED + 1000 * k + g)
        row, segs = doc_len, []
        for j in rk.permutation(len(rl)):
            L = int(rl[j])
            if row + L <= cap:
                segs.append((int(rs[j]), L)); row += L
                if row == cap:
                    break
        plans.append((segs, row))
    return plans


def build_draw_cache(cap: int = 512, K: int = 10, out_dir: Path | None = None, data_dir: Path = DATA_DIR,
                     games=None, log=print):
    """Materialise the K evaluation packs of every game at budget ``cap`` as standardised fp16 arrays.

    Writes to <out_dir>/draws_cap<cap>/:  draw<k>.npy (NG, cap, 1024) fp16, len<k>.npy (NG,), rows<k>.npy
    (NG, cap) int64 full-pool row of every sentence (-1 for doc prefix / padding), doclen.npy (NG,).
    Resumable: progress.json records the next game.  One pass over the games, reading each game's
    block of the full pool once (merged reads), so it is sequential enough for the USB disk.
    """
    out = Path(out_dir or CACHE_DIR) / f"draws_cap{cap}"
    out.mkdir(parents=True, exist_ok=True)
    F = Fixtures(data_dir, load_wiki_docs=False, load_queries=False)
    P = FullPool(data_dir)
    NG = F.NG if games is None else int(games)
    prog = out / "progress.json"
    if prog.exists():
        old = json.loads(prog.read_text())
        assert (old["NG"], old["cap"], old["K"]) == (NG, cap, K), f"existing cache at {out} was built with {old}; use another directory"
    mode = "r+" if prog.exists() else "w+"
    arrs = [np.lib.format.open_memmap(out / f"draw{k}.npy", mode=mode, dtype=np.float16, shape=(NG, cap, 1024)) for k in range(K)]
    lens = [np.lib.format.open_memmap(out / f"len{k}.npy", mode=mode, dtype=np.int64, shape=(NG,)) for k in range(K)]
    rows_ = [np.lib.format.open_memmap(out / f"rows{k}.npy", mode=mode, dtype=np.int64, shape=(NG, cap)) for k in range(K)]
    doclen = np.lib.format.open_memmap(out / "doclen.npy", mode=mode, dtype=np.int64, shape=(NG,))
    g0 = json.loads(prog.read_text())["done"] if prog.exists() else 0
    if g0 >= NG:
        log(f"draw cache complete at {out}"); return out
    import time
    t0 = time.time()
    for g in range(g0, NG):
        rs, rl = P.reviews_of(F.names[g])
        doc = F.doc(g); dl = 0 if doc is None else len(doc)
        plans = draw_plans(rs, rl, g, cap, K, dl)
        rows = [np.concatenate([np.arange(s, s + L) for s, L in segs]) if segs else np.zeros(0, np.int64) for segs, _ in plans]
        uni = np.unique(np.concatenate(rows))
        raw = read_rows(P.V, uni) if len(uni) else None
        doclen[g] = dl
        for k, (segs, row) in enumerate(plans):
            a = arrs[k][g]; a[:] = 0
            r = rows_[k][g]; r[:] = -1
            if doc is not None:
                a[:dl] = doc
            if len(rows[k]):
                a[dl:row] = rown(raw[np.searchsorted(uni, rows[k])]).astype(np.float16)
                r[dl:row] = rows[k]
            lens[k][g] = row
        if (g + 1) % 50 == 0 or g + 1 == NG:
            for x in arrs + lens + rows_ + [doclen]:
                x.flush()
            prog.write_text(json.dumps({"done": g + 1, "cap": cap, "K": K, "NG": NG}))
            el = time.time() - t0
            log(f"  draws {g + 1}/{NG}  {el / 60:.1f} min, {el / (g + 1 - g0) * (NG - g - 1) / 60:.0f} min to go")
    return out


def load_draw(cap: int, k: int, cache_dir: Path | None = None):
    """one cached draw: (fp16 (NG, cap, 1024) memmap, lens (NG,), rows (NG, cap), doclen (NG,))"""
    d = Path(cache_dir or CACHE_DIR) / f"draws_cap{cap}"
    prog = json.loads((d / "progress.json").read_text())
    assert prog["done"] == prog["NG"], f"draw cache at {d} is incomplete ({prog['done']}/{prog['NG']})"
    return (np.load(d / f"draw{k}.npy", mmap_mode="r"), np.load(d / f"len{k}.npy"), np.load(d / f"rows{k}.npy"),
            np.load(d / "doclen.npy"))
