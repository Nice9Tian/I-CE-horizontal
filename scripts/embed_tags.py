# -*- coding: utf-8 -*-
"""Embed the 23 coarse tag names with the paper's embedder (Qwen3-Embedding-0.6B, last-token
pooling, no instruction), raw and as short phrases, and store them raw and row-standardised.

    python scripts/embed_tags.py [--out D:\\ice_horizontal_cache\\tag_emb.npz]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ice_h.data import CACHE_DIR, DATA_DIR, Fixtures, rown  # noqa: E402


def phrase(name: str) -> str:
    """the uniform expansion rule: 'Roguelike' -> 'a roguelike game'; slashes read as 'or'"""
    t = name.replace("/", " or ").lower()
    article = "an" if (t[0] in "aeiou" or t.startswith("rpg")) else "a"
    return f"{article} {t}" if t.endswith("game") else f"{article} {t} game"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(CACHE_DIR) / "tag_emb.npz"))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    F = Fixtures(Path(a.data_dir), load_wiki_docs=False, load_queries=False)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(a.model, device="cuda")
    names = F.tag_names
    phrases = [phrase(n) for n in names]
    e_name = m.encode(names, convert_to_numpy=True, normalize_embeddings=False, batch_size=32).astype(np.float32)
    e_phr = m.encode(phrases, convert_to_numpy=True, normalize_embeddings=False, batch_size=32).astype(np.float32)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, names=np.array(names, dtype=object), phrases=np.array(phrases, dtype=object),
             name_raw=e_name, phrase_raw=e_phr, name_std=rown(e_name), phrase_std=rown(e_phr))
    print("pooling:", m[1].get_config_dict() if len(m) > 1 else "?")
    print("wrote", a.out, e_name.shape)
    for n, p in zip(names, phrases):
        print(f"  {n:24s} -> {p}")
