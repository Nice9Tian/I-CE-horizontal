#!/usr/bin/env bash
# Pod bootstrap for the design-B experiments (plan sections 5, 9, 12: the 2048 budget).
#   ROOT=/workspace/iceh bash setup.sh
# 1. code: this repository and the paper's public repository (the tag probe imports its text_variant_eval.py)
# 2. data from the public Hugging Face dataset at the pinned revision: the 2048-sentence review pool and the small
#    files (training), the 140 GiB full pool in four parts + join (evaluation draws)
# 3. the evaluation caches: ten 512-sentence and ten 2048-sentence draws of every game, the tag-name embeddings
set -euo pipefail
ROOT="${ROOT:-/workspace/iceh}"
REPO=chironYu/latenarray-i2ce-data
REV="${HF_REV:-35cc5dd4e3573c45842d06d8f8f9a29c85f49abb}"
D="$ROOT/data/fusion_cache_w9"
export PIP_BREAK_SYSTEM_PACKAGES=1
export ICEH_DATA_DIR="$D" ICEH_CACHE_DIR="$ROOT/cache" ICEH_PUBLIC_REPO="$ROOT/I2CE-GameRepresention-by-SteamReviews"
mkdir -p "$D" "$ROOT/cache"
cd "$ROOT"
[ -d I-CE-horizontal ] || git clone --depth 1 https://github.com/Nice9Tian/I-CE-horizontal.git
[ -d I2CE-GameRepresention-by-SteamReviews ] || git clone --depth 1 https://github.com/Nice9Tian/I2CE-GameRepresention-by-SteamReviews.git
python -m pip install -q "huggingface_hub>=0.34" numpy scipy scikit-learn h5py pytest sentence-transformers
hfget() { hf download "$REPO" "$@" --repo-type dataset --revision "$REV" --local-dir "$D"; }
echo "== small and medium files"
hfget games.npz wiki_eval_split.json sp_raw_views.npz wiki_clean_views.npz wiki_eval.npz tag_labels.npz \
      wscan_pool_rev.npy wscan_pool_rev_rid.npy wscan_pool_rev_len.npy full_pool_meta.npz SHA256SUMS join_full_pool.py
echo "== full pool (4 parts, 150 GB, then joined and hashed by join_full_pool.py)"
POOL_BYTES=150096930944
size_of() { [ -f "$1" ] && stat -c %s "$1" || echo 0; }
if [ "$(size_of "$D/full_pool_fp16.npy")" != "$POOL_BYTES" ]; then
  df -h "$D"
  hfget full_pool_fp16.npy.part00 full_pool_fp16.npy.part01 full_pool_fp16.npy.part02 full_pool_fp16.npy.part03
  (cd "$D" && python join_full_pool.py)
fi
touch "$D/full_pool_READY"
cd "$ROOT/I-CE-horizontal"
python -m pytest tests -q
echo "== evaluation caches"
python scripts/build_draws.py --cap 2048 --k 10 --data-dir "$D" --out "$ROOT/cache"
python scripts/build_draws.py --cap 512 --k 10 --data-dir "$D" --out "$ROOT/cache"
python scripts/build_pool_draw.py --cap 512 --data-dir "$D" --cache-dir "$ROOT/cache_pool512"
python scripts/build_pool_draw.py --cap 2048 --data-dir "$D" --cache-dir "$ROOT/cache_pool2048"
python scripts/embed_tags.py --data-dir "$D" --out "$ROOT/cache/tag_emb.npz"
echo "setup done"
