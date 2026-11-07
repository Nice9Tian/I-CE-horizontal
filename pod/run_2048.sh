#!/usr/bin/env bash
# The 2048-budget comparison on the pod (after setup.sh): the original I-CE checkpoints rescored on the pod's draws,
# then design B at the 2048 budget (B0, gallery tau 0.05, B-frozen, gallery tau 0.03), ten-draw evaluations of each.
#   ROOT=/workspace/iceh nohup bash run_2048.sh > run_2048.log 2>&1 &
set -uo pipefail
ROOT="${ROOT:-/workspace/iceh}"
D="$ROOT/data/fusion_cache_w9"
export ICEH_DATA_DIR="$D" ICEH_CACHE_DIR="$ROOT/cache" ICEH_PUBLIC_REPO="$ROOT/I2CE-GameRepresention-by-SteamReviews"
cd "$ROOT/I-CE-horizontal" && git pull -q
R="$ROOT/results"; mkdir -p "$R"
log() { echo "[$(date '+%H:%M:%S')] $*"; }
TAG="--tag-emb $ROOT/cache/tag_emb.npz"
log "B2 rescoring: paper 512 checkpoint on 512 draws (pipeline check against the local 0.921 / 0.653), docfix512 on 512 and 2048, the paper's _g2048 on 2048"
python scripts/baseline_ice.py --ckpt "$ROOT/ckpts/ckpt_w9cv_wcle_i2ce_icetf_fold0_d512_fp_ep400.pt" --family pub512 --cap 512 --draws 10 --cache-dir "$ROOT/cache" --data-dir "$D" $TAG --out "$R/baseline_pub512_fold0_cap512.json"
python scripts/baseline_ice.py --ckpt "$ROOT/ckpts/ckpt_w9cv_wcle_i2ce_icetf_fold0_d512_fp_ep1400.pt" --cap 512 --draws 10 --cache-dir "$ROOT/cache" --data-dir "$D" $TAG --out "$R/baseline_docfix512_fold0_cap512.json"
python scripts/baseline_ice.py --ckpt "$ROOT/ckpts/ckpt_w9cv_wcle_i2ce_icetf_fold0_d512_fp_ep1400.pt" --cap 2048 --draws 10 --cache-dir "$ROOT/cache" --data-dir "$D" $TAG --out "$R/baseline_docfix512_fold0_cap2048.json"
python scripts/baseline_ice.py --ckpt "$ROOT/ckpts/ckpt_w9cv_wcle_i2ce_icetf_fold0_g2048_fp_ep1750.pt" --cap 2048 --draws 10 --cache-dir "$ROOT/cache" --data-dir "$D" $TAG --out "$R/baseline_g2048_fold0_cap2048.json"
TRAIN="--K 64 --cap 2048 --data-dir $D --cache-dir $ROOT/cache_pool2048 --out-root $ROOT/runs"
EVAL="--cap 2048 --draws 10 --data-dir $D --cache-dir $ROOT/cache $TAG"
log "B0 at 2048: k-means codebook, step 0 + evaluation"
python scripts/train.py --name b64_k0_2048 --steps 0 $TRAIN > "$ROOT/runs_b64_k0_2048.log" 2>&1 && \
python scripts/eval_run.py --run "$ROOT/runs/b64_k0_2048" --step 0 $EVAL > "$ROOT/runs_b64_k0_2048_eval.log" 2>&1 || log "b64_k0_2048 failed"
for TAU in 0.05 0.03; do
  NAME="b64_gallery2048_t${TAU#0.}"
  log "B-gallery at 2048, tau $TAU: 10k steps + evaluation"
  python scripts/train.py --name "$NAME" --gallery full --tau "$TAU" --steps 10000 --warmup 500 --eval-every 1000 --save-every 1000 $TRAIN > "$ROOT/runs_$NAME.log" 2>&1 && \
  python scripts/eval_run.py --run "$ROOT/runs/$NAME" --step best $EVAL > "$ROOT/runs_${NAME}_eval.log" 2>&1 || log "$NAME failed"
  if [ "$TAU" = "0.05" ]; then
    log "B-frozen at 2048: 10k steps + evaluation"
    python scripts/train.py --name b64_frozen_2048 --steps 10000 --warmup 500 --eval-every 1000 --save-every 1000 $TRAIN > "$ROOT/runs_b64_frozen_2048.log" 2>&1 && \
    python scripts/eval_run.py --run "$ROOT/runs/b64_frozen_2048" --step best $EVAL > "$ROOT/runs_b64_frozen_2048_eval.log" 2>&1 || log "b64_frozen_2048 failed"
  fi
done
cp "$ROOT"/runs/*/eval_cap2048_step*.json "$R/" 2>/dev/null
log "run_2048 done"
