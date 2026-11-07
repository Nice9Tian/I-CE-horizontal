# -*- coding: utf-8 -*-
"""Checkpoint selection after the fact (for runs trained with --no-eval): encode the cached draw 0 and the wiki queries
with every saved checkpoint, score the validation fold with the paper's rank rule (routed scoring for design B, slot
for A), and write runs/<name>/selection.json plus one eval.jsonl line per checkpoint.

    python scripts/select_ckpt.py --run runs/b64_frozen [--cap 512] [--steps 1000,2000,...]
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
from ice_h.evaluate import encode_queries, encode_sets, encode_topics, retrieval_metrics, topic_retrieval_metrics  # noqa: E402
from ice_h.runs import build_model, load_state  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--steps", default=None, help="comma-separated steps; default every ckpt_step*.pt in the run")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    args = ap.parse_args()
    dev = torch.device("cuda")
    run = Path(args.run)
    ckpts = sorted(run.glob("ckpt_step*.pt"), key=lambda p: int(p.stem.split("step")[-1]))
    if args.steps:
        want = {int(x) for x in args.steps.split(",")}
        ckpts = [p for p in ckpts if int(p.stem.split("step")[-1]) in want]
    assert ckpts, "no checkpoints"
    F = Fixtures(Path(args.data_dir))
    X0, L0, _, _ = load_draw(args.cap, 0, Path(args.cache_dir)); X0 = np.ascontiguousarray(X0)
    SQ, _ = F.queries_std()
    selection = {"best_step": None, "best_score": -1.0, "scores": {}}           # recomputed from scratch for the checkpoints scored here
    model = None
    with open(run / "eval.jsonl", "a", encoding="utf-8") as ef:
        for p in ckpts:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            a = ck["args"]
            design = a.get("design", "A")
            if model is None:
                model = build_model(a, dev)
                fold = a["fold"]
            load_state(model, ck["model"], design); model.eval()
            t0 = time.time()
            if design == "B":
                m = topic_retrieval_metrics(encode_topics(model, SQ, F.QL, dev, batch=128), encode_topics(model, X0, L0, dev), F, fold, device=dev)
                score = m["routed_norm_val_rank_score"]
                line = f"routed_norm h1 name {m['routed_norm_h1_neutral']:.3f} stripped {m['routed_norm_h1_noname']:.3f} | shrink {m['routed_h1_noname']:.3f} mean {m['mean_h1_noname']:.3f} hist {m['hist_h1_noname']:.3f}"
            else:
                m = retrieval_metrics(encode_queries(model, F, dev), encode_sets(model, X0, L0, dev), F, fold, device=dev)
                score = m["slot_val_rank_score"]
                line = f"slot h1 name {m['slot_h1_neutral']:.3f} stripped {m['slot_h1_noname']:.3f} | mean {m['mean_h1_noname']:.3f} chamfer {m['chamfer_h1_noname']:.3f}"
            step = int(ck["step"])
            rec = {"step": step, **{k: round(v, 4) for k, v in m.items()}, "sec": round(time.time() - t0, 1), "source": "select_ckpt"}
            ef.write(json.dumps(rec) + "\n"); ef.flush()
            selection["scores"][str(step)] = score
            if score > selection["best_score"] or (score == selection["best_score"] and selection["best_step"] is not None and step < selection["best_step"]):
                selection.update(best_step=step, best_score=score)
            print(f"step {step:6d}: val rank {score:8.2f} | test {line}", flush=True)
    (run / "selection.json").write_text(json.dumps(selection, indent=1), encoding="utf-8")
    print("best step", selection["best_step"], "score", round(selection["best_score"], 2), "->", run / "selection.json")


if __name__ == "__main__":
    main()
