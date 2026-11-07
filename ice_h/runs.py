# -*- coding: utf-8 -*-
"""Building a model from a run's saved arguments and loading its checkpoints (both designs)."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .codebook import TopicCodebook
from .model import LatentArray


def build_model(a: dict, dev):
    if a.get("design", "A") == "B":
        return TopicCodebook(K=a["K"], c=a["c"], eps=a["eps"], lam=a["lam"], beta=a["beta"], rho=a.get("rho", 0.0) or None,
                             assign_map=a.get("assign_map", False), film=a.get("film", False), iters=a["sinkhorn_iters"],
                             gain_init=a.get("gain_init", 3.0), gain_min=a.get("gain_min", 2.0), gain_max=a.get("gain_max", 8.0),
                             dual_lr=a.get("dual_lr", 0.2), implicit_grad=not a.get("no_implicit_grad", False),
                             center=a.get("center", True)).to(dev)
    return LatentArray(K=a["K"], d_model=a["d_model"], heads=a["heads"], rounds=a["rounds"], dropout=a["dropout"], c=a["c"],
                       eps=a["eps"], sinkhorn_iters=a["sinkhorn_iters"], rho=a["rho"], self_attn=a.get("self_attn", False),
                       ot=a.get("ot", "fermi"), anchor_mode=a.get("anchor_mode", "decode"), center=a.get("center", False)).to(dev)


def load_state(model, sd: dict, design: str):
    sd = dict(sd)
    if "W.weight" in sd:                                                       # design-A checkpoints from before W = I + dW
        w = sd.pop("W.weight")
        sd["dW.weight"] = w - torch.eye(w.shape[0])
    if design == "B":
        model.load_state_dict(sd, strict=True)                                 # buffers (mu, g_bar, proto_mean) must all be there
    else:
        model.load_state_dict({k: v for k, v in sd.items() if k in model.state_dict()}, strict=False)


def load_model(run: Path, step, dev):
    """(model in eval mode, step, args dict); step 'best' reads runs/<name>/selection.json"""
    run = Path(run)
    if step in (None, "best"):
        step = json.loads((run / "selection.json").read_text())["best_step"]
    ck = torch.load(run / f"ckpt_step{step}.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    m = build_model(a, dev)
    load_state(m, ck["model"], a.get("design", "A"))
    m.eval()
    return m, int(step), a
