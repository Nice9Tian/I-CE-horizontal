# -*- coding: utf-8 -*-
"""Print the plan's section-9 comparison table from the evaluation JSONs.

    python scripts/compare.py --model runs/k6_fold0/eval_cap512_step*.json \
        --baseline D:\\ice_horizontal_cache\\baseline_docfix512_i2ce_icetf_fold0_cap512.json \
        [--baseline <pub512 json>] [--md out.md]

Each row: mean ± std over the draws.  Model rows: slot / mean / chamfer scoring; baseline rows: the
paper's tower (B2) and the frozen mean (B1) from the same draws; B3 random groups from the model file.
"""
import argparse
import glob
import json
from pathlib import Path


def fmt(v):
    return "—" if v is None else f"{v['mean']:.3f} ± {v['std']:.3f}"


def pick(d, *keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--baseline", nargs="*", default=[])
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    cols = [("Name@1", "h1_neutral"), ("Name@5", "h5_neutral"), ("Stripped@1", "h1_noname"), ("Stripped@5", "h5_noname"),
            ("tag F1 (F1 均值)", "tag_f1_mean"), ("tag F1 (F3 拼接)", "tag_f1_concat"), ("R3 零样本 mAP", "zs_map")]
    lines = ["| 方法 | 锚点预算 | " + " | ".join(c for c, _ in cols) + " |", "| --- | --- | " + " | ".join("---" for _ in cols) + " |"]
    rows = []
    for pat in a.baseline:
        for f in glob.glob(pat):
            b = json.loads(Path(f).read_text(encoding="utf-8"))
            t = b["tower"]
            rows.append((f"B2 I-CE@512 {b.get('family', '')} {b['arm']}（重打分）", b["cap"],
                         {"h1_neutral": t["h1_neutral"], "h5_neutral": t["h5_neutral"], "h1_noname": t["h1_noname"], "h5_noname": t["h5_noname"], "tag_f1_mean": t["tag_f1"]}))
            fz = b["frozen"]
            rows.append(("B1 原始 Qwen 均值", b["cap"], {"h1_neutral": fz["h1_neutral"], "h5_neutral": fz["h5_neutral"], "h1_noname": fz["h1_noname"], "h5_noname": fz["h5_noname"],
                                                     "tag_f1_mean": fz["tag_f1"], "zs_map": fz.get("zs_map")}))
    for pat in a.model:
        for f in glob.glob(pat):
            m = json.loads(Path(f).read_text(encoding="utf-8"))
            K, cap, mm = m["K"], m["cap"], m["model"]
            for mode, label in (("slot", "槽位余弦"), ("mean", "均值池化"), ("chamfer", "Chamfer")):
                r = {k: mm.get(f"{mode}_{k}") for k in ("h1_neutral", "h5_neutral", "h1_noname", "h5_noname")}
                if mode == "slot":
                    r.update(tag_f1_mean=mm.get("tag_f1_mean"), tag_f1_concat=mm.get("tag_f1_concat"), zs_map=mm.get("zs_map"))
                rows.append((f"本方法（K={K}，{label}）step {m['step']}", cap, r))
            if "random" in m:
                rr = m["random"]
                rows.append((f"B3 随机分组（K={K}，Chamfer）", cap, {k: rr.get(f"chamfer_{k}") for k in ("h1_neutral", "h5_neutral", "h1_noname", "h5_noname")}
                             | {"tag_f1_concat": rr.get("tag_f1_concat"), "zs_map": rr.get("zs_map")}))
    for name, cap, r in rows:
        lines.append(f"| {name} | {cap} | " + " | ".join(fmt(r.get(k)) for _, k in cols) + " |")
    table = "\n".join(lines)
    print(table)
    if a.md:
        Path(a.md).write_text(table + "\n", encoding="utf-8"); print("wrote", a.md)


if __name__ == "__main__":
    main()
