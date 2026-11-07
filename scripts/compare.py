# -*- coding: utf-8 -*-
"""Print the plan's section-9 comparison table from the evaluation JSONs (designs A and B).

    python scripts/compare.py --model runs/b64_frozen/eval_cap512_step*.json runs/k6_mm_center_tau03_v2/eval_cap512_step*.json \
        --baseline D:\\ice_horizontal_cache\\baseline_docfix512_i2ce_icetf_fold0_cap512.json [--md out.md]

Each row: mean ± std over the draws.  Design-B rows: routed / routed+coverage / mean / topic-distribution scoring,
tag F1 on the mean (F1), the topic distribution (F2), both, and the topic concatenation (F3), zero-shot mAP routed
(max in brackets); B1' (centred mean) from the same file.  Design-A rows: slot / mean / chamfer.  Baseline files:
the paper's tower (B2) and the frozen mean (B1) from the same draws.
"""
import argparse
import glob
import json
from pathlib import Path


def fmt(v):
    return "—" if v is None else f"{v['mean']:.3f} ± {v['std']:.3f}"


COLS = [("Name@1", "h1_neutral"), ("Name@5", "h5_neutral"), ("Stripped@1", "h1_noname"), ("Stripped@5", "h5_noname"),
        ("tag F1 (F1 均值)", "tag_f1_mean"), ("tag F2 (话题分布)", "tag_f1_hist"), ("tag F1⊕F2", "tag_f1_mean_hist"),
        ("tag F3 (拼接)", "tag_f1_concat"), ("R3 零样本 mAP", "zs_map")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--baseline", nargs="*", default=[])
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    lines = ["| 方法 | 锚点预算 | " + " | ".join(c for c, _ in COLS) + " |", "| --- | --- | " + " | ".join("---" for _ in COLS) + " |"]
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
            K, cap, mm, design = m["K"], m["cap"], m["model"], m.get("design", "A")
            name = Path(f).parent.name
            if design == "B":
                if "frozen" in m and "h1_neutral_c" in m["frozen"]:
                    fz = m["frozen"]
                    rows.append(("B1' 中心化均值", cap, {"h1_neutral": fz["h1_neutral_c"], "h5_neutral": fz["h5_neutral_c"], "h1_noname": fz["h1_noname_c"], "h5_noname": fz["h5_noname_c"], "zs_map": fz.get("zs_map_c")}))
                routed_idx = len(rows)
                for mode, label in (("routed", "话题路由"), ("routed_norm", "路由归一"), ("routed_raw", "路由无覆盖"), ("mean", "均值"), ("hist", "话题分布")):
                    r = {k: mm.get(f"{mode}_{k}") for k in ("h1_neutral", "h5_neutral", "h1_noname", "h5_noname")}
                    if mode == "routed":
                        r.update(tag_f1_mean=mm.get("tag_f1_mean"), tag_f1_hist=mm.get("tag_f1_hist"), tag_f1_mean_hist=mm.get("tag_f1_mean_hist"),
                                 tag_f1_concat=mm.get("tag_f1_concat"), zs_map=mm.get("zs_map_routed"))
                    rows.append((f"设计 B {name}（K={K}，{label}）step {m['step']}", cap, r))
                if mm.get("zs_map_max"):
                    rows[routed_idx] = (rows[routed_idx][0] + f"；零样本 max 读法 {fmt(mm['zs_map_max'])}", rows[routed_idx][1], rows[routed_idx][2])
            else:
                for mode, label in (("slot", "槽位余弦"), ("mean", "均值池化"), ("chamfer", "Chamfer")):
                    r = {k: mm.get(f"{mode}_{k}") for k in ("h1_neutral", "h5_neutral", "h1_noname", "h5_noname")}
                    if mode == "slot":
                        r.update(tag_f1_mean=mm.get("tag_f1_mean"), tag_f1_concat=mm.get("tag_f1_concat"), zs_map=mm.get("zs_map"))
                    rows.append((f"设计 A {name}（K={K}，{label}）step {m['step']}", cap, r))
                if "random" in m:
                    rr = m["random"]
                    rows.append((f"B3 随机分组（K={K}，Chamfer）", cap, {k: rr.get(f"chamfer_{k}") for k in ("h1_neutral", "h5_neutral", "h1_noname", "h5_noname")}
                                 | {"tag_f1_concat": rr.get("tag_f1_concat"), "zs_map": rr.get("zs_map")}))
    for name, cap, r in rows:
        lines.append(f"| {name} | {cap} | " + " | ".join(fmt(r.get(k)) for _, k in COLS) + " |")
    table = "\n".join(lines)
    print(table)
    if a.md:
        Path(a.md).write_text(table + "\n", encoding="utf-8"); print("wrote", a.md)


if __name__ == "__main__":
    main()
