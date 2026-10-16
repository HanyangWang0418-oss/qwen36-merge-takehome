#!/usr/bin/env python
"""Render the one-page report.pdf (摘要 150–200 字 / 过程 / 结果) from results/summary.csv,
analysis/macro_average.json and logs/resource_summary.csv. Chinese text uses a system CJK
font (PingFang/Hiragino on macOS, Noto Sans CJK on Linux). Content placeholders live in
report/content.yaml so the prose is reviewed, not generated.

    uv run python scripts/make_report.py            # -> report.pdf
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BENCH = [("swebench_pro_v2_hard", "SWE-bench Pro v2-hard"), ("terminal_bench_2", "Terminal-Bench 2.0"), ("hle", "HLE (text-only)"), ("asi_bench", "ASI-Bench (B3)")]


def cjk_font() -> str:
    for name in ("PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei", "Songti SC"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


def load_summary() -> dict:
    cells = {}
    p = ROOT / "results/summary.csv"
    if p.exists():
        for r in csv.DictReader(p.open()):
            cells.setdefault(r["model"], {})[r["benchmark"]] = r
    return cells


def main() -> int:
    content = yaml.safe_load((ROOT / "report/content.yaml").read_text())
    cells = load_summary()
    macro = json.loads((ROOT / "analysis/macro_average.json").read_text()) if (ROOT / "analysis/macro_average.json").exists() else {}
    res = list(csv.DictReader((ROOT / "logs/resource_summary.csv").open())) if (ROOT / "logs/resource_summary.csv").exists() else []
    plt.rcParams["font.family"] = cjk_font()
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.text(0.5, 0.965, content["title"], ha="center", va="top", fontsize=15, weight="bold")
    fig.text(0.5, 0.945, content["subtitle"], ha="center", va="top", fontsize=8.5, color="#444")
    y = 0.915
    for head, key in (("摘要", "abstract"), ("过程", "process")):
        fig.text(0.06, y, head, fontsize=11, weight="bold", va="top"); y -= 0.018
        txt = content[key].strip()
        fig.text(0.06, y, txt, fontsize=8.6, va="top", wrap=True, linespacing=1.45, transform=fig.transFigure)
        y -= 0.012 + 0.0145 * (len(txt) // 58 + txt.count("\n") + 1)
    fig.text(0.06, y, "结果", fontsize=11, weight="bold", va="top"); y -= 0.02

    models = content["models"]  # ordered list of model ids to show
    rows, labels = [], []
    for m in models:
        row = []
        for b, _ in BENCH:
            c = cells.get(m, {}).get(b)
            row.append(f"{float(c['score']):.1f} [{float(c['ci_low']):.0f},{float(c['ci_high']):.0f}]" if c else ("–" if m != "base" and not m.startswith("merged") else "n/a"))
        mm = macro.get(m, {})
        row.append(f"{mm['macro_mean']:.2f}" if mm.get("macro_mean") is not None else "部分评测*")
        d = mm.get("delta_vs_base")
        row.append(f"{d['mean']:+.2f} [{d['ci_low']:+.1f},{d['ci_high']:+.1f}]" if d else "")
        fr = [cells.get(m, {}).get(b) for b, _ in BENCH]
        ef = [float(c["env_failure_rate"]) for c in fr if c]
        row.append(f"{100 * max(ef):.1f}%" if ef else "")
        cost = sum(float(c["total_cost_usd"]) for c in fr if c and c["total_cost_usd"])
        row.append(f"${cost:.0f}" if cost else "")
        rows.append(row); labels.append(content.get("model_labels", {}).get(m, m))
    cols = [n for _, n in BENCH] + ["宏平均", "Δ vs Base [95% CI]", "运行失败率(最高项)", "评测成本"]
    ax = fig.add_axes([0.04, y - 0.03 - 0.028 * len(rows), 0.92, 0.028 * len(rows) + 0.03]); ax.axis("off")
    t = ax.table(cellText=rows, rowLabels=labels, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(6.6); t.scale(1, 1.25)
    y = y - 0.04 - 0.028 * len(rows)
    notes = content["results_notes"].strip()
    fig.text(0.06, y, notes, fontsize=7.8, va="top", wrap=True, linespacing=1.4)
    y -= 0.012 + 0.013 * (len(notes) // 62 + notes.count("\n") + 1)
    if res:
        gpu_h = sum(float(r["gpu_hours"] or 0) for r in res); cost = sum(float(r["cost_usd"] or 0) for r in res)
        fig.text(0.06, y, f"资源：GPU 小时 {gpu_h:.0f}，训练 token {sum(float(r['tokens'] or 0) for r in res) / 1e6:.0f}M，总费用约 ${cost:.0f}（含训练、推理、沙箱、判分；明细 logs/resource_summary.csv）", fontsize=7.8, va="top", wrap=True)
    fig.text(0.5, 0.025, content["footer"], ha="center", fontsize=7, color="#555")
    fig.savefig(ROOT / "report.pdf")
    print("wrote report.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
