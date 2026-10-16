#!/usr/bin/env python
"""Plot train/dev loss for every run in logs/<run_id>/metrics.jsonl and the
merge-ablation comparison from analysis/uncertainty.json.

Writes curves/<run_id>_loss.png and curves/source/<run_id>_loss.csv (raw, no
smoothing), plus curves/merge_ablation.png + curves/source/merge_ablation.csv.

    uv run python scripts/plot_curves.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "curves"
SRC = OUT / "source"
SRC.mkdir(parents=True, exist_ok=True)


def plot_run(run_dir: Path) -> None:
    rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").open() if l.strip()]
    if not rows:
        return
    # A resumed run re-logs steps after the resume point; keep the last record per step
    # (the earlier ones belong to the lost session) and note it on the plot.
    by_step = {}
    for r in rows:
        by_step[r["step"]] = r
    resumed = len(by_step) != len(rows)
    rows = [by_step[k] for k in sorted(by_step)]
    meta = json.loads((run_dir / "metadata.json").read_text()) if (run_dir / "metadata.json").exists() else {}
    with (SRC / f"{run_dir.name}_loss.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "train_loss", "dev_loss", "lr", "checkpoint"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["step"] for r in rows], [r["train_loss"] for r in rows], label="train loss (per step, unsmoothed)", lw=0.8)
    dev = [r for r in rows if "dev_loss" in r]
    if dev:
        ax.plot([r["step"] for r in dev], [r["dev_loss"] for r in dev], "o-", label="dev loss (independent dev split)")
        best = min(dev, key=lambda r: r["dev_loss"])
        ax.axvline(best["step"], ls="--", c="grey", lw=0.8)
        ax.annotate(f"selected ckpt step {best['step']}", (best["step"], best["dev_loss"]), fontsize=7, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("token-weighted cross-entropy (assistant tokens)")
    sel = (meta.get("selected_checkpoint") or "").split("/")[-1]  # basename; full tinker path is in metadata.json
    note = "  (resumed from state checkpoint after a lost session)" if resumed else ""
    ax.set_title(f"{run_dir.name}  base={meta.get('config', {}).get('base', {}).get('tinker_model', '?')}{note}\nselected={sel}", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"{run_dir.name}_loss.png", dpi=150)
    plt.close(fig)
    print(f"plotted {run_dir.name}")


def plot_merge_ablation() -> None:
    p = ROOT / "analysis/uncertainty.json"
    if not p.exists():
        return
    cells = json.loads(p.read_text())["cells"]
    merged = {m: v for m, v in cells.items() if m.startswith("merged_") or m == "base"}
    if len(merged) < 2:
        return
    benches = ["swebench_pro_v2_hard", "terminal_bench_2", "hle", "asi_bench"]
    with (SRC / "merge_ablation.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "benchmark", "score", "ci_low", "ci_high"])
        for m, v in merged.items():
            for b in benches:
                if b in v:
                    w.writerow([m, b, v[b]["mean"], v[b]["ci_low"], v[b]["ci_high"]])
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=False)
    for ax, b in zip(axes, benches):
        names = [m for m in merged if b in merged[m]]
        means = [merged[m][b]["mean"] for m in names]
        err = [[merged[m][b]["mean"] - merged[m][b]["ci_low"] for m in names], [merged[m][b]["ci_high"] - merged[m][b]["mean"] for m in names]]
        ax.bar(range(len(names)), means, yerr=err, capsize=3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_title(b, fontsize=9)
        ax.set_ylabel("primary metric (0-100), 95% bootstrap CI", fontsize=7)
    fig.suptitle("Merge-method ablation (frozen checkpoints, identical eval config)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "merge_ablation.png", dpi=150)
    plt.close(fig)
    print("plotted merge_ablation")


def main() -> int:
    for run_dir in sorted((ROOT / "logs").glob("*/")):
        if (run_dir / "metrics.jsonl").exists():
            plot_run(run_dir)
    plot_merge_ablation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
