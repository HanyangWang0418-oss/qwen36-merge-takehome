#!/usr/bin/env python
"""Rebuild every table in the report from results/*/<benchmark>/per_task.jsonl.

Outputs:
  results/summary.csv            model x benchmark primary metric, CI, failure rate, cost
  results/summary.md             same as a markdown table (pasted into README/report)
  analysis/uncertainty.json      bootstrap CIs per cell + paired deltas vs base
  analysis/macro_average.json    4-way macro average per model, paired delta vs base

Benchmark scores are normalised to 0-100 before averaging (SWE/TB/HLE: pass
fraction x 100; ASI: final_score already 0-100). Terminal-Bench uses the mean
over the 5 seeded runs, and run-to-run std is reported separately.

    uv run python scripts/summarize.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.bootstrap import bootstrap_mean, load_per_task, paired_bootstrap_delta  # noqa: E402

BENCHMARKS = ["swebench_pro_v2_hard", "terminal_bench_2", "hle", "asi_bench"]
SCALE = {"swebench_pro_v2_hard": 100, "terminal_bench_2": 100, "hle": 100, "asi_bench": 1}
BASE = "base"


def discover() -> dict[str, dict[str, Path]]:
    found: dict[str, dict[str, Path]] = defaultdict(dict)
    for group in ("base", "experts", "merged"):
        for p in (ROOT / "results" / group).glob("*/*/per_task.jsonl"):
            model, bench = p.parts[-3], p.parts[-2]
            if bench in BENCHMARKS:
                found[model][bench] = p
    return found


def run_stats(path: Path) -> dict:
    rows = [json.loads(l) for l in path.open() if l.strip()]
    status = defaultdict(int)
    for r in rows:
        status[r.get("status", "ok")] += 1
    n = len(rows)
    per_run = defaultdict(list)
    for r in rows:
        per_run[r.get("run", 0)].append(float(r.get("score", 0.0)))
    run_means = [float(np.mean(v)) for v in per_run.values()]
    cost = sum(float(r.get("cost_usd", 0.0)) for r in rows)
    wall = [float(r["wallclock_s"]) for r in rows if "wallclock_s" in r]
    return {
        "n_rows": n, "n_runs": len(per_run),
        "env_failure_rate": status.get("env_failure", 0) / max(n, 1),
        "timeout_rate": status.get("timeout", 0) / max(n, 1),
        "model_failure_rate": status.get("model_failure", 0) / max(n, 1),
        "run_std": float(np.std(run_means, ddof=1)) if len(run_means) > 1 else None,
        "total_cost_usd": cost, "mean_wallclock_s": float(np.mean(wall)) if wall else None,
    }


def main() -> int:
    found = discover()
    if BASE not in found:
        print("no results/base/base/*/per_task.jsonl yet; nothing to summarise")
    cells, deltas, macro = {}, {}, {}
    base_scores = {b: load_per_task(found[BASE][b]) for b in found.get(BASE, {})}
    for model, benches in sorted(found.items()):
        cells[model], deltas[model] = {}, {}
        per_bench_mean = {}
        for b, path in benches.items():
            scores = load_per_task(path)
            arr = np.array(list(scores.values())) * SCALE[b]
            cells[model][b] = {**bootstrap_mean(arr), **run_stats(path)}
            per_bench_mean[b] = cells[model][b]["mean"]
            if model != BASE and b in base_scores:
                bs = {k: v * SCALE[b] for k, v in base_scores[b].items()}
                ms = {k: v * SCALE[b] for k, v in scores.items()}
                deltas[model][b] = paired_bootstrap_delta(bs, ms)
        if all(b in per_bench_mean for b in BENCHMARKS):
            macro[model] = {"macro_mean": float(np.mean([per_bench_mean[b] for b in BENCHMARKS])), "benchmarks": per_bench_mean}
            if model != BASE and BASE in macro:
                # paired macro delta: resample each benchmark's tasks jointly, average the 4 deltas
                rng = np.random.default_rng(0)
                ds = []
                for _ in range(10000):
                    acc = []
                    for b in BENCHMARKS:
                        ids = sorted(set(base_scores[b]) & set(load_per_task(found[model][b])))
                        if not ids:
                            continue
                        sc = load_per_task(found[model][b])
                        idx = rng.integers(0, len(ids), len(ids))
                        acc.append(np.mean([(sc[ids[i]] - base_scores[b][ids[i]]) * SCALE[b] for i in idx]))
                    ds.append(np.mean(acc))
                macro[model]["delta_vs_base"] = {"mean": macro[model]["macro_mean"] - macro[BASE]["macro_mean"],
                                                 "ci_low": float(np.quantile(ds, 0.025)), "ci_high": float(np.quantile(ds, 0.975))}
        else:
            macro[model] = {"macro_mean": None, "note": f"evaluated on {sorted(per_bench_mean)} only", "benchmarks": per_bench_mean}

    (ROOT / "analysis").mkdir(exist_ok=True)
    (ROOT / "analysis/uncertainty.json").write_text(json.dumps({"cells": cells, "paired_delta_vs_base": deltas}, indent=2))
    (ROOT / "analysis/macro_average.json").write_text(json.dumps(macro, indent=2))

    header = ["model", "benchmark", "score", "ci_low", "ci_high", "delta_vs_base", "delta_ci_low", "delta_ci_high",
              "n_tasks", "n_runs", "run_std", "env_failure_rate", "timeout_rate", "total_cost_usd", "mean_wallclock_s"]
    with (ROOT / "results/summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for model in cells:
            for b in BENCHMARKS:
                c = cells[model].get(b)
                if not c:
                    continue
                d = deltas[model].get(b, {})
                w.writerow([model, b, f"{c['mean']:.2f}", f"{c['ci_low']:.2f}", f"{c['ci_high']:.2f}",
                            f"{d['delta_mean']:.2f}" if d else "", f"{d['ci_low']:.2f}" if d else "", f"{d['ci_high']:.2f}" if d else "",
                            c["n_tasks"], c["n_runs"], f"{c['run_std']:.2f}" if c["run_std"] is not None else "",
                            f"{c['env_failure_rate']:.3f}", f"{c['timeout_rate']:.3f}", f"{c['total_cost_usd']:.2f}",
                            f"{c['mean_wallclock_s']:.0f}" if c["mean_wallclock_s"] else ""])
    lines = ["| model | " + " | ".join(BENCHMARKS) + " | macro avg | Δ vs base [95% CI] |", "|" + "---|" * (len(BENCHMARKS) + 3)]
    for model in cells:
        row = [model]
        for b in BENCHMARKS:
            c = cells[model].get(b)
            row.append(f"{c['mean']:.1f} [{c['ci_low']:.1f}, {c['ci_high']:.1f}]" if c else "n/a")
        m = macro.get(model, {})
        row.append(f"{m['macro_mean']:.2f}" if m.get("macro_mean") is not None else "partial")
        d = m.get("delta_vs_base")
        row.append(f"{d['mean']:+.2f} [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}]" if d else "")
        lines.append("| " + " | ".join(row) + " |")
    (ROOT / "results/summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
