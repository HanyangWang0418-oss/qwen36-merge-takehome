#!/usr/bin/env python
"""Table of dev-proxy results (results_proxy/<group>/<model>/<proxy>/per_task.jsonl) -> analysis/proxy_summary.{json,md}.
Paired bootstrap vs the base model on each proxy. Proxies are for screening only."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.bootstrap import load_per_task, paired_bootstrap_delta, bootstrap_mean  # noqa: E402

PROX = ["hle_proxy", "swebench_proxy", "terminal_bench_proxy", "asi_bench"]
found = {}
for p in (ROOT / "results_proxy").glob("*/*/*/per_task.jsonl"):
    found.setdefault(p.parts[-3], {})[p.parts[-2]] = p
base = {k: load_per_task(v) for k, v in found.get("base", {}).items()}
out = {}
for model, bench in sorted(found.items()):
    out[model] = {}
    for b, p in bench.items():
        s = load_per_task(p); arr = np.array(list(s.values())) * (1 if b == "asi_bench" else 100)
        cell = bootstrap_mean(arr, n=5000)
        if model != "base" and b in base:
            scale = 1 if b == "asi_bench" else 100
            cell["delta_vs_base"] = paired_bootstrap_delta({k: v * scale for k, v in base[b].items()}, {k: v * scale for k, v in s.items()}, n=5000)
        out[model][b] = cell
(ROOT / "analysis").mkdir(exist_ok=True)
(ROOT / "analysis/proxy_summary.json").write_text(json.dumps(out, indent=1))
lines = ["| model | " + " | ".join(PROX) + " |", "|" + "---|" * (len(PROX) + 1)]
for model, cells in out.items():
    row = []
    for b in PROX:
        c = cells.get(b)
        if not c: row.append("–"); continue
        d = c.get("delta_vs_base")
        row.append(f"{c['mean']:.1f}" + (f" ({d['delta_mean']:+.1f} [{d['ci_low']:+.0f},{d['ci_high']:+.0f}])" if d else ""))
    lines.append(f"| {model} | " + " | ".join(row) + " |")
(ROOT / "analysis/proxy_summary.md").write_text("_Dev-proxy screening only (not benchmark results)._\n\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
