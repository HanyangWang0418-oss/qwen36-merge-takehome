#!/usr/bin/env bash
# Rebuild summary tables, uncertainty estimates and plots from results/**/per_task.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python scripts/summarize.py
uv run python scripts/plot_curves.py
[ -d results_proxy ] && uv run python scripts/summarize_proxies.py || true
uv run python scripts/make_report.py || true
echo "results/summary.md, analysis/uncertainty.json, analysis/macro_average.json, curves/merge_ablation.png, report.pdf"
