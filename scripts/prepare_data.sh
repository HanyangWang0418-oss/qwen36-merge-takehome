#!/usr/bin/env bash
# Build every expert's training set. Requires data/manifests/* from
# scripts/lock_benchmarks.py (blocklists) and HF_TOKEN for gated sources.
set -euo pipefail
cd "$(dirname "$0")/.."
for e in e1_code_repair e2_terminal_agent e3_reasoning; do
  uv run python src/data/build_expert_data.py "$e" "$@"
done
