#!/usr/bin/env bash
# Terminal-Bench 2.0, terminus-2, 5 seeded runs, model-card-aligned settings.
# usage: scripts/eval_terminal_bench.sh <model_name_on_vllm> <results_group>
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=$1; GROUP=$2
TBV=$(python -c "import json;print(json.load(open('data/manifests/terminal_bench_2_tasks.json'))['version'])")
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
export OPENAI_API_BASE=${VLLM_BASE:-http://localhost:8000/v1}
for SEED in 1 2 3 4 5; do
  JOB="tb2-${MODEL}-seed${SEED}"
  harbor run -d "terminal-bench/terminal-bench@${TBV}" -a terminus-2 -m "openai/${MODEL}" \
    -e "${HARBOR_ENV:-e2b}" -n "${CONCURRENCY:-30}" \
    --ae OPENAI_API_KEY="$OPENAI_API_KEY" --ae OPENAI_API_BASE="$OPENAI_API_BASE" \
    --ak temperature=1.0 --ak top_p=0.95 --ak top_k=20 --ak max_tokens=81920 --ak seed="$SEED" \
    --timeout-multiplier 1.0 --task-timeout 10800 --cpus 32 --memory 48G \
    --job-name "$JOB" 2>&1 | tee "logs/eval_${JOB}.log"
done
uv run python src/eval/collect_harbor.py "jobs/tb2-${MODEL}-seed*" "results/$GROUP/$MODEL/terminal_bench_2"
