#!/usr/bin/env bash
# ASI-Bench via the official asibench CLI, opencode adapter pointed at our vLLM server.
# usage: scripts/eval_asi_bench.sh <model_name_on_vllm> <results_group>
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=$1; GROUP=$2
TRACK=$(python -c "import json;print(json.load(open('data/manifests/asi_bench_tasks.json'))['track'])")
cat > /tmp/asi_agent_${MODEL}.json <<JSON
{"model": "${MODEL}", "api_base": "${VLLM_BASE:-http://localhost:8000/v1}", "api_key_env": "OPENAI_API_KEY", "api_protocol": "openai", "supports_image_input": false}
JSON
asibench task pull --repo "$TRACK" --output-dir "hf_instances_${TRACK}/"
asibench run --agent opencode --agent-config "/tmp/asi_agent_${MODEL}.json" --sandbox os \
  --instances-dir "hf_instances_${TRACK}/" --output-dir "out_${TRACK}_${MODEL}/" \
  --judge-api-base "$ASIBENCH_JUDGE_API_BASE" --judge-api-key-env ASIBENCH_JUDGE_API_KEY --judge-api-protocol openai \
  2>&1 | tee "logs/eval_asi_${MODEL}.log"
if [ "$TRACK" = "seed31415" ]; then
  asibench score --repo "$TRACK" --results-dir "out_${TRACK}_${MODEL}/" --instances-dir "hf_instances_${TRACK}/" \
    --tasks-dir third_party/ASI-Bench/tasks/ --parallel 4 2>&1 | tee -a "logs/eval_asi_${MODEL}.log"
else
  echo "seed42: run 'asibench submit --results-dir out_${TRACK}_${MODEL}/ --benchmark-repo Apexintelligence-AI/ASI-Bench-seed42' and download the scored results"
fi
uv run python src/eval/collect_asi.py "out_${TRACK}_${MODEL}/" "results/$GROUP/$MODEL/asi_bench"
