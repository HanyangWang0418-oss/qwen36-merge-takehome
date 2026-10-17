#!/usr/bin/env bash
# ASI-Bench with the official asibench CLI; Docker sandbox on this machine (not Modal).
# usage: eval/run_asi_bench.sh <model_name_on_vllm> <group> [--track seed42|seed31415] [--instances "a b"]
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . eval/.env; set +a
MODEL=$1; GROUP=$2; shift 2
TRACK=$(python3 -c "import json;print(json.load(open('data/manifests/asi_bench_tasks.json'))['track'])"); INST=""
while [ $# -gt 0 ]; do case $1 in --track) TRACK=$2; shift 2;; --instances) INST=$2; shift 2;; *) shift;; esac; done
docker info >/dev/null 2>&1 || { echo "asibench --sandbox os needs a Docker daemon on this machine"; exit 1; }
mkdir -p logs/eval eval/agent_configs
cat > "eval/agent_configs/opencode_${MODEL}.json" <<JSON
{"model": "${MODEL}", "api_base": "${VLLM_BASE}", "api_key_env": "OPENAI_API_KEY", "api_protocol": "openai", "supports_image_input": false,
 "temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}
JSON
[ -d "hf_instances_${TRACK}" ] || asibench task pull --repo "$TRACK" --output-dir "hf_instances_${TRACK}/"
INST_ARGS=(); for i in $INST; do INST_ARGS+=(--instance "$i"); done
echo "[$(date -u +%FT%TZ)] model=$MODEL track=$TRACK judge=${ASIBENCH_JUDGE_MODEL}" | tee -a logs/eval/asi_${MODEL}.log
OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}" asibench run --agent opencode --agent-config "eval/agent_configs/opencode_${MODEL}.json" --sandbox os \
  --instances-dir "hf_instances_${TRACK}/" --output-dir "out_${TRACK}_${MODEL}/" "${INST_ARGS[@]}" \
  --judge-api-base "$ASIBENCH_JUDGE_API_BASE" --judge-api-key-env ASIBENCH_JUDGE_API_KEY --judge-api-protocol openai \
  2>&1 | tee -a logs/eval/asi_${MODEL}.log
if [ "$TRACK" = "seed31415" ]; then
  asibench score --repo "$TRACK" --results-dir "out_${TRACK}_${MODEL}/" --instances-dir "hf_instances_${TRACK}/" --tasks-dir third_party/ASI-Bench/tasks/ --parallel 4 2>&1 | tee -a logs/eval/asi_${MODEL}.log
else
  echo "seed42: submit with 'asibench login && asibench submit --results-dir out_${TRACK}_${MODEL}/ --benchmark-repo Apexintelligence-AI/ASI-Bench-seed42', then download the scored results into out_${TRACK}_${MODEL}/"
fi
uv run python src/eval/collect_asi.py "out_${TRACK}_${MODEL}/" "results/$GROUP/$MODEL/asi_bench"
