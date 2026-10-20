#!/usr/bin/env bash
# ASI-Bench seed31415 (public references, local scoring) with the official
# asibench 0.2.0 CLI; Docker sandbox on this machine (not Modal).
# usage: eval/run_asi_bench.sh <model_name_on_vllm> <group> \
#          [--instances "a,b"] [--levels b1,b2,b3,b4] [--parallel N]
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . eval/.env; set +a
MODEL=$1; GROUP=$2; shift 2
SERVED=$MODEL; [ "$MODEL" = base ] && SERVED=${BASE_MODEL_NAME:-base}
EP=$(python3 eval/resolve_endpoint.py "$MODEL") || exit 2; [ -n "$EP" ] && eval "$EP"   # per-model endpoint (eval/endpoints.yaml)
TRACK=seed31415; INST=""; LEVELS=b1,b2,b3,b4; PAR=${ASI_PARALLEL:-2}
while [ $# -gt 0 ]; do case $1 in
  --instances) INST=$2; shift 2;;
  --levels) LEVELS=$2; shift 2;;
  --parallel) PAR=$2; shift 2;;
  --track) TRACK=$2; shift 2;;
  *) shift;;
esac; done
docker info >/dev/null 2>&1 || { echo "asibench --sandbox os needs a Docker daemon on this machine"; exit 1; }
# asibench creates each workspace via tempfile.mkdtemp() and bind-mounts it into
# the container. colima only shares $HOME, so the default macOS TMPDIR
# (/var/folders/...) mounts as an empty dir and the agent sees no data files.
export TMPDIR="$PWD/.asibench_tmp"
mkdir -p "$TMPDIR"
mkdir -p logs/eval
[ -d "hf_instances_${TRACK}" ] || \
  HTTPS_PROXY="${HTTPS_PROXY:-http://10.217.147.40:8080}" HTTP_PROXY="${HTTP_PROXY:-http://10.217.147.40:8080}" \
  asibench task pull --repo "$TRACK" --output-dir "hf_instances_${TRACK}/"
AGENT_CFG=$(python3 - "$SERVED" "$VLLM_BASE" <<'PY'
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "api_base": sys.argv[2],
    "api_key_env": "OPENAI_API_KEY",
    "api_protocol": "openai",
}))
PY
)
# Judge: basicai gpt-5.6-sol through the local header-injecting proxy.
export ASIBENCH_JUDGE_API_BASE="${ASI_JUDGE_BASE:-http://127.0.0.1:8799/v1}"
export ASIBENCH_JUDGE_API_KEY_ENV=ASIBENCH_JUDGE_API_KEY
export ASIBENCH_JUDGE_API_KEY="$(cat "${BASICAI_KEY_FILE:-/tmp/basicai_key}")"
echo "[$(date -u +%FT%TZ)] model=$MODEL track=$TRACK tasks=${INST:-all} levels=$LEVELS judge=${BASICAI_MODEL_OVERRIDE:-gpt-5.6-sol}@$ASIBENCH_JUDGE_API_BASE" | tee -a logs/eval/asi_${MODEL}.log
OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}" asibench run-score \
  --instances-dir "hf_instances_${TRACK}/" --tasks-dir third_party/ASI-Bench/tasks/ \
  --tasks "${INST:-all}" --prompt-levels "$LEVELS" --parallel "$PAR" --sandbox os \
  --agent opencode_cli --agent-config "$AGENT_CFG" \
  --judge-api-base "$ASIBENCH_JUDGE_API_BASE" --judge-api-key-env ASIBENCH_JUDGE_API_KEY \
  --judge-api-protocol openai \
  --output-dir "out_${TRACK}_${MODEL}/" 2>&1 | tee -a logs/eval/asi_${MODEL}.log
uv run python src/eval/collect_asi.py "out_${TRACK}_${MODEL}/" "results/$GROUP/$MODEL/asi_bench"
