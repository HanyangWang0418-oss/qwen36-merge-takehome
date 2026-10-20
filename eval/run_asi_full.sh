#!/usr/bin/env bash
# Full ASI-Bench seed31415 run: all tasks x b1..b4, per-task resume, N workers.
#
# `asibench run-score` clamps its outer parallelism to 1 on this host and has
# no resume, so a ~2-day run would restart from zero on any interruption.
# This wrapper drives the same official `asibench run` per task (all four
# prompt levels), keeps completed tasks in <out>/<task>, and finishes with the
# official `asibench score` over the whole results dir.
#
# usage: eval/run_asi_full.sh <model_name_on_vllm> <group>
#   ASI_WORKERS=2 ASI_LEVELS=b1,b2,b3,b4 bash eval/run_asi_full.sh qwen36-35b-base full
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . eval/.env; set +a
MODEL=${1:-qwen36-35b-base}; GROUP=${2:-full}
SERVED=$MODEL; [ "$MODEL" = base ] && SERVED=${BASE_MODEL_NAME:-base}
TRACK=seed31415
OUT="out_${TRACK}_${MODEL}"
WORKERS=${ASI_WORKERS:-2}
LEVELS=${ASI_LEVELS:-b1,b2,b3,b4}
LOG="logs/eval/asi_${MODEL}.log"
mkdir -p "$OUT/.jobs" logs/eval "$OUT/_workspaces"

# The workspace is bind-mounted from tempfile.mkdtemp(); colima only shares
# $HOME, so keep TMPDIR inside the repo (not /var/folders).
export TMPDIR="$PWD/.asibench_tmp"; mkdir -p "$TMPDIR"

# The agent must not inherit the judge credential/selector.
unset ASIBENCH_JUDGE_API_BASE ASIBENCH_JUDGE_API_KEY_ENV ASIBENCH_JUDGE_API_PROTOCOL ASIBENCH_JUDGE_API_KEY
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

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

# macOS ships bash 3.2 (no mapfile); task IDs contain no spaces.
TASKS=($(for d in hf_instances_${TRACK}/*/; do
  b=$(basename "$d"); echo "${b%__${TRACK}}"; done | sort))
echo "[$(date -u +%FT%TZ)] ASI full: ${#TASKS[@]} tasks, levels=$LEVELS, workers=$WORKERS" | tee -a "$LOG"

run_one() {
  local t=$1
  if [ -d "$OUT/$t" ]; then
    echo "[$(date -u +%FT%TZ)] skip $t (done)" >> "$LOG"
    return 0
  fi
  echo "[$(date -u +%FT%TZ)] start $t" >> "$LOG"
  asibench run \
    --instances-dir "hf_instances_${TRACK}/" --tasks-dir third_party/ASI-Bench/tasks/ \
    --tasks "$t" --prompt-levels "$LEVELS" --parallel 1 --sandbox os \
    --agent opencode_cli --agent-config "$AGENT_CFG" --timeout 10800 \
    --resume "$OUT/.jobs/$t" \
    --output-dir "$OUT/.jobs/$t" >> "$LOG" 2>&1
  local rc=$?
  if [ -d "$OUT/.jobs/$t/$t" ]; then
    mv "$OUT/.jobs/$t/$t" "$OUT/$t"
    echo "[$(date -u +%FT%TZ)] done $t (rc=$rc)" >> "$LOG"
  else
    echo "[$(date -u +%FT%TZ)] FAILED $t (rc=$rc)" >> "$LOG"
  fi
}

worker() {
  local w=$1 i
  for i in "${!TASKS[@]}"; do
    [ $((i % WORKERS)) -eq "$w" ] || continue
    run_one "${TASKS[$i]}"
  done
}

for w in $(seq 0 $((WORKERS - 1))); do worker "$w" & done
wait
echo "[$(date -u +%FT%TZ)] all run workers finished" | tee -a "$LOG"

# Official local scoring (seed31415) with the basicai judge via the local proxy.
# NB: judge model is overridden to gpt-5.6-sol by eval/judge_proxy_basicai.py.
export ASIBENCH_JUDGE_API_BASE="${ASI_JUDGE_BASE:-http://127.0.0.1:8799/v1}"
export ASIBENCH_JUDGE_API_KEY_ENV=ASIBENCH_JUDGE_API_KEY
export ASIBENCH_JUDGE_API_KEY="$(cat "${BASICAI_KEY_FILE:-/tmp/basicai_key}")"
asibench score --repo seed31415 --results-dir "$OUT" \
  --instances-dir "hf_instances_${TRACK}/" --tasks-dir third_party/ASI-Bench/tasks/ \
  --output "$OUT/local_score_seed31415.json" --parallel "$WORKERS" \
  --judge-api-base "$ASIBENCH_JUDGE_API_BASE" \
  --judge-api-key-env ASIBENCH_JUDGE_API_KEY --judge-api-protocol openai \
  >> "$LOG" 2>&1
echo "[$(date -u +%FT%TZ)] score exit $?" | tee -a "$LOG"
uv run python src/eval/collect_asi.py "$OUT/" "results/$GROUP/$MODEL/asi_bench"
