#!/usr/bin/env bash
# SWE-bench Pro v2 HARD-51 on Modal with the locked mini-swe-agent, then the authoritative fresh-sandbox regrade.
# usage: eval/run_swebench_pro.sh <model_name_on_vllm> <group: base|experts|merged> [--subset ids.txt]
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . eval/.env; set +a
MODEL=$1; GROUP=$2; SUBSET=${4:-}
SERVED=$MODEL; [ "$MODEL" = base ] && SERVED=${BASE_MODEL_NAME:-base}
EP=$(python3 eval/resolve_endpoint.py "$MODEL" $([ "${HARBOR_ENV:-modal}" = modal ] && echo --modal)) || exit 2; [ -n "$EP" ] && eval "$EP"   # per-model endpoint (eval/endpoints.yaml)
SBP=third_party/SWE-bench_Pro-os
IDS=${SUBSET:-$SBP/v2/hard51_ids.txt}
JOB="sbp-hard51-${MODEL}"
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy} OPENAI_API_BASE="$VLLM_BASE"
mkdir -p logs/eval
echo "[$(date -u +%FT%TZ)] model=$MODEL tasks=$(wc -l < "$IDS") env=modal agent=locked_mini_swe" | tee -a logs/eval/${JOB}.log
# agent sandbox is offline except the model host; verifier phase keeps network (Go modules etc.)
PYTHONPATH=$SBP/v2/tooling harbor run -p $SBP/v2/tasks $(sed 's/^/-i /' "$IDS") \
  -e "${HARBOR_ENV:-modal}" -n "${CONCURRENCY:-20}" -k 1 \
  -a locked_mini_swe:LockedMiniSwe -m "openai/${SERVED}" \
  --ak config_file=$SBP/v2/tooling/configs/mini_toolcall.yaml \
  --ae OPENAI_API_KEY="$OPENAI_API_KEY" --ae OPENAI_API_BASE="$OPENAI_API_BASE" \
  --allow-agent-host "$VLLM_HOST" \
  --job-name "$JOB" -o jobs 2>&1 | tee -a logs/eval/${JOB}.log
PYTHONPATH=$SBP/v2/tooling harbor run -p $SBP/v2/tasks $(sed 's/^/-i /' "$IDS") \
  -e "${HARBOR_ENV:-modal}" -n "${CONCURRENCY:-20}" -a patch_replay:PatchReplayAgent -m replay \
  --ak source_job="jobs/$JOB" --job-name "${JOB}-regrade" -o jobs 2>&1 | tee -a logs/eval/${JOB}-regrade.log
uv run python src/eval/collect_harbor.py "jobs/${JOB}-regrade" "results/$GROUP/$MODEL/swebench_pro_v2_hard" --original "jobs/$JOB"
