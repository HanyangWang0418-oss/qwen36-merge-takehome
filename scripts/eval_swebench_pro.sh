#!/usr/bin/env bash
# SWE-bench Pro v2 HARD-51 with the locked mini-swe-agent, then authoritative regrade.
# usage: scripts/eval_swebench_pro.sh <model_name_on_vllm> <results_group: base|experts|merged>
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=$1; GROUP=$2
SBP=${SBP_DIR:-third_party/SWE-bench_Pro-os}
JOB="sbp-hard51-${MODEL}"
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
export OPENAI_API_BASE=${VLLM_BASE:-http://localhost:8000/v1}
PYTHONPATH=$SBP/v2/tooling harbor run -p $SBP/v2/tasks $(sed 's/^/-i /' $SBP/v2/hard51_ids.txt) \
  -e "${HARBOR_ENV:-modal}" -n "${CONCURRENCY:-20}" \
  -a locked_mini_swe:LockedMiniSwe -m "openai/${MODEL}" \
  --ae OPENAI_API_KEY="$OPENAI_API_KEY" --ae OPENAI_API_BASE="$OPENAI_API_BASE" \
  --allow-agent-host "${VLLM_HOST:-localhost}" \
  --ak temperature=0.6 --ak top_p=0.95 --ak max_tokens=32768 \
  --job-name "$JOB" 2>&1 | tee "logs/eval_${JOB}.log"
PYTHONPATH=$SBP/v2/tooling harbor run -p $SBP/v2/tasks $(sed 's/^/-i /' $SBP/v2/hard51_ids.txt) \
  -e "${HARBOR_ENV:-modal}" -n "${CONCURRENCY:-20}" -a patch_replay:PatchReplayAgent -m replay \
  --ak source_job="jobs/$JOB" --job-name "${JOB}-regrade" 2>&1 | tee "logs/eval_${JOB}-regrade.log"
uv run python src/eval/collect_harbor.py "jobs/${JOB}-regrade" "results/$GROUP/$MODEL/swebench_pro_v2_hard" --original "jobs/$JOB"
