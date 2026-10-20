#!/usr/bin/env bash
# Retry b3/b4 for tasks whose levels failed while containers were forced
# through the corporate proxy (403 ip access not allowed) during the vLLM
# endpoint migration. Resumes against existing b1/b2 results.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . eval/.env; set +a
MODEL=${1:-qwen36-35b-base}
SERVED=$MODEL
TRACK=seed31415
OUT="out_${TRACK}_${MODEL}"
export TMPDIR="$PWD/.asibench_tmp"; mkdir -p "$TMPDIR"
unset ASIBENCH_JUDGE_API_BASE ASIBENCH_JUDGE_API_KEY_ENV ASIBENCH_JUDGE_API_PROTOCOL ASIBENCH_JUDGE_API_KEY
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
AGENT_CFG=$(python3 - "$SERVED" "$VLLM_BASE" <<'PY'
import json, sys
print(json.dumps({"model": sys.argv[1], "api_base": sys.argv[2],
                  "api_key_env": "OPENAI_API_KEY", "api_protocol": "openai"}))
PY
)
for T in astronomy.planet_activity_sep astronomy.nbody_close_encounters; do
  echo "=== fixing $T"
  if [ -d "$OUT/$T" ]; then
    mkdir -p "$OUT/.jobs/$T"
    rm -rf "$OUT/.jobs/$T/$T"
    mv "$OUT/$T" "$OUT/.jobs/$T/$T"
  fi
  rm -f "$OUT/.jobs/$T/$T"/*__b3.json "$OUT/.jobs/$T/$T"/*__b3.trajectory.json \
        "$OUT/.jobs/$T/$T"/*__b3.model_calls.json "$OUT/.jobs/$T/$T"/*__b3.agent_stdout.jsonl \
        "$OUT/.jobs/$T/$T"/*__b4.json "$OUT/.jobs/$T/$T"/*__b4.trajectory.json \
        "$OUT/.jobs/$T/$T"/*__b4.model_calls.json "$OUT/.jobs/$T/$T"/*__b4.agent_stdout.jsonl
  asibench run --instances-dir "hf_instances_${TRACK}/" --tasks-dir third_party/ASI-Bench/tasks/ \
    --tasks "$T" --prompt-levels b3,b4 --parallel 1 --sandbox os \
    --agent opencode_cli --agent-config "$AGENT_CFG" --timeout 10800 \
    --resume "$OUT/.jobs/$T" --output-dir "$OUT/.jobs/$T" \
    >> "logs/eval/asi_${MODEL}.log" 2>&1
  echo "=== fix rc=$? for $T"
  [ -d "$OUT/.jobs/$T/$T" ] && mv "$OUT/.jobs/$T/$T" "$OUT/$T"
done
echo "=== fix2 done"
