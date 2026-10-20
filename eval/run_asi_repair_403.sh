#!/usr/bin/env bash
# Generic repair for ASI-Bench prompt levels that failed with
# "403 ip access not allowed" (containers were forced through the corporate
# proxy while the vLLM base moved to 10.164.228.13).  Scans completed tasks in
# the output dir, deletes only the failed level artifacts, and re-runs those
# levels with --resume so successful levels are kept.
#
# Waits for any other fixer (run_asi_fix_b34.sh) to exit first so we never run
# two repairs over the same results tree.
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

# Serialize with the other fixer.
while pgrep -f "run_asi_fix_b34.sh" >/dev/null 2>&1; do sleep 30; done

TASKS=$(python3 - "$OUT" <<'PY'
import glob, json, os, sys
out = sys.argv[1]
for d in sorted(glob.glob(os.path.join(out, "*"))):
    if not os.path.isdir(d) or os.path.basename(d).startswith((".", "_")):
        continue
    levels = []
    for f in glob.glob(os.path.join(d, "*__b*.json")):
        base = os.path.basename(f)
        if base.endswith((".model_calls.json", ".trajectory.json", ".agent_stdout.jsonl")):
            continue
        try:
            rec = json.load(open(f))
        except Exception:
            continue
        log = (rec.get("agent_output") or {}).get("log") or ""
        if rec.get("status") != "completed" and "ip access not allowed" in log:
            levels.append(base.split("__")[-1].split(".")[0])
    if levels:
        print(os.path.basename(d) + "\t" + ",".join(sorted(set(levels))))
PY
)

if [ -z "$TASKS" ]; then
  echo "[$(date -u +%FT%TZ)] no ip403 failures found"
  exit 0
fi

echo "$TASKS" | while IFS=$'\t' read -r T LEVELS; do
  [ -n "$T" ] || continue
  echo "=== repairing $T levels=$LEVELS"
  mkdir -p "$OUT/.jobs/$T"
  if [ -d "$OUT/$T" ]; then
    rm -rf "$OUT/.jobs/$T/$T"
    mv "$OUT/$T" "$OUT/.jobs/$T/$T"
  fi
  for L in ${LEVELS//,/ }; do
    rm -f "$OUT/.jobs/$T/$T"/*__${L}.json "$OUT/.jobs/$T/$T"/*__${L}.trajectory.json \
          "$OUT/.jobs/$T/$T"/*__${L}.model_calls.json "$OUT/.jobs/$T/$T"/*__${L}.agent_stdout.jsonl
  done
  asibench run --instances-dir "hf_instances_${TRACK}/" --tasks-dir third_party/ASI-Bench/tasks/ \
    --tasks "$T" --prompt-levels "$LEVELS" --parallel 1 --sandbox os \
    --agent opencode_cli --agent-config "$AGENT_CFG" --timeout 10800 \
    --resume "$OUT/.jobs/$T" --output-dir "$OUT/.jobs/$T" \
    >> "logs/eval/asi_${MODEL}.log" 2>&1
  echo "=== repair rc=$? for $T"
  [ -d "$OUT/.jobs/$T/$T" ] && mv "$OUT/.jobs/$T/$T" "$OUT/$T"
done
echo "[$(date -u +%FT%TZ)] repair done"
