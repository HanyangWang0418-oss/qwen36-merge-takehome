#!/usr/bin/env bash
# Terminal-Bench 2.0 on Modal, terminus-2, five seeded runs, model-card-aligned settings.
# usage: eval/run_terminal_bench.sh <model_name_on_vllm> <group> [--seeds "1 2 3 4 5"] [--tasks "t1 t2"]
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . eval/.env; set +a
MODEL=$1; GROUP=$2; shift 2
SERVED=$MODEL; [ "$MODEL" = base ] && SERVED=${BASE_MODEL_NAME:-base}
EP=$(python3 eval/resolve_endpoint.py "$MODEL" $([ "${HARBOR_ENV:-modal}" = modal ] && echo --modal)) || exit 2; [ -n "$EP" ] && eval "$EP"   # per-model endpoint (eval/endpoints.yaml)
SEEDS="1 2 3 4 5"; TASKS=""
while [ $# -gt 0 ]; do case $1 in --seeds) SEEDS=$2; shift 2;; --tasks) TASKS=$2; shift 2;; *) shift;; esac; done
TBV=$(python3 -c "import json;print(json.load(open('data/manifests/terminal_bench_2_tasks.json'))['version'])")
TBC=$(python3 -c "import json;print(json.load(open('data/manifests/terminal_bench_2_tasks.json'))['dataset_git_commit'][0])")
TBP=third_party/terminal-bench-2   # laude-institute/terminal-bench-2 at the locked commit (eval/setup.sh); avoids Harbor Hub tag resolution
[ "$(git -C $TBP rev-parse HEAD)" = "$TBC" ] || { echo "dataset checkout mismatch; run eval/setup.sh"; exit 1; }
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy} OPENAI_API_BASE="$VLLM_BASE"
TASK_ARGS=(); for t in $TASKS; do TASK_ARGS+=(-i "$t"); done
mkdir -p logs/eval
for SEED in $SEEDS; do
  JOB="tb2-${MODEL}-seed${SEED}"
  echo "[$(date -u +%FT%TZ)] model=$MODEL seed=$SEED cpus=${TB_CPUS} mem_mb=${TB_MEM_MB} env=modal agent=terminus-2 dataset=terminal-bench-2@${TBC:0:8} (v${TBV})" | tee -a logs/eval/${JOB}.log
  harbor run -p "$TBP" "${TASK_ARGS[@]}" -a terminus-2 -m "openai/${SERVED}" \
    -e "${HARBOR_ENV:-modal}" -n "${CONCURRENCY:-20}" -k 1 \
    --override-cpus "${TB_CPUS}" --override-memory-mb "${TB_MEM_MB}" \
    --ak api_base="$VLLM_BASE" --ak temperature=1.0 --ak max_thinking_tokens=81920 \
    --ak 'llm_call_kwargs={"extra_body":{"top_k":20,"seed":'"$SEED"'},"top_p":0.95,"max_tokens":81920}' \
    --ae OPENAI_API_KEY="$OPENAI_API_KEY" --ae OPENAI_API_BASE="$OPENAI_API_BASE" \
    --allow-agent-host "$VLLM_HOST" \
    --job-name "$JOB" -o jobs 2>&1 | tee -a logs/eval/${JOB}.log
done
uv run python src/eval/collect_harbor.py "jobs/tb2-${MODEL}-seed*" "results/$GROUP/$MODEL/terminal_bench_2"
python3 - "$MODEL" "$GROUP" <<'PY'
import json, os, sys
m, g = sys.argv[1:3]
meta = {"cpus": int(os.environ["TB_CPUS"]), "memory_mb": int(os.environ["TB_MEM_MB"]), "timeout_s": 10800, "env": "modal",
        "seeds": os.environ.get("SEEDS", "1 2 3 4 5"), "note": "model-card spec is 32 CPU / 48 GiB; values here are what was actually used"}
json.dump(meta, open(f"results/{g}/{m}/terminal_bench_2/resources_used.json", "w"), indent=2)
PY
