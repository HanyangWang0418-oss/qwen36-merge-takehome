#!/usr/bin/env bash
# Pre-flight before spending: endpoint reachable (locally and from Modal), one TB2 oracle task on Modal,
# one SWE-bench Pro task with the locked agent, three HLE questions. Exits non-zero on the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . eval/.env; set +a
echo "1) endpoint"; curl -sf "$VLLM_BASE/models" | python3 -c "import sys,json;print('  models:',[m['id'] for m in json.load(sys.stdin)['data']])"
curl -sf "$VLLM_BASE/chat/completions" -H 'Content-Type: application/json' -d '{"model":"base","messages":[{"role":"user","content":"Say OK."}],"max_tokens":64}' | python3 -c "import sys,json;print('  base reply:',json.load(sys.stdin)['choices'][0]['message']['content'][:80])"
echo "2) Modal oracle on one TB2 task"
TBV=$(python3 -c "import json;print(json.load(open('data/manifests/terminal_bench_2_tasks.json'))['version'])")
T1=$(python3 -c "import json;print(json.load(open('data/manifests/terminal_bench_2_tasks.json'))['task_ids'][0])")
harbor run -p third_party/terminal-bench-2 -i "$T1" -a oracle -e modal -n 1 --job-name smoke-oracle -o jobs_smoke -y 2>&1 | tail -5
echo "3) one TB2 task with terminus-2 against the endpoint (model-card settings, 20 min cap)"
harbor run -p third_party/terminal-bench-2 -i "$T1" -a terminus-2 -m "openai/base" -e modal -n 1 \
  --override-cpus "${TB_CPUS}" --override-memory-mb "${TB_MEM_MB}" --agent-timeout-multiplier 0.12 \
  --ak api_base="$VLLM_BASE" --ak temperature=1.0 --ak 'llm_call_kwargs={"extra_body":{"top_k":20},"top_p":0.95,"max_tokens":81920}' \
  --ae OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}" --ae OPENAI_API_BASE="$VLLM_BASE" --allow-agent-host "$VLLM_HOST" \
  --job-name smoke-tb2 -o jobs_smoke -y 2>&1 | tail -8
echo "4) one SWE-bench Pro task with the locked agent"
SBP=third_party/SWE-bench_Pro-os; S1=$(head -1 $SBP/v2/hard51_ids.txt)
PYTHONPATH=$SBP/v2/tooling OPENAI_API_KEY=${OPENAI_API_KEY:-dummy} OPENAI_API_BASE="$VLLM_BASE" harbor run -p $SBP/v2/tasks -i "$S1" -e modal -n 1 \
  -a locked_mini_swe:LockedMiniSwe -m "openai/base" --ak config_file=$SBP/v2/tooling/configs/mini_toolcall.yaml \
  --ae OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}" --ae OPENAI_API_BASE="$VLLM_BASE" --allow-agent-host "$VLLM_HOST" \
  --job-name smoke-sbp -o jobs_smoke -y 2>&1 | tail -8
echo "5) HLE: 3 questions (prediction only)"
HLE=third_party/hle; [ -f "$HLE/hle_text_only/test-00000-of-00001.parquet" ] || uv run python src/eval/hle_textonly_dataset.py data/manifests/hle_text_only_ids.json "$HLE/hle_text_only"
( cd "$HLE/hle_eval" && OPENAI_BASE_URL="$VLLM_BASE" OPENAI_API_KEY=dummy python run_model_predictions.py --dataset ../hle_text_only --model base --max_completion_tokens 4096 --num_workers 3 --max_samples 3 ) 2>&1 | tail -3
echo "smoke OK"
