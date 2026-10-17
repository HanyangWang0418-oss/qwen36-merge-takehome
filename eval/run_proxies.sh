#!/usr/bin/env bash
# Dev-proxy screening for merge ablations / controls (configs/eval/dev_proxies.json). Never reported as benchmark numbers.
# usage: eval/run_proxies.sh <model_name_on_vllm> <group>   -> results_proxy/<group>/<model>/...
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . eval/.env; set +a
MODEL=$1; GROUP=$2
P=configs/eval/dev_proxies.json
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy} OPENAI_API_BASE="$VLLM_BASE"
mkdir -p "results_proxy/$GROUP/$MODEL" logs/eval
# 1. HLE proxy: MMLU-Pro STEM 300 via the vLLM endpoint (exact-match letter)
uv run python src/eval/proxy_mmlu_pro.py --model "$MODEL" --base "$VLLM_BASE" --out "results_proxy/$GROUP/$MODEL/hle_proxy" 2>&1 | tee -a logs/eval/proxy_${MODEL}.log
# 2. SWE-bench Verified 15 (Modal, harbor dataset)
IDS=$(python3 -c "import json;print(' '.join('-i '+t for t in json.load(open('$P'))['swebench_proxy']['tasks']))")
harbor run -d swebench-verified@1.0 $IDS -a terminus-2 -m "openai/${MODEL}" -e modal -n "${CONCURRENCY:-15}" -k 1 \
  --ak api_base="$VLLM_BASE" --ak temperature=0.6 --ak 'llm_call_kwargs={"extra_body":{"top_k":20},"top_p":0.95,"max_tokens":32768}' \
  --ae OPENAI_API_KEY="$OPENAI_API_KEY" --ae OPENAI_API_BASE="$OPENAI_API_BASE" --allow-agent-host "$VLLM_HOST" \
  --job-name "proxy-swev-${MODEL}" -o jobs_proxy 2>&1 | tee -a logs/eval/proxy_${MODEL}.log
uv run python src/eval/collect_harbor.py "jobs_proxy/proxy-swev-${MODEL}" "results_proxy/$GROUP/$MODEL/swebench_proxy"
# 3. Terminal-Bench-Pro 20 (Modal), one run, same terminus-2 settings but task.toml resources (screening only)
IDS=$(python3 -c "import json;print(' '.join('-i '+t for t in json.load(open('$P'))['terminal_bench_proxy']['tasks']))")
harbor run -d terminal-bench-pro@1.0 $IDS -a terminus-2 -m "openai/${MODEL}" -e modal -n "${CONCURRENCY:-20}" -k 1 \
  --ak api_base="$VLLM_BASE" --ak temperature=1.0 --ak max_thinking_tokens=81920 --ak 'llm_call_kwargs={"extra_body":{"top_k":20,"seed":1},"top_p":0.95,"max_tokens":81920}' \
  --ae OPENAI_API_KEY="$OPENAI_API_KEY" --ae OPENAI_API_BASE="$OPENAI_API_BASE" --allow-agent-host "$VLLM_HOST" \
  --job-name "proxy-tbpro-${MODEL}" -o jobs_proxy 2>&1 | tee -a logs/eval/proxy_${MODEL}.log
uv run python src/eval/collect_harbor.py "jobs_proxy/proxy-tbpro-${MODEL}" "results_proxy/$GROUP/$MODEL/terminal_bench_proxy"
# 4. ASI seed31415, 12 instances, local scoring (docker on this machine)
INST=$(python3 -c "import json;print(' '.join(sorted(json.load(open('data/manifests/asi_bench_tasks.json'))['instance_ids'])[:12]))" | sed 's/__seed42/__seed31415/g')
eval/run_asi_bench.sh "$MODEL" "proxy_$GROUP" --track seed31415 --instances "$INST" || echo "ASI proxy skipped/failed"
uv run python scripts/summarize_proxies.py
