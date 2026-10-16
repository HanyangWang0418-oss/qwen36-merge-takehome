#!/usr/bin/env bash
# HLE text-only subset with the official predict + judge scripts.
# usage: scripts/eval_hle.sh <model_name_on_vllm> <results_group>
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=$1; GROUP=$2
HLE=${HLE_DIR:-third_party/hle}
export OPENAI_BASE_URL=${VLLM_BASE:-http://localhost:8000/v1}   # predictions hit our server
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
uv run python src/eval/hle_textonly_dataset.py data/manifests/hle_text_only_ids.json "$HLE/hle_text_only"   # local filtered copy, never committed
( cd "$HLE/hle_eval" && python run_model_predictions.py --dataset ../hle_text_only --model "$MODEL" \
    --max_completion_tokens 32768 --num_workers "${WORKERS:-64}" --temperature 1.0 ) 2>&1 | tee "logs/eval_hle_${MODEL}.log"
# judge uses the real OpenAI endpoint
( cd "$HLE/hle_eval" && OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY="$OPENAI_JUDGE_KEY" \
    python run_judge_results.py --dataset ../hle_text_only --predictions "hle_${MODEL}.json" --judge o3-mini-2025-01-31 --num_workers 32 ) 2>&1 | tee -a "logs/eval_hle_${MODEL}.log"
uv run python src/eval/collect_hle.py "$HLE/hle_eval/judged_hle_${MODEL}.json" "results/$GROUP/$MODEL/hle"
