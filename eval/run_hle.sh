#!/usr/bin/env bash
# HLE text-only subset with the official predict + judge scripts (no sandbox).
# usage: eval/run_hle.sh <model_name_on_vllm> <group>
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . eval/.env; set +a
MODEL=$1; GROUP=$2
HLE=third_party/hle
[ -f "$HLE/hle_text_only/test-00000-of-00001.parquet" ] || uv run python src/eval/hle_textonly_dataset.py data/manifests/hle_text_only_ids.json "$HLE/hle_text_only"
mkdir -p logs/eval
echo "[$(date -u +%FT%TZ)] model=$MODEL n=$(python3 -c "import json;print(json.load(open('data/manifests/hle_text_only_ids.json'))['n_text_only'])") judge=o3-mini-2025-01-31" | tee -a logs/eval/hle_${MODEL}.log
( cd "$HLE/hle_eval" && OPENAI_BASE_URL="$VLLM_BASE" OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}" \
    python run_model_predictions.py --dataset ../hle_text_only --model "$MODEL" --max_completion_tokens 32768 --temperature 1.0 \
    --num_workers "${HLE_WORKERS:-48}" ) 2>&1 | tee -a logs/eval/hle_${MODEL}.log
( cd "$HLE/hle_eval" && OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY="$OPENAI_JUDGE_KEY" \
    python run_judge_results.py --dataset ../hle_text_only --predictions "hle_${MODEL}.json" --judge o3-mini-2025-01-31 --num_workers 32 ) 2>&1 | tee -a logs/eval/hle_${MODEL}.log
uv run python src/eval/collect_hle.py "$HLE/hle_eval/judged_hle_${MODEL}.json.json" "results/$GROUP/$MODEL/hle"   # official script names it judged_<predictions>.json
