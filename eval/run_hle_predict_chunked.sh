#!/usr/bin/env bash
# Full HLE text-only predictions with periodic checkpoints.
#
# run_model_predictions.py writes its JSON only after all questions finish, so a
# 20h run would lose everything on interruption.  Feeding an increasing
# --max_samples value makes each invocation save one more chunk (the script
# skips ids already present in the output file), giving resumable checkpoints.
#
# usage: eval/run_hle_predict_chunked.sh [n_questions]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$ROOT/.venv/bin:$PATH"
cd "$ROOT/third_party/hle/hle_eval"

TOTAL=${1:-2158}
CHUNK=${HLE_CHUNK:-150}
WORKERS=${HLE_WORKERS:-16}
MODEL=${HLE_MODEL:-qwen36-35b-base}

N=0
while [ "$N" -lt "$TOTAL" ]; do
  N=$((N + CHUNK))
  [ "$N" -gt "$TOTAL" ] && N=$TOTAL
  echo "[$(date -u +%FT%TZ)] chunk -> $N"
  ok=0
  for attempt in 1 2 3; do
    OPENAI_BASE_URL="${VLLM_BASE:-http://10.164.228.13:8000/v1}" OPENAI_API_KEY=dummy \
      python run_model_predictions.py --dataset ../hle_text_only --model "$MODEL" \
        --max_completion_tokens 65536 --temperature 1.0 \
        --num_workers "$WORKERS" --max_samples "$N" && { ok=1; break; }
    echo "[$(date -u +%FT%TZ)] chunk $N attempt $attempt failed; retrying in 60s"
    sleep 60
  done
  if [ "$ok" -ne 1 ]; then
    echo "[$(date -u +%FT%TZ)] chunk $N failed 3x; stopping"
    exit 1
  fi
  have=$(python3 -c "import json;print(len(json.load(open('hle_${MODEL}.json'))))" 2>/dev/null || echo 0)
  echo "[$(date -u +%FT%TZ)] chunk $N done; predictions=$have"
done
