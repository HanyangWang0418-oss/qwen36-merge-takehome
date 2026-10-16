#!/usr/bin/env bash
# Serve the frozen base + every adapter behind one OpenAI-compatible endpoint.
# All four benchmarks point at this server; the `model` field selects the adapter.
# Requires: 2x H100 80GB (or 1x H200). bf16, no quantization, 256K context for Terminal-Bench.
set -euo pipefail
cd "$(dirname "$0")/.."
REV=$(python -c "import json;print(json.load(open('data/manifests/base_model.json'))['hf_revision'])")
LORA_ARGS=()
for d in models/weights/*/; do
  name=$(basename "$d")
  [ -f "$d/adapter_config.json" ] && LORA_ARGS+=("$name=$d")
done
exec vllm serve Qwen/Qwen3.6-35B-A3B --revision "$REV" \
  --dtype bfloat16 --tensor-parallel-size "${TP:-2}" \
  --max-model-len 262144 --max-num-seqs "${MAX_SEQS:-64}" \
  --enable-lora --max-lora-rank 128 --max-loras 4 --lora-modules "${LORA_ARGS[@]}" \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes \
  --port "${PORT:-8000}" 2>&1 | tee -a logs/vllm_server.log
