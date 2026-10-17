#!/usr/bin/env bash
# On the GPU box. Starts vLLM with the base + every adapter from data/manifests/hf_models.json (keys remapped
# for vLLM) and exposes it through a cloudflared quick tunnel so Modal sandboxes can reach it.
# Prints VLLM_BASE / VLLM_HOST to paste into eval/.env. Run inside tmux.
#   ADAPTERS="base e1_code_repair merged_ta_l0.7" eval/serve.sh    # subset (max 3-4 LoRAs resident is plenty)
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f eval/.env ] && set -a && . eval/.env && set +a
command -v cloudflared >/dev/null || { echo "install cloudflared: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"; exit 1; }
export MAX_LORAS=${MAX_LORAS:-4}
if [ -n "${ADAPTERS:-}" ]; then
  python3 - "$ADAPTERS" <<'PY'
import json, sys
m = json.load(open("data/manifests/hf_models.json")); keep = set(sys.argv[1].split())
json.dump({k: v for k, v in m.items() if k in keep}, open("/tmp/hf_models_subset.json", "w"))
PY
  export HF_MODELS_FILE=/tmp/hf_models_subset.json
fi
PORT=${PORT:-8000}
( cloudflared tunnel --url "http://localhost:$PORT" --no-autoupdate 2>&1 | tee logs/cloudflared.log | grep -m1 -oE "https://[a-z0-9-]+\.trycloudflare\.com" | sed 's#https://##' > /tmp/tunnel_host.txt ) &
PORT=$PORT scripts/serve_vllm.sh &
VPID=$!
for i in $(seq 1 60); do [ -s /tmp/tunnel_host.txt ] && break; sleep 2; done
HOST=$(cat /tmp/tunnel_host.txt 2>/dev/null || true)
echo "=== VLLM_BASE=https://$HOST/v1"; echo "=== VLLM_HOST=$HOST"
echo "waiting for vLLM to be ready..."; until curl -sf "http://localhost:$PORT/v1/models" >/dev/null; do sleep 10; done
curl -s "http://localhost:$PORT/v1/models" | python3 -c "import sys,json;print('served models:',[m['id'] for m in json.load(sys.stdin)['data']])"
wait $VPID
