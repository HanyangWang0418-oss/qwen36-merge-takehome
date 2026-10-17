#!/usr/bin/env bash
# Serve the frozen base + every adapter behind one OpenAI-compatible endpoint.
# All four benchmarks point at this server; the `model` field selects the adapter.
# Hardware: 2x A100/H100 80GB (TP=2). bf16, no quantization, 256K context for Terminal-Bench.
#
# Adapters are downloaded from the HF repos listed in data/manifests/hf_models.json and remapped once
# (src/merge/remap_keys.py: in_proj_q/k/v -> in_proj_qkv, unembed_tokens -> lm_head) into models/weights/<name>_vllm.
# Routed-expert LoRA tensors are fused 3D (vllm/lora/utils.py handles experts.w1/w2/w3; set
# --enable-mixed-moe-lora-format if vLLM reports a 2D/3D layout mismatch).
set -euo pipefail
cd "$(dirname "$0")/.."
REV=$(python3 -c "import json;print(json.load(open('data/manifests/base_model.json'))['hf_revision'])")
mkdir -p models/weights
python3 - <<'PY'
import json, os, subprocess, sys
from huggingface_hub import snapshot_download
m = json.load(open(os.environ.get("HF_MODELS_FILE", "data/manifests/hf_models.json")))
for name, rec in m.items():
    src = f"models/weights/{name}"
    dst = f"models/weights/{name}_vllm"
    if not os.path.exists(f"{dst}/adapter_config.json"):
        snapshot_download(rec["repo_id"], revision=rec.get("revision"), local_dir=src, allow_patterns=["adapter_config.json", "adapter_model*.safetensors"])
        subprocess.run([sys.executable, "src/merge/remap_keys.py", src, dst], check=True)
    print("ready", name, dst)
PY
LORA_ARGS=()
for d in models/weights/*_vllm/; do
  name=$(basename "$d" _vllm)
  [ -f "$d/adapter_config.json" ] && LORA_ARGS+=("$name=$d")
done
MAX_RANK=$(python3 - <<'PY'
import glob, json
r = 0
for f in glob.glob("models/weights/*_vllm/adapter_config.json"):
    c = json.load(open(f)); r = max([r, c["r"]] + list((c.get("rank_pattern") or {}).values()))
print(r)
PY
)
echo "adapters: ${LORA_ARGS[*]} | max rank $MAX_RANK"
exec vllm serve Qwen/Qwen3.6-35B-A3B --revision "$REV" \
  --dtype bfloat16 --tensor-parallel-size "${TP:-2}" \
  --max-model-len "${MAX_LEN:-262144}" --max-num-seqs "${MAX_SEQS:-32}" \
  --enable-lora --max-lora-rank "$MAX_RANK" --max-loras "${MAX_LORAS:-3}" --lora-modules "${LORA_ARGS[@]}" \
  ${MIXED_MOE_LORA:+--enable-mixed-moe-lora-format} \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes \
  --generation-config auto \
  --served-model-name base \
  --port "${PORT:-8000}" 2>&1 | tee -a logs/vllm_server.log
