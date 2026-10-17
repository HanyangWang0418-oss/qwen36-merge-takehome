"""vLLM on Modal serving Qwen3.6-35B-A3B (frozen base) + our LoRA adapters behind one OpenAI-compatible
endpoint. Model and sandboxes then both live on Modal: no tunnel, scale-to-zero when idle.

    modal secret create qwen36-vllm VLLM_API_KEY=<random>  HF_TOKEN=<hf token>      # once
    modal run eval/modal_vllm.py::prepare_weights                                      # base + adapters -> Volume (~110 GB)
    modal deploy eval/modal_vllm.py                                                    # prints the https URL
    modal run eval/modal_vllm.py                                                       # health check + one completion per served model

Then in eval/.env: VLLM_BASE=<url>/v1, VLLM_HOST=<url host>, OPENAI_API_KEY=<VLLM_API_KEY>.
Adapters to keep resident are chosen with LORAS (space-separated names from data/manifests/hf_models.json);
vLLM's --max-loras caps how many are on the GPU at once, the rest swap in on demand.
Cost reference (Modal list prices, Oct 2026): H200 ≈ $4.54/h; the server scales to zero after SCALEDOWN_MIN idle minutes.
"""
import json
import os
import subprocess
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
ROOT = Path(__file__).resolve().parents[1]
BASE_REV = json.loads((ROOT / "data/manifests/base_model.json").read_text())["hf_revision"]
HF_MODELS = json.loads((ROOT / "data/manifests/hf_models.json").read_text())
DEFAULT_LORAS = os.environ.get("LORAS", "e1_code_repair e2_terminal_agent e3_reasoning merged_ta_l0.7 merged_ties_d0.2").split()
GPU = os.environ.get("VLLM_GPU", "H200")          # 141 GB: 70 GB bf16 base + adapters + 256K-context KV
SCALEDOWN_MIN = int(os.environ.get("SCALEDOWN_MIN", "10"))
VLLM_PORT = 8000
MIN = 60

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.31.0", "huggingface_hub[hf_transfer]", "safetensors")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_LOG_STATS_INTERVAL": "5"})
    .add_local_file(str(ROOT / "src/merge/remap_keys.py"), "/opt/remap_keys.py")
)
weights = modal.Volume.from_name("qwen36-weights", create_if_missing=True)
vllm_cache = modal.Volume.from_name("qwen36-vllm-cache", create_if_missing=True)
secret = modal.Secret.from_name("qwen36-vllm")   # VLLM_API_KEY, HF_TOKEN
app = modal.App("qwen36-eval-vllm")


@app.function(image=image, volumes={"/weights": weights}, secrets=[secret], timeout=4 * 3600, memory=32768, cpu=8)
def prepare_weights(adapters: str = " ".join(HF_MODELS.keys())):
    """Download the base at the locked revision and every adapter, remap adapter keys for vLLM, commit to the Volume."""
    import sys
    from huggingface_hub import snapshot_download
    sys.path.insert(0, "/opt")
    from remap_keys import remap  # noqa: E402

    tok = os.environ.get("HF_TOKEN")
    base_dir = "/weights/base"
    if not os.path.exists(f"{base_dir}/config.json"):
        snapshot_download(MODEL_ID, revision=BASE_REV, local_dir=base_dir, token=tok,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.py", "*.jinja", "merges.txt", "vocab.json"])
        print("base downloaded", BASE_REV)
    for name in adapters.split():
        rec = HF_MODELS[name]
        src, dst = f"/weights/adapters/{name}", f"/weights/adapters/{name}_vllm"
        if os.path.exists(f"{dst}/adapter_config.json"):
            print("exists", name); continue
        snapshot_download(rec["repo_id"], revision=rec.get("revision"), local_dir=src, token=tok,
                          allow_patterns=["adapter_config.json", "adapter_model*.safetensors"])
        stats = remap(Path(src), Path(dst))
        subprocess.run(["rm", "-rf", src])
        print("remapped", name, {k: stats[k] for k in ("fused_qkv", "renamed_unembed", "n_tensors_out")})
        weights.commit()
    weights.commit()
    print(subprocess.run(["du", "-sh", "/weights/base", "/weights/adapters"], capture_output=True, text=True).stdout)


@app.server(
    image=image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_MIN * MIN,
    startup_timeout=30 * MIN,
    timeout=4 * 3600,
    volumes={"/weights": weights, "/root/.cache/vllm": vllm_cache},
    secrets=[secret],
    port=VLLM_PORT,
    target_concurrency=48,
    unauthenticated=True,          # protected by vLLM --api-key instead (agents send it as OPENAI_API_KEY)
)
class Server:
    @modal.enter()
    def start(self):
        loras = []
        for name in DEFAULT_LORAS:
            p = f"/weights/adapters/{name}_vllm"
            if os.path.exists(f"{p}/adapter_config.json"):
                loras.append(f"{name}={p}")
        max_rank = 32
        for l in loras:
            cfg = json.load(open(l.split("=", 1)[1] + "/adapter_config.json"))
            max_rank = max([max_rank, cfg["r"]] + list((cfg.get("rank_pattern") or {}).values()))
        cmd = ["vllm", "serve", "/weights/base", "--served-model-name", "base", "--host", "0.0.0.0", "--port", str(VLLM_PORT),
               "--dtype", "bfloat16", "--max-model-len", os.environ.get("MAX_LEN", "262144"), "--max-num-seqs", "32",
               "--enable-lora", "--max-lora-rank", str(max_rank), "--max-loras", os.environ.get("MAX_LORAS", "3"),
               "--reasoning-parser", "qwen3", "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
               "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0}),
               "--api-key", os.environ["VLLM_API_KEY"], "--uvicorn-log-level=info"]
        if loras:
            cmd += ["--lora-modules", *loras]
        if os.environ.get("MIXED_MOE_LORA"):
            cmd += ["--enable-mixed-moe-lora-format"]
        print(" ".join(cmd), flush=True)
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.local_entrypoint()
async def test(prompt: str = "List files sorted by size, largest first. Answer briefly."):
    import asyncio, time
    import aiohttp
    url = await Server.get_url.aio()
    key = os.environ.get("VLLM_API_KEY") or "unset"
    print("server url:", url)
    async with aiohttp.ClientSession(base_url=url, headers={"Authorization": f"Bearer {key}"}) as s:
        deadline = time.time() + 29 * MIN
        while time.time() < deadline:
            async with s.get("/health", timeout=aiohttp.ClientTimeout(total=60)) as r:
                if r.status == 200:
                    break
                await asyncio.sleep(5)
        async with s.get("/v1/models") as r:
            models = [m["id"] for m in (await r.json())["data"]]
        print("served:", models)
        for m in models:
            t0 = time.time()
            async with s.post("/v1/chat/completions", json={"model": m, "messages": [{"role": "user", "content": prompt}], "max_tokens": 400,
                                                             "temperature": 1.0, "top_p": 0.95, "extra_body": {"top_k": 20}}) as r:
                d = await r.json()
            msg = d["choices"][0]["message"]
            print(f"[{m}] {time.time()-t0:.1f}s | reasoning {len(msg.get('reasoning_content') or msg.get('reasoning') or '')} chars | {(msg.get('content') or '')[:120]!r}")
