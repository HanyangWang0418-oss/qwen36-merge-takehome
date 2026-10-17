"""Run an eval/*.sh script inside a Modal CPU container (up to 24 h), so that Harbor's `-e modal` sandboxes and the
vLLM server are all in Modal and nothing depends on the laptop's network. The repo is snapshotted into the image at
launch; results are committed back to GitHub from the container (needs GH_TOKEN in the qwen36-eval secret) or written
to the `qwen36-eval-results` Volume (/results) as a fallback.

    modal run --detach eval/modal_driver.py --script "eval/run_all.sh base"
"""
import os
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "docker.io", "jq")
    .uv_pip_install("harbor[modal]==0.24.0", "e2b==2.45.0", "modal==1.6.2", "huggingface_hub", "pyyaml", "numpy", "matplotlib", "pandas", "openai", "datasets")
    .add_local_dir(str(ROOT), "/repo", ignore=[".venv", ".venv-modal", "data/raw", "data/processed", "models/weights", "third_party", "jobs", "jobs_smoke", "jobs_proxy", ".git"])
)
results = modal.Volume.from_name("qwen36-eval-results", create_if_missing=True)
app = modal.App("qwen36-eval-driver")


@app.function(image=image, cpu=4, memory=16384, timeout=24 * 3600, volumes={"/results": results},
              secrets=[modal.Secret.from_name("qwen36-eval")])
def run(script: str):
    os.chdir("/repo")
    env = dict(os.environ)
    # eval/.env equivalent from the secret + serving manifest
    import json
    url = json.load(open("data/manifests/serving.json"))["vllm_url"].rstrip("/")
    envfile = (f"VLLM_BASE={url}/v1\nVLLM_HOST={url.split('://')[1].split('/')[0]}\nOPENAI_API_KEY={env['OPENAI_API_KEY']}\n"
               f"OPENAI_JUDGE_KEY={env.get('OPENAI_JUDGE_KEY','')}\nASIBENCH_JUDGE_API_BASE=https://api.openai.com/v1\nASIBENCH_JUDGE_API_KEY={env.get('ASIBENCH_JUDGE_API_KEY','')}\n"
               f"ASIBENCH_JUDGE_MODEL={env.get('ASIBENCH_JUDGE_MODEL','gpt-5-mini')}\nHF_TOKEN={env['HF_TOKEN']}\nHARBOR_ENV=modal\nCONCURRENCY={env.get('CONCURRENCY','16')}\n"
               f"TB_CPUS={env.get('TB_CPUS','4')}\nTB_MEM_MB={env.get('TB_MEM_MB','8192')}\n")
    Path("eval/.env").write_text(envfile)
    subprocess.run(["bash", "-lc", "uv --version || (curl -LsSf https://astral.sh/uv/install.sh | sh)"], check=False)
    env["PATH"] = f"/root/.local/bin:{env['PATH']}"
    subprocess.run(["bash", "-lc", "eval/setup.sh"], env=env, check=True)
    rc = subprocess.run(["bash", "-lc", script], env=env).returncode
    # persist outputs
    stamp = subprocess.run(["date", "-u", "+%Y%m%dT%H%M%SZ"], capture_output=True, text=True).stdout.strip()
    dst = f"/results/{stamp}_{script.replace('/', '_').replace(' ', '_')}"
    subprocess.run(["bash", "-lc", f"mkdir -p '{dst}' && cp -r results logs analysis '{dst}/' 2>/dev/null; cp -r jobs '{dst}/' 2>/dev/null || true"], check=False)
    results.commit()
    if env.get("GH_TOKEN"):
        subprocess.run(["bash", "-lc", (
            'git config user.name "modal-driver" && git config user.email "modal-driver@users.noreply.github.com" && '
            'git remote set-url origin "https://x-access-token:${GH_TOKEN}@github.com/HanyangWang0418-oss/qwen36-merge-takehome.git" && '
            'git add results analysis logs/eval configs/eval 2>/dev/null; git commit -m "eval results: ' + script + '" || true; '
            'for i in 1 2 3; do git pull --rebase -X theirs origin main && git push && break || sleep 10; done')], env=env, check=False)
    print(f"script exit code {rc}; outputs in {dst}")
    return rc


@app.local_entrypoint()
def main(script: str = "eval/smoke.sh"):
    print("launching:", script)
    print("exit", run.remote(script))
