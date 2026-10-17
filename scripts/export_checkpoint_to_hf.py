#!/usr/bin/env python
"""Export a Tinker sampler checkpoint to a public Hugging Face PEFT repo without `tinker checkpoint push-hf`.

Order of attempts:
  1. `save_weights_external` from the matching training-state checkpoint + `get_external_weights_urls`
     (recommended by Tinker; ~1 min; yields adapter_config.json + adapter_model.safetensors directly)
  2. archive URL (`get_checkpoint_archive_url_from_tinker_path`, deprecated), waiting up to --wait-min for the
     server-side archive to be created (it can take >10 min for fresh checkpoints)
Then: verify, add TINKER_SOURCE.txt + a minimal card, upload with HfApi, print repo@sha.

    TINKER_API_KEY=... HF_TOKEN=... python scripts/export_checkpoint_to_hf.py <tinker sampler path> <hf repo id> [--wait-min 45]
"""
import argparse
import json
import os
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=600) as r, open(dst, "wb") as f:
        while True:
            b = r.read(16 * 1024 * 1024)
            if not b:
                break
            f.write(b)


def via_external(sc, rc, tinker_path: str, out: Path) -> bool:
    state = tinker_path.replace("/sampler_weights/", "/weights/")
    tc = sc.create_training_client_from_state(path=state)
    res = tc.save_weights_external(name=tinker_path.rsplit("/", 1)[-1] + "_ext").result()
    urls = rc.get_external_weights_urls(res.path).result()
    for rel, u in urls.urls.items():
        download(u, out / rel)
    print(f"external-weights files: {sorted(urls.urls)}", flush=True)
    return (out / "adapter_config.json").exists()


def via_archive(rc, tinker_path: str, out: Path, wait_min: int) -> bool:
    deadline = time.time() + wait_min * 60
    while time.time() < deadline:
        try:
            url = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path).result(timeout=max(60, deadline - time.time())).url
            tar_path = out.parent / "ckpt.tar"
            download(url, tar_path)
            with tarfile.open(tar_path) as tf:
                tf.extractall(out, filter="data")
            return (out / "adapter_config.json").exists()
        except Exception as e:  # noqa: BLE001
            print(f"archive attempt failed ({repr(e)[:120]}); retrying in 30s", flush=True)
            time.sleep(30)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tinker_path")
    ap.add_argument("repo_id")
    ap.add_argument("--wait-min", type=int, default=45)
    a = ap.parse_args()
    import tinker
    from huggingface_hub import HfApi

    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()
    work = Path(tempfile.mkdtemp(prefix="export_"))
    out = work / "adapter"
    out.mkdir()
    t0 = time.time()
    got = False
    try:
        got = via_external(sc, rc, a.tinker_path, out)
        print(f"external-weights path {'ok' if got else 'incomplete'} in {time.time() - t0:.0f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"external-weights path failed ({repr(e)[:160]}); falling back to archive", flush=True)
    if not got:
        got = via_archive(rc, a.tinker_path, out, a.wait_min)
        print(f"archive path {'ok' if got else 'failed'} in {time.time() - t0:.0f}s", flush=True)
    if not got:
        raise SystemExit("no adapter obtained")
    files = sorted(p.name for p in out.iterdir())
    assert any(f.startswith("adapter_model") for f in files), files
    (out / "TINKER_SOURCE.txt").write_text(a.tinker_path + "\n")
    if not (out / "README.md").exists():
        (out / "README.md").write_text(
            f"---\nlicense: apache-2.0\nbase_model: Qwen/Qwen3.6-35B-A3B\nbase_model_relation: adapter\nlibrary_name: peft\n---\n"
            f"# {a.repo_id.split('/')[-1]}\n\nLoRA adapter exported from Tinker checkpoint `{a.tinker_path}`. Model card to be completed; "
            f"code, data and evaluation: https://github.com/HanyangWang0418-oss/qwen36-merge-takehome\n")
    if (out / "checkpoint_complete").exists():
        (out / "checkpoint_complete").unlink()
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(a.repo_id, private=False, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=a.repo_id, commit_message=f"adapter from {a.tinker_path}")
    sha = api.model_info(a.repo_id).sha
    cfg = json.loads((out / "adapter_config.json").read_text())
    print(json.dumps({"repo": a.repo_id, "sha": sha, "r": cfg.get("r"), "lora_alpha": cfg.get("lora_alpha"),
                      "target_modules": cfg.get("target_modules"), "files": files, "seconds": round(time.time() - t0)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
