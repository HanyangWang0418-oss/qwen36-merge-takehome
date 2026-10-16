#!/usr/bin/env python
"""Upload an adapter directory to a public Hugging Face model repo with a rendered model card,
then verify it downloads anonymously. Records repo id + revision in data/manifests/hf_models.json.

    HF_TOKEN=... uv run python scripts/upload_hf.py models/weights/e1_code_repair ssyhw7/qwen36-35b-e1-code-repair --card models/cards/e1_code_repair.md
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter_dir"); ap.add_argument("repo_id"); ap.add_argument("--card", required=True)
    a = ap.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(a.repo_id, repo_type="model", private=False, exist_ok=True)
    d = Path(a.adapter_dir)
    assert (d / "adapter_config.json").exists() and (d / "adapter_model.safetensors").exists(), "not a PEFT adapter dir"
    (d / "README.md").write_text(Path(a.card).read_text())
    info = api.upload_folder(folder_path=str(d), repo_id=a.repo_id, repo_type="model", commit_message=f"upload {d.name}")
    rev = api.model_info(a.repo_id).sha
    # anonymous verification: no token, fresh cache dir
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = snapshot_download(a.repo_id, revision=rev, token=False, cache_dir=tmp, allow_patterns=["adapter_config.json", "README.md"])
        ok = (Path(p) / "adapter_config.json").exists()
    rec_path = ROOT / "data/manifests/hf_models.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {}
    rec[d.name] = {"repo_id": a.repo_id, "revision": rev, "url": f"https://huggingface.co/{a.repo_id}", "anonymous_download_ok": ok,
                   "uploaded_at": datetime.now(timezone.utc).isoformat()}
    rec_path.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"{a.repo_id} @ {rev} anonymous_download_ok={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
