#!/usr/bin/env python
"""Make a copy of the SWE-bench Pro V2 HARD-51 task directories whose task.toml points at the
ACR mirror instead of ghcr.io (harbor's ack backend uses `[environment] docker_image` as-is).
Nothing else in the task is changed; the per-task diff is written next to the copy.

    ACR_REGISTRY=... uv run python scripts/prepare_swebench_tasks_acr.py third_party/SWE-bench_Pro-os/v2/tasks eval_tasks/swebench_pro_hard51_acr
"""
import difflib
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src_root, dst_root = Path(sys.argv[1]), Path(sys.argv[2])
acr = os.environ["ACR_REGISTRY"]
ids = json.loads((ROOT / "data/manifests/swebench_pro_v2_hard.json").read_text())["task_ids"]
digests = {}
p = ROOT / "data/manifests/images_acr_swebench_pro.json"
if p.exists():
    digests = {r["task_id"]: r["acr_digest"] for r in json.loads(p.read_text())["images"]}
dst_root.mkdir(parents=True, exist_ok=True)
diffs = []
for tid in ids:
    s, d = src_root / tid, dst_root / tid
    if not s.exists():
        raise SystemExit(f"missing task dir {s}")
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(s, d)
    toml = d / "task.toml"
    old = toml.read_text()
    ref = f"{acr}/swe-bench_pro-v2:{tid}"
    if digests.get(tid):
        ref = f"{acr}/swe-bench_pro-v2@{digests[tid]}"   # pin by digest when known
    new = re.sub(r'docker_image\s*=\s*"ghcr\.io/scaleapi/swe-bench_pro-v2:[^"]+"', f'docker_image = "{ref}"', old)
    assert new != old, f"docker_image line not found in {toml}"
    toml.write_text(new)
    # environment/Dockerfile is `FROM ghcr.io/...`; harbor ack uses docker_image, but keep it consistent
    df = d / "environment" / "Dockerfile"
    if df.exists():
        df.write_text(re.sub(r"FROM ghcr\.io/scaleapi/swe-bench_pro-v2:\S+", f"FROM {ref}", df.read_text()))
    diffs += list(difflib.unified_diff(old.splitlines(), new.splitlines(), f"{tid}/task.toml", f"{tid}/task.toml (acr)", lineterm=""))
(dst_root / "ACR_REWRITE.diff").write_text("\n".join(diffs) + "\n")
print(f"{len(ids)} tasks copied to {dst_root}; only docker_image/FROM lines changed (see ACR_REWRITE.diff)")
