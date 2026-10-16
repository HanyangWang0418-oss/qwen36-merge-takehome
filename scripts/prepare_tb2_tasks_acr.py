#!/usr/bin/env python
"""Copy the Terminal-Bench 2.0 tasks at the locked commit and add
`[environment] docker_image = "<ACR>/<task>:<commit8>"` so harbor's e2b backend uses the
pre-built ACR image (from scripts/build_tb2_images.sh) instead of trying to build the
Dockerfile, which the Alibaba Cloud builder does not support. Nothing else is changed; the
unified diff of every task.toml is written to ACR_REWRITE.diff.

    ACR_REGISTRY=... uv run python scripts/prepare_tb2_tasks_acr.py third_party/terminal-bench-2 eval_tasks/terminal_bench_2_acr
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
m = json.loads((ROOT / "data/manifests/terminal_bench_2_tasks.json").read_text())
c8 = m["dataset_git_commit"][0][:8]
dst_root.mkdir(parents=True, exist_ok=True)
diffs = []
for t in m["task_ids"]:
    s, d = src_root / t, dst_root / t
    if not s.exists():
        raise SystemExit(f"missing task dir {s} (clone the dataset at commit {m['dataset_git_commit'][0]})")
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(s, d, ignore=shutil.ignore_patterns(".git"))
    toml = d / "task.toml"
    old = toml.read_text()
    ref = f"{acr}/{t}:{c8}"
    if re.search(r"^\s*docker_image\s*=", old, re.M):
        new = re.sub(r'^(\s*docker_image\s*=\s*)"[^"]*"', rf'\1"{ref}"', old, count=1, flags=re.M)
    else:
        new = re.sub(r"^\[environment\]\s*$", f'[environment]\ndocker_image = "{ref}"', old, count=1, flags=re.M)
        if new == old:  # no [environment] table at all
            new = old.rstrip("\n") + f'\n\n[environment]\ndocker_image = "{ref}"\n'
    toml.write_text(new)
    diffs += list(difflib.unified_diff(old.splitlines(), new.splitlines(), f"{t}/task.toml", f"{t}/task.toml (acr)", lineterm=""))
(dst_root / "ACR_REWRITE.diff").write_text("\n".join(diffs) + "\n")
print(f"{len(m['task_ids'])} tasks copied to {dst_root}; only docker_image lines changed (ACR_REWRITE.diff)")
