#!/usr/bin/env python
"""Pre-build Alibaba Cloud Agent Sandbox templates with the exact alias Harbor's `e2b`
backend derives, so `harbor run -e e2b` finds them and skips its own build.

Why: the Alibaba Cloud E2B-compatible builder (a) only supports `from_image`, not Dockerfiles,
(b) requires registry credentials and push rights on the source registry, and (c) harbor's e2b
backend calls `Template().from_image(image)` without credentials. So every task image is first
pushed to ACR (scripts/mirror_images.sh / build_tb2_images.sh), the task copies point at ACR
(scripts/prepare_*_tasks_acr.py), and this script builds one template per task from ACR with
credentials, at the CPU/memory the evaluation needs.

Run with harbor's own interpreter so the alias hash is computed by harbor's code:
    export E2B_API_KEY=... E2B_API_URL=https://api.cn-hangzhou.sandbox.aliyuncs.com E2B_DOMAIN=cn-hangzhou.sandbox.aliyuncs.com
    export ACR_USERNAME=... ACR_PASSWORD=...
    ~/.local/share/uv/tools/harbor/bin/python scripts/prebuild_sandbox_templates.py eval_tasks/swebench_pro_hard51_acr --cpus 1 --memory-mb 4096
    ~/.local/share/uv/tools/harbor/bin/python scripts/prebuild_sandbox_templates.py eval_tasks/terminal_bench_2_acr --cpus 32 --memory-mb 49152 --parallel 4

Writes data/manifests/sandbox_templates_<dir>.json (alias, image, spec, template id, build seconds).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def harbor_alias(task_dir: Path, docker_image: str | None, environment_name: str) -> str:
    """Mirror harbor.environments.e2b.E2BEnvironment.__init__ exactly."""
    from harbor.environments.definition import SNAPSHOT_HASH_LEN, environment_content_hash

    env_hash = environment_content_hash(task_dir / "environment", docker_image=docker_image)[:SNAPSHOT_HASH_LEN]
    return f"{environment_name}__{env_hash}".replace("/", "__").replace(".", "-")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks_dir")
    ap.add_argument("--cpus", type=int, required=True)
    ap.add_argument("--memory-mb", type=int, required=True)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--only", default=None, help="comma-separated task dir names")
    ap.add_argument("--dry-run", action="store_true", help="print aliases only")
    a = ap.parse_args()

    from e2b import Template

    tasks_dir = Path(a.tasks_dir)
    only = set(a.only.split(",")) if a.only else None
    user, pwd = os.environ.get("ACR_USERNAME"), os.environ.get("ACR_PASSWORD")
    jobs = []
    for td in sorted(p for p in tasks_dir.iterdir() if (p / "task.toml").exists()):
        if only and td.name not in only:
            continue
        # Use harbor's own Task model so environment_name == trial.task.short_name exactly.
        from harbor.models.task.task import Task

        task = Task(td)
        image = task.config.environment.docker_image
        if not image:
            print(f"SKIP {td.name}: task.toml has no [environment] docker_image (run prepare_*_tasks_acr.py first)")
            continue
        env_name = task.short_name
        jobs.append((td.name, env_name, image, harbor_alias(td, image, env_name)))

    print(f"{len(jobs)} templates at {a.cpus} vCPU / {a.memory_mb} MiB")
    if a.dry_run:
        for name, env_name, image, alias in jobs:
            print(f"  {alias}  <- {image}")
        return 0

    def build(job):
        name, env_name, image, alias = job
        t0 = time.time()
        try:
            if Template.alias_exists(alias):
                return {"task": name, "alias": alias, "image": image, "status": "exists", "build_s": 0}
            tpl = Template().from_image(image, username=user, password=pwd) if user else Template().from_image(image)
            info = Template.build(tpl, alias=alias, cpu_count=a.cpus, memory_mb=a.memory_mb)
            return {"task": name, "alias": alias, "image": image, "status": "built", "build_s": round(time.time() - t0, 1),
                    "template_id": getattr(info, "template_id", None), "build_id": getattr(info, "build_id", None)}
        except Exception as e:  # noqa: BLE001
            return {"task": name, "alias": alias, "image": image, "status": "error", "build_s": round(time.time() - t0, 1), "error": repr(e)[:500]}

    results = []
    with ThreadPoolExecutor(a.parallel) as ex:
        for fut in as_completed([ex.submit(build, j) for j in jobs]):
            r = fut.result()
            results.append(r)
            print(f"{r['status']:6} {r['task'][:60]} {r['build_s']}s {r.get('error', '')[:120]}")
    out = ROOT / "data/manifests" / f"sandbox_templates_{tasks_dir.name}.json"
    out.write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), "api_url": os.environ.get("E2B_API_URL"),
                               "cpus": a.cpus, "memory_mb": a.memory_mb, "n": len(results),
                               "n_error": sum(r["status"] == "error" for r in results), "templates": sorted(results, key=lambda r: r["task"])}, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")
    return 1 if any(r["status"] == "error" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
