#!/usr/bin/env python
"""Record the manifest digests of the task images actually used for evaluation.

    uv run python scripts/lock_images.py swebench_pro|terminal_bench_2

Writes data/manifests/images_acr_<bench>.json with, per task: source reference,
destination reference, and both digests (via `skopeo inspect`), so reviewers can
verify the ACR copies are byte-identical to the public images / locked builds."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACR = os.environ.get("ACR_REGISTRY", "<ACR_REGISTRY unset>")


def digest(ref: str) -> str | None:
    try:
        out = subprocess.run(["skopeo", "inspect", "--format", "{{.Digest}}", f"docker://{ref}"],
                             capture_output=True, text=True, timeout=120, check=True).stdout.strip()
        return out or None
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    bench = sys.argv[1]
    if bench == "swebench_pro":
        m = json.loads((ROOT / "data/manifests/swebench_pro_v2_hard.json").read_text())
        pairs = [(f"ghcr.io/scaleapi/swe-bench_pro-v2:{i}", f"{ACR}/swe-bench_pro-v2:{i}", i) for i in m["task_ids"]]
    elif bench == "terminal_bench_2":
        m = json.loads((ROOT / "data/manifests/terminal_bench_2_tasks.json").read_text())
        c8 = m["dataset_git_commit"][0][:8]
        pairs = [(f"build:{m['dataset_git_url'][0]}@{m['dataset_git_commit'][0]}/{t}/environment", f"{ACR}/tb2-{t}:{c8}", t) for t in m["task_ids"]]
    else:
        raise SystemExit("bench must be swebench_pro or terminal_bench_2")

    def one(p):
        src, dst, tid = p
        return {"task_id": tid, "source": src, "acr_ref": dst,
                "source_digest": digest(src) if not src.startswith("build:") else None,
                "acr_digest": digest(dst)}

    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(one, pairs))
    missing = [r["task_id"] for r in rows if not r["acr_digest"]]
    mismatch = [r["task_id"] for r in rows if r["source_digest"] and r["acr_digest"] and r["source_digest"] != r["acr_digest"]]
    out = {"locked_at": datetime.now(timezone.utc).isoformat(), "registry": ACR, "n": len(rows),
           "n_missing_in_acr": len(missing), "missing": missing, "digest_mismatch": mismatch, "images": rows}
    path = ROOT / f"data/manifests/images_acr_{bench}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"{len(rows)} images, {len(missing)} missing in ACR, {len(mismatch)} digest mismatches -> {path.relative_to(ROOT)}")
    return 1 if missing or mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
