#!/usr/bin/env bash
# One-time setup on the machine that drives the evaluation (GPU box or laptop with good network).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f eval/.env ] && set -a && . eval/.env && set +a
M=data/manifests
uv tool install --force 'harbor[modal]==0.24.0' --with 'e2b==2.45.0' >/dev/null && harbor --version
modal profile current >/dev/null || { echo "run: modal token new"; exit 1; }
mkdir -p third_party
# SWE-bench Pro locked tooling at the locked commit
SBP=$(python3 -c "import json;print(json.load(open('$M/swebench_pro_v2_hard.json'))['harness_commit'])")
[ -d third_party/SWE-bench_Pro-os/.git ] || git clone -q https://github.com/scaleapi/SWE-bench_Pro-os third_party/SWE-bench_Pro-os
git -C third_party/SWE-bench_Pro-os fetch -q origin "$SBP" && git -C third_party/SWE-bench_Pro-os checkout -q "$SBP"
echo "SWE-bench_Pro-os @ $(git -C third_party/SWE-bench_Pro-os rev-parse --short HEAD)"
# HLE official scripts at the locked commit
HLE=$(python3 -c "import json;print(json.load(open('$M/hle_text_only_ids.json'))['harness_commit'])")
[ -d third_party/hle/.git ] || git clone -q https://github.com/centerforaisafety/hle third_party/hle
git -C third_party/hle fetch -q origin "$HLE" && git -C third_party/hle checkout -q "$HLE"
echo "hle @ $(git -C third_party/hle rev-parse --short HEAD)"
# ASI-Bench CLI (pinned) + repo for local scoring
uv tool install --force 'asibench==0.2.0' >/dev/null && asibench --version || true
ASI=$(python3 -c "import json;print(json.load(open('$M/asi_bench_tasks.json'))['harness_commit'])")
[ -d third_party/ASI-Bench/.git ] || git clone -q https://github.com/apexin-ai/ASI-Bench third_party/ASI-Bench
git -C third_party/ASI-Bench fetch -q origin "$ASI" && git -C third_party/ASI-Bench checkout -q "$ASI"
echo "ASI-Bench @ $(git -C third_party/ASI-Bench rev-parse --short HEAD)"
# Pin tool versions into the lock config
python3 - <<'PY'
import json, re, subprocess, pathlib
def ver(cmd):
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip().split()[-1]
    except Exception: return "unknown"
pins = {"harbor_version": ver(["harbor","--version"]), "asibench_version": ver(["asibench","--version"]),
        "mini_swe_agent_version": "installed in-sandbox by LockedMiniSwe (see v2/tooling/locked_mini_swe.py pin)",
        "terminus_version": "bundled with harbor " + ver(["harbor","--version"])}
for f in ["swebench_pro_v2_hard.yaml","terminal_bench_2.yaml","asi_bench.yaml"]:
    p = pathlib.Path("configs/eval")/f; s = p.read_text()
    for k, v in pins.items():
        s = re.sub(rf"^(\s*{k}:\s*)LOCK\b.*$", lambda m: f"{m.group(1)}{v}", s, flags=re.M)
    p.write_text(s)
print("pinned:", pins)
PY
echo "setup done"
