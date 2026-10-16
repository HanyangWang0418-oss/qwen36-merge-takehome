#!/usr/bin/env python
"""One-off probe of Alibaba Cloud Agent Sandbox (E2B-compatible, Hangzhou) template building:
A) from_image on a public ghcr SWE-bench Pro task image at 1 vCPU / 4 GiB -> build time, sandbox boots, /app present
B) spec probe: 32 vCPU / 48 GiB template from a small public image -> accepted or error text
Writes logs/sandbox/template_probe.json. Deletes what it creates."""
import json, os, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
key = [l.split("=", 1)[1].strip() for l in (ROOT / ".env").open() if l.startswith("E2B_API_KEY=")][0]
os.environ.update(E2B_API_KEY=key, E2B_API_URL="https://api.cn-hangzhou.sandbox.aliyuncs.com", E2B_DOMAIN="cn-hangzhou.sandbox.aliyuncs.com")
from e2b import Sandbox, Template  # noqa: E402

out = {"started": time.strftime("%Y-%m-%dT%H:%M:%S")}
ids = json.loads((ROOT / "data/manifests/swebench_pro_v2_hard.json").read_text())["task_ids"]
image = f"ghcr.io/scaleapi/swe-bench_pro-v2:{ids[0]}"

def build(alias, image, cpu, mem):
    t0 = time.time()
    tpl = Template().from_image(image)
    res = Template.build(tpl, alias=alias, cpu_count=cpu, memory_mb=mem)
    return res, round(time.time() - t0, 1)

# A
try:
    res, secs = build("probe-sbp-hard51-a", image, 1, 4096)
    out["A"] = {"image": image, "build_s": secs, "result": str(res)[:300]}
    sbx = Sandbox.create(template="probe-sbp-hard51-a", timeout=300, allow_internet_access=True)
    try:
        r = sbx.commands.run("nproc; free -m | sed -n 2p; ls /app | head -5; ls /testbed 2>/dev/null | head -3; git -C /app rev-parse --short HEAD 2>/dev/null; true", timeout=60)
        out["A"]["sandbox"] = {"id": sbx.sandbox_id, "stdout": r.stdout[:600], "info": str(sbx.get_info())[:400]}
    finally:
        sbx.kill()
except Exception as e:  # noqa: BLE001
    out["A"] = {"image": image, "error": repr(e)[:800], "trace": traceback.format_exc()[-800:]}
(ROOT / "logs/sandbox/template_probe.json").write_text(json.dumps(out, indent=2))

# B
for cpu, mem in ((32, 49152), (16, 32768), (8, 16384)):
    k = f"B_{cpu}c_{mem}m"
    try:
        res, secs = build(f"probe-spec-{cpu}c", "debian:bookworm-slim", cpu, mem)
        out[k] = {"build_s": secs, "result": str(res)[:300]}
        sbx = Sandbox.create(template=f"probe-spec-{cpu}c", timeout=120)
        try:
            r = sbx.commands.run("nproc; free -m | sed -n 2p", timeout=30)
            out[k]["sandbox"] = r.stdout.strip()
        finally:
            sbx.kill()
        break
    except Exception as e:  # noqa: BLE001
        out[k] = {"error": repr(e)[:600]}
    (ROOT / "logs/sandbox/template_probe.json").write_text(json.dumps(out, indent=2))

# cleanup templates (best effort)
try:
    from e2b import Template as T
    for alias in ("probe-sbp-hard51-a", "probe-spec-32c", "probe-spec-16c", "probe-spec-8c"):
        try:
            T.delete(alias) if hasattr(T, "delete") else None
        except Exception as e:  # noqa: BLE001
            out.setdefault("cleanup", []).append(f"{alias}: {repr(e)[:120]}")
except Exception:  # noqa: BLE001
    pass
out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
(ROOT / "logs/sandbox/template_probe.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
