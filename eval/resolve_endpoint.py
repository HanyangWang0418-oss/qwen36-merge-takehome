#!/usr/bin/env python
"""Print shell exports for one model from eval/endpoints.yaml.
    eval "$(python3 eval/resolve_endpoint.py <model> [--modal])"
Sets VLLM_BASE, VLLM_HOST, SERVED. With --modal the public_url is required (Modal cannot reach private IPs)."""
import ipaddress, sys, urllib.parse
from pathlib import Path
import yaml

model = sys.argv[1]; need_public = "--modal" in sys.argv
eps = yaml.safe_load((Path(__file__).parent / "endpoints.yaml").read_text())
if model not in eps:
    sys.exit(0)  # fall back to eval/.env values
e = eps[model]
url = e.get("public_url") or e["base_url"]
host = urllib.parse.urlparse(url).hostname or ""
try:
    private = ipaddress.ip_address(host).is_private
except ValueError:
    private = host in ("localhost",)
if need_public and private:
    sys.stderr.write(f"ERROR: {model} endpoint {url} is a private address; Modal sandboxes cannot reach it. "
                     f"Set public_url in eval/endpoints.yaml (e.g. cloudflared tunnel) or run with HARBOR_ENV=docker.\n")
    sys.exit(2)
print(f"export VLLM_BASE='{url.rstrip('/')}'; export VLLM_HOST='{host}'; export SERVED='{e['served']}'")
