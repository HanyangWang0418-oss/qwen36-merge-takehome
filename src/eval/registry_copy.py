#!/usr/bin/env python
"""Pure-Python OCI/Docker image copy between registries (no docker, no skopeo).

Copies one tag from a source registry (anonymous or basic auth, token auth handled)
to a destination registry (basic auth), streaming each blob; existing blobs are
skipped (HEAD), so re-runs are cheap. For a multi-platform index the linux/amd64
manifest is selected and pushed as a single-platform tag (what the sandbox
builder needs). Prints the pushed manifest digest.

    python registry_copy.py SRC_REF DST_REF [--src-user U --src-pass P] [--dst-user U --dst-pass P] [--chunk-mb 16]

Only stdlib + requests. Intended to run inside an Agent Sandbox (which reaches both ghcr.io and ACR).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time

import requests

MT_INDEX = ("application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json")
MT_MANIFEST = ("application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json")
ACCEPT = ", ".join(MT_INDEX + MT_MANIFEST)


def parse_ref(ref: str) -> tuple[str, str, str]:
    ref = ref.removeprefix("docker://")
    host, _, rest = ref.partition("/")
    if "@" in rest:
        repo, tag = rest.split("@", 1)
    elif ":" in rest.rsplit("/", 1)[-1]:
        repo, tag = rest.rsplit(":", 1)
    else:
        repo, tag = rest, "latest"
    return host, repo, tag


class Registry:
    def __init__(self, host: str, user: str | None, password: str | None):
        self.base = f"https://{host}"
        self.auth = (user, password) if user else None
        self.s = requests.Session()
        self.token: str | None = None

    def _hdr(self, extra: dict | None = None) -> dict:
        h = dict(extra or {})
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _req(self, method: str, path: str, **kw) -> requests.Response:
        url = path if path.startswith("http") else self.base + path
        headers = kw.pop("headers", None)
        timeout = kw.pop("timeout", 300)
        r = self.s.request(method, url, headers=self._hdr(headers), auth=None if self.token else self.auth, timeout=timeout, **kw)
        if r.status_code == 401 and "www-authenticate" in r.headers:
            self._token_from_challenge(r.headers["www-authenticate"])
            r = self.s.request(method, url, headers=self._hdr(headers), timeout=timeout, **kw)
        return r

    def _token_from_challenge(self, challenge: str) -> None:
        params = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
        realm = params.pop("realm")
        r = self.s.get(realm, params=params, auth=self.auth, timeout=60)
        r.raise_for_status()
        self.token = r.json().get("token") or r.json().get("access_token")

    def manifest(self, repo: str, ref: str) -> tuple[bytes, str]:
        r = self._req("GET", f"/v2/{repo}/manifests/{ref}", headers={"Accept": ACCEPT})
        r.raise_for_status()
        return r.content, r.headers.get("Content-Type", "")

    def blob_exists(self, repo: str, digest: str) -> bool:
        return self._req("HEAD", f"/v2/{repo}/blobs/{digest}").status_code == 200

    def open_blob(self, repo: str, digest: str) -> requests.Response:
        r = self._req("GET", f"/v2/{repo}/blobs/{digest}", stream=True)
        r.raise_for_status()
        return r

    def push_blob_stream(self, repo: str, digest: str, stream, size: int | None, chunk: int) -> None:
        r = self._req("POST", f"/v2/{repo}/blobs/uploads/")
        if r.status_code not in (202,):
            raise RuntimeError(f"upload init failed {r.status_code}: {r.text[:200]}")
        loc = r.headers["Location"]
        if not loc.startswith("http"):
            loc = self.base + loc
        # monolithic PUT with streamed body (registries accept chunked transfer)
        h = {"Content-Type": "application/octet-stream"}
        if size is not None:
            h["Content-Length"] = str(size)
        sep = "&" if "?" in loc else "?"
        r = self.s.put(f"{loc}{sep}digest={digest}", data=stream, headers=self._hdr(h), timeout=3600)
        if r.status_code not in (201, 204):
            raise RuntimeError(f"blob put failed {r.status_code}: {r.text[:200]}")

    def push_manifest(self, repo: str, tag: str, body: bytes, media_type: str) -> str:
        r = self._req("PUT", f"/v2/{repo}/manifests/{tag}", data=body, headers={"Content-Type": media_type})
        if r.status_code not in (201, 202):
            raise RuntimeError(f"manifest put failed {r.status_code}: {r.text[:300]}")
        return r.headers.get("Docker-Content-Digest", "sha256:" + hashlib.sha256(body).hexdigest())


def download_with_resume(src: "Registry", repo: str, digest: str, size: int | None, chunk_mb: int, retries: int) -> str:
    """Stream a blob to a temp file, resuming with HTTP Range on stalls; verifies the sha256."""
    fd, path = tempfile.mkstemp(prefix="blob-", dir=os.environ.get("BLOB_TMP", None))
    os.close(fd)
    got, attempt, h = 0, 0, hashlib.sha256()
    while True:
        try:
            hdr = {"Range": f"bytes={got}-"} if got else {}
            r = src._req("GET", f"/v2/{repo}/blobs/{digest}", headers=hdr, stream=True, timeout=(30, 60))
            if r.status_code not in (200, 206):
                raise RuntimeError(f"blob get {r.status_code}")
            if r.status_code == 200 and got:  # server ignored Range: restart
                got, h = 0, hashlib.sha256()
                open(path, "wb").close()
            with open(path, "ab") as f:
                for part in r.iter_content(chunk_mb * 1024 * 1024):
                    f.write(part); h.update(part); got += len(part)
            if size is not None and got < size:
                raise RuntimeError(f"short read {got}/{size}")
            break
        except Exception as e:  # noqa: BLE001
            attempt += 1
            if attempt > retries:
                raise
            print(f"    resume {attempt}/{retries} at {got / 1e6:.0f} MB ({type(e).__name__})", flush=True)
            time.sleep(min(30, 2 * attempt))
    if h.hexdigest() != digest.split(":", 1)[1]:
        raise RuntimeError(f"digest mismatch for {digest}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--src-user"); ap.add_argument("--src-pass"); ap.add_argument("--dst-user"); ap.add_argument("--dst-pass")
    ap.add_argument("--chunk-mb", type=int, default=16)
    ap.add_argument("--retries", type=int, default=20, help="per-blob resume attempts")
    a = ap.parse_args()
    sh, srepo, sref = parse_ref(a.src)
    dh, drepo, dtag = parse_ref(a.dst)
    src, dst = Registry(sh, a.src_user, a.src_pass), Registry(dh, a.dst_user, a.dst_pass)

    body, mt = src.manifest(srepo, sref)
    m = json.loads(body)
    if mt in MT_INDEX or "manifests" in m:
        cands = [x for x in m["manifests"] if x.get("platform", {}).get("os") == "linux" and x.get("platform", {}).get("architecture") == "amd64"]
        if not cands:
            raise SystemExit("no linux/amd64 manifest in index")
        body, mt = src.manifest(srepo, cands[0]["digest"])
        m = json.loads(body)
    layers = m.get("layers", []) + [m["config"]]
    total = sum(l.get("size", 0) for l in layers)
    print(f"{a.src} -> {a.dst}: {len(layers)} blobs, {total / 1e6:.0f} MB", flush=True)
    t0 = time.time(); copied = 0
    for i, l in enumerate(layers):
        d = l["digest"]
        if dst.blob_exists(drepo, d):
            print(f"  [{i + 1}/{len(layers)}] exists {d[:19]}", flush=True); continue
        t1 = time.time()
        # Download to a temp file with Range-resume (ghcr's CDN stalls from some regions), then upload.
        tmp = download_with_resume(src, srepo, d, l.get("size"), a.chunk_mb, a.retries)
        try:
            with open(tmp, "rb") as fh:
                dst.push_blob_stream(drepo, d, fh, l.get("size"), a.chunk_mb)
        finally:
            os.unlink(tmp)
        copied += l.get("size", 0)
        dt = time.time() - t1
        print(f"  [{i + 1}/{len(layers)}] copied {d[:19]} {l.get('size', 0) / 1e6:.0f} MB in {dt:.0f}s ({l.get('size', 0) / 1e6 / max(dt, 1e-6):.1f} MB/s)", flush=True)
    digest = dst.push_manifest(drepo, dtag, body, mt or MT_MANIFEST[0])
    print(f"done: {copied / 1e6:.0f} MB copied in {time.time() - t0:.0f}s; manifest {digest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
