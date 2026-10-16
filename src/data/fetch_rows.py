#!/usr/bin/env python
"""Fetch a bandwidth-bounded *sample* of each HF source via the datasets-server
rows API (https://datasets-server.huggingface.co/rows), not via full downloads.

Why: this machine sees ~10-15 KB/s to huggingface.co with frequent connection
resets, so the shard files (0.3-28 GB per source) are out of reach. The rows API
returns complete rows (no cell truncation was observed on any source we use),
accepts any offset, and gzips the JSON, so we can take pages spread evenly over
the whole split instead of the first N rows.

Each page is written atomically to data/raw/<source>/rows_<offset>.jsonl and
skipped when present, so the fetcher is resumable and can run in the background
while the builder reads whatever has arrived so far.

    uv run python src/data/fetch_rows.py            # all experts, priority order
    uv run python src/data/fetch_rows.py e3_reasoning --workers 2
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
API = "https://datasets-server.huggingface.co/rows"
RAW = ROOT / "data" / "raw"


def _get(url: str, token: str | None, timeout: int = 180) -> dict:
    headers = {"Accept-Encoding": "gzip", "User-Agent": "qwen36-merge-takehome/fetch_rows"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
    return json.loads(data)


def fetch_page(repo: str, config: str, split: str, offset: int, length: int, token: str | None) -> dict:
    q = urllib.parse.urlencode({"dataset": repo, "config": config, "split": split, "offset": offset, "length": length})
    delay = 10
    for attempt in range(12):
        try:
            return _get(f"{API}?{q}", token)
        except Exception as e:  # flaky link: resets, SSL EOF, IncompleteRead, 502s — all retried with backoff
            code = getattr(e, "code", None)
            if code in (400, 404, 422):  # bad offset / unknown config: never retried
                raise
            print(f"  [retry {attempt}] {repo}@{offset}: {e!r}", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 120)
    raise RuntimeError(f"gave up on {repo}@{offset}")


def page_path(src: dict, offset: int) -> Path:
    return RAW / src["name"] / f"rows_{offset:09d}.jsonl"


def plan_pages(src: dict, n_total: int) -> list:
    """(offset, length) pages: a seeded shuffle of all page-aligned offsets, taking
    the first n_pages. Raising n_rows later only *extends* the set (prefix
    property), so pages already on disk stay valid and are skipped."""
    import random

    f = src["fetch"]
    n_rows, page = int(f["n_rows"]), int(f.get("page", 100))
    n_rows = min(n_rows, n_total)
    n_pages = max(1, -(-n_rows // page))
    starts = list(range(0, max(1, n_total - page + 1), page))
    random.Random(int(f.get("seed", 0))).shuffle(starts)
    if n_total % page and n_total > page:  # trailing partial page last, so the shuffled prefix is stable
        starts.append(n_total - n_total % page)
    return [(off, min(page, n_total - off)) for off in starts[:n_pages]]


def convert_local_parquet(src: dict) -> dict:
    """Small sources whose rows API is broken are downloaded whole (one parquet,
    a few MB) and converted to the same page files, so the builder sees one input shape."""
    import pyarrow.parquet as pq

    f = src["fetch"]
    path = ROOT / f["local_parquet"]
    out_dir = RAW / src["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    page = int(f.get("page", 100))
    table = pq.read_table(path)
    rows = table.to_pylist()
    n = 0
    for off in range(0, len(rows), page):
        pth = page_path(src, off)
        if pth.exists():
            n += page
            continue
        tmp = pth.with_suffix(".tmp")
        with tmp.open("w") as fh:
            for r in rows[off:off + page]:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
                n += 1
        tmp.rename(pth)
    (out_dir / "_fetch_meta.json").write_text(json.dumps({"hf_repo": src["hf_repo"], "local_parquet": f["local_parquet"],
                                                           "num_rows_total": len(rows), "bytes": path.stat().st_size,
                                                           "features": [str(x) for x in table.schema]}, indent=1))
    return {"rows": n, "bytes": path.stat().st_size, "truncated_rows": 0, "num_rows_total": len(rows)}


def fetch_source(src: dict, token: str | None, workers: int) -> dict:
    f = src["fetch"]
    if f.get("local_parquet") and (ROOT / f["local_parquet"]).exists():
        return convert_local_parquet(src)
    repo, config, split = src["hf_repo"], f.get("config", "default"), f.get("split", src.get("split", "train"))
    out_dir = RAW / src["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "_fetch_meta.json"
    if meta_path.exists():
        n_total = json.loads(meta_path.read_text())["num_rows_total"]
    else:
        first = fetch_page(repo, config, split, 0, 1, token)
        n_total = first["num_rows_total"]
        meta_path.write_text(json.dumps({"hf_repo": repo, "config": config, "split": split, "num_rows_total": n_total,
                                         "features": first["features"], "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    pages = [(o, l) for o, l in plan_pages(src, n_total) if not page_path(src, o).exists()]
    print(f"{src['name']}: total={n_total} pages_todo={len(pages)} page={f.get('page', 100)}", flush=True)

    def one(ol):
        off, length = ol
        t = time.time()
        d = fetch_page(repo, config, split, off, length, token)
        rows = [r["row"] for r in d["rows"]]
        trunc = sum(1 for r in d["rows"] if r.get("truncated_cells"))
        tmp = page_path(src, off).with_suffix(".tmp")
        with tmp.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.rename(page_path(src, off))
        size = page_path(src, off).stat().st_size
        print(f"  {src['name']} @{off} rows={len(rows)} trunc={trunc} {size/1e6:.2f}MB {time.time()-t:.0f}s", flush=True)
        return len(rows), size, trunc

    got = [0, 0, 0]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n, s, t in ex.map(one, pages):
            got[0] += n; got[1] += s; got[2] += t
    return {"rows": got[0], "bytes": got[1], "truncated_rows": got[2], "num_rows_total": n_total}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experts", nargs="*")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/data/sources.yaml").read_text())
    token = os.environ.get("HF_TOKEN")
    todo = []
    for expert, ecfg in cfg.items():
        if args.experts and expert not in args.experts:
            continue
        for src in ecfg["sources"]:
            if src.get("hf_repo", "LOCK") != "LOCK" and src.get("fetch"):
                todo.append((src["fetch"].get("priority", 99), expert, src))
    todo.sort(key=lambda t: t[0])
    for prio, expert, src in todo:
        print(f"== [{prio}] {expert}/{src['name']} {src['hf_repo']}", flush=True)
        try:
            print(f"   done: {fetch_source(src, token, args.workers)}", flush=True)
        except Exception as e:  # keep going: a broken source must not block the others
            print(f"   FAILED {src['name']}: {e!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
