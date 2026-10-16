#!/usr/bin/env python
"""Snapshot the *question text* of HLE's text-only items (ids locked in
data/manifests/hle_text_only_ids.json) to data/raw/hle/hle_text_only_questions.jsonl,
which the e3 builder uses only as a 13-gram contamination blocklist.

Answers, rationales and images are deliberately not written. The output lives
under data/raw/ (gitignored) because cais/hle is gated and must not be redistributed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data"))
from fetch_rows import fetch_page  # noqa: E402


def main() -> int:
    lock = json.loads((ROOT / "data/manifests/hle_text_only_ids.json").read_text())
    ids = set(lock["question_ids"])
    out = ROOT / "data/raw/hle/hle_text_only_questions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    page, total, n = 100, lock["n_total"], 0
    with out.open("w") as f:
        for off in range(0, total, page):
            d = fetch_page(lock["dataset_hf_repo"], "default", lock["split"], off, min(page, total - off), token)
            for r in d["rows"]:
                row = r["row"]
                if row["id"] in ids and not row.get("image"):
                    f.write(json.dumps({"id": row["id"], "question": row["question"]}, ensure_ascii=False) + "\n")
                    n += 1
            print(f"  @{off}: {n} questions so far", flush=True)
    print(f"wrote {n} / {len(ids)} text-only questions to {out.relative_to(ROOT)}")
    return 0 if n == len(ids) else 1


if __name__ == "__main__":
    sys.exit(main())
