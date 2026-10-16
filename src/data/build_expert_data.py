#!/usr/bin/env python
"""Build one expert's training set from configs/data/sources.yaml.

Pipeline per source: load -> convert -> contamination filter -> exact dedup ->
near dedup -> length cap -> difficulty tier; then weighted mix, train/dev split,
jsonl + manifest. Nothing from the four benchmarks (answers, patches, hidden
tests, question text) is ever read here except as a *blocklist*.

    uv run python src/data/build_expert_data.py e1_code_repair [--limit 200]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.common import (ContaminationFilter, NearDedup, difficulty_tier, exact_key,  # noqa: E402
                             example_text, shingles, split_train_dev, write_jsonl, write_manifest)
from src.data.loaders import LOADERS, fetched_pages, iter_source  # noqa: E402

TIER_CUTS = {"n_turns": (10, 30), "response_tokens": (2000, 8000)}


def load_blocklists(c: dict) -> ContaminationFilter:
    repos, ids, names, sh = set(), set(), set(), set()

    def manifest(key):
        p = c.get(key)
        if p and (ROOT / p).exists():
            return json.loads((ROOT / p).read_text())
        if p:
            print(f"  [warn] blocklist {p} missing — run scripts/lock_benchmarks.py first")
        return {}

    m = manifest("exclude_repos_from")
    repos |= set(m.get("repos", []))
    # SWE-bench Pro ids look like "instance_<owner>__<repo>-<sha>-v<sha>"; derive owner/repo
    # from them too, because the locked `repos` list keeps the "instance_" prefix.
    for tid in m.get("task_ids", []):
        head = tid[len("instance_"):] if tid.startswith("instance_") else tid
        head = head.split("-", 1)[0]
        if "__" in head:
            repos.add(head.replace("__", "/", 1))
    m = manifest("exclude_instance_ids_from")
    ids |= set(m.get("task_ids", []))
    m = manifest("exclude_task_names_from")
    names |= set(m.get("task_ids", []))
    m = manifest("exclude_project_ids_from")
    names |= set(m.get("project_ids", []))
    q = c.get("exclude_ngram_overlap_with")
    if q and (ROOT / q).exists():
        for line in (ROOT / q).open():
            sh |= shingles(json.loads(line)["question"], c.get("ngram", 13))
    elif q:
        print(f"  [warn] {q} missing — n-gram contamination check disabled until built")
    return ContaminationFilter(repos, ids, names, c.get("canary"), sh, c.get("ngram", 13))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("expert")
    ap.add_argument("--limit", type=int, default=None, help="rows per source (smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chars", type=int, default=120_000)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs/data/sources.yaml").read_text())[args.expert]
    filt = load_blocklists(cfg.get("contamination", {}))
    near = NearDedup()
    seen: set[str] = set()
    tier_field = cfg["difficulty_field"]
    per_source: dict[str, list[dict]] = {}
    stats: dict[str, Counter] = {}

    for src in cfg["sources"]:
        loader = LOADERS[src["loader"]]
        kept, st = [], Counter()
        for row in iter_source(src, args.limit, os.environ.get("HF_TOKEN")):
            st["raw"] += 1
            ex = loader(row)
            if ex is None:
                st["unparsable"] += 1
                continue
            if ex == "unresolved":  # source-level quality/licence rule (see loader docstring)
                st["unresolved_or_excluded"] += 1
                continue
            ex["meta"]["source"] = src["name"]
            if not filt.keep(ex):
                st["contaminated"] += 1
                continue
            k = exact_key(ex)
            if k in seen:
                st["exact_dup"] += 1
                continue
            seen.add(k)
            if near.is_dup(ex):
                st["near_dup"] += 1
                continue
            if len(example_text(ex)) > args.max_chars:
                st["too_long"] += 1
                continue
            if tier_field == "response_tokens":
                ex["meta"]["response_tokens"] = len(ex["messages"][-1]["content"]) // 4
            ex["meta"]["tier"] = difficulty_tier(float(ex["meta"].get(tier_field, 0)), TIER_CUTS[tier_field])
            kept.append(ex)
            st["kept"] += 1
        per_source[src["name"]] = kept
        stats[src["name"]] = st
        print(f"{src['name']}: {dict(st)}")

    # weighted mix, capped at max_examples; shortfall in one source is not back-filled
    rng = random.Random(args.seed)
    total = cfg["max_examples"]
    mixed = []
    for src in cfg["sources"]:
        pool = per_source[src["name"]]
        rng.shuffle(pool)
        take = min(len(pool), int(round(total * src["weight"])))
        mixed += pool[:take]
    rng.shuffle(mixed)
    train, dev = split_train_dev(mixed, min(cfg["dev_size"], len(mixed) // 10), args.seed)

    out_dir = ROOT / "data/processed"
    n_tr = write_jsonl(out_dir / f"{args.expert}_train.jsonl", train)
    n_dev = write_jsonl(out_dir / f"{args.expert}_dev.jsonl", dev)
    tiers = Counter(e["meta"]["tier"] for e in train)
    manifest = write_manifest(args.expert, {
        "expert": args.expert,
        "sources": [{k: src.get(k) for k in ("name", "hf_repo", "local_file", "split", "license", "url", "weight", "fetch")}
                    | {"loader": src["loader"], "loader_rule": (LOADERS[src["loader"]].__doc__ or "").strip(),
                       "fetched_pages": fetched_pages(src) if src.get("hf_repo") else [],
                       "status": ("skipped: file missing" if src.get("local_file") and not (ROOT / src["local_file"]).exists()
                                  else "skipped: hf_repo not locked" if src.get("hf_repo", "x") == "LOCK"
                                  else "skipped: no rows fetched" if src.get("hf_repo") and not fetched_pages(src)
                                  else "used"),
                       "stats": dict(stats[src["name"]])} for src in cfg["sources"]],
        "ingestion": "stratified row sample via datasets-server rows API (src/data/fetch_rows.py); shard files were not downloadable at this site's bandwidth",
        "contamination_blocklist": {"repos": len(filt.repos), "instance_ids": len(filt.ids), "task_names": len(filt.task_names),
                                    "benchmark_shingles": len(filt.bench_shingles), "canary": bool(filt.canary)},
        "contamination_drops_by_reason": dict(filt.reasons),
        "n_train": n_tr, "n_dev": n_dev, "difficulty_tiers_train": dict(tiers),
        "dedup": "on assistant-turn text only (dedup_text): sha1 exact on normalised text, then 64-perm minhash LSH (8 bands x 8 rows, 5-gram shingles, threshold ~0.77 Jaccard)",
        "seed": args.seed, "limit": args.limit,
    })
    print(f"train={n_tr} dev={n_dev} tiers={dict(tiers)} manifest={manifest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
