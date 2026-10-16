"""Shared data utilities: normalisation, exact + near dedup, n-gram
contamination checks, difficulty tiers, train/dev split and manifest writing."""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[2]

_ws = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _ws.sub(" ", text.lower()).strip()


def example_text(ex: dict) -> str:
    return "\n".join(m["content"] for m in ex["messages"] if isinstance(m.get("content"), str))


def exact_key(ex: dict) -> str:
    return hashlib.sha1(normalize(example_text(ex)).encode()).hexdigest()


def shingles(text: str, n: int = 13) -> set[int]:
    toks = normalize(text).split()
    return {hash(" ".join(toks[i : i + n])) for i in range(max(0, len(toks) - n + 1))}


def minhash_sig(sh: set[int], k: int = 64, seed: int = 0) -> tuple[int, ...]:
    if not sh:
        return tuple([0] * k)
    rng = random.Random(seed)
    salts = [rng.getrandbits(61) for _ in range(k)]
    return tuple(min((s ^ h) & 0x1FFFFFFFFFFFFFFF for h in sh) for s in salts)


class NearDedup:
    """Band-LSH over 64-perm minhash; near-duplicates (est. Jaccard >= ~0.8) dropped."""

    def __init__(self, bands: int = 16, rows: int = 4):
        self.bands, self.rows = bands, rows
        self.seen: set[tuple[int, int]] = set()

    def is_dup(self, ex: dict) -> bool:
        sig = minhash_sig(shingles(example_text(ex), n=5))
        keys = [(b, hash(sig[b * self.rows : (b + 1) * self.rows])) for b in range(self.bands)]
        dup = any(k in self.seen for k in keys)
        self.seen.update(keys)
        return dup


class ContaminationFilter:
    """Drop examples that touch a benchmark: by repo name, instance/task id,
    canary string, or 13-gram overlap with benchmark prompt text."""

    def __init__(self, repos: set[str] = frozenset(), ids: set[str] = frozenset(),
                 task_names: set[str] = frozenset(), canary: str | None = None,
                 benchmark_shingles: set[int] | None = None, ngram: int = 13):
        self.repos = {r.lower() for r in repos}
        self.ids = {i.lower() for i in ids}
        self.task_names = {t.lower() for t in task_names}
        self.canary = canary
        self.bench_shingles = benchmark_shingles or set()
        self.ngram = ngram
        self.reasons: Counter = Counter()

    def reason(self, ex: dict) -> str | None:
        meta = ex.get("meta", {})
        repo = str(meta.get("repo", "")).lower()
        if repo and any(repo == r or repo.endswith("/" + r.split("/")[-1]) for r in self.repos):
            return "repo"
        if str(meta.get("instance_id", "")).lower() in self.ids:
            return "instance_id"
        tn = str(meta.get("task_name", "")).lower()
        # TB task variants are named "<task>.easy" / "<task>.hard"; the base name is what the TB2 list holds
        if tn and (tn in self.task_names or tn.split(".")[0] in self.task_names):
            return "task_name"
        text = example_text(ex)
        low = text.lower()
        if any(t in low for t in self.task_names if len(t) > 6):
            return "task_name"
        if self.canary and self.canary.lower() in low:
            return "canary"
        if self.bench_shingles and shingles(text, self.ngram) & self.bench_shingles:
            return "ngram_overlap"
        return None

    def keep(self, ex: dict) -> bool:
        r = self.reason(ex)
        if r:
            self.reasons[r] += 1
            return False
        return True


def difficulty_tier(value: float, cuts: tuple[float, float]) -> str:
    return "easy" if value < cuts[0] else "medium" if value < cuts[1] else "hard"


def split_train_dev(rows: list[dict], dev_size: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    dev = [rows[i] for i in idx[:dev_size]]
    train = [rows[i] for i in idx[dev_size:]]
    return train, dev


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path, limit: int | None = None) -> Iterator[dict]:
    with path.open() as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if line.strip():
                yield json.loads(line)


def write_manifest(name: str, payload: dict) -> Path:
    path = ROOT / "data" / "manifests" / f"{name}_manifest.json"
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), **payload}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path
