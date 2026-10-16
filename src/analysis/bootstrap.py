"""Paired bootstrap over shared task ids.

Per-task result files are results/<model>/<benchmark>/per_task.jsonl with one
row per task (and per run for Terminal-Bench):
    {"task_id": str, "run": int, "score": float (0/1 or 0-100), "status": "ok|env_failure|timeout|model_failure", ...}
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_per_task(path: Path) -> dict[str, float]:
    """task_id -> mean score over runs. Failed tasks keep score 0 (never dropped)."""
    acc: dict[str, list[float]] = defaultdict(list)
    for line in path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        acc[str(r["task_id"])].append(float(r.get("score", 0.0)))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def bootstrap_mean(scores: np.ndarray, n: int = 10000, seed: int = 0, ci: float = 0.95) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(scores), size=(n, len(scores)))
    means = scores[idx].mean(1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"mean": float(scores.mean()), "ci_low": float(lo), "ci_high": float(hi), "n_tasks": int(len(scores)), "n_boot": n}


def paired_bootstrap_delta(a: dict[str, float], b: dict[str, float], n: int = 10000, seed: int = 0, ci: float = 0.95) -> dict:
    """Delta = b - a over the intersection of task ids (same tasks, resampled together)."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"error": "no shared task ids"}
    da = np.array([a[i] for i in ids])
    db = np.array([b[i] for i in ids])
    diff = db - da
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(n, len(ids)))
    deltas = diff[idx].mean(1)
    lo, hi = np.quantile(deltas, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {
        "delta_mean": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
        "p_delta_le_0": float((deltas <= 0).mean()), "n_tasks": len(ids), "n_boot": n,
        "n_wins": int((diff > 0).sum()), "n_losses": int((diff < 0).sum()), "n_ties": int((diff == 0).sum()),
    }
