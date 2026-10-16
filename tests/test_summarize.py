import json
import numpy as np
from src.analysis.bootstrap import bootstrap_mean, paired_bootstrap_delta


def test_paired_delta_sign():
    a = {f"t{i}": float(i % 2) for i in range(60)}
    b = {k: min(1.0, v + (0.0 if i % 3 else 1.0)) for i, (k, v) in enumerate(a.items())}
    d = paired_bootstrap_delta(a, b, n=2000)
    assert d["delta_mean"] > 0 and d["ci_low"] > 0
    assert d["n_tasks"] == 60


def test_bootstrap_mean_ci_contains_mean():
    s = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 1], dtype=float)
    r = bootstrap_mean(s, n=2000)
    assert r["ci_low"] <= r["mean"] <= r["ci_high"]
