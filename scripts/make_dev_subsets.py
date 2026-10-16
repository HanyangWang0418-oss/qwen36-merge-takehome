#!/usr/bin/env python
"""Fixed, seeded dev subsets used ONLY to screen merge ablations (never to select
checkpoints or to report headline numbers). Sizes are chosen so one screening pass
of all merge configs costs a few GPU-hours.

    uv run python scripts/make_dev_subsets.py   # writes configs/eval/dev_subsets.json
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "data/manifests"
rng = random.Random(20261017)
sw = json.loads((M / "swebench_pro_v2_hard.json").read_text())["task_ids"]
tb = json.loads((M / "terminal_bench_2_tasks.json").read_text())["task_ids"]
hle = json.loads((M / "hle_text_only_ids.json").read_text())["question_ids"]
asi = json.loads((M / "asi_bench_tasks.json").read_text())["instance_ids"]
sub = {
    "seed": 20261017,
    "purpose": "merge-ablation screening only; full splits are used for all reported numbers",
    "swebench_pro_v2_hard": sorted(rng.sample(sw, 12)),
    "terminal_bench_2": sorted(rng.sample(tb, 20)),
    "hle": sorted(rng.sample(hle, 200)),
    "asi_bench": sorted(rng.sample(asi, 12)),
    "terminal_bench_2_runs": 1,
}
(ROOT / "configs/eval/dev_subsets.json").write_text(json.dumps(sub, indent=2) + "\n")
print({k: (len(v) if isinstance(v, list) else v) for k, v in sub.items()})
