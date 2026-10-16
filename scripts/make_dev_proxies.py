#!/usr/bin/env python
"""Independent development proxies for tuning merge recipes (configs/eval/dev_proxies.json).

The task statement forbids using the four benchmarks' items to tune or select models, so merge
ablations are screened on *different* datasets of the same task form:
  HLE            -> MMLU-Pro (TIGER-Lab/MMLU-Pro, test), 300 STEM questions, 10-way MC, exact-match letter;
                    13-gram overlap with the locked HLE text-only questions removed
  SWE-bench Pro  -> Harbor `swebench-verified@1.0`, 15 tasks (Python repos; different benchmark)
  Terminal-Bench -> Harbor `terminal-bench-pro@1.0`, 20 tasks (Alibaba's TB-Pro; same harness, disjoint tasks)
  ASI-Bench      -> `Apexintelligence-AI/ASI-Bench-seed31415`, 12 instances (public references, local scoring)
Seeded sampling; the proxies are never reported as benchmark numbers.

    uv run python scripts/make_dev_proxies.py
"""
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.common import shingles  # noqa: E402

SEED = 20261017
rng = random.Random(SEED)
SP = Path("/private/tmp/claude-501/-Users-hanyang-Desktop-Apex-interview/d014b14b-ebea-42e8-9593-75199bd4d5c7/scratchpad")
reg_path = next((p for p in [ROOT / "data/raw/harbor_registry.json", SP / "harbor_registry.json"] if p.exists()), None)
reg = json.loads(reg_path.read_text()) if reg_path else []


def registry_tasks(name: str, version: str) -> list[str]:
    for e in reg:
        if e.get("name") == name and str(e.get("version")) == version:
            return sorted(t["name"] if isinstance(t, dict) else t for t in e["tasks"])
    raise SystemExit(f"{name}@{version} not in harbor registry")


# HLE proxy: MMLU-Pro STEM
df = pd.read_parquet(next((ROOT / "data/raw/mmlu_pro").rglob("*test*.parquet")))
stem = df[df["category"].isin(["math", "physics", "chemistry", "biology", "computer science", "engineering", "health"])]
hle_sh = set()
for line in (ROOT / "data/raw/hle/hle_text_only_questions.jsonl").open():
    hle_sh |= shingles(json.loads(line)["question"], 13)
rows = stem.sample(frac=1.0, random_state=SEED)
picked, dropped = [], 0
per_cat = {}
for _, r in rows.iterrows():
    if per_cat.get(r["category"], 0) >= 45:
        continue
    if shingles(r["question"], 13) & hle_sh:
        dropped += 1
        continue
    picked.append({"id": int(r["question_id"]), "category": r["category"], "question": r["question"],
                   "options": list(r["options"]), "answer": r["answer"], "answer_index": int(r["answer_index"])})
    per_cat[r["category"]] = per_cat.get(r["category"], 0) + 1
    if len(picked) >= 300:
        break
(ROOT / "configs/eval").mkdir(exist_ok=True)
(ROOT / "configs/eval/dev_proxy_hle_mmlu_pro.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in picked))

out = {
    "seed": SEED,
    "purpose": "merge-ablation screening only; independent of the four evaluation benchmarks",
    "hle_proxy": {"dataset": "TIGER-Lab/MMLU-Pro test (STEM categories, <=45 per category)", "n": len(picked),
                  "file": "configs/eval/dev_proxy_hle_mmlu_pro.jsonl", "metric": "exact-match answer letter",
                  "hle_13gram_overlap_dropped": dropped, "per_category": per_cat},
    "swebench_proxy": {"dataset": "harbor swebench-verified@1.0", "tasks": sorted(rng.sample(registry_tasks("swebench-verified", "1.0"), 15)), "metric": "resolved rate"},
    "terminal_bench_proxy": {"dataset": "harbor terminal-bench-pro@1.0", "tasks": sorted(rng.sample(registry_tasks("terminal-bench-pro", "1.0"), 20)), "runs": 1, "metric": "success rate"},
    "asi_proxy": {"dataset": "Apexintelligence-AI/ASI-Bench-seed31415", "n_instances": 12, "selection": "first 12 instance ids in sorted order (seed31415 is the public track)", "metric": "B3 mean final_score (local scoring)"},
}
(ROOT / "configs/eval/dev_proxies.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: (vv if not isinstance(vv, list) else len(vv)) for kk, vv in v.items()}) for k, v in out.items()}, indent=1, ensure_ascii=False))
