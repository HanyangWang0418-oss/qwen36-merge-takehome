#!/usr/bin/env python
"""Copy values from data/manifests/*.json into the LOCK fields of configs/eval/*.yaml.
Idempotent: only replaces fields whose value is still LOCK. Fields with no manifest
source (tool versions pinned at eval-environment setup) are left for scripts/pin_eval_tools.py."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = lambda n: json.loads((ROOT / "data/manifests" / n).read_text())  # noqa: E731

sw, tb, asi, base = M("swebench_pro_v2_hard.json"), M("terminal_bench_2_tasks.json"), M("asi_bench_tasks.json"), M("base_model.json")
hle = M("hle_text_only_ids.json") if (ROOT / "data/manifests/hle_text_only_ids.json").exists() else {}

FILLS = {
    "swebench_pro_v2_hard.yaml": {
        "hf_revision": sw["dataset_hf_revision"], "commit": sw["harness_commit"],
        "max_steps": "0  # mini_toolcall.yaml step_limit=0: unbounded, time-bounded by per_task_timeout_s",
        "token_budget": "cost_limit 3.0 (litellm USD; 0 for a self-hosted endpoint, so effectively time-bounded only)",
    },
    "terminal_bench_2.yaml": {
        "version": tb["version"], "n_tasks": tb["n_tasks"], "commit": tb["harness_commit"],
    },
    "asi_bench.yaml": {
        "track_used": asi["track"] + "  # pending interviewer confirmation of multi-submission; else switch to seed31415 and re-lock",
        "hf_revision": asi["dataset_hf_revision"], "commit": asi["harness_commit"],
    },
    "base_model.yaml": {"hf_revision": base["hf_revision"], "chat_template_sha256": base["chat_template_sha256"]},
    "hle.yaml": {"hf_revision": hle.get("dataset_hf_revision"), "n_questions": hle.get("n_text_only"), "commit": hle.get("harness_commit")},
}

for fname, fills in FILLS.items():
    p = ROOT / "configs/eval" / fname
    text = p.read_text()
    for key, val in fills.items():
        if val is None:
            continue
        text, n = re.subn(rf"^(\s*{re.escape(key)}:\s*)LOCK\b", lambda m: f"{m.group(1)}{val}", text, count=1, flags=re.M)
        if n:
            print(f"{fname}: {key} <- {str(val)[:60]}")
    p.write_text(text)

left = [(str(p.relative_to(ROOT)), i, l.strip()) for p in sorted((ROOT / "configs/eval").glob("*.yaml"))
        for i, l in enumerate(p.read_text().splitlines(), 1) if re.search(r":\s*LOCK\b", l)]
print(f"\n{len(left)} LOCK fields remain:")
for f, i, l in left:
    print(f"  {f}:{i}: {l}")
