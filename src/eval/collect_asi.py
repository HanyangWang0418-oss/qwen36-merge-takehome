#!/usr/bin/env python
"""asibench results dir -> per_task.jsonl with task_id=<project>/<level>, score=final_score (0-100)."""
import json
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
rows = []
for f in sorted(src.rglob("*.json")):
    try:
        r = json.loads(f.read_text())
    except Exception:  # noqa: BLE001
        continue
    if "final_score" not in r:
        continue
    level = r.get("level") or r.get("guidance_level") or "?"
    invalid = r.get("status") == "NOT SCORED" or r.get("valid") is False
    rows.append({"task_id": f"{r.get('project_id', f.stem)}/{level}", "level": level, "run": 1,
                 "score": 0.0 if invalid else float(r["final_score"]), "status": "env_failure" if invalid else "ok",
                 "wallclock_s": r.get("duration_s"), "cost_usd": r.get("cost_usd", 0.0)})
(out / "per_task.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
# primary metric (B3) is computed by scripts/summarize.py via the level field; write per-level means here too
levels = {}
for r in rows:
    levels.setdefault(r["level"], []).append(r["score"])
(out / "per_level.json").write_text(json.dumps({k: sum(v) / len(v) for k, v in levels.items()}, indent=2))
(out / "env_failures.json").write_text(json.dumps([r for r in rows if r["status"] != "ok"], indent=2))
print(f"{len(rows)} evaluations -> {out}")
