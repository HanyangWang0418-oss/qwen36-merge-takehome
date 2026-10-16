#!/usr/bin/env python
"""Judged HLE json -> per_task.jsonl (score 1 if judge said yes)."""
import json
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
data = json.loads(src.read_text())
rows = []
for qid, r in data.items():
    correct = str(r.get("judge_response", {}).get("correct", r.get("correct", "no"))).lower() == "yes"
    st = "ok" if r.get("response") else "model_failure"
    rows.append({"task_id": qid, "run": 1, "score": 1.0 if correct else 0.0, "status": st,
                 "confidence": r.get("judge_response", {}).get("confidence"), "n_output_tokens": r.get("usage", {}).get("completion_tokens")})
(out / "per_task.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
(out / "failed_tasks.json").write_text(json.dumps([r for r in rows if r["score"] == 0.0], indent=2))
(out / "env_failures.json").write_text(json.dumps([r for r in rows if r["status"] != "ok"], indent=2))
print(f"{len(rows)} questions, acc={sum(r['score'] for r in rows)/max(len(rows),1):.4f} -> {out}")
