#!/usr/bin/env python
"""Convert Harbor job output(s) to results/<group>/<model>/<bench>/per_task.jsonl.

Harbor writes jobs/<job>/<task>/result.json (schema checked at first real run;
field names below are the ones used by harbor 0.2x: `verifier_result.rewards`,
`exception_info`, `agent_result.n_input_tokens/n_output_tokens`)."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def status_of(r: dict) -> str:
    if r.get("exception_info"):
        kind = str(r["exception_info"].get("exception_type", "")).lower()
        return "timeout" if "timeout" in kind else "env_failure"
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs_glob")
    ap.add_argument("out_dir")
    ap.add_argument("--original", default=None, help="pre-regrade job dir (SWE-bench Pro) for side-by-side numbers")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for run_i, job in enumerate(sorted(glob.glob(a.jobs_glob)), 1):
        for rf in sorted(Path(job).glob("*/result.json")):
            r = json.loads(rf.read_text())
            rewards = (r.get("verifier_result") or {}).get("rewards") or {}
            score = float(rewards.get("reward", 0.0)) if rewards else 0.0
            st = status_of(r)
            ag = r.get("agent_result") or {}
            row = {"task_id": r.get("task_name") or rf.parent.name, "run": run_i, "score": score, "status": st,
                   "job": Path(job).name, "n_input_tokens": ag.get("n_input_tokens"), "n_output_tokens": ag.get("n_output_tokens"),
                   "wallclock_s": r.get("duration_s") or r.get("elapsed_s"), "cost_usd": ag.get("cost_usd", 0.0)}
            rows.append(row)
            if st != "ok" or score == 0.0:
                failures.append(row)
    (out / "per_task.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "failed_tasks.json").write_text(json.dumps(failures, indent=2))
    (out / "env_failures.json").write_text(json.dumps([f for f in failures if f["status"] != "ok"], indent=2))
    if a.original:
        orig = [json.loads(Path(f).read_text()) for f in glob.glob(f"{a.original}/*/result.json")]
        n_ok = sum(1 for r in orig if ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward"))
        (out / "original_sandbox_score.json").write_text(json.dumps({"resolved": n_ok, "n": len(orig), "note": "pre-regrade, non-authoritative"}))
    print(f"{len(rows)} rows, {len(failures)} failed -> {out}")


if __name__ == "__main__":
    main()
