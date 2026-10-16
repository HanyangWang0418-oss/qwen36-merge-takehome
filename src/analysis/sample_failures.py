#!/usr/bin/env python
"""Sample failures of a merged model, stratified by benchmark, and pre-fill the
failures.md table with the expert's and base's result on the same task so that
merge-induced regressions are visible immediately."""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERT_FOR = {"swebench_pro_v2_hard": "e1_code_repair", "terminal_bench_2": "e2_terminal_agent", "hle": "e3_reasoning", "asi_bench": "e2_terminal_agent"}


def scores(group: str, model: str, bench: str) -> dict[str, float]:
    p = ROOT / "results" / group / model / bench / "per_task.jsonl"
    if not p.exists():
        return {}
    out: dict[str, list[float]] = {}
    for l in p.open():
        r = json.loads(l)
        out.setdefault(r["task_id"], []).append(r["score"])
    return {k: sum(v) / len(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="merged_ta_l0.7")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []
    per = a.n // 4
    for bench, expert in EXPERT_FOR.items():
        f = ROOT / "results/merged" / a.model / bench / "failed_tasks.json"
        if not f.exists():
            continue
        fails = json.loads(f.read_text())
        rng.shuffle(fails)
        es, bs = scores("experts", expert, bench), scores("base", "base", bench)
        for r in fails[:per]:
            t = r["task_id"]
            rows.append(f"| | {bench} | {t} | | {es.get(t, 'n/a')} | {bs.get(t, 'n/a')} | results/merged/{a.model}/{bench} | status={r['status']} |")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
