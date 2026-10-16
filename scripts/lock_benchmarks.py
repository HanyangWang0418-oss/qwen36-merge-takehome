#!/usr/bin/env python
"""Snapshot the four benchmarks' task lists and the base model revision.

Writes data/manifests/*.json and prints which LOCK fields in configs/eval/*.yaml
still need a value. Run once before training, commit the output, never re-run
to a different value afterwards.

Usage:
    uv run python scripts/lock_benchmarks.py [--skip-hle] [--skip-asi]

HLE is gated: set HF_TOKEN and accept the terms on huggingface.co/datasets/cais/hle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
MANIFESTS.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(name: str, payload: dict) -> None:
    path = MANIFESTS / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


def github_head_sha(repo: str, branch: str = "main") -> str:
    r = requests.get(f"https://api.github.com/repos/{repo}/commits/{branch}", timeout=30)
    r.raise_for_status()
    return r.json()["sha"]


def hf_revision(repo_id: str, repo_type: str = "model") -> str:
    from huggingface_hub import HfApi

    info = HfApi().repo_info(repo_id, repo_type=repo_type, token=os.environ.get("HF_TOKEN"))
    return info.sha


# ---------------------------------------------------------------- SWE-bench Pro
def lock_swebench_pro() -> None:
    repo = "scaleapi/SWE-bench_Pro-os"
    sha = github_head_sha(repo)
    raw = requests.get(
        f"https://raw.githubusercontent.com/{repo}/{sha}/v2/hard51_ids.txt", timeout=30
    )
    raw.raise_for_status()
    ids = [line.strip() for line in raw.text.splitlines() if line.strip()]
    assert len(ids) == 51, f"expected 51 ids, got {len(ids)}"
    write(
        "swebench_pro_v2_hard.json",
        {
            "locked_at": now(),
            "harness_repo": repo,
            "harness_commit": sha,
            "dataset_hf_repo": "ScaleAI/SWE-bench_Pro",
            "dataset_hf_revision": hf_revision("ScaleAI/SWE-bench_Pro", "dataset"),
            "hf_config": "hard",
            "split": "test",
            "n_tasks": len(ids),
            "task_ids": ids,
            "image_pattern": "ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>",
            "repos": sorted({i.split("__")[0] for i in ids}),
        },
    )


# ---------------------------------------------------------------- Terminal-Bench
def lock_terminal_bench() -> None:
    repo = "harbor-framework/terminal-bench"
    sha = github_head_sha(repo)
    # Task list: use the harbor CLI when installed, else fall back to the GitHub tree.
    tasks: list[str] = []
    version = os.environ.get("TB_VERSION", "")
    try:
        out = subprocess.run(
            ["harbor", "datasets", "tasks", f"terminal-bench/terminal-bench@{version or '2.0'}"],
            capture_output=True, text=True, timeout=120, check=True,
        ).stdout
        tasks = [t.strip() for t in out.splitlines() if t.strip()]
    except Exception as exc:  # noqa: BLE001
        print(f"harbor CLI unavailable ({exc}); falling back to GitHub tree listing")
        r = requests.get(
            f"https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1", timeout=60
        )
        r.raise_for_status()
        tasks = sorted(
            {
                p["path"].split("/")[1]
                for p in r.json()["tree"]
                if p["path"].startswith("tasks/") and p["path"].count("/") == 1 and p["type"] == "tree"
            }
        )
    write(
        "terminal_bench_2_tasks.json",
        {
            "locked_at": now(),
            "harness_repo": repo,
            "harness_commit": sha,
            "registry": "terminal-bench/terminal-bench",
            "version": version or "LOCK: set TB_VERSION to the exact 2.0.x tag",
            "n_tasks": len(tasks),
            "task_ids": tasks,
            "seeds": [1, 2, 3, 4, 5],
        },
    )


# ---------------------------------------------------------------- HLE
def lock_hle() -> None:
    from datasets import load_dataset

    sha = github_head_sha("centerforaisafety/hle")
    rev = hf_revision("cais/hle", "dataset")
    ds = load_dataset("cais/hle", split="test", revision=rev, token=os.environ.get("HF_TOKEN"))
    text_only = [row["id"] for row in ds if not row.get("image")]
    write(
        "hle_text_only_ids.json",
        {
            "locked_at": now(),
            "harness_repo": "centerforaisafety/hle",
            "harness_commit": sha,
            "dataset_hf_repo": "cais/hle",
            "dataset_hf_revision": rev,
            "split": "test",
            "n_total": len(ds),
            "n_text_only": len(text_only),
            "modality": "text_only",
            "question_ids": text_only,
            "judge_model": "o3-mini-2025-01-31",
        },
    )


# ---------------------------------------------------------------- ASI-Bench
def lock_asi_bench(track: str) -> None:
    repo = "apexin-ai/ASI-Bench"
    sha = github_head_sha(repo)
    hf_repo = f"Apexintelligence-AI/ASI-Bench-{track}"
    rev = hf_revision(hf_repo, "dataset")
    r = requests.get(f"https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1", timeout=60)
    r.raise_for_status()
    task_dirs = sorted(
        {
            p["path"].split("/")[1]
            for p in r.json()["tree"]
            if p["path"].startswith("tasks/") and p["path"].count("/") == 1 and p["type"] == "tree"
        }
    )
    write(
        "asi_bench_tasks.json",
        {
            "locked_at": now(),
            "harness_repo": repo,
            "harness_commit": sha,
            "track": track,
            "dataset_hf_repo": hf_repo,
            "dataset_hf_revision": rev,
            "levels": ["B1", "B2", "B3", "B4"],
            "n_projects": len(task_dirs),
            "project_ids": task_dirs,
            "primary_metric": "b3_mean_final_score",
        },
    )


# ---------------------------------------------------------------- base model
def lock_base_model() -> None:
    from huggingface_hub import hf_hub_download

    rev = hf_revision("Qwen/Qwen3.6-35B-A3B")
    path = hf_hub_download("Qwen/Qwen3.6-35B-A3B", "tokenizer_config.json", revision=rev,
                           token=os.environ.get("HF_TOKEN"))
    cfg = json.loads(Path(path).read_text())
    template = cfg.get("chat_template", "")
    if isinstance(template, list):
        template = json.dumps(template, sort_keys=True)
    write(
        "base_model.json",
        {
            "locked_at": now(),
            "model_id": "Qwen/Qwen3.6-35B-A3B",
            "hf_revision": rev,
            "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        },
    )


def report_unlocked() -> None:
    pending = []
    for cfg in sorted((ROOT / "configs" / "eval").glob("*.yaml")):
        for i, line in enumerate(cfg.read_text().splitlines(), 1):
            if re.search(r":\s*LOCK\b", line):
                pending.append(f"{cfg.relative_to(ROOT)}:{i}: {line.strip()}")
    print(f"\n{len(pending)} LOCK fields still unfilled:")
    print("\n".join(pending))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-hle", action="store_true")
    ap.add_argument("--skip-asi", action="store_true")
    ap.add_argument("--asi-track", default="seed42", choices=["seed42", "seed31415"])
    ap.add_argument("--skip-base", action="store_true")
    args = ap.parse_args()

    steps = [("swebench_pro", lock_swebench_pro), ("terminal_bench", lock_terminal_bench)]
    if not args.skip_hle:
        steps.append(("hle", lock_hle))
    if not args.skip_asi:
        steps.append(("asi_bench", lambda: lock_asi_bench(args.asi_track)))
    if not args.skip_base:
        steps.append(("base_model", lock_base_model))

    failed = []
    for name, fn in steps:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] FAILED: {exc}", file=sys.stderr)
            failed.append(name)
    report_unlocked()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
