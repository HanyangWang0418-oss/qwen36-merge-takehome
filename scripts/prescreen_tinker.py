#!/usr/bin/env python
"""Cheap pre-screen on the HLE dev proxy (MMLU-Pro STEM) using Tinker's sampling API:
compare the base model with an expert checkpoint before any GPU is available.
Not an evaluation result; used only to decide whether e3 is worth keeping / retraining.

    set -a; . ./.env; set +a
    uv run python scripts/prescreen_tinker.py --models base=Qwen/Qwen3.6-35B-A3B e3=tinker://.../sampler_weights/... --n 100

Sampling: thinking on, temperature 1.0 / top_p 0.95 / top_k 20 (Qwen "thinking general"), max 8192 new
tokens, one sample per question; answer letter extracted from the text after </think> ("Answer: X" or a
lone letter), exact match. Writes analysis/prescreen_<tag>.json with per-question results.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_prompt(q: dict) -> str:
    letters = "ABCDEFGHIJ"
    opts = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(q["options"]))
    return (f"{q['question']}\n\n{opts}\n\nAnswer with the letter of the correct option only, in the form \"Answer: X\".")


def extract(text: str) -> str | None:
    t = text.split("</think>")[-1]
    m = re.findall(r"Answer\s*[:：]\s*\(?([A-J])\)?", t)
    if m:
        return m[-1]
    m = re.findall(r"\b([A-J])\b", t.strip()[-40:])
    return m[-1] if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="tag=model_path (base model name or tinker:// sampler path)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    import tinker
    from tinker import types
    from transformers import AutoTokenizer

    qs = [json.loads(l) for l in (ROOT / "configs/eval/dev_proxy_hle_mmlu_pro.jsonl").open()][: a.n]
    rev = json.loads((ROOT / "data/manifests/base_model.json").read_text())["hf_revision"]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B", revision=rev)
    sc = tinker.ServiceClient()
    params = types.SamplingParams(max_tokens=a.max_tokens, temperature=1.0, top_p=0.95, top_k=20)
    summary = {}
    for spec in a.models:
        tag, path = spec.split("=", 1)
        client = sc.create_sampling_client(model_path=path) if path.startswith("tinker://") else sc.create_sampling_client(base_model=path)
        t0 = time.time()

        def one(q):
            prompt = tok.apply_chat_template([{"role": "user", "content": build_prompt(q)}], add_generation_prompt=True, tokenize=False)
            ids = tok(prompt, add_special_tokens=False)["input_ids"]
            try:
                res = client.sample(prompt=tinker.ModelInput.from_ints(ids), sampling_params=params, num_samples=1).result()
                text = tok.decode(res.sequences[0].tokens)
                pred = extract(text)
                return {"id": q["id"], "category": q["category"], "gold": q["answer"], "pred": pred, "correct": pred == q["answer"],
                        "n_tokens": len(res.sequences[0].tokens), "truncated": len(res.sequences[0].tokens) >= a.max_tokens, "status": "ok"}
            except Exception as e:  # noqa: BLE001
                return {"id": q["id"], "category": q["category"], "gold": q["answer"], "pred": None, "correct": False, "status": f"error: {repr(e)[:120]}"}

        with ThreadPoolExecutor(a.workers) as ex:
            rows = list(ex.map(one, qs))
        acc = sum(r["correct"] for r in rows) / len(rows)
        errs = sum(r["status"] != "ok" for r in rows)
        trunc = sum(r.get("truncated", False) for r in rows)
        toks = sum(r.get("n_tokens", 0) for r in rows)
        summary[tag] = {"model": path, "n": len(rows), "accuracy": acc, "errors": errs, "truncated": trunc, "output_tokens": toks,
                        "est_cost_usd": toks / 1e6 * 1.335, "seconds": round(time.time() - t0)}
        (ROOT / "analysis").mkdir(exist_ok=True)
        (ROOT / f"analysis/prescreen_{tag}.json").write_text(json.dumps({"summary": summary[tag], "rows": rows}, indent=1))
        print(tag, json.dumps(summary[tag]), flush=True)
    # paired comparison if two models
    tags = list(summary)
    if len(tags) >= 2:
        from src.analysis.bootstrap import paired_bootstrap_delta
        A = {r["id"]: float(r["correct"]) for r in json.loads((ROOT / f"analysis/prescreen_{tags[0]}.json").read_text())["rows"]}
        B = {r["id"]: float(r["correct"]) for r in json.loads((ROOT / f"analysis/prescreen_{tags[1]}.json").read_text())["rows"]}
        d = paired_bootstrap_delta(A, B)
        print(f"paired delta {tags[1]} - {tags[0]}:", json.dumps(d))
        summary["paired_delta"] = d
    (ROOT / "analysis/prescreen_summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
