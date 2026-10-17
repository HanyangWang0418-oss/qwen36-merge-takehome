#!/usr/bin/env python
"""HLE dev proxy: MMLU-Pro STEM (configs/eval/dev_proxy_hle_mmlu_pro.jsonl) against an OpenAI-compatible endpoint.
Same prompt/extraction as scripts/prescreen_tinker.py; writes <out>/per_task.jsonl (score 0/1) + summary.json."""
import argparse, json, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=20000); ap.add_argument("--workers", type=int, default=32); ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    from openai import OpenAI
    client = OpenAI(base_url=a.base, api_key="dummy", timeout=1800)
    qs = [json.loads(l) for l in (ROOT / "configs/eval/dev_proxy_hle_mmlu_pro.jsonl").open()][: a.n]
    letters = "ABCDEFGHIJ"

    def one(q):
        opts = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(q["options"]))
        prompt = f"{q['question']}\n\n{opts}\n\nAnswer with the letter of the correct option only, in the form \"Answer: X\"."
        t0 = time.time()
        try:
            r = client.chat.completions.create(model=a.model, messages=[{"role": "user", "content": prompt}], max_tokens=a.max_tokens,
                                               temperature=1.0, top_p=0.95, extra_body={"top_k": 20})
            text = r.choices[0].message.content or ""
            m = re.findall(r"Answer\s*[:：]\s*\(?([A-J])\)?", text)
            pred = m[-1] if m else (re.findall(r"\b([A-J])\b", text.strip()[-40:]) or [None])[-1]
            return {"task_id": q["id"], "run": 1, "score": float(pred == q["answer"]), "status": "ok", "pred": pred, "gold": q["answer"],
                    "category": q["category"], "n_output_tokens": getattr(r.usage, "completion_tokens", None), "wallclock_s": round(time.time() - t0, 1)}
        except Exception as e:  # noqa: BLE001
            return {"task_id": q["id"], "run": 1, "score": 0.0, "status": "model_failure", "error": repr(e)[:160], "category": q["category"]}

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, qs))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "per_task.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    acc = sum(r["score"] for r in rows) / len(rows)
    (out / "summary.json").write_text(json.dumps({"model": a.model, "n": len(rows), "accuracy": acc, "failures": sum(r["status"] != "ok" for r in rows)}, indent=1))
    print(f"{a.model}: MMLU-Pro STEM proxy acc={acc:.3f} n={len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
