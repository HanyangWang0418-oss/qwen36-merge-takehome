#!/usr/bin/env python
"""Pre-flight for the evaluation endpoints in eval/endpoints.yaml. Run on the evaluation box:

    python3 eval/check_endpoints.py               # all checks except Modal reachability
    python3 eval/check_endpoints.py --modal       # also curl each URL from inside a Modal sandbox

Per endpoint:
  1 models      served id matches, max_model_len (TB2 needs 262144; SWE/HLE/ASI need >= 131072)
  2 adapter     greedy outputs on fixed prompts differ from the base (identical => adapter/merge not applied)
  3 reasoning   thinking separated into reasoning_content (reasoning parser on), no '<think>' leaking into content
  4 tools       OpenAI tool calling returns tool_calls (needed by mini-swe-agent toolcall config)
  5 sampling    top_k accepted; same sampling knobs work on every endpoint
  6 throughput  16 concurrent requests: aggregate tokens/s, p95 latency
  7 modal       (--modal) the URL answers from inside a Modal sandbox (required for harbor -e modal)
Writes logs/eval/endpoint_check.json.
"""
import argparse, json, statistics, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
EPS = yaml.safe_load((ROOT / "eval/endpoints.yaml").read_text())
PROMPTS = ["Write a bash one-liner that prints the 5 largest files under the current directory.",
           "A git rebase failed with a conflict in setup.py. List the exact commands to inspect, resolve and continue.",
           "Compute 17*23 and explain in one sentence."]
TOOLS = [{"type": "function", "function": {"name": "bash", "description": "Run a bash command",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]


def post(base, payload, timeout=600, key="dummy"):
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.time() - t0


def get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def greedy(base, served, p):
    d, _ = post(base, {"model": served, "messages": [{"role": "user", "content": p}], "max_tokens": 160, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}})
    return d["choices"][0]["message"].get("content") or ""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--modal", action="store_true"); a = ap.parse_args()
    report, base_out = {}, None
    names = list(EPS)
    if "base" in names:
        names.remove("base"); names = ["base"] + names
    for name in names:
        e = EPS[name]; base, served = e["base_url"], e["served"]; r = {"base_url": base, "served": served}
        print(f"\n== {name}  {base}  ({served})")
        try:
            ms = get(base.rstrip("/") + "/models")["data"]
            ids = [m["id"] for m in ms]; mlen = max((m.get("max_model_len") or 0) for m in ms)
            r["models"] = {"ids": ids, "max_model_len": mlen, "served_present": served in ids}
            print(f"  1 models     ids={ids} max_model_len={mlen} {'OK' if served in ids else 'MISSING served id'}"
                  + ("" if mlen >= 262144 else "  WARN <262144: Terminal-Bench needs 256K context"))
        except Exception as ex:
            r["models"] = {"error": repr(ex)[:200]}; print("  1 models     UNREACHABLE", repr(ex)[:160]); report[name] = r; continue
        try:
            outs = [greedy(base, served, p) for p in PROMPTS]; r["greedy"] = outs
            if name == "base":
                base_out = outs; print("  2 adapter    (reference)")
            elif base_out:
                same = sum(o == b for o, b in zip(outs, base_out)); r["identical_to_base"] = same
                print(f"  2 adapter    {same}/{len(PROMPTS)} greedy outputs identical to base" + ("  FAIL: adapter/merge likely NOT applied" if same == len(PROMPTS) else "  OK"))
        except Exception as ex:
            print("  2 adapter    error", repr(ex)[:160])
        try:
            d, dt = post(base, {"model": served, "messages": [{"role": "user", "content": "Is 391 prime? Answer yes or no."}], "max_tokens": 2048, "temperature": 1.0, "top_p": 0.95, "top_k": 20})
            m = d["choices"][0]["message"]; rc = m.get("reasoning_content") or m.get("reasoning") or ""; c = m.get("content") or ""
            ok = bool(rc) and "<think>" not in c
            r["reasoning"] = {"reasoning_chars": len(rc), "think_in_content": "<think>" in c, "top_k_accepted": True}
            print(f"  3 reasoning  reasoning_chars={len(rc)} think_in_content={'<think>' in c} {'OK' if ok else 'WARN: start vLLM with --reasoning-parser qwen3'}")
            print(f"  5 sampling   top_k accepted ({dt:.1f}s)")
        except Exception as ex:
            print("  3/5          error", repr(ex)[:160])
        try:
            d, _ = post(base, {"model": served, "messages": [{"role": "user", "content": "List the files in /tmp using the bash tool."}], "tools": TOOLS, "tool_choice": "auto", "max_tokens": 1024})
            tc = d["choices"][0]["message"].get("tool_calls") or []
            r["tools"] = {"n_tool_calls": len(tc)}
            print(f"  4 tools      tool_calls={len(tc)} {'OK' if tc else 'WARN: start vLLM with --enable-auto-tool-choice --tool-call-parser qwen3_coder (or hermes)'}")
        except Exception as ex:
            print("  4 tools      error", repr(ex)[:160])
        try:
            def one(i):
                d, dt = post(base, {"model": served, "messages": [{"role": "user", "content": f"Count from 1 to 60 (run {i})."}], "max_tokens": 256, "temperature": 0.7,
                                    "chat_template_kwargs": {"enable_thinking": False}})
                return d["usage"]["completion_tokens"], dt
            t0 = time.time()
            with ThreadPoolExecutor(16) as ex:
                res = list(ex.map(one, range(16)))
            wall = time.time() - t0; toks = sum(t for t, _ in res); lat = sorted(dt for _, dt in res)
            r["throughput"] = {"tokens_per_s": toks / wall, "p95_latency_s": lat[int(0.95 * len(lat)) - 1]}
            print(f"  6 throughput {toks / wall:.0f} tok/s aggregate at 16 concurrent, p95 {lat[int(0.95 * len(lat)) - 1]:.1f}s")
        except Exception as ex:
            print("  6 throughput error", repr(ex)[:160])
        report[name] = r
    if a.modal:
        import modal
        app = modal.App.lookup("qwen36-endpoint-check", create_if_missing=True)
        sb = modal.Sandbox.create(app=app, image=modal.Image.debian_slim().apt_install("curl"), timeout=300)
        try:
            for name, e in EPS.items():
                url = (e.get("public_url") or e["base_url"]).rstrip("/") + "/models"
                p = sb.exec("bash", "-c", f"curl -s -m 15 -o /dev/null -w '%{{http_code}}' {url} || echo FAIL"); p.wait()
                code = p.stdout.read().strip()
                report.setdefault(name, {})["modal_reachable"] = code == "200"
                print(f"  7 modal      {name}: {url} -> {code} {'OK' if code == '200' else 'FAIL: not reachable from Modal (use a public tunnel or HARBOR_ENV=docker)'}")
        finally:
            sb.terminate()
    (ROOT / "logs/eval").mkdir(parents=True, exist_ok=True)
    (ROOT / "logs/eval/endpoint_check.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("\nwrote logs/eval/endpoint_check.json")


if __name__ == "__main__":
    sys.exit(main())
