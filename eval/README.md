# Evaluation with Modal sandboxes

Self-contained runbook for the four benchmarks. Sandboxes run on **Modal** (the environment used by the
SWE-bench Pro locked protocol and the Terminal-Bench leaderboard); the model is served by **vLLM on our GPU
box** and reached from Modal through a public HTTPS endpoint. HLE needs no sandbox; ASI-Bench needs a local
Docker daemon and runs on the GPU box itself.

```
eval/
  README.md            this file
  env.example          environment variables (copy to eval/.env)
  setup.sh             harnesses at locked commits, harbor[modal], asibench, hle deps, modal auth check
  serve.sh             vLLM (base + all adapters, keys remapped) + cloudflared tunnel -> public URL
  smoke.sh             endpoint check, 1 TB2 task with the oracle agent on Modal, 1 HLE question, 1 SWE task
  matrix.yaml          which model runs which benchmark (full / target / proxy-screen)
  run_swebench_pro.sh  Modal, locked mini-swe-agent, fresh-sandbox regrade
  run_terminal_bench.sh Modal, terminus-2, 5 seeds, model-card settings
  run_hle.sh           official predict + judge (text-only ids), no sandbox
  run_asi_bench.sh     asibench CLI, docker sandbox on the GPU box
  run_proxies.sh       dev-proxy screening (MMLU-Pro STEM, SWE-bench Verified 15, TB-Pro 20, ASI seed31415 12)
  run_all.sh           drives matrix.yaml for one model
  collect.sh           harbor/hle/asi outputs -> results/<group>/<model>/<bench>/per_task.jsonl; summary + plots
```

## Chosen setup
vLLM runs on **our own 2× A100 80GB box** (`eval/serve.sh`, TP=2) and is exposed to Modal sandboxes through a cloudflared
tunnel. `eval/modal_vllm.py` / the `modal-ops` workflow (vLLM on a Modal H200) are an **optional alternative that is not
used**; `eval/modal_driver.py` is only needed if the driving machine cannot reach api.modal.com.

## Order of operations
1. `cp eval/env.example eval/.env` and fill it in; `eval/setup.sh`.
2. On the GPU box: `eval/serve.sh` (prints the public URL; keep it running under tmux).
3. `eval/smoke.sh` — must pass before spending on full runs.
4. `eval/run_all.sh base base` (all four, full), then experts on their targets, then the two main merges (all four).
5. `eval/run_proxies.sh <model>` for the ablation/control merges; promote winners to full runs.
6. `eval/collect.sh` after each run; `results/summary.md`, `analysis/uncertainty.json`, `curves/merge_ablation.png`.

## Fixed settings (see configs/eval/*.yaml; identical for every model within a benchmark)
| benchmark | tasks | agent / harness | per-task resources | sampling |
|---|---|---|---|---|
| SWE-bench Pro v2-hard | 51 (`data/manifests/swebench_pro_v2_hard.json`) | `locked_mini_swe:LockedMiniSwe` from SWE-bench_Pro-os @ locked commit, 50 min, agent offline except the model host, regrade with `patch_replay` | task.toml (1 CPU / 4 GiB) | T 0.6, top_p 0.95, top_k 20, max_tokens 32768 |
| Terminal-Bench 2.0 | 89 (`terminal-bench/terminal-bench@2.0`) | terminus-2, 5 seeds (1–5) | 32 CPU / 48 GiB, 3 h | T 1.0, top_p 0.95, top_k 20, max_tokens 81920, ctx 256K |
| HLE | 2158 text-only ids | official `run_model_predictions.py` + `run_judge_results.py`, judge o3-mini-2025-01-31 | – | T 1.0, top_p 0.95, top_k 20, max_tokens 32768 |
| ASI-Bench | seed42 60 × B1–B4 (or seed31415 local) | `asibench run --agent opencode --sandbox os` | asibench defaults | T 1.0, top_p 0.95, top_k 20, max_tokens 32768 |

Thinking on for all; `preserve_thinking` on for agent benchmarks; bf16; no quantization.

## Costs (Modal, approximate)
CPU $0.000038/core·s, memory $0.0000067/GiB·s. TB2 at 32 CPU/48 GiB ≈ $5.5/h per task; 89 × 5 seeds × ~0.75 h ≈
$1,800 per model at full spec. **Default here: `TB_CPUS=32 TB_MEM_MB=49152` as the model card; set `TB_CPUS=4
TB_MEM_MB=8192` to run at task.toml-like resources (~8× cheaper) and disclose it.** SWE-bench Pro ≈ $5 per model.
vLLM box time is separate.

## Disclosures to carry into the README/report
Sandbox provider Modal (official), model served by vLLM with LoRA hot-swap (bf16), HLE text-only subset, SWE-bench
Pro single locked agent (mini-swe-agent), Terminal-Bench per-task CPU/memory as actually used (printed by run script).
