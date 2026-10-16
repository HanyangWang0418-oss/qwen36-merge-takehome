# Evaluation infrastructure: Alibaba Cloud Agent Sandbox 极速版 (E2B-compatible) + Harbor `-e e2b`

Decision (2026-10-16): run the Harbor-based benchmarks (SWE-bench Pro v2-hard, Terminal-Bench 2.0)
on Alibaba Cloud **Agent Sandbox 极速版** (Speed Edition, Function-Compute hosted, E2B-compatible API)
instead of Modal, using Harbor's built-in `e2b` backend. Reason: price (Default QoS, mainland:
0.078 元/vCPU·h + 0.039 元/GiB·h, i.e. ≈ 4.4 元/h for a 32 vCPU / 48 GiB task vs ≈ $5.5/h on Modal)
and zero cluster administration. The ACS "容器版" + `-e ack` route is kept as fallback (section at the end).

Disclosure: SWE-bench Pro's locked protocol and the Terminal-Bench leaderboard use Modal. We use the
same task images, agents, budgets and prompts, but a different sandbox provider and a weaker
network isolation (below). Recorded as "本实验设置" in configs/eval/*.yaml and the README.

## Verified facts (probed 2026-10-16 with the project key, region cn-hangzhou)
| item | result |
|---|---|
| API | `https://api.cn-hangzhou.sandbox.aliyuncs.com`, domain `cn-hangzhou.sandbox.aliyuncs.com`; the key is bound to one region (401 elsewhere) |
| E2B API version | **v1 only**: `POST /sandboxes` works, `POST /v2/sandboxes` → 405. e2b SDK ≥ 2.5x uses v2 → pinned `e2b==2.45.0` (v1 create + Template API). Harbor is installed with `uv tool install harbor==0.24.0 --with e2b==2.45.0` |
| sandbox boot | ≈ 2.5 s; built-in `code-interpreter-v1` = 2 vCPU / 2 GiB / 15 GiB, Debian 13, bash/python3/git/curl present, no docker, no tmux |
| commands | foreground + background (`commands.run(background=True)` + `wait()`), files.write/read OK |
| timeout | `set_timeout(4 h)` accepted → Terminal-Bench's 3 h limit is fine |
| egress | ghcr.io, pypi.org, github.com reachable; **proxy.golang.org and huggingface.co not reachable** (relevant for SWE-bench Pro Go tasks' verifier: set `GOPROXY=https://goproxy.cn,direct` in the task env) |
| network policy | `update_network({"allow_internet_access": False})` is accepted but **not enforced** (pypi still 200). Harbor's `allow_internet_access=False` for the agent phase therefore has no effect → disclosed; mitigations: host-side agent variant (`host_mini_swe`, LLM loop on the host, no agent install in the sandbox) and the regrade step on a pristine sandbox |
| templates | only `Template().from_image(ref, username, password)`; no Dockerfile/steps. Builder needs **pull AND push** rights on the source registry (it pushes the converted image back) → public ghcr/Docker Hub images cannot be used directly; everything goes through our own **ACR** namespace. Fully-qualified refs only (`docker.io/library/...`) |
| Harbor alias | `e2b` backend builds/looks up alias `{task.short_name}__{sha256(environment dir)[:12]}` ("/"→"__", "."→"-"); Harbor's own build would fail (no credentials), so templates are **pre-built** with `scripts/prebuild_sandbox_templates.py`, which computes the same alias through Harbor's code and builds from ACR with credentials. Harbor then only creates sandboxes |
| spec limits | not yet probed (needs an ACR image); the default template is 2 vCPU / 2 GiB; TB2 needs 32 vCPU / 48 GiB per the model card → probe before the full run, fall back to the largest allowed spec and disclose |

## Pipeline
1. **Images → ACR** (ECS build box, or a sandbox with skopeo for the mirror step):
   `scripts/mirror_images.sh` (51 SWE-bench Pro images, 62 GB compressed, ghcr → ACR, digest-verified) and
   `scripts/build_tb2_images.sh` (89 Terminal-Bench 2.0 Dockerfiles at the locked commit → `ACR/<task>:<commit8>` + `:latest`).
   `scripts/lock_images.py` records digests in `data/manifests/images_acr_*.json`.
2. **Task copies pointing at ACR**: `scripts/prepare_swebench_tasks_acr.py` (rewrites `docker_image` + Dockerfile FROM, digest-pinned) and `scripts/prepare_tb2_tasks_acr.py` (adds `docker_image`). Diffs saved as `ACR_REWRITE.diff` next to the copies.
3. **Templates**: `scripts/prebuild_sandbox_templates.py <tasks_dir> --cpus N --memory-mb M` (run with `~/.local/share/uv/tools/harbor/bin/python`, needs `E2B_*` and `ACR_USERNAME/ACR_PASSWORD`).
4. **Run**: `scripts/eval_swebench_pro.sh` / `scripts/eval_terminal_bench.sh` with `HARBOR_ENV=e2b` (default), `E2B_API_KEY/E2B_API_URL/E2B_DOMAIN` exported. Harbor finds the aliases and skips building.
5. **ASI-Bench**: `asibench --sandbox os` needs a Docker daemon → runs on an ECS VM, not in Agent Sandbox. **HLE**: no sandbox.

## Model serving
vLLM, base + LoRA hot-swap, ≥ 2 × 80 GB-class GPUs for 256K context. Mainland: H20 (96 GB) / A800; H100 only outside the mainland. Place it in the same region (Hangzhou) and expose one HTTPS endpoint; it is the only host the agent may call (enforced by the agent config, not by the sandbox network).

## What the account owner needs to provide
1. **ACR** (personal edition should suffice since we pass credentials; EE only if the builder rejects personal — to be probed) in **cn-hangzhou**: namespace, username, password → `ACR_REGISTRY`, `ACR_USERNAME`, `ACR_PASSWORD` in `.env`.
2. One **ECS VM with Docker** (8 vCPU / 16 GiB, 200 GB disk, cn-hangzhou) for image mirroring, TB2 builds, running harbor itself and `asibench` (avoids the laptop's bandwidth).
3. GPU instance for vLLM (see above).

## Cost estimate (Default QoS, mainland)
- SWE-bench Pro: 51 × ≤50 min × 1 vCPU / 4 GiB × 4 models ≈ 10–20 元.
- Terminal-Bench 2.0: 89 × 5 × 4 = 1,780 task-runs × ~0.75 h × 4.4 元/h ≈ 5,900 元 at 32 vCPU / 48 GiB (≈ 4,300 元 at Economy QoS; far less if the platform caps the spec lower).
- ASI-Bench: ECS time ≈ 100–200 元. GPU serving: H20 × 2 ≈ 60–80 元/h × 60–100 h.

## Fallback: ACS 容器版 + `harbor -e ack`
Kubernetes API via kubeconfig, images built in-cluster (privileged DinD or BuildKit) and pushed to ACR, SandboxSet/SandboxClaim pools, per-task CPU/memory from task.toml. More setup (cluster, add-ons `acs-virtual-node ≥ 2.17.0`, `ack-agent-sandbox-controller ≥ 0.5.14`, RAM role), but real NetworkPolicy enforcement and no 极速版 spec limits. Use if the 极速版 spec cap or network isolation turns out unacceptable.

## Bandwidth findings (2026-10-17) and the image pipeline that follows
- Hangzhou sandbox → ghcr.io blob CDN: **~14 KB/s** (62 GB would take ~50 days); Docker Hub and gcr unreachable; quay.io and public.ecr.aws reachable. ACR (public endpoint) reachable from the sandbox (401 in 40 ms), unreachable from the laptop's corporate network.
- Therefore images are moved by **GitHub Actions** (foreign runners, fast ghcr): `.github/workflows/mirror_swebench_images.yml` (skopeo, 6 shards) and `.github/workflows/build_tb2_images.yml` (buildx, 10 shards, locked dataset commit). Repository secrets: `ACR_HOST`, `ACR_NAMESPACE`, `ACR_USERNAME`, `ACR_PASSWORD`. Image names: `<ns>/swe-bench_pro-v2:<instance_id>` and `<ns>/tb2-<task>:<commit8>`.
- `src/eval/registry_copy.py` (pure-Python copier with Range-resume) remains for small fixes from inside a sandbox; `scripts/mirror_images.sh` / `build_tb2_images.sh` remain for an ECS box.
- ACR namespace in use: `qwen36-1017` (personal edition, auto-create repositories on).
