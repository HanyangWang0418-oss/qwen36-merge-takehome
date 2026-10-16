# Evaluation infrastructure: Alibaba Cloud Agent Sandbox (ACS) + Harbor `ack`

Decision (2026-10-16): run the Harbor-based benchmarks (SWE-bench Pro v2-hard, Terminal-Bench 2.0)
and the asibench docker sandbox on Alibaba Cloud instead of Modal. Reason: price. At the Qwen
model-card Terminal-Bench spec (32 vCPU / 48 GiB per task) Agent Sandbox "Default" QoS is
0.078 元/vCPU·h + 0.039 元/GiB·h ≈ 4.4 元/h ≈ $0.60/h per task (mainland), versus ≈ $5.5/h on Modal.
Disclosure: the SWE-bench Pro locked protocol and the Terminal-Bench leaderboard use Modal; we use
the same task images, agents, budgets and network policy, but a different sandbox provider. This is
recorded as "本实验设置" in configs/eval/*.yaml and in the README.

## What Harbor needs (`-e ack`, src/harbor/environments/ack.py)
- Any Kubernetes cluster reachable through a kubeconfig (`--ek kubeconfig=~/.kube/acs.yaml --ek namespace=harbor`).
- Images: either built in-cluster (privileged DinD job, or BuildKit via `--ek use_buildkit=true --ek buildkit_address=tcp://...`) and pushed to a registry (`--ek registry=<acr>.cn-hangzhou.personal.cr.aliyuncs.com/<ns>`), or pre-built images referenced directly.
- Optional SandboxClaim mode (`--ek use_sandbox_claim=true`) to draw warm sandboxes from a SandboxSet pool.
- Per-task CPU/memory come from task.toml (`cpus`, `memory_mb`); Terminal-Bench is overridden to 32 / 49152 with harbor's resource flags. ACS rounds unsupported vCPU:memory ratios up and bills the rounded spec; the actual spec is read back from the pod and logged.

## Plan per benchmark
| benchmark | images | per-task spec | sandbox mode |
|---|---|---|---|
| SWE-bench Pro hard-51 | 51 prebuilt `ghcr.io/scaleapi/swe-bench_pro-v2:<id>` (62 GB total) → mirrored once into ACR with `scripts/mirror_images.sh` | 1 vCPU / 4 GiB (task.toml), agent offline, verifier online | plain pods |
| Terminal-Bench 2.0 | 89 task Dockerfiles built once (BuildKit job or an ECS build box with docker) → ACR | 32 vCPU / 48 GiB, 3 h | plain pods or SandboxSet pool |
| ASI-Bench | asibench `--sandbox os` needs a Docker daemon → run the asibench CLI on an ECS VM with Docker (not inside Agent Sandbox) | per asibench defaults | ECS |
| HLE | no sandbox | – | – |

## Model serving
vLLM with the base + LoRA hot-swap needs ≥ 2 × 80 GB-class GPUs for 256K context. Mainland regions
offer H20 (96 GB) / A800; H100 only outside the mainland. Options: PAI-DSW/EAS GPU instance, or an ECS
gn8is/H20 spot instance in the same VPC as the ACS cluster so sandboxes reach the endpoint privately.
The endpoint host is the only host allowed in the agent-phase network policy.

## One-time setup (account owner)
1. Alibaba Cloud account with billing enabled; RAM user/role for the CLI (`aliyun configure`).
2. Create an **ACS cluster** in the chosen region (mainland is cheapest; Hong Kong if GPUs/egress matter), enable **Agent Sandbox**: add-ons `acs-virtual-node ≥ 2.17.0`, `ack-agent-sandbox-controller ≥ 0.5.14`, grant `AliyunCSManagedAgentSandboxRole`. (The E2B-compatible gateway, `ack-sandbox-manager` + Ingress + domain + TLS, is NOT required for Harbor, which talks to the Kubernetes API directly.)
3. Download the cluster kubeconfig to `~/.kube/acs.yaml`; `kubectl get sandboxsets -A` must work.
4. Create an **ACR** (personal edition is enough) namespace for the task images; docker login credentials → k8s image pull secret in namespace `harbor` (`--ek image_pull_secret=acr-pull`).
5. Decide the egress policy: sandboxes need public egress for the verifier phase (Go modules, pip) and for image pulls from ACR; agent phase must be restricted to the vLLM host (Harbor applies `network_mode = no-network` + `--allow-agent-host`; ACS enforces via NetworkPolicy, capability to be verified on the first run).
6. One ECS VM with Docker (8 vCPU is enough) for: mirroring/building images, running `asibench`, and running the harbor CLI itself (avoids the laptop's bandwidth).

## Cost estimate (Default QoS, mainland)
- SWE-bench Pro: 51 tasks × ≤ 50 min × 1 vCPU/4 GiB × 4 models ≈ 10–20 元.
- Terminal-Bench 2.0: 89 × 5 runs × 4 models = 1,780 task-runs × ~0.75 h avg × 4.4 元/h ≈ 5,900 元 (≈ $820); Economy QoS ≈ 4,300 元. The 3 h cap is the main risk; a per-run wall-clock budget is enforced by Harbor.
- ASI-Bench: ECS VM time only, ≈ 100–200 元.
- GPU serving: H20 ×2 ≈ 60–80 元/h × 60–100 h ≈ 4,000–8,000 元; spot/preemptible cuts this by half or more.
