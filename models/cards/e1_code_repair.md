---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
base_model_relation: adapter
library_name: peft
tags: [lora, qwen3.6, moe, code,swe-agent]
language: [en, zh]
---

# qwen36-35b-e1_code_repair

LoRA adapter (rank 32, all linear projections incl. MoE expert matrices) on
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) revision `995ad96eacd98c81ed38be0c5b274b04031597b0`,
trained on Tinker for the take-home "multi-expert training and merging" study.
Expert role: **code repair (issue→patch agent trajectories)** (target benchmark: SWE-bench Pro v2-hard).

## Training
| item | value |
|---|---|
| method | LoRA (Tinker managed), bf16 |
| rank / alpha | 32 / 32 (scaling 1.0; exported adapter_config lora_alpha=32) |
| target modules | all linear projections: self_attn q/k/v/o, Gated-DeltaNet in_proj_q/k/v/z + out_proj, shared_expert gate/up/down, routed experts w1/w2/w3 (fused 3D, one side shared across 256 experts), unembed (lm_head); router not adapted |
| data | Issue-to-patch agent trajectories (SWE-Gym, SWE-smith, SWE-rebench), all 11 SWE-bench Pro repos excluded |
| examples / loss tokens | 1206 / 4,051,850 |
| steps / batch / LR / schedule | 76 / 32 / 0.0001 / cosine, 20 warmup |
| context length | 32768 |
| seed | 42 |
| checkpoint selection | lowest dev loss on a held-out split of the training mixture (0.1600 at step 76); benchmark data never used |
| Tinker source | `tinker://1bb45f40-b515-525f-ab74-a323d9454e75:train:0/sampler_weights/e1_code_repair_r32_s42_20261016_step00076` |

Loss curves and raw metrics: `curves/` in the code repository.

## Data sources and licences
| source | location | licence | rows |
|---|---|---|---|
| SWE-smith-trajectories | SWE-bench/SWE-smith-trajectories | MIT | 805 kept of 2500 fetched |
| SWE-Gym-trajectories | SWE-Gym/OpenHands-Sampled-Trajectories | MIT (code repos keep their own licenses) | 114 kept of 1400 fetched |
| SWE-Gym-SFT-trajectories | SWE-Gym/OpenHands-SFT-Trajectories | MIT (code repos keep their own licenses) | 435 kept of 491 fetched |
| nemotron-swe | nvidia/Nemotron-SWE-v1 | CC-BY-4.0 | 202 kept of 600 fetched |

Contamination controls: SWE-bench Pro repos and instance ids, Terminal-Bench 2.0 task names,
ASI-Bench project ids excluded; HLE canary + 13-gram overlap filter. Details: `data/README.md` in the code repository.

## Results
See the code repository README; headline numbers are reported there with bootstrap confidence intervals.
Per-benchmark results for this expert: pending (evaluation not yet run)

## How to load
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-35B-A3B", revision="995ad96eacd98c81ed38be0c5b274b04031597b0", torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "ssyhw7/qwen36-35b-e1_code_repair")
```
vLLM: `vllm serve Qwen/Qwen3.6-35B-A3B --enable-lora --lora-modules e1_code_repair=<path to this adapter>`

## Limitations
Trained on bandwidth-bounded samples of public datasets (see data README); no claim of improvement is made
beyond what the evaluation tables show. Some upstream data (TMax) contains Gemini-generated content subject to Google's terms.

Code repository: https://github.com/HanyangWang0418-oss/qwen36-merge-takehome
