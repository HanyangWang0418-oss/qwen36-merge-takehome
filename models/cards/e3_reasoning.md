---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
base_model_relation: adapter
library_name: peft
tags: [lora, qwen3.6, moe, reasoning,math]
language: [en, zh]
---

# qwen36-35b-e3_reasoning

LoRA adapter (rank 32, all linear projections incl. MoE expert matrices) on
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) revision `995ad96eacd98c81ed38be0c5b274b04031597b0`,
trained on Tinker for the take-home "multi-expert training and merging" study.
Expert role: **long-form STEM reasoning** (target benchmark: HLE (text-only)).

## Training
| item | value |
|---|---|
| method | LoRA (Tinker managed), bf16 |
| rank / alpha | 32 / 32 (scaling 1.0; exported adapter_config lora_alpha=32) |
| target modules | all linear projections: self_attn q/k/v/o, Gated-DeltaNet in_proj_q/k/v/z + out_proj, shared_expert gate/up/down, routed experts w1/w2/w3 (fused 3D, one side shared across 256 experts), unembed (lm_head); router not adapted |
| data | Hard STEM short-answer reasoning with <think> traces (OpenThoughts, NuminaMath, GPQA-style synthetic); HLE canary filtered |
| examples / loss tokens | 4831 / 50,870,818 |
| steps / batch / LR / schedule | 151 / 32 / 0.0001 / cosine, 20 warmup |
| context length | 32768 |
| seed | 42 |
| checkpoint selection | lowest dev loss on a held-out split of the training mixture (0.8987 at step 151); benchmark data never used |
| Tinker source | `tinker://df9e7c47-fa5c-597e-a8d5-ea20edaa9035:train:0/sampler_weights/e3_reasoning_r32_s42_third_step00151` |

Loss curves and raw metrics: `curves/` in the code repository.

## Data sources and licences
| source | location | licence | rows |
|---|---|---|---|
| openthoughts3 | open-thoughts/OpenThoughts3-1.2M | Apache-2.0 | 9993 kept of 10000 fetched |
| numinamath-cot | AI-MO/NuminaMath-CoT | Apache-2.0 (CC-BY-NC for some sub-sources, excluded) | 21314 kept of 22400 fetched |

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
model = PeftModel.from_pretrained(base, "ssyhw7/qwen36-35b-e3_reasoning")
```
vLLM: `vllm serve Qwen/Qwen3.6-35B-A3B --enable-lora --lora-modules e3_reasoning=<path to this adapter>`

## Limitations
Trained on bandwidth-bounded samples of public datasets (see data README); no claim of improvement is made
beyond what the evaluation tables show. Some upstream data (TMax) contains Gemini-generated content subject to Google's terms.

Code repository: https://github.com/HanyangWang0418-oss/qwen36-merge-takehome


Note: training was resumed once from the step-100 optimizer state after a lost Tinker session; the loss curve in the code repository marks this.
