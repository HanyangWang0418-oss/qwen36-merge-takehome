---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
base_model_relation: adapter
library_name: peft
tags: [lora, qwen3.6, moe, {{TAGS}}]
language: [en, zh]
---

# {{REPO_NAME}}

LoRA adapter (rank {{RANK}}, all linear projections incl. MoE expert matrices) on
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) revision `{{BASE_REV}}`,
trained on Tinker for the take-home "multi-expert training and merging" study.
Expert role: **{{ROLE}}** (target benchmark: {{TARGET}}).

## Training
| item | value |
|---|---|
| method | LoRA (Tinker managed), bf16 |
| rank / alpha | {{RANK}} / {{ALPHA}} |
| target modules | {{TARGET_MODULES}} |
| data | {{DATA}} |
| examples / loss tokens | {{N_TRAIN}} / {{LOSS_TOKENS}} |
| steps / batch / LR / schedule | {{STEPS}} / {{BS}} / {{LR}} / cosine, {{WARMUP}} warmup |
| context length | {{SEQ}} |
| seed | {{SEED}} |
| checkpoint selection | lowest dev loss on a held-out split of the training mixture ({{DEV_LOSS}}); benchmark data never used |
| Tinker source | `{{TINKER_PATH}}` |

Loss curves and raw metrics: `curves/` in the code repository.

## Data sources and licences
{{DATA_TABLE}}

Contamination controls: SWE-bench Pro repos and instance ids, Terminal-Bench 2.0 task names,
ASI-Bench project ids excluded; HLE canary + 13-gram overlap filter. Details: `data/README.md` in the code repository.

## Results
See the code repository README; headline numbers are reported there with bootstrap confidence intervals.
Per-benchmark results for this expert: {{RESULTS}}

## How to load
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-35B-A3B", revision="{{BASE_REV}}", torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "{{REPO_ID}}")
```
vLLM: `vllm serve Qwen/Qwen3.6-35B-A3B --enable-lora --lora-modules {{SHORT}}=<path to this adapter>`

## Limitations
Trained on bandwidth-bounded samples of public datasets (see data README); no claim of improvement is made
beyond what the evaluation tables show. Some upstream data (TMax) contains Gemini-generated content subject to Google's terms.

Code repository: {{CODE_REPO}}
