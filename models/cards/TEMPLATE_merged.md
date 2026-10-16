---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
base_model_relation: merge
library_name: peft
tags: [lora, merge, {{METHOD}}, qwen3.6, moe]
language: [en, zh]
---

# {{REPO_NAME}}

A single LoRA adapter obtained by **merging** three expert adapters on
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) revision `{{BASE_REV}}`:
{{EXPERT_LIST}}

## Merge recipe
| item | value |
|---|---|
| method | {{METHOD}} |
| coefficients / density | {{COEFFS}} |
| output rank | {{OUT_RANK}} ({{RANK_NOTE}}) |
| conflict handling | {{CONFLICT}} |
| SVD reconstruction error (mean / p90 / max) | {{SVD_ERR}} |
| script / config | `src/merge/merge_adapters.py`, `{{CONFIG}}` |

`merge_recipe.json` in this repository holds the exact inputs (expert adapter revisions, lambdas, layer scopes).

## Results
{{RESULTS_TABLE}}

Paired-bootstrap deltas vs the base model and run-to-run variance are in the code repository (`analysis/`).

## How to load
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-35B-A3B", revision="{{BASE_REV}}", torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "{{REPO_ID}}")
```

## Limitations
{{LIMITATIONS}}

Code repository: {{CODE_REPO}}
