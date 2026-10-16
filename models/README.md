# Model checkpoints

All adapters are PEFT LoRA on `Qwen/Qwen3.6-35B-A3B` revision `<fill from data/manifests/base_model.json>`.

| name | HF repo | revision | type | merge recipe |
|---|---|---|---|---|
| e1_code_repair | https://huggingface.co/<user>/qwen36-35b-e1-code-repair | | LoRA r32 | – |
| e2_terminal_agent | https://huggingface.co/<user>/qwen36-35b-e2-terminal-agent | | LoRA r32 | – |
| e3_reasoning | https://huggingface.co/<user>/qwen36-35b-e3-reasoning | | LoRA r32 | – |
| merged_ta_l0.7 | https://huggingface.co/<user>/qwen36-35b-merged-ta | | LoRA r96 | `configs/merge/ta_l0.7.yaml`, `merge_recipe.json` in repo |
| merged_ties_d0.2 | https://huggingface.co/<user>/qwen36-35b-merged-ties | | LoRA r96 (SVD) | `configs/merge/ties_d0.2.yaml` |

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-35B-A3B", revision="<rev>", torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "<user>/qwen36-35b-merged-ta", revision="<rev>")
```

vLLM: `vllm serve Qwen/Qwen3.6-35B-A3B --enable-lora --lora-modules merged=<local path>`.

Each HF repo contains `adapter_config.json`, `adapter_model.safetensors`, `merge_recipe.json` (merged models), `TINKER_SOURCE.txt` (experts), and a model card with base revision, data sources/licences, results and limitations. License: Apache-2.0.
