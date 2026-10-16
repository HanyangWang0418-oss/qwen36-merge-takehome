# Model checkpoints

All adapters are PEFT LoRA on `Qwen/Qwen3.6-35B-A3B` revision `<fill from data/manifests/base_model.json>`.

| name | HF repo | revision | type | merge recipe |
|---|---|---|---|---|
| e1_code_repair | https://huggingface.co/ssyhw7/qwen36-35b-e1_code_repair | `a29d47f231ee7f2432dbce2b4b1581164a2df909` | LoRA r32 α32, fp32, 2.24 GB | – |
| e2_terminal_agent | https://huggingface.co/ssyhw7/qwen36-35b-e2_terminal_agent | `7c7b78419db4af225c8fbf28a1e4d3d5a2e57bcf` | LoRA r32 α32, fp32, 2.24 GB | – |
| e3_reasoning | https://huggingface.co/ssyhw7/qwen36-35b-e3_reasoning | `014a1c5a70a69130dd2ce81c4e2bf027e58924bc` | LoRA r32 α32, fp32, 2.24 GB | – |
| merged_ta_l0.7 | https://huggingface.co/ssyhw7/qwen36-35b-merged-ta_l0.7 | `d8da199b6ed6fa6a0a6dadd418692d210e0614e4` | LoRA r96, fp32, 6.74 GB (exact rank concatenation, no approximation) | `configs/merge/ta_l0.7.yaml`, `data/manifests/merge_recipes/merged_ta_l0.7.json` |
| merged_ties_d0.2 | https://huggingface.co/ssyhw7/qwen36-35b-merged-ties_d0.2 | `d9ce9d15a9ec5a6de5acc0f7b0cb8682ff862aa5` | LoRA r96, fp32, 6.74 GB (joint Gram-SVD re-expression; rel. Frobenius error mean 0.49 / max 0.56 — TIES trimming makes the merged delta high-rank, so ~24% of its energy is lost at rank 96) | `configs/merge/ties_d0.2.yaml`, `data/manifests/merge_recipes/merged_ties_d0.2.json` |

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-35B-A3B", revision="<rev>", torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "<user>/qwen36-35b-merged-ta", revision="<rev>")
```

vLLM: `vllm serve Qwen/Qwen3.6-35B-A3B --enable-lora --lora-modules merged=<local path>`.

Each HF repo contains `adapter_config.json`, `adapter_model.safetensors`, `merge_recipe.json` (merged models), `TINKER_SOURCE.txt` (experts), and a model card with base revision, data sources/licences, results and limitations. License: Apache-2.0.

## Tinker source checkpoints (before export)

| expert | tinker path (sampler weights, selected by dev loss) | size | selection | run id |
|---|---|---|---|---|
| e1_code_repair | `tinker://1bb45f40-b515-525f-ab74-a323d9454e75:train:0/sampler_weights/e1_code_repair_r32_s42_20261016_step00076` | 2.24 GB (sampler) / 6.71 GB (training state) | dev loss 0.1600 | e1_code_repair_r32_s42_20261016 |
| e2_terminal_agent | `tinker://423b1516-45ff-5480-b185-cc72ee3a4fa1:train:0/sampler_weights/e2_terminal_agent_r32_s42_20261016_step00170` | 2.24 GB (sampler) / 6.71 GB (training state) | dev loss 0.2031 | e2_terminal_agent_r32_s42_20261016 |
| e3_reasoning | `tinker://df9e7c47-fa5c-597e-a8d5-ea20edaa9035:train:0/sampler_weights/e3_reasoning_r32_s42_third_step00151` | 2.24 GB (sampler) / 6.71 GB (training state) | dev loss 0.8987 | e3_reasoning_r32_s42_third |

Export: `src/train/export_adapter.py <tinker path> models/weights/<expert>` on the cloud build box (2.24 GB each; TTL set to 365 days on 2026-10-16).

## Adapter tensor layout (from the exported checkpoints, verified 2026-10-17)
`target_modules: all-linear`, 862 tensors, fp32. Dense 2D LoRA on `self_attn.{q,k,v,o}_proj` (10 layers),
`linear_attn.in_proj_{q,k,v,z}` + `out_proj` (30 Gated-DeltaNet layers), `mlp.shared_expert.{gate,up,down}_proj`
and `unembed_tokens` (lm_head). Routed experts are **fused 3D** with one side shared across the 256 experts:
`mlp.experts.w1/w3`: A `[1,32,2048]`, B `[256,512,32]`; `mlp.experts.w2`: A `[256,32,512]`, B `[1,2048,32]`.
The MoE router (`mlp.gate`) and the GDN `in_proj_a/b`, conv1d, norms are **not** adapted.

Key names follow Tinker's export, not the HF checkpoint (`model.language_model.layers.N.linear_attn.in_proj_qkv`,
`mlp.experts.gate_up_proj/down_proj`, `lm_head`). Loading with vLLM `--enable-lora` must be verified on the GPU box;
if vLLM does not map these names, `src/merge/remap_keys.py` (to be written) fuses q/k/v→qkv and w1/w3→gate_up and
renames unembed→lm_head. For 3D expert adapters vLLM needs `--enable-mixed-moe-lora-format` / `is_3d_lora_weight`.
