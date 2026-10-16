#!/usr/bin/env python
"""Rename/fuse a Tinker-exported PEFT adapter into the layout vLLM's LoRA loader expects for
Qwen3.5/3.6 (vllm/model_executor/models/qwen3_5.py, vllm/lora/utils.py):

  * linear_attn.in_proj_{q,k,v}  ->  linear_attn.in_proj_qkv   (vLLM packs in_proj_qkv + in_proj_z into in_proj_qkvz)
        exact fusion: A_qkv = cat(A_q, A_k, A_v) on the rank axis; B_qkv is block-diagonal
        [[B_q 0 0],[0 B_k 0],[0 0 B_v]] over (out rows of q,k,v) x (rank blocks), so B_qkv @ A_qkv == blockdiag deltas
  * unembed_tokens -> lm_head  (vLLM embedding_modules maps PEFT `lm_head` to output_embeddings)
  * experts.w1/w2/w3 (3D), in_proj_z, out_proj, self_attn.*, shared_expert.* : unchanged (already what vLLM expects)
Rank grows for in_proj_qkv only (3r); other modules keep r, so `rank_pattern` is set for the fused module and
adapter_config.r stays the base rank. target_modules becomes an explicit list.

    uv run python src/merge/remap_keys.py models/weights/e1_code_repair models/weights/e1_code_repair_vllm
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

QKV = ("in_proj_q", "in_proj_k", "in_proj_v")


def remap(src: Path, dst: Path) -> dict:
    cfg = json.loads((src / "adapter_config.json").read_text())
    files = sorted(src.glob("adapter_model*.safetensors"))
    handles = [safe_open(str(f), framework="pt") for f in files]
    key2h = {k: h for h in handles for k in h.keys()}
    out: dict[str, torch.Tensor] = {}
    fused_prefixes: set[str] = set()
    stats = {"renamed_unembed": 0, "fused_qkv": 0, "kept": 0}
    for k in key2h:
        m = re.match(r"^(.*\.linear_attn\.)in_proj_[qkv]\.lora_[AB]\.weight$", k)
        if m:
            fused_prefixes.add(m.group(1))
            continue
        nk = k.replace(".unembed_tokens.", ".lm_head.")
        if nk != k:
            stats["renamed_unembed"] += 1
        else:
            stats["kept"] += 1
        out[nk] = key2h[k].get_tensor(k)
    rank_pattern = dict(cfg.get("rank_pattern") or {})
    for p in sorted(fused_prefixes):
        As = [key2h[f"{p}{n}.lora_A.weight"].get_tensor(f"{p}{n}.lora_A.weight") for n in QKV]
        Bs = [key2h[f"{p}{n}.lora_B.weight"].get_tensor(f"{p}{n}.lora_B.weight") for n in QKV]
        A = torch.cat(As, 0)                                  # [3r, in]
        r_tot = A.shape[0]
        B = A.new_zeros(sum(b.shape[0] for b in Bs), r_tot)  # block-diagonal
        ro = co = 0
        for b in Bs:
            B[ro: ro + b.shape[0], co: co + b.shape[1]] = b
            ro += b.shape[0]
            co += b.shape[1]
        out[f"{p}in_proj_qkv.lora_A.weight"] = A.contiguous()
        out[f"{p}in_proj_qkv.lora_B.weight"] = B.contiguous()
        # PEFT rank_pattern keys are regex-matched against module names; use the module path without prefix
        mod = (p + "in_proj_qkv").replace("base_model.model.", "", 1)
        rank_pattern[mod] = r_tot
        stats["fused_qkv"] += 1
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    save_file(out, str(dst / "adapter_model.safetensors"), metadata={"format": "pt"})
    mods = sorted({k.rsplit(".lora_", 1)[0].split(".")[-1] for k in out})
    cfg.update({"target_modules": mods, "rank_pattern": rank_pattern})
    # alpha must scale with rank for the fused module to keep scaling == alpha/r of the originals
    base_scale = float(cfg.get("lora_alpha", cfg["r"])) / float(cfg["r"])
    cfg["alpha_pattern"] = {m: base_scale * r for m, r in rank_pattern.items()} if rank_pattern else {}
    (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    for extra in ("merge_recipe.json", "TINKER_SOURCE.txt", "README.md"):
        if (src / extra).exists():
            shutil.copy(src / extra, dst / extra)
    stats["n_tensors_out"] = len(out)
    stats["target_modules"] = mods
    return stats


if __name__ == "__main__":
    s = remap(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps(s, indent=1))
