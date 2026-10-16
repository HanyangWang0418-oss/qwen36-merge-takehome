import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import load_file, save_file  # noqa: E402

from src.merge.remap_keys import remap  # noqa: E402


def test_qkv_fusion_is_exact_and_unembed_renamed(tmp_path):
    g = torch.Generator().manual_seed(0)
    r, din = 4, 16
    p = "base_model.model.model.layers.0.linear_attn."
    t = {}
    outs = {"in_proj_q": 8, "in_proj_k": 8, "in_proj_v": 12}
    for n, o in outs.items():
        t[f"{p}{n}.lora_A.weight"] = torch.randn(r, din, generator=g)
        t[f"{p}{n}.lora_B.weight"] = torch.randn(o, r, generator=g)
    t[f"{p}in_proj_z.lora_A.weight"] = torch.randn(r, din, generator=g)
    t[f"{p}in_proj_z.lora_B.weight"] = torch.randn(12, r, generator=g)
    t["base_model.model.model.unembed_tokens.lora_A.weight"] = torch.randn(r, din, generator=g)
    t["base_model.model.model.unembed_tokens.lora_B.weight"] = torch.randn(32, r, generator=g)
    t["base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight"] = torch.randn(1, r, din, generator=g)
    t["base_model.model.model.layers.0.mlp.experts.w1.lora_B.weight"] = torch.randn(3, 6, r, generator=g)
    src = tmp_path / "src"; src.mkdir()
    save_file(t, str(src / "adapter_model.safetensors"))
    (src / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "x", "r": r, "lora_alpha": 2 * r, "target_modules": "all-linear", "peft_type": "LORA"}))
    stats = remap(src, tmp_path / "dst")
    o = load_file(str(tmp_path / "dst" / "adapter_model.safetensors"))
    cfg = json.loads((tmp_path / "dst" / "adapter_config.json").read_text())
    # fused qkv reproduces the stacked per-projection deltas exactly
    A, B = o[f"{p}in_proj_qkv.lora_A.weight"], o[f"{p}in_proj_qkv.lora_B.weight"]
    fused = B @ A
    expect = torch.cat([t[f"{p}{n}.lora_B.weight"] @ t[f"{p}{n}.lora_A.weight"] for n in outs], 0)
    assert torch.allclose(fused, expect, atol=1e-5)
    assert A.shape[0] == 3 * r and "in_proj_q.lora_A.weight" not in " ".join(o)
    assert "base_model.model.model.lm_head.lora_B.weight" in o and not any("unembed" in k for k in o)
    assert cfg["rank_pattern"] == {"model.layers.0.linear_attn.in_proj_qkv": 3 * r}
    assert cfg["alpha_pattern"] == {"model.layers.0.linear_attn.in_proj_qkv": 2.0 * 3 * r}
    assert set(cfg["target_modules"]) == {"in_proj_qkv", "in_proj_z", "lm_head", "w1"}
    assert stats["fused_qkv"] == 1 and stats["renamed_unembed"] == 2
