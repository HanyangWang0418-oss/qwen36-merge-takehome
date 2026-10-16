"""Unit tests for src/merge/merge_adapters.py on tiny random adapters."""
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
MODULES = [f"base_model.model.model.layers.{l}.{m}" for l in range(4) for m in
           ("self_attn.q_proj", "mlp.experts.0.gate_proj")]
SHAPES = {"self_attn.q_proj": (64, 32), "mlp.experts.0.gate_proj": (16, 32)}


def make_adapter(path: Path, r: int, alpha: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    t = {}
    for m in MODULES:
        out_dim, in_dim = SHAPES[m.split(".", 5)[-1]]
        t[f"{m}.lora_A.weight"] = torch.randn(r, in_dim, generator=g)
        t[f"{m}.lora_B.weight"] = torch.randn(out_dim, r, generator=g)
    path.mkdir(parents=True)
    save_file(t, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "Qwen/Qwen3.6-35B-A3B", "r": r, "lora_alpha": alpha,
        "target_modules": ["q_proj", "gate_proj"], "peft_type": "LORA"}))


def dense(path: Path) -> dict[str, torch.Tensor]:
    cfg = json.loads((path / "adapter_config.json").read_text())
    s = cfg["lora_alpha"] / cfg["r"]
    t = load_file(str(path / "adapter_model.safetensors"))
    return {m: s * (t[f"{m}.lora_B.weight"].float() @ t[f"{m}.lora_A.weight"].float()) for m in MODULES}


def run(cfg: dict, tmp: Path) -> Path:
    cfg_path = tmp / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp / "out"
    subprocess.run([sys.executable, str(ROOT / "src/merge/merge_adapters.py"), str(cfg_path), "--out", str(out)],
                   check=True, capture_output=True)
    return out


def test_task_arithmetic_is_exact(tmp_path):
    a1, a2 = tmp_path / "a1", tmp_path / "a2"
    make_adapter(a1, 4, 8, 1)
    make_adapter(a2, 8, 8, 2)
    out = run({"method": "task_arithmetic", "experts": [{"path": str(a1), "lambda": 0.5}, {"path": str(a2), "lambda": 1.0}]}, tmp_path)
    d1, d2, dm = dense(a1), dense(a2), dense(out)
    for m in MODULES:
        assert torch.allclose(dm[m], 0.5 * d1[m] + d2[m], atol=0.1, rtol=0.02)  # bf16 storage
    assert json.loads((out / "adapter_config.json").read_text())["r"] == 12


def test_layer_scope(tmp_path):
    a1, a2 = tmp_path / "a1", tmp_path / "a2"
    make_adapter(a1, 4, 4, 1)
    make_adapter(a2, 4, 4, 2)
    out = run({"method": "task_arithmetic", "experts": [{"path": str(a1), "lambda": 1.0, "layers": "2-3"}, {"path": str(a2), "lambda": 1.0}]}, tmp_path)
    d1, d2, dm = dense(a1), dense(a2), dense(out)
    for m in MODULES:
        layer = int(m.split(".layers.")[1].split(".")[0])
        expect = d1[m] + d2[m] if layer >= 2 else d2[m]
        assert torch.allclose(dm[m], expect, atol=0.1, rtol=0.02)


def test_ties_single_adapter_full_density_roundtrips(tmp_path):
    a1 = tmp_path / "a1"
    make_adapter(a1, 4, 8, 3)
    out = run({"method": "ties", "density": 1.0, "out_rank": 4, "experts": [{"path": str(a1), "lambda": 1.0}]}, tmp_path)
    d1, dm = dense(a1), dense(out)
    for m in MODULES:
        assert torch.allclose(dm[m], d1[m], atol=0.1, rtol=0.02)
    rec = json.loads((out / "merge_recipe.json").read_text())
    assert rec["svd_rel_error"]["max"] < 1e-3


def test_ties_sign_election(tmp_path):
    # two identical adapters with opposite sign on one, ties should elect and keep magnitude
    a1, a2 = tmp_path / "a1", tmp_path / "a2"
    make_adapter(a1, 4, 4, 5)
    make_adapter(a2, 4, 4, 5)
    t = load_file(str(a2 / "adapter_model.safetensors"))
    t = {k: (-v if "lora_B" in k else v) for k, v in t.items()}
    save_file(t, str(a2 / "adapter_model.safetensors"))
    out = run({"method": "ties", "density": 1.0, "out_rank": 4,
               "experts": [{"path": str(a1), "lambda": 1.0}, {"path": str(a2), "lambda": 0.5}]}, tmp_path)
    d1, dm = dense(a1), dense(out)
    for m in MODULES:  # 1.0*d vs -0.5*d: sign of d wins, mean over agreeing = d
        assert torch.allclose(dm[m], d1[m], atol=0.1, rtol=0.02)
