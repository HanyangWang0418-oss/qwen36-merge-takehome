"""Unit tests for src/merge/merge_adapters.py on tiny random adapters."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # merge path only; install with `uv sync --extra merge`
import yaml
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
MODULES = [f"base_model.model.model.layers.{l}.{m}" for l in range(4) for m in
           ("self_attn.q_proj", "mlp.shared_expert.gate_proj", "mlp.experts.w1", "mlp.experts.w2")]
SHAPES = {"self_attn.q_proj": (64, 32), "mlp.shared_expert.gate_proj": (16, 32)}
E = 5  # experts in the 3D fixture


def make_adapter(path: Path, r: int, alpha: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    t = {}
    for m in MODULES:
        kind = m.split(".", 5)[-1]
        if kind == "mlp.experts.w1":    # shared A [1,r,in], per-expert B [E,out,r]
            t[f"{m}.lora_A.weight"] = torch.randn(1, r, 32, generator=g)
            t[f"{m}.lora_B.weight"] = torch.randn(E, 8, r, generator=g)
        elif kind == "mlp.experts.w2":  # per-expert A [E,r,in], shared B [1,out,r]
            t[f"{m}.lora_A.weight"] = torch.randn(E, r, 8, generator=g)
            t[f"{m}.lora_B.weight"] = torch.randn(1, 32, r, generator=g)
        else:
            out_dim, in_dim = SHAPES[kind]
            t[f"{m}.lora_A.weight"] = torch.randn(r, in_dim, generator=g)
            t[f"{m}.lora_B.weight"] = torch.randn(out_dim, r, generator=g)
    path.mkdir(parents=True)
    save_file(t, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "Qwen/Qwen3.6-35B-A3B", "r": r, "lora_alpha": alpha,
        "target_modules": ["q_proj", "gate_proj"], "peft_type": "LORA"}))


def dense(path: Path) -> dict[str, torch.Tensor]:
    """Dense deltas; 3D modules are expanded per expert to [E, out, in]."""
    cfg = json.loads((path / "adapter_config.json").read_text())
    s = cfg["lora_alpha"] / cfg["r"]
    t = load_file(str(path / "adapter_model.safetensors"))
    out = {}
    for m in MODULES:
        A, B = t[f"{m}.lora_A.weight"].float(), t[f"{m}.lora_B.weight"].float()
        if A.dim() == 3:
            n = max(A.shape[0], B.shape[0])
            out[m] = torch.stack([s * (B[e if B.shape[0] > 1 else 0] @ A[e if A.shape[0] > 1 else 0]) for e in range(n)])
        else:
            out[m] = s * (B @ A)
    return out


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
        assert torch.allclose(dm[m], 0.5 * d1[m] + d2[m], atol=1e-3, rtol=1e-3)
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
        assert torch.allclose(dm[m], expect, atol=1e-3, rtol=1e-3)


def test_ties_single_adapter_full_density_roundtrips(tmp_path):
    a1 = tmp_path / "a1"
    make_adapter(a1, 4, 8, 3)
    out = run({"method": "ties", "density": 1.0, "out_rank": 4, "experts": [{"path": str(a1), "lambda": 1.0}]}, tmp_path)
    d1, dm = dense(a1), dense(out)
    for m in MODULES:
        assert torch.allclose(dm[m], d1[m], atol=1e-3, rtol=1e-3)
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
        assert torch.allclose(dm[m], d1[m], atol=1e-3, rtol=1e-3)
