#!/usr/bin/env python
"""Merge several PEFT LoRA adapters (same base, same target modules) into one PEFT LoRA adapter.
No base weights are needed, so this runs on a CPU box.

Tensor layouts handled (as exported by Tinker for Qwen3.6-35B-A3B):
  * dense 2D modules (attention, Gated-DeltaNet projections, shared expert, unembed):
        lora_A: [r, in]            lora_B: [out, r]
  * fused 3D MoE expert modules, one side shared across experts (broadcast dim 1):
        w1/w3:  lora_A: [1, r, in]  lora_B: [E, out, r]
        w2:     lora_A: [E, r, in]  lora_B: [1, out, r]
    expert e's delta is scale * B[e or 0] @ A[e or 0].

Methods (configs/merge/*.yaml):
  task_arithmetic : D = sum_i lambda_i D_i, exact as a LoRA pair by concatenation on the rank axis
                    (keeps the broadcast layout; rank = sum of ranks)
  ties            : trim to top-`density` magnitude per expert adapter, sign election by summed
                    magnitude, disjoint mean of agreeing entries (Yadav et al. 2023); re-expressed
                    by truncated SVD to `out_rank` per (module, expert); for 3D modules the result is
                    stored per expert on both sides ([E, r, in] / [E, out, r]), so the adapter grows
  dare_ties / dare_ta : Bernoulli-drop with p = 1-density, rescale by 1/density, then ties / sum
Per-expert `layers` (e.g. "20-39") restricts an expert adapter to a layer range (layer-scope ablation).
Output dtype follows the inputs (Tinker exports fp32). SVD relative errors are logged per module.

    uv run python src/merge/merge_adapters.py configs/merge/ta_l0.7.yaml [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def parse_layers(spec) -> set[int] | None:
    if not spec:
        return None
    out: set[int] = set()
    for part in str(spec).split(","):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


class Adapter:
    def __init__(self, path: Path, lam: float, layers):
        self.path = path
        self.cfg = json.loads((path / "adapter_config.json").read_text())
        self.r = int(self.cfg["r"])
        self.alpha = float(self.cfg.get("lora_alpha", self.r))
        self.scale = self.alpha / self.r
        self.lam = lam
        self.layers = parse_layers(layers)
        files = sorted(path.glob("adapter_model*.safetensors"))
        assert files, f"no adapter_model*.safetensors in {path}"
        self.handles = [safe_open(str(f), framework="pt") for f in files]
        self.key2h = {k: h for h in self.handles for k in h.keys()}
        self.modules = sorted({k.rsplit(".lora_", 1)[0] for k in self.key2h if ".lora_A" in k})

    def active(self, module: str) -> bool:
        if self.layers is None:
            return True
        m = LAYER_RE.search(module)
        return m is not None and int(m.group(1)) in self.layers

    def ab(self, module: str) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.key2h[f"{module}.lora_A.weight"].get_tensor(f"{module}.lora_A.weight")
        b = self.key2h[f"{module}.lora_B.weight"].get_tensor(f"{module}.lora_B.weight")
        return a, b

    def dtype(self, module: str) -> torch.dtype:
        return self.key2h[f"{module}.lora_A.weight"].get_tensor(f"{module}.lora_A.weight").dtype


# ------------------------------------------------------------------ dense ops (2D)
def dare(d: torch.Tensor, density: float, gen: torch.Generator) -> torch.Tensor:
    if density >= 1.0:
        return d
    mask = torch.rand(d.shape, generator=gen) < density
    return d * mask / density


def ties(deltas: list[torch.Tensor], lams: list[float], density: float) -> torch.Tensor:
    trimmed = []
    for d in deltas:
        if density < 1.0:
            k = max(1, int(density * d.numel()))
            thresh = d.abs().flatten().kthvalue(d.numel() - k + 1).values
            d = torch.where(d.abs() >= thresh, d, torch.zeros_like(d))
        trimmed.append(d)
    stacked = torch.stack([l * d for l, d in zip(lams, trimmed)])
    sign = torch.sign(stacked.sum(0))
    agree = (torch.sign(stacked) == sign) & (stacked != 0)
    num = (stacked * agree).sum(0)
    cnt = agree.sum(0).clamp(min=1)
    return num / cnt


def svd_lowrank(d: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """(A [rank,in], B [out,rank]) with B@A ≈ d, plus relative Frobenius error."""
    u, s, vh = torch.linalg.svd(d, full_matrices=False)
    k = min(rank, s.numel())
    b = u[:, :k] * s[:k]
    a = vh[:k]
    err = (torch.linalg.norm(d - b @ a) / torch.linalg.norm(d).clamp(min=1e-12)).item()
    return a.contiguous(), b.contiguous(), err


def expert_delta(a: torch.Tensor, b: torch.Tensor, e: int, scale: float) -> torch.Tensor:
    """Dense delta of expert e for a 3D module with broadcast on either side."""
    ae = a[e if a.shape[0] > 1 else 0].float()
    be = b[e if b.shape[0] > 1 else 0].float()
    return scale * (be @ ae)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    method = cfg["method"]
    density = float(cfg.get("density", 1.0))
    out_rank = int(cfg.get("out_rank", 64))
    seed = int(cfg.get("seed", 0))
    out = Path(args.out or (ROOT / cfg["output_dir"]))

    adapters = [Adapter(ROOT / e["path"] if not Path(e["path"]).is_absolute() else Path(e["path"]),
                        float(e.get("lambda", 1.0)), e.get("layers")) for e in cfg["experts"]]
    base = adapters[0].cfg["base_model_name_or_path"]
    for a in adapters:
        assert a.cfg["base_model_name_or_path"] == base, f"base mismatch: {a.path}"
    modules = sorted(set().union(*(set(a.modules) for a in adapters)))
    gen = torch.Generator().manual_seed(seed)

    tensors: dict[str, torch.Tensor] = {}
    errors: dict[str, float] = {}
    per_layer_err: dict[str, list[float]] = defaultdict(list)
    ranks_out: set[int] = set()
    skipped = n3d = 0
    for mi, m in enumerate(modules):
        contrib = [a for a in adapters if m in a.modules and a.active(m)]
        if not contrib:
            skipped += 1
            continue
        a0, b0 = contrib[0].ab(m)
        is3d = a0.dim() == 3
        dt = a0.dtype
        if method == "task_arithmetic":
            # exact: A_merged = cat_i A_i on the rank axis, B_merged = cat_i (lam_i*scale_i) B_i on the rank axis
            As, Bs = [], []
            for a in contrib:
                A, B = a.ab(m)
                As.append(A)
                Bs.append((a.lam * a.scale) * B.float())
            if is3d:
                A_m = torch.cat(As, 1)
                B_m = torch.cat(Bs, 2)
                r_out = A_m.shape[1]
            else:
                A_m = torch.cat(As, 0)
                B_m = torch.cat(Bs, 1)
                r_out = A_m.shape[0]
            err = 0.0
        else:
            lams = [a.lam for a in contrib]
            if is3d:
                E = max(a0.shape[0], b0.shape[0])
                A_list, B_list, errs = [], [], []
                for e in range(E):
                    deltas = [expert_delta(*a.ab(m), e, a.scale) for a in contrib]
                    if method.startswith("dare"):
                        deltas = [dare(d, density, gen) for d in deltas]
                    D = ties(deltas, lams, density if method == "ties" else 1.0) if method in ("ties", "dare_ties") else sum(l * d for l, d in zip(lams, deltas))
                    Ae, Be, ee = svd_lowrank(D, out_rank)
                    A_list.append(Ae); B_list.append(Be); errs.append(ee)
                A_m = torch.stack(A_list)      # [E, r, in]
                B_m = torch.stack(B_list)      # [E, out, r]
                r_out = out_rank
                err = float(sum(errs) / len(errs))
            else:
                deltas = [a.scale * (a.ab(m)[1].float() @ a.ab(m)[0].float()) for a in contrib]
                if method.startswith("dare"):
                    deltas = [dare(d, density, gen) for d in deltas]
                if method in ("ties", "dare_ties"):
                    D = ties(deltas, lams, density if method == "ties" else 1.0)
                elif method == "dare_ta":
                    D = sum(l * d for l, d in zip(lams, deltas))
                else:
                    raise ValueError(method)
                A_m, B_m, err = svd_lowrank(D, out_rank)
                r_out = out_rank
        n3d += int(is3d)
        # scaling folded into B: alpha == r so PEFT scaling is 1.0
        tensors[f"{m}.lora_A.weight"] = A_m.to(dt).contiguous()
        tensors[f"{m}.lora_B.weight"] = B_m.to(dt).contiguous()
        ranks_out.add(int(r_out))
        errors[m] = err
        lm = LAYER_RE.search(m)
        per_layer_err[lm.group(1) if lm else "other"].append(err)
        if mi % 100 == 0:
            print(f"{mi}/{len(modules)} {m.split('.model.')[-1]} 3d={is3d} rank={r_out} err={err:.4f}", flush=True)

    if len(ranks_out) != 1:
        raise RuntimeError(f"PEFT needs one rank per adapter, got {sorted(ranks_out)}")
    r = ranks_out.pop()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    save_file(tensors, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    acfg = dict(adapters[0].cfg)
    acfg.update({"r": r, "lora_alpha": r, "lora_dropout": 0.0, "rank_pattern": {}, "alpha_pattern": {}})
    (out / "adapter_config.json").write_text(json.dumps(acfg, indent=2))
    errs = torch.tensor(list(errors.values())) if errors else torch.zeros(1)
    recipe = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": args.config,
        "config": cfg,
        "base_model": base,
        "method": method,
        "density": density,
        "out_rank": r,
        "experts": [{"path": str(a.path), "r": a.r, "alpha": a.alpha, "lambda": a.lam,
                     "layers": sorted(a.layers) if a.layers else "all"} for a in adapters],
        "n_modules_merged": len(errors),
        "n_modules_3d": n3d,
        "n_modules_skipped": skipped,
        "svd_rel_error": {"mean": errs.mean().item(), "max": errs.max().item(), "p90": errs.quantile(0.9).item()},
        "svd_rel_error_by_layer": {k: sum(v) / len(v) for k, v in sorted(per_layer_err.items(), key=lambda kv: kv[0])},
        "conflict_handling": {
            "task_arithmetic": "no conflict resolution; opposing deltas cancel additively",
            "ties": "trim to top-density magnitude, sign election by summed magnitude, disjoint mean of agreeing entries",
            "dare_ties": "random drop + rescale before TIES sign election",
            "dare_ta": "random drop + rescale, then additive",
        }[method],
        "layout_note": "3D MoE modules: task_arithmetic keeps the shared-side broadcast layout; SVD-based methods store per-expert A and B",
    }
    (out / "merge_recipe.json").write_text(json.dumps(recipe, indent=2))
    print(json.dumps({k: recipe[k] for k in ("method", "out_rank", "n_modules_merged", "n_modules_3d", "svd_rel_error")}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
