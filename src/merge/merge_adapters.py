#!/usr/bin/env python
"""Merge several PEFT LoRA adapters (same base, same target modules) into one
PEFT LoRA adapter. No base weights are needed, so this runs on a CPU box.

For every adapted module m, expert i contributes a dense delta
    D_i[m] = (alpha_i / r_i) * B_i[m] @ A_i[m].
Methods (configs/merge/*.yaml):
  task_arithmetic : D = sum_i lambda_i * D_i            (Ilharco et al. 2023)
  ties            : trim each D_i to top-`density` magnitude, elect sign by
                    summed magnitude, average agreeing entries, scale by lambda
                    (Yadav et al. 2023)
  dare_ties / dare_ta : Bernoulli-drop each D_i entry with p = 1-density and
                    rescale survivors by 1/density, then ties / task_arithmetic
                    (Yu et al. 2024)
Output is re-expressed as a LoRA pair (A, B):
  task_arithmetic -> exact, by row/column concatenation (rank = sum of ranks)
  ties / dare_*   -> truncated SVD to `out_rank`; the relative Frobenius
                    reconstruction error is logged per module and summarised,
                    so the approximation is auditable.
Per-expert `layers` (e.g. "0-19") restricts an expert's contribution to a
layer range, for the layer-scope ablation. Modules not adapted by any kept
expert are omitted (base weights are always retained by construction).

    uv run python src/merge/merge_adapters.py configs/merge/ta_l0.7.yaml
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


def parse_layers(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    out: set[int] = set()
    for part in str(spec).split(","):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


class Adapter:
    def __init__(self, path: Path, lam: float, layers: str | None):
        self.path = path
        self.cfg = json.loads((path / "adapter_config.json").read_text())
        self.r = int(self.cfg["r"])
        self.alpha = float(self.cfg.get("lora_alpha", self.r))
        self.scale = self.alpha / self.r
        self.lam = lam
        self.layers = parse_layers(layers)
        self.f = safe_open(str(path / "adapter_model.safetensors"), framework="pt")
        keys = list(self.f.keys())
        self.modules = sorted({k.rsplit(".lora_", 1)[0] for k in keys if ".lora_A" in k})

    def active(self, module: str) -> bool:
        if self.layers is None:
            return True
        m = LAYER_RE.search(module)
        return m is not None and int(m.group(1)) in self.layers

    def ab(self, module: str) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.f.get_tensor(f"{module}.lora_A.weight").float()
        b = self.f.get_tensor(f"{module}.lora_B.weight").float()
        return a, b

    def delta(self, module: str) -> torch.Tensor:
        a, b = self.ab(module)
        return self.scale * (b @ a)


# ------------------------------------------------------------------ dense ops
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
    """Return (A, B) with B@A ≈ d, A: rank×in, B: out×rank, plus relative Frobenius error."""
    u, s, vh = torch.linalg.svd(d, full_matrices=False)
    k = min(rank, s.numel())
    b = u[:, :k] * s[:k]
    a = vh[:k]
    err = (torch.linalg.norm(d - b @ a) / torch.linalg.norm(d).clamp(min=1e-12)).item()
    return a.contiguous(), b.contiguous(), err


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default=None, help="override output dir")
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
    skipped = 0
    for mi, m in enumerate(modules):
        contrib = [a for a in adapters if m in a.modules and a.active(m)]
        if not contrib:
            skipped += 1
            continue
        if method == "task_arithmetic":
            # exact: B_merged = [lam_i * scale_i * B_i], A_merged = [A_i] stacked on rank dim
            As, Bs = [], []
            for a in contrib:
                A, B = a.ab(m)
                As.append(A)
                Bs.append(a.lam * a.scale * B)
            A_m = torch.cat(As, 0)
            B_m = torch.cat(Bs, 1)
            err = 0.0
        else:
            deltas = [a.delta(m) for a in contrib]
            lams = [a.lam for a in contrib]
            if method.startswith("dare"):
                deltas = [dare(d, density, gen) for d in deltas]
            if method in ("ties", "dare_ties"):
                D = ties(deltas, lams, density if method == "ties" else 1.0)
            elif method == "dare_ta":
                D = sum(l * d for l, d in zip(lams, deltas))
            else:
                raise ValueError(method)
            A_m, B_m, err = svd_lowrank(D, out_rank)
        # store with scaling folded in: alpha == r so PEFT scaling is 1.0
        tensors[f"{m}.lora_A.weight"] = A_m.to(torch.bfloat16)
        tensors[f"{m}.lora_B.weight"] = B_m.to(torch.bfloat16)
        ranks_out.add(A_m.shape[0])
        errors[m] = err
        lm = LAYER_RE.search(m)
        per_layer_err[lm.group(1) if lm else "other"].append(err)
        if mi % 500 == 0:
            print(f"{mi}/{len(modules)} {m} rank={A_m.shape[0]} err={err:.4f}")

    if len(ranks_out) != 1:
        raise RuntimeError(f"PEFT needs one rank per adapter, got {sorted(ranks_out)}; set out_rank or equal expert ranks")
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
        "experts": [{"path": str(a.path.relative_to(ROOT)) if a.path.is_relative_to(ROOT) else str(a.path), "r": a.r, "alpha": a.alpha,
                     "lambda": a.lam, "layers": sorted(a.layers) if a.layers else "all"} for a in adapters],
        "n_modules_merged": len(errors),
        "n_modules_skipped": skipped,
        "svd_rel_error": {"mean": errs.mean().item(), "max": errs.max().item(), "p90": errs.quantile(0.9).item()},
        "svd_rel_error_by_layer": {k: sum(v) / len(v) for k, v in sorted(per_layer_err.items(), key=lambda kv: kv[0])},
        "conflict_handling": {
            "task_arithmetic": "no conflict resolution; opposing deltas cancel additively",
            "ties": "trim to top-density magnitude, sign election by summed magnitude, disjoint mean of agreeing entries",
            "dare_ties": "random drop + rescale before TIES sign election",
            "dare_ta": "random drop + rescale, then additive",
        }[method],
    }
    (out / "merge_recipe.json").write_text(json.dumps(recipe, indent=2))
    print(json.dumps({k: recipe[k] for k in ("method", "out_rank", "n_modules_merged", "svd_rel_error")}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
