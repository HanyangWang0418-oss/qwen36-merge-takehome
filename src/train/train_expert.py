#!/usr/bin/env python
"""Supervised LoRA fine-tuning of one expert on Tinker.

Reads configs/experts/<name>.yaml, streams data/processed/<name>_train.jsonl
through Tinker's forward_backward / optim_step loop, evaluates dev loss every
`save_every_steps`, saves a sampler checkpoint at each eval point, and writes
logs/<run_id>/{metadata.json, metrics.jsonl, train.log, resources.csv}.

The structure follows the tinker-cookbook supervised loop; the SDK surface
(ServiceClient, LoraConfig, Datum, forward_backward, optim_step,
save_weights_for_sampler) must be re-checked against the pinned `tinker`
version in requirements.lock before the first real run.

Usage:
    TINKER_API_KEY=... uv run python src/train/train_expert.py configs/experts/e1_code_repair.yaml
    uv run python src/train/train_expert.py configs/experts/e1_code_repair.yaml --dry-run   # tokenizes only
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.chat_format import render_example  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def make_run_id(cfg: dict) -> str:
    return cfg.get("run_id") or "{}_r{}_s{}_{}".format(
        cfg["expert_name"], cfg["lora"]["rank"], cfg["seed"], datetime.now(timezone.utc).strftime("%Y%m%d")
    )


def tokenize_split(tokenizer, rows: list[dict], max_len: int, loss_on_assistant_only: bool) -> list[dict]:
    """Returns list of {"tokens": [...], "weights": [...]} with weight 1 on assistant tokens."""
    out, dropped = [], 0
    for row in rows:
        tokens, weights = render_example(tokenizer, row["messages"], loss_on_assistant_only)
        if len(tokens) > max_len:
            dropped += 1
            continue
        out.append({"tokens": tokens, "weights": weights})
    print(f"tokenized {len(out)} examples, dropped {dropped} over max_seq_len={max_len}")
    return out


def to_datum(tinker, ex: dict):
    tokens = ex["tokens"]
    # next-token prediction: input = tokens[:-1], target = tokens[1:], weight aligned to target
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "weights": ex["weights"][1:]},
    )


def mean_loss(result, batch: list[dict]) -> float:
    # Tinker returns per-token logprobs in result.loss_fn_outputs; normalise by weighted tokens.
    total, n = 0.0, 0
    for out, ex in zip(result.loss_fn_outputs, batch):
        lp = out["logprobs"].tolist()  # tinker.types.TensorData
        w = ex["weights"][1:]
        total += -sum(l * wi for l, wi in zip(lp, w))
        n += sum(w)
    return total / max(n, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true", help="tokenize and report sizes, no Tinker calls")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_id = make_run_id(cfg)
    log_dir = ROOT / "logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cfg["seed"])

    from transformers import AutoTokenizer  # local import: heavy

    tokenizer = AutoTokenizer.from_pretrained(cfg["base"]["hf_model"], revision=cfg["base"]["hf_revision"] if cfg["base"]["hf_revision"] != "LOCK" else None)

    train_rows = load_jsonl(ROOT / cfg["data"]["train_file"])
    dev_rows = load_jsonl(ROOT / cfg["data"]["dev_file"])
    if args.limit:
        train_rows, dev_rows = train_rows[: args.limit], dev_rows[: max(8, args.limit // 10)]
    assistant_only = cfg["data"]["loss_on"] == "assistant_only"
    train = tokenize_split(tokenizer, train_rows, cfg["optim"]["max_seq_len"], assistant_only)
    dev = tokenize_split(tokenizer, dev_rows, cfg["optim"]["max_seq_len"], assistant_only)

    bs = cfg["optim"]["batch_size_examples"]
    steps_per_epoch = math.ceil(len(train) / bs)
    max_steps = cfg["optim"]["max_steps"] or steps_per_epoch * cfg["optim"]["epochs"]
    n_train_tokens = sum(len(e["tokens"]) for e in train)
    n_loss_tokens = sum(sum(e["weights"]) for e in train)

    metadata = {
        "run_id": run_id,
        "config": cfg,
        "config_path": args.config,
        "n_train_examples": len(train),
        "n_dev_examples": len(dev),
        "n_train_tokens": n_train_tokens,
        "n_loss_tokens": n_loss_tokens,
        "steps_per_epoch": steps_per_epoch,
        "max_steps": max_steps,
        "launch_command": " ".join(sys.argv),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "hardware": "Tinker managed cluster (GPU type not exposed by the API)",
    }
    (log_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({k: v for k, v in metadata.items() if k != "config"}, indent=2))
    if args.dry_run:
        return 0

    import tinker  # noqa: F401  (requires TINKER_API_KEY)
    from tinker import types

    service = tinker.ServiceClient()
    lora_flags = {"train_mlp": True, "train_attn": True, "train_unembed": True, **cfg["lora"].get("tinker_flags", {})}
    client = service.create_lora_training_client(
        base_model=cfg["base"]["tinker_model"],
        rank=cfg["lora"]["rank"],
        seed=cfg["seed"],
        user_metadata={"run_id": run_id, "expert": cfg["expert_name"]},
        **lora_flags,
    )
    metadata["lora_flags"] = lora_flags
    lr0 = cfg["optim"]["learning_rate"]
    warmup = cfg["optim"]["warmup_steps"]

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr0 * (step + 1) / warmup
        if cfg["optim"]["lr_schedule"] == "cosine":
            p = (step - warmup) / max(1, max_steps - warmup)
            return lr0 * 0.5 * (1 + math.cos(math.pi * p))
        return lr0

    metrics = (log_dir / "metrics.jsonl").open("a")
    resources = (log_dir / "resources.csv").open("a")
    resources.write("step,wallclock_s,train_tokens_cum,tokens_per_s\n")
    t0 = time.time()
    tokens_cum = 0
    best_dev, best_ckpt = float("inf"), None
    order = list(range(len(train)))

    def eval_dev() -> float:
        losses = []
        for i in range(0, len(dev), bs):
            batch = dev[i : i + bs]
            fut = client.forward([to_datum(tinker, e) for e in batch], loss_fn="cross_entropy")
            losses.append(mean_loss(fut.result(), batch) * len(batch))
        return sum(losses) / len(dev)

    step = 0
    while step < max_steps:
        random.shuffle(order)
        for i in range(0, len(order), bs):
            if step >= max_steps:
                break
            batch = [train[j] for j in order[i : i + bs]]
            fb = client.forward_backward([to_datum(tinker, e) for e in batch], loss_fn="cross_entropy")
            lr = lr_at(step)
            opt = client.optim_step(types.AdamParams(learning_rate=lr, grad_clip_norm=cfg["optim"]["grad_clip"],
                                                     weight_decay=cfg["optim"].get("weight_decay", 0.0)))
            loss = mean_loss(fb.result(), batch)
            opt.result()
            tokens_cum += sum(len(e["tokens"]) for e in batch)
            step += 1
            elapsed = time.time() - t0
            rec = {"step": step, "train_loss": loss, "lr": lr, "wallclock_s": elapsed, "tokens_cum": tokens_cum}
            if step % cfg["checkpointing"]["save_every_steps"] == 0 or step == max_steps:
                dev_loss = eval_dev()
                name = f"{run_id}_step{step:05d}"
                client.save_state(name=name)
                ck = client.save_weights_for_sampler(name=name).result()
                rec.update({"dev_loss": dev_loss, "checkpoint": getattr(ck, "path", name)})
                if dev_loss < best_dev:
                    best_dev, best_ckpt = dev_loss, rec["checkpoint"]
            metrics.write(json.dumps(rec) + "\n")
            metrics.flush()
            resources.write(f"{step},{elapsed:.1f},{tokens_cum},{tokens_cum / max(elapsed, 1):.1f}\n")
            resources.flush()
            print(json.dumps(rec))

    metadata.update({"finished_at": datetime.now(timezone.utc).isoformat(), "best_dev_loss": best_dev,
                     "selected_checkpoint": best_ckpt, "selection_rule": cfg["checkpointing"]["select_by"]})
    (log_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"selected checkpoint by dev loss: {best_ckpt} ({best_dev:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
