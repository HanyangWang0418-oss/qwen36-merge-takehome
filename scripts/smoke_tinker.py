#!/usr/bin/env python
"""End-to-end smoke test of the Tinker training path on the real base model:
list server capabilities, create a rank-8 LoRA client on Qwen/Qwen3.6-35B-A3B,
run 3 forward_backward/optim steps on toy chat examples, save a sampler
checkpoint, sample once, and print the cost-relevant counters. Takes a minute;
costs cents. Records everything to logs/smoke_<date>/.

    set -a; . ./.env; set +a; uv run python scripts/smoke_tinker.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = "Qwen/Qwen3.6-35B-A3B"
REV = json.loads((ROOT / "data/manifests/base_model.json").read_text())["hf_revision"]

TOY = [
    [{"role": "user", "content": "Rename every .txt file in the current directory to .md."},
     {"role": "assistant", "content": "<think>\nUse a for loop with parameter expansion.\n</think>\n\n```bash\nfor f in *.txt; do mv \"$f\" \"${f%.txt}.md\"; done\n```"}],
    [{"role": "user", "content": "What is 17 * 23?"},
     {"role": "assistant", "content": "<think>\n17*23 = 17*20 + 17*3 = 340 + 51 = 391.\n</think>\n\n391"}],
    [{"role": "user", "content": "Show the last 20 lines of app.log and follow new output."},
     {"role": "assistant", "content": "<think>\ntail -n 20 -f.\n</think>\n\n```bash\ntail -n 20 -f app.log\n```"}],
    [{"role": "user", "content": "Git: undo the last commit but keep the changes staged."},
     {"role": "assistant", "content": "<think>\nsoft reset.\n</think>\n\n```bash\ngit reset --soft HEAD~1\n```"}],
]


def _to_list(x):
    for attr in ("to_numpy", "tolist"):
        if hasattr(x, attr):
            v = getattr(x, attr)()
            return v.tolist() if hasattr(v, "tolist") else list(v)
    if hasattr(x, "data"):
        return list(x.data)
    return list(x)


def main() -> int:
    import tinker
    from tinker import types
    from transformers import AutoTokenizer

    from src.data.chat_format import render_example

    log_dir = ROOT / "logs" / f"smoke_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True)
    rec: dict = {"base": BASE, "hf_revision": REV, "started_at": datetime.now(timezone.utc).isoformat()}

    service = tinker.ServiceClient()
    caps = service.get_server_capabilities()
    models = [m.model_name for m in caps.supported_models]
    rec["n_supported_models"] = len(models)
    rec["qwen36_models"] = [m for m in models if "Qwen3.6" in m]
    print("Qwen3.6 on Tinker:", rec["qwen36_models"])
    assert BASE in models, f"{BASE} not in Tinker model list"

    tok = AutoTokenizer.from_pretrained(BASE, revision=REV)
    data, raw_w = [], []
    for msgs in TOY:
        tokens, weights = render_example(tok, msgs, assistant_only=True)
        data.append(tinker.Datum(model_input=tinker.ModelInput.from_ints(tokens[:-1]),
                                 loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights[1:]}))
        raw_w.append(weights[1:])
    rec["toy_tokens"] = [len(w) + 1 for w in raw_w]
    rec["toy_loss_tokens"] = [int(sum(w)) for w in raw_w]
    print("tokens per example", rec["toy_tokens"], "loss tokens", rec["toy_loss_tokens"])

    t0 = time.time()
    client = service.create_lora_training_client(base_model=BASE, rank=8, seed=0)
    rec["create_client_s"] = round(time.time() - t0, 1)
    losses = []
    for step in range(3):
        t1 = time.time()
        fb = client.forward_backward(data, loss_fn="cross_entropy")
        opt = client.optim_step(types.AdamParams(learning_rate=1e-4))
        out = fb.result()
        opt.result()
        total, n = 0.0, 0
        for o, w in zip(out.loss_fn_outputs, raw_w):
            lp = _to_list(o["logprobs"])
            total += -sum(l * wi for l, wi in zip(lp, w)); n += sum(w)
        losses.append(total / n)
        print(f"step {step} loss {losses[-1]:.4f} ({time.time() - t1:.1f}s)")
    rec["losses"] = losses
    name = f"smoke_{log_dir.name}"
    ck = client.save_weights_for_sampler(name=name).result()
    rec["sampler_checkpoint"] = getattr(ck, "path", str(ck))
    st = client.save_state(name=name).result()
    rec["state_checkpoint"] = getattr(st, "path", str(st))
    print("checkpoints:", rec["sampler_checkpoint"], rec["state_checkpoint"])

    sampler = service.create_sampling_client(model_path=rec["sampler_checkpoint"])
    prompt_text = tok.apply_chat_template([{"role": "user", "content": "List files sorted by size, largest first."}],
                                          add_generation_prompt=True, tokenize=False)
    prompt = tok(prompt_text, add_special_tokens=False)["input_ids"]
    res = sampler.sample(prompt=tinker.ModelInput.from_ints(prompt),
                         sampling_params=types.SamplingParams(max_tokens=200, temperature=0.7, top_p=0.95),
                         num_samples=1).result()
    text = tok.decode(res.sequences[0].tokens)
    rec["sample"] = text
    print("sample:", text[:400])
    rec["finished_at"] = datetime.now(timezone.utc).isoformat()
    (log_dir / "smoke.json").write_text(json.dumps(rec, indent=2))
    print("wrote", log_dir / "smoke.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
