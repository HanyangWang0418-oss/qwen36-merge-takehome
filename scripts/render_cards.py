#!/usr/bin/env python
"""Render models/cards/TEMPLATE_expert.md for each expert from logs/<run>/metadata.json, the data
manifests and configs, write models/cards/<expert>.md, and (with --upload) push README.md to the HF repo.

    HF_TOKEN=... uv run python scripts/render_cards.py [--upload]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"e1_code_repair": "e1_code_repair_r32_s42_20261016", "e2_terminal_agent": "e2_terminal_agent_r32_s42_20261016", "e3_reasoning": "e3_reasoning_r32_s42_third"}
ROLES = {"e1_code_repair": ("code repair (issue→patch agent trajectories)", "SWE-bench Pro v2-hard", "code,swe-agent"),
         "e2_terminal_agent": ("terminal / research-execution agent", "Terminal-Bench 2.0, ASI-Bench", "terminal,agent"),
         "e3_reasoning": ("long-form STEM reasoning", "HLE (text-only)", "reasoning,math")}
CODE_REPO = "https://github.com/HanyangWang0418-oss/qwen36-merge-takehome"
TARGET_MODULES = ("all linear projections: self_attn q/k/v/o, Gated-DeltaNet in_proj_q/k/v/z + out_proj, shared_expert gate/up/down, "
                  "routed experts w1/w2/w3 (fused 3D, one side shared across 256 experts), unembed (lm_head); router not adapted")


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--upload", action="store_true"); a = ap.parse_args()
    tpl = (ROOT / "models/cards/TEMPLATE_expert.md").read_text()
    hf = json.loads((ROOT / "data/manifests/hf_models.json").read_text())
    base = json.loads((ROOT / "data/manifests/base_model.json").read_text())
    for expert, run in RUNS.items():
        meta = json.loads((ROOT / f"logs/{run}/metadata.json").read_text())
        cfg = meta["config"]
        man = json.loads((ROOT / f"data/manifests/{expert}_manifest.json").read_text())
        rows = [f"| {s['name']} | {s.get('hf_repo') or s.get('local_file')} | {s.get('license')} | {s.get('stats', {}).get('kept', 0)} kept of {s.get('stats', {}).get('raw', 0)} fetched |"
                for s in man["sources"] if s.get("stats")]
        data_table = "| source | location | licence | rows |\n|---|---|---|---|\n" + "\n".join(rows)
        role, target, tags = ROLES[expert]
        repo = hf[expert]["repo_id"]
        steps = meta["max_steps"]
        card = (tpl.replace("{{REPO_NAME}}", repo.split("/")[-1]).replace("{{REPO_ID}}", repo).replace("{{RANK}}", str(cfg["lora"]["rank"]))
                .replace("{{ALPHA}}", "32 (scaling 1.0; exported adapter_config lora_alpha=32)").replace("{{BASE_REV}}", base["hf_revision"])
                .replace("{{ROLE}}", role).replace("{{TARGET}}", target).replace("{{TAGS}}", tags).replace("{{TARGET_MODULES}}", TARGET_MODULES)
                .replace("{{DATA}}", cfg["data_description"]).replace("{{N_TRAIN}}", str(meta["n_train_examples"])).replace("{{LOSS_TOKENS}}", f"{int(meta['n_loss_tokens']):,}")
                .replace("{{STEPS}}", str(steps)).replace("{{BS}}", str(cfg["optim"]["batch_size_examples"])).replace("{{LR}}", str(cfg["optim"]["learning_rate"]))
                .replace("{{WARMUP}}", str(cfg["optim"]["warmup_steps"])).replace("{{SEQ}}", str(cfg["optim"]["max_seq_len"])).replace("{{SEED}}", str(cfg["seed"]))
                .replace("{{DEV_LOSS}}", f"{meta['best_dev_loss']:.4f} at step {steps}").replace("{{TINKER_PATH}}", meta["selected_checkpoint"])
                .replace("{{DATA_TABLE}}", data_table).replace("{{RESULTS}}", "pending (evaluation not yet run)").replace("{{SHORT}}", expert)
                .replace("{{CODE_REPO}}", CODE_REPO))
        if meta.get("resumes"):
            card += f"\n\nNote: training was resumed once from the step-{meta['resumes'][0]['start_step']} optimizer state after a lost Tinker session; the loss curve in the code repository marks this.\n"
        out = ROOT / f"models/cards/{expert}.md"
        out.write_text(card)
        print("rendered", out.relative_to(ROOT))
        if a.upload:
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ["HF_TOKEN"])
            api.upload_file(path_or_fileobj=str(out), path_in_repo="README.md", repo_id=repo, commit_message="model card")
            hf[expert]["revision"] = api.model_info(repo).sha
            print("  uploaded README to", repo, hf[expert]["revision"][:10])
    if a.upload:
        (ROOT / "data/manifests/hf_models.json").write_text(json.dumps(hf, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
