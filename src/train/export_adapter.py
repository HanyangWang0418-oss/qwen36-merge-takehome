#!/usr/bin/env python
"""Download a Tinker sampler checkpoint and export it as a PEFT adapter.

    uv run python src/train/export_adapter.py tinker://<run>/sampler_weights/<step> models/weights/e1_code_repair

Afterwards inspect models/weights/<name>/adapter_config.json and copy
`target_modules` into the expert config (this answers which modules Tinker
actually adapted, including MoE expert matrices and the router).
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path


def main() -> int:
    tinker_path, out = sys.argv[1], Path(sys.argv[2])
    base = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen3.6-35B-A3B"
    from tinker import weights

    with tempfile.TemporaryDirectory() as tmp:
        weights.download(tinker_path=tinker_path, output_dir=tmp)
        if out.exists():
            shutil.rmtree(out)
        weights.build_lora_adapter(base_model=base, adapter_path=tmp, output_path=str(out))
    cfg = json.loads((out / "adapter_config.json").read_text())
    print(json.dumps({k: cfg[k] for k in ("r", "lora_alpha", "target_modules") if k in cfg}, indent=2))
    (out / "TINKER_SOURCE.txt").write_text(tinker_path + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
