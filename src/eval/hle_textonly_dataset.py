#!/usr/bin/env python
"""Write a local HF dataset containing only the locked text-only HLE ids
(gated upstream; output dir is gitignored and never redistributed)."""
import json
import os
import sys

from datasets import load_dataset

ids_file, out = sys.argv[1], sys.argv[2]
m = json.loads(open(ids_file).read())
keep = set(m["question_ids"])
ds = load_dataset("cais/hle", split="test", revision=m["dataset_hf_revision"], token=os.environ.get("HF_TOKEN"))
sub = ds.filter(lambda r: r["id"] in keep)
assert len(sub) == len(keep), (len(sub), len(keep))
sub.save_to_disk(out)
print(f"{len(sub)} text-only questions -> {out}")
