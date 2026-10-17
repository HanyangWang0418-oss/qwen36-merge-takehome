#!/usr/bin/env python
"""Write a local dataset containing only the locked text-only HLE ids, in a layout the official scripts can
load with `load_dataset(<dir>, split="test")` (a `test-*.parquet` file). Columns are kept as-is (the `image`
field is "" for text-only rows, which is what the official prompt builder checks). Gated upstream; the output
directory is gitignored and never redistributed."""
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
assert all(not r for r in sub["image"]), "non-empty image field in the text-only subset"
os.makedirs(out, exist_ok=True)
sub.to_parquet(os.path.join(out, "test-00000-of-00001.parquet"))
chk = load_dataset(out, split="test")
assert len(chk) == len(keep)
print(f"{len(sub)} text-only questions -> {out} (loadable with load_dataset(dir, split='test'))")
