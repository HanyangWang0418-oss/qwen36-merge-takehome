# Data

Training sets are built by `src/data/build_expert_data.py` from the sources in
`configs/data/sources.yaml`; per-expert manifests (`data/manifests/<expert>_manifest.json`)
record source, licence, URL, counts at every filter stage, dedup settings,
difficulty tiers and contamination drops by reason. Raw and processed jsonl are
not committed (regenerable, some licences forbid redistribution).

## Cleaning and dedup
1. Loader converts each source to `{"messages": [...], "meta": {...}}`; unparsable rows dropped.
2. Contamination filter (blocklists from `data/manifests/*`): repo name, instance id, task name, canary string, 13-gram overlap with benchmark prompt text.
3. Exact dedup: sha1 of whitespace-normalised lowercase text.
4. Near dedup: 64-perm MinHash, 16 bands × 4 rows, 5-gram shingles (≈ Jaccard ≥ 0.8).
5. Length cap 120k chars (fits 32k tokens).
6. Difficulty tier: agent data by number of assistant turns (<10 / 10–30 / >30), reasoning data by response length (<2k / 2k–8k / >8k tokens).

## Contamination policy
- SWE-bench Pro: every task from all 11 V2 repos is excluded (not only the 51 hard ones), plus any trajectory whose instance id is in the manifest.
- Terminal-Bench 2.0: task names excluded; TB 1.x trajectories deduped against the 2.0 task list.
- HLE: canary-string filter on every source; 13-gram overlap against the locked question text (built locally from the gated dataset, never committed).
- ASI-Bench: project ids excluded; no ASI data (either seed) is used for training.
- Benchmark answers, reference patches, hidden tests and test feedback are never read by any training or selection script. Checkpoints are selected by dev loss on a held-out split of the training mixture.

## Mixture sizes and why they differ
| expert | examples | rationale |
|---|---|---|
| e1 | 16k | long trajectories (median ~8k tokens); more examples than this exceeds budget |
| e2 | 12k | public terminal data is scarcer; synthetic sandbox-verified tasks fill 40% |
| e3 | 20k | short examples, high variance subjects; rank-32 needs more loss tokens (Tinker rule: LoRA params ≳ loss tokens) |
