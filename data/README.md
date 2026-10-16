# Data

Training sets are built by `src/data/build_expert_data.py` from the sources in
`configs/data/sources.yaml`; per-expert manifests (`data/manifests/<expert>_manifest.json`)
record source, licence, URL, the pages that were actually read, counts at every
filter stage, dedup settings, difficulty tiers and contamination drops by reason.
Raw and processed jsonl are not committed (regenerable, some licences forbid redistribution).

## How the data was obtained (and why it is a sample)

This machine reaches `huggingface.co/.../resolve/` at ~13 KB/s with frequent
connection resets (hf-mirror.com is slower), so the shard files behind the
sources (0.3 GB SWE-Gym, 4.2 GB SWE-smith, 11 GB Nemotron-SWE, 28 GB
OpenThoughts3, 1.2 GB NuminaMath) could not be downloaded. Instead
`src/data/fetch_rows.py` reads rows through the datasets-server rows API
(`/rows?dataset=&config=&split=&offset=&length=`), which here sustains
~100–170 KB/s gzipped, returns complete rows (no truncated cells were seen on
any source used) and accepts arbitrary offsets. For each source it takes a
seeded shuffle of page-aligned offsets over the *whole* split and fetches the
first `n_rows / page` of them (so raising `n_rows` later only extends the set),
writing one `data/raw/<source>/rows_<offset>.jsonl` per page; the builder reads
whatever pages exist. The `fetch:` block of every source in `sources.yaml`
records `n_rows`, `page`, `priority`, and the manifest lists the page files read.
One small source whose rows API was down (`terminal-bench-traces-local`,
6 MB) was downloaded as a single parquet and converted to the same page format.

Consequence: every HF source is a **stratified random sample**, sized by
bandwidth, not the full split. Sizes are in the build report below.

## Sources, formats, filters

| expert | source | split used | loader | rows kept rule | message format |
|---|---|---|---|---|---|
| e1 | SWE-bench/SWE-smith-trajectories (MIT) | `tool` | `swe_agent_trajectory` | `resolved == true` | SWE-agent messages stored as a JSON string with content blocks → flattened; assistant `tool_calls` kept (OpenAI style), `tool` turns with `tool_call_id` |
| e1 | SWE-Gym/OpenHands-Sampled-Trajectories (MIT) | `train.raw` | `openhands_trajectory` | `resolved == true` (~8% of rows) | OpenAI-style with native `tool_calls`; `tools` schemas kept at top level |
| e1 | SWE-Gym/OpenHands-SFT-Trajectories (MIT) | `train.success.oss` | `openhands_sft_trajectory` | all 491 rows are resolved by construction | system/user/assistant text, actions rendered as text (no tool calls); repo parsed from the `/workspace/<owner>__<repo>__<ver>` path |
| e1 | nvidia/Nemotron-SWE-v1 (CC-BY-4.0, per-row repo licence kept in `meta.row_license`) | `r2e_gym` | `nemotron_swe` | no resolved flag in the release (already a curated SFT set) → all kept | OpenAI-style with native `tool_calls`, tool turns keyed by `id` → `tool_call_id`; `tools` kept |
| e2 | mlfoundations-dev/terminal-bench-traces-local (licence not stated on card) | `train` | `terminus_trajectory` | one row per step holding the conversation prefix → collapsed to the longest row per (task, trial) = 80 trajectories; TB2-overlapping task names dropped | plain text: Terminus prompt in the user turn, JSON action block in the assistant turn |
| e2 | mlfoundations-dev/swesmith_with_plain_docker-sandboxes-traces-terminus-2 (licence not stated on card) | `train` | `terminus_trajectory` | collapsed by `trial_name`; no success flag in the release | plain text Terminus-2 episodes (gpt-5-nano) on SWE-smith tasks |
| e2 | zai-org/AgentInstruct, formerly THUDM/AgentInstruct (Apache-2.0) | `os` | `agentinstruct` | all 195 rows | ShareGPT human/gpt turns; `Think:/Act: bash` text protocol, no tool calls |
| e3 | open-thoughts/OpenThoughts3-1.2M (Apache-2.0) | `train` | `openthoughts` | all | ShareGPT; `<think>…</think>` + answer kept inside the assistant turn |
| e3 | AI-MO/NuminaMath-CoT (Apache-2.0) | `train` | `numina` | `source` ∉ {amc_aime, aops_forum} (restrictive terms) | user problem / assistant solution |

Sources listed in `sources.yaml` but **not used** in this build: the three
synthetic sources (`synthetic-shell-tasks`, `synthetic-research-exec`,
`gpqa-style-synthetic`) have no generator yet; the builder skips them and the
manifests say `skipped: file missing`.

No public "Terminal-Bench 1.x trajectory release" exists as such; the closest
is the `terminal-bench-traces-local` run above (Terminus + claude-3-7-sonnet on
80 TB-1.x tasks). 27 of those 80 tasks were carried into Terminal-Bench 2.0 and
are removed by the blocklist (see contamination below).

## Cleaning and dedup
1. Loader converts each source to `{"messages": [...], "meta": {...}}` (+ optional `tools`); rows with an unknown shape are counted `unparsable`, rows failing a source rule (unresolved, excluded sub-source) are counted `unresolved_or_excluded`.
2. Contamination filter (blocklists from `data/manifests/*`): repo name, instance id, task name (`meta.task_name`, with `.easy/.hard` variant suffixes stripped, and as a substring of the text), canary string, 13-gram overlap with the HLE question text.
3. Exact dedup: sha1 of whitespace-normalised lowercase text.
4. Near dedup: 64-perm MinHash, 16 bands × 4 rows, 5-gram shingles (≈ Jaccard ≥ 0.8).
5. Length cap 120k chars (fits 32k tokens).
6. Difficulty tier: agent data by number of assistant turns (<10 / 10–30 / >30), reasoning data by response length (<2k / 2k–8k / >8k tokens, chars/4).

## Contamination policy
- SWE-bench Pro: every task from all 11 V2 repos is excluded (not only the 51 hard ones), plus any trajectory whose instance id is in the manifest. The locked `repos` list carries an `instance_` prefix, so the builder also derives `owner/repo` from the task ids.
- Terminal-Bench 2.0: task names excluded; TB 1.x trajectories deduped against the 2.0 task list (27 of 80 TB1 tasks are in TB2 and are dropped).
- HLE: canary-string filter on every source; 13-gram overlap against the question text of the 2158 locked text-only items. The snapshot is built locally by `src/data/lock_hle_questions.py` into `data/raw/hle/` (gated dataset, gitignored, questions only, never answers).
- ASI-Bench: project ids excluded; no ASI data (either seed) is used for training.
- Benchmark answers, reference patches, hidden tests and test feedback are never read by any training or selection script. Checkpoints are selected by dev loss on a held-out split of the training mixture.

## Mixture sizes and why they differ
The `max_examples` / `weight` fields in `sources.yaml` are the intended mixture
(e1 16k, e2 12k, e3 20k) once full shards can be pulled; in this build every
source pool is smaller than its share, so each pool is used whole and the
weights do not bind. The build report below has the real numbers.

