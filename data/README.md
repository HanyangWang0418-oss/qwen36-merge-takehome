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
| e2 | mlfoundations-dev/terminal-bench-traces-local (licence not stated on card) | `train` | `terminus_trajectory` | one row per step holding the conversation prefix → collapsed to the longest row per (task, trial) = 80 trajectories; kept only if the final turn declares `is_task_complete: true`; TB2-overlapping task names dropped | plain text: Terminus prompt in the user turn, JSON action block in the assistant turn |
| e2 | mlfoundations-dev/swesmith_with_plain_docker-sandboxes-traces-terminus-2 (licence not stated on card) | `train` | `terminus_trajectory` | no verifier result in the release; kept only if the final turn declares `task_complete: true` (≈13%; most episodes stop at the 8-turn cap) | plain text Terminus-2 episodes (gpt-5-nano) on SWE-smith tasks |
| e2 | NiuNiu0110/TMax-Agent-SFT-terminus (ODC-BY) — a re-rendering of `allenai/tmax-sft`, config `…thinking_only_success` (the **verifier-labelled successful** rollouts of AI2's TMax synthetic terminal tasks), 5,445 train trajectories, several per task group | `train` | `tmax_trajectory` | all rows (success is upstream-verified); trailing submit-acknowledgement user turn dropped | OpenAI-style messages; Hermes-style `bash` tool declared in the system prompt, calls/`<tool_response>` as text, `<think>` reasoning on every assistant turn |
| e2 | zai-org/AgentInstruct, formerly THUDM/AgentInstruct (Apache-2.0) | `os` | `agentinstruct` | all 195 rows | ShareGPT human/gpt turns; `Think:/Act: bash` text protocol, no tool calls |
| e3 | open-thoughts/OpenThoughts3-1.2M (Apache-2.0) | `train` | `openthoughts` | all | ShareGPT; `<think>…</think>` + answer kept inside the assistant turn |
| e3 | AI-MO/NuminaMath-CoT (Apache-2.0) | `train` | `numina` | `source` ∉ {amc_aime, aops_forum} (restrictive terms) | user problem / assistant solution |

Sources listed in `sources.yaml` but **not used** in this build: the three
synthetic sources (`synthetic-shell-tasks`, `synthetic-research-exec`,
`gpqa-style-synthetic`) have no generator yet; the builder skips them and the
manifests say `skipped: file missing`.

Provenance check: no Terminal-Bench 2.0 task id (all 89 are >6 chars) occurs anywhere in the text of any kept e2 example (train or dev), verified by a direct grep in addition to the filter. The mlfoundations-dev cards are auto-generated (no licence, no description); TMax is ODC-BY end to end.

No public "Terminal-Bench 1.x trajectory release" exists as such; the closest
is the `terminal-bench-traces-local` run above (Terminus + claude-3-7-sonnet on
80 TB-1.x tasks). 27 of those 80 tasks were carried into Terminal-Bench 2.0 and
are removed by the blocklist (see contamination below).

## Cleaning and dedup
1. Loader converts each source to `{"messages": [...], "meta": {...}}` (+ optional `tools`). Assistant `tool_calls[].function.arguments` are emitted as **dicts** (sources store JSON strings): the Qwen3.6 chat template iterates `arguments | items`; `src/data/chat_format.py` also coerces strings defensively; rows with an unknown shape are counted `unparsable`, rows failing a source rule (unresolved, excluded sub-source) are counted `unresolved_or_excluded`.
2. Contamination filter (blocklists from `data/manifests/*`): repo name, instance id, task name (`meta.task_name`, with `.easy/.hard` variant suffixes stripped, and as a substring of the text), canary string, 13-gram overlap with the HLE question text.
3. Exact dedup: sha1 of the whitespace-normalised lowercase *dedup text* = the assistant turns only (content + tool-call arguments), i.e. the tokens the loss is computed on. Prompts and tool observations are excluded: agent and dialogue sources share multi-kB instructions and boilerplate outputs that made unrelated examples collide.
4. Near dedup on the same text: 64-perm MinHash, 8 bands × 8 rows, 5-gram shingles (threshold ≈ Jaccard 0.77; the earlier 16 × 4 layout fired at ≈ 0.5 and removed distinct successful runs of the same issue).
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

## Build report (this run)

### Rows fetched per source (datasets-server rows API unless noted)
| source | HF split | rows in split | rows fetched | bytes on disk (jsonl) | pages |
|---|---|---|---|---|---|
| SWE-Gym-SFT-trajectories | train.success.oss | 491 | 491 | 33 MB | 50 |
| SWE-Gym-trajectories | train.raw | 6,055 | 1,400 | 301 MB | 140 |
| SWE-smith-trajectories | tool | 24,100 | 2,500 | 392 MB | 250 |
| agenttuning | os | 195 | 195 | 1 MB | 4 |
| nemotron-swe | r2e_gym | 24,875 | 600 | 129 MB | 300 |
| numinamath-cot | train | 859,494 | 22,400 | 70 MB | 224 |
| openthoughts3 | train | 1,200,000 | 10,000 | 517 MB | 500 |
| terminal-bench-1-trajectories (single parquet download, 6.3 MB) | train | 1,189 | 1,189 | 50 MB | 24 |
| terminus-swesmith-sandbox-traces | train | 10,000 | 3,000 | 162 MB | 60 |
| tmax-agent-terminus | train | 5,445 | 3,000 | 80 MB | 60 |
| HLE question-text snapshot (blocklist only) | test, text-only | 2,158 | 2,158 | 2.0 MB | – |

Total jsonl written to `data/raw/`: **1736 MB** (44,775 rows); bytes over the wire were lower (gzip).

### e1_code_repair: train 1,401 / dev 155  (generated 2026-10-16T09:41Z)
| source | status | raw | unparsable | unresolved/excluded | contaminated | exact dup | near dup | too long | kept |
|---|---|---|---|---|---|---|---|---|---|
| SWE-smith-trajectories | used | 2500 | 0 | 1530 | 0 | 0 | 0 | 165 | **805** |
| SWE-Gym-trajectories | used | 1400 | 0 | 1273 | 0 | 7 | 0 | 6 | **114** |
| SWE-Gym-SFT-trajectories | used | 491 | 0 | 0 | 0 | 0 | 22 | 34 | **435** |
| nemotron-swe | used | 600 | 0 | 0 | 0 | 0 | 0 | 398 | **202** |

- contamination drops by reason: `{}`; blocklist sizes: `{'repos': 19, 'instance_ids': 51, 'task_names': 0, 'benchmark_shingles': 0, 'canary': False}`
- difficulty tiers (train): `{'medium': 935, 'easy': 95, 'hard': 371}`

### e2_terminal_agent: train 2,722 / dev 302  (generated 2026-10-16T09:42Z)
| source | status | raw | unparsable | unresolved/excluded | contaminated | exact dup | near dup | too long | kept |
|---|---|---|---|---|---|---|---|---|---|
| terminal-bench-1-trajectories | used | 80 | 0 | 8 | 29 | 0 | 0 | 1 | **42** |
| terminus-swesmith-sandbox-traces | used | 3000 | 0 | 2613 | 0 | 0 | 0 | 0 | **387** |
| tmax-agent-terminus | used | 3000 | 0 | 0 | 0 | 0 | 1 | 0 | **2999** |
| agenttuning | used | 195 | 0 | 0 | 0 | 0 | 0 | 0 | **195** |
| synthetic-shell-tasks | skipped: file missing | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| synthetic-research-exec | skipped: file missing | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

- contamination drops by reason: `{'task_name': 29}`; blocklist sizes: `{'repos': 0, 'instance_ids': 0, 'task_names': 154, 'benchmark_shingles': 0, 'canary': False}`
- difficulty tiers (train): `{'medium': 1217, 'easy': 1374, 'hard': 131}`

### e3_reasoning: train 14,493 / dev 500  (generated 2026-10-16T09:45Z)
| source | status | raw | unparsable | unresolved/excluded | contaminated | exact dup | near dup | too long | kept |
|---|---|---|---|---|---|---|---|---|---|
| openthoughts3 | used | 10000 | 0 | 0 | 7 | 0 | 0 | 0 | **9993** |
| numinamath-cot | used | 22400 | 0 | 903 | 7 | 168 | 8 | 0 | **21314** |
| gpqa-style-synthetic | skipped: file missing | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

- contamination drops by reason: `{'ngram_overlap': 14}`; blocklist sizes: `{'repos': 0, 'instance_ids': 0, 'task_names': 0, 'benchmark_shingles': 247183, 'canary': True}`
- difficulty tiers (train): `{'hard': 8886, 'easy': 4870, 'medium': 737}`

### Known gaps
- Every HF source is a bandwidth-bounded random sample of its split (table above), not the full data; `fetch.n_rows` in `sources.yaml` is the knob, and re-running `fetch_rows.py` only adds pages.
- e1 is far below the intended 16k: resolved SWE trajectories are 100–300 KB each as JSON, SWE-smith's `tool` split is 39% resolved and SWE-Gym's sampled runs only 8%. Nemotron-SWE trajectories are long (median 143k chars, 60 turns): 398 of the 600 sampled exceed the 120k-char cap; raising `--max-chars` recovers them if the trainer's context allows.
- e2 public terminal data is thin: 80 TB-1.x trajectories (42 kept after the 29 TB2-overlap drops, 8 incomplete runs and 1 over-length), 195 AgentInstruct `os` dialogues, 387 self-declared-complete Terminus-2 episodes on SWE-smith sandboxes, and 2,999 TMax trajectories, which are therefore ~80% of e2. The two synthetic sources (40% of the planned e2 mix) and the e3 synthetic STEM source were not generated.
- e3 is the only expert where the `weight` caps bind: NuminaMath is capped at 5,000 (0.25 × 20k) although 21k rows passed the filters; OpenThoughts3 contributes all 9,993.
- Licences: the two mlfoundations-dev trace sets state no licence on their cards (underlying tasks are Apache-2.0 / MIT); treat them as research-only until clarified.
- Terminal-bench-traces-local rows are per-step prefixes; only the longest row per trial is used, so n_turns tiers for e2 reflect full trajectories, while AgentInstruct dialogues are short by nature.
- The HLE 13-gram blocklist covers only the question ids locked in `hle_text_only_ids.json` and only as many as the snapshot script could fetch (count above).
- No shard files were downloaded; `uv.lock` dependencies (pyarrow, datasets) were not available in the project venv at build time, so the pipeline runs on the stdlib + pyyaml (pyarrow only for the one local parquet).
