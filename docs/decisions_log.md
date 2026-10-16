# Decisions log (autonomous stretch, night of 2026-10-16/17)

Rules I followed while the account owner was away: no new paid cloud resources, no GPU, no control-expert
training, no e3 retraining; only work that certainly moves the project forward or settles a design question.

| # | decision | why | cost |
|---|---|---|---|
| 1 | Export + merge + HF upload moved to GitHub Actions (`publish_and_merge.yml`) | laptop bandwidth ~100 KB/s; runners reach Tinker/HF at full speed | free (private-repo minutes) |
| 2 | Three expert adapters published to HF as public PEFT repos via `tinker checkpoint push-hf`; model cards rendered from training metadata | deliverable; also needed as the merge input | 0 |
| 3 | `merge_adapters.py` rewritten for fused 3D MoE LoRA (shared-side broadcast) after the first merge failed on rank mismatch; layer-scoped merges zero-padded to a global rank | the exported layout was unknown until the adapters existed | 0 |
| 4 | Dev subsets drawn from the official benchmarks **replaced** by independent proxies (MMLU-Pro STEM, SWE-bench Verified sample, Terminal-Bench-Pro sample, ASI seed31415) | task statement: tune on independent dev sets, never on the evaluation items | 0 |
| 5 | `remap_keys.py` written from vLLM source: fuse `in_proj_q/k/v`→`in_proj_qkv`, rename `unembed_tokens`→`lm_head`; experts `w1/w2/w3` and the `language_model` prefix are handled by vLLM itself | Tinker's export uses its own names; serving would fail otherwise | 0 |
| 6 | Tinker-sampling pre-screen of base vs e3 on 100 MMLU-Pro STEM questions (8K budget): base 66%, e3 80%, paired Δ +14 [5, 23] | cheap signal on whether e3 is worth keeping; **confound**: base truncated 27/100 vs e3 12/100 at 8K, so re-run at 20K | ≈ $1 |
| 7 | 20K-budget re-run of the same 100 questions | removes the truncation confound | ≈ $1.5 |

Open for the owner: GPU box (2× A100) SSH; OpenAI key (HLE judge); ASI judge key; interviewer confirmations;
whether to train the control experts (~$100) and whether to pursue e3 RFT.
