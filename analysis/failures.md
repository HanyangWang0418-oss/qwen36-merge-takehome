# Failure analysis (merged model)

Target: ≥ 20 sampled failures of `merged_ta_l0.7` across the four benchmarks, drawn with
`uv run python src/analysis/sample_failures.py --n 24 --seed 0` from `results/merged/merged_ta_l0.7/*/failed_tasks.json`.

Categories: localisation error · patch error · test-understanding error · merge-induced regression (expert solves it, merge does not) · environment/tool failure · timeout · other.

| # | benchmark | task id | category | expert on this task | base on this task | evidence (log path) | note |
|---|---|---|---|---|---|---|---|

## Category counts

## Next experiments (≥ 3)
