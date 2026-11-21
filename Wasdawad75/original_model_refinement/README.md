# Original ensemble refinement — Wasdawad75

The original four-component passenger-demand ensemble consistently underpredicted volume in the development backtests. Small, isolated adjustments reduced its three-fold mean directional-market WAPE from **19.8811% to 19.0826%**, a **4.02% relative error reduction**. This folder records the selected `model_v10` configuration and the evidence behind it.

The target is monthly **OD_PAX passenger demand** for a directed airport pair, up to 24 months ahead. These results are development backtests. The refined model's later holdout evaluation is incomplete.

## Completed comparison

Lower WAPE is better. For example, `0.190826` means 19.0826% WAPE; it is not a classification accuracy score.

| Historical window | Refined model_v10 WAPE | Reproduced baseline WAPE |
|---|---:|---:|
| Recovery: Aug 2021–Jul 2023 | 0.213189 | 0.225555 |
| Transition: Feb 2022–Jan 2024 | 0.189884 | 0.200459 |
| Steady: Aug 2022–Jul 2024 | 0.169406 | 0.170419 |
| **Three-fold mean** | **0.190826** | **0.198811** |

All three periods improved. Most of the gain comes from recovery and transition; improvement in the steady period is small.

There were **34 completed refinement evaluations**, one further attempt discarded at the time limit, and **10 local checkpoints**, including the baseline. Each trial changed one setting from the latest retained configuration. Worse results were discarded. [experiments.jsonl](experiments.jsonl) records every attempt's hypothesis, change, score and decision. It excludes raw observations and local filesystem paths.

## What changed

| Setting | Original | model_v10 |
|---|---:|---:|
| 03 seasonal component weight | 35% | 53.4211% |
| 14 multilevel component weight | 35% | 21.9231% |
| 12 direct XGBoost weight | 15% | 12.3279% |
| 20b cross-sectional XGBoost weight | 15% | 12.3279% |
| Final volume multiplier | 1.00 | 1.05 |
| 03 route-to-segment prior strength | 1500 | 500 |
| 03 recent-season weight for recovered routes | 0.50 | 0.35 |
| 20b log-correction clipping limit | 0.28 | 0.14 |
| 20b segment-centering multiplier | 0.50 | 0.25 |

In plain English: correct a persistent volume underestimate, rely more on the seasonal forecast, give individual routes more influence over their group average, and constrain the tree model's corrections. The last centering adjustment produced only a very small additional development improvement.

[best_config.json](best_config.json) contains the exact values and source substitutions. [baseline_config.json](baseline_config.json) reproduces the original blend. The original shared evaluator, validation splits and input CSVs were unchanged.

## Later windows: still unverified for model_v10

| Window | Refined model_v10 WAPE | Archived original model 21 WAPE |
|---|---:|---:|
| Year 1: Aug 2024–Jul 2025 | Not evaluated | 0.146583 |
| Clean 17 months: Aug 2024–Dec 2025 | Not evaluated | 0.158822 |
| Pre-anomaly 21 months: Aug 2024–Apr 2026 | Not evaluated | 0.173720 |
| Full 24 months: Aug 2024–Jul 2026 | Not evaluated | 0.192983 |

The archived column comes from the repository's original `21_diverse_family_ensemble/outputs/final_holdout_metrics.json`; it was not recomputed in this run. The refined model's holdout rerun was stopped before producing scores. Its development score cannot be directly compared with the archived holdout score. The project had already inspected this later period, so a rerun is an audit rather than a new blind test.

## Run locally

Use Python 3.12 from the repository root. Numerical-library versions used for the recorded run are pinned in [requirements.txt](requirements.txt).

```powershell
python -m pip install -r Wasdawad75/original_model_refinement/requirements.txt
python Wasdawad75/original_model_refinement/run_model.py --check-only
```

The source check verifies the original implementation files and imports every component without training. The dependency files are already in `abraham/agent_workspace`; [source_manifest.json](source_manifest.json) records their content hashes and original source commit `71b891442d7eefca65c307045b707335f2c5b19d`. Unrelated additions elsewhere in the repository do not affect the check.

Required local inputs:

- `pax_forecasting.csv` and `airport_info.csv`, in the original pipe-delimited export format with the saved index column.
- Prepared public feature files: `airport_external_annual.parquet`, `pair_migration_2015.parquet`, and `macro_monthly_clean.parquet`. Use the team's original `abraham/agent_workspace/data_external/prepare_external_tables.py` preparation workflow.

Supply the actual local directories; the paths below are placeholders:

```powershell
python Wasdawad75/original_model_refinement/run_model.py --data-dir "C:\path\to\temp" --external-dir "C:\path\to\prepared_features" --output Wasdawad75/original_model_refinement/local_runs/model_v10_backtest
```

Omit the directory options to use the team's original default locations. For a baseline rerun, add `--config Wasdawad75/original_model_refinement/baseline_config.json` and choose a new output directory.

To evaluate the frozen candidate on the later holdout:

```powershell
python Wasdawad75/original_model_refinement/run_model.py --mode holdout --data-dir "C:\path\to\temp" --external-dir "C:\path\to\prepared_features" --output Wasdawad75/original_model_refinement/local_runs/model_v10_holdout
python Wasdawad75/original_model_refinement/summarize_holdout.py --refined Wasdawad75/original_model_refinement/local_runs/model_v10_holdout/outputs/final_holdout_metrics.json --baseline abraham/agent_workspace/21_diverse_family_ensemble/outputs/final_holdout_metrics.json
```

The summarizer prints all four windows from the same 24-month forecast and labels the comparison as a reference. Use a freshly reproduced baseline report instead when making a comparison with identical local feature snapshots. Do not retune against these audit scores.

The runner compiles configuration changes in memory; it does not edit teammates' model or evaluator files. It uses fresh fits, preserves fitted boosters locally and refuses to overwrite an existing output directory. Component 20b uses one CPU worker; other components default to four. Retraining can be slow and memory intensive. The recorded development run reused unchanged component predictions with fixed inputs; the portable runner always computes fresh predictions, so small numerical differences are possible.

## Verification and limits

- The reproduced baseline passed the original all-fold feature and Fold-3 external/internal future-poisoning invariance check, with maximum prediction difference `0.00e+00`.
- The selected model's separate poisoning check was stopped before completion. It has not been reported as passed.
- All saved local checkpoint artifacts and protected data/evaluator fingerprints passed integrity checks. Fitted weights and raw inputs remain local; they are not included here.
- Public external inputs were rebuilt from official/public sources. Their vintages may differ from the author's snapshot, and some original per-year availability metadata predates the downloaded source vintages. A poisoning-invariance pass does not resolve this real-world publication-vintage issue.
- The fixed loader fills absent route-months with zero and tracks an observation flag. The meaning of missing records and the operational reporting delay for OD_PAX remain unconfirmed.
- Only routes observed by the forecast origin are evaluated. Discovery of entirely new markets is outside this benchmark.

The next evidence needed is a completed, fixed-configuration later-period comparison and the selected model's leakage check. Until then, `model_v10` is a candidate supported by historical backtest improvements.
