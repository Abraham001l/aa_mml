# Keshav’s route-recovery model results

The model achieved **17.34% WAPE** on the full 24-month holdout and **19.08% mean WAPE** across the three development backtests.

## July 2024 forecast

All 24 months were forecast using demand through July 2024. The evaluation covers 17,119 directional airport pairs and treats absent market-month records as zero passengers.

| Forecast period | Market-months | WAPE | MAE (passengers) | RMSE (passengers) | R² | Total bias |
|---|---:|---:|---:|---:|---:|---:|
| Clean 17 months (Aug 2024–Dec 2025) | 291,023 | 14.73% | 475.38 | 1,189.40 | 0.9728 | +1.81% |
| Pre-anomaly 21 months (Aug 2024–Apr 2026) | 359,499 | 15.77% | 497.95 | 1,249.47 | 0.9691 | +3.00% |
| Full 24 months (Aug 2024–Jul 2026) | 410,856 | 17.34% | 545.32 | 1,408.51 | 0.9600 | +4.77% |
| Aug 2024–Jul 2025 | 205,428 | 13.89% | 451.38 | 1,126.99 | 0.9758 | +1.42% |
| Aug 2025–Jul 2026 | 205,428 | 21.02% | 639.27 | 1,642.46 | 0.9422 | +8.36% |

WAPE measures summed absolute passenger error as a percentage of actual demand. Positive total bias indicates overprediction.

## Development backtests

| Forecast cutoff | WAPE over the following 24 months |
|---|---:|
| 2021-07 | 20.43% |
| 2022-01 | 19.73% |
| 2022-07 | 17.07% |

These overlapping windows were used to select the model settings. The selected model uses a three-month demand window, 12-month recovery damping, and a 25% contribution from the reverse route when available.

Reproduce the scores with `python3 keshav/ml_pipeline/seasonal_challenger.py` from the repository root. The [evaluation script](seasonal_challenger.py) writes the detailed metrics to `results/seasonal_challenger/metrics.csv`.
