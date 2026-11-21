# Keshav’s monthly passenger demand model

The route-recovery model forecasts **monthly passengers (`OD_PAX`) for each directional airport pair, 1–24 months ahead**. It is a statistical model. Each forecast combines the route’s pre-pandemic seasonal pattern, its recent demand, and evidence from traffic in the opposite direction. Historical backtests select the settings shared across routes.

On the July 2024 forecast, the model achieved **17.34% WAPE over the following 24 months**. Its **19.08% WAPE** is the average from three earlier development backtests. [Implementation](seasonal_challenger.py) · [Results and metrics](MODEL_RESULTS.md)

## Run the model

Place the local pipe-delimited passenger extract at `keshav/ml_pipeline/pax_forecasting.csv`. From the repository root, run:

```sh
python3 -m pip install -r keshav/ml_pipeline/requirements.txt
python3 keshav/ml_pipeline/seasonal_challenger.py
python3 -m unittest discover -s keshav/ml_pipeline -p 'test_*.py' -v
```

The script searches the development configurations and reproduces the historical evaluation. It writes these local files under `keshav/ml_pipeline/results/seasonal_challenger/`:

| File | Contents |
|---|---|
| `development_search.csv` | WAPE for each candidate at each development cutoff |
| `settings.json` | Selected model family, demand window, recent seasonal weight, damping and directional weight |
| `metrics.csv` | Forecast errors for the selected model and seasonal-repeat benchmark by cutoff and evaluation window |
| `holdout_forecasts.csv` | July 2024 forecast, one row per directional route and one column per target month |

## Input and assumptions

Each row of the passenger file supplies `ORG_AIRPRT`, `DES_AIRPRT`, `DEP_YEAR`, `DEP_MONTH`, and `OD_PAX`. The loader expects one record per observed route-month, nonnegative demand, and consecutive calendar months across the network starting in January 2019. The evaluation uses January 2019–July 2026 and covers 17,119 routes. The seasonal model requires only the passenger file.

IAH → LAX and LAX → IAH occupy separate rows in the model’s route-by-month table. The model treats an absent route-month as zero in both forecasting and scoring. If a record is absent because of a reporting problem, that assumption affects the result. Forecasts also assume that demand through the selected cutoff was available when the forecast was issued.

## How one forecast works

Consider the forecast issued in July 2024 for **IAH → LAX in December 2024**.

1. **Find the route’s December patterns.** Divide each monthly passenger count by its number of calendar days. Look up average daily demand in December 2019 and in December 2023, the latest December available. Working in daily units lets the model account for the length of the target month, including leap years.
2. **Estimate the route’s current level.** Compare the sum of daily rates in May–July 2024 with the sum for May–July 2019. Matching the months controls for the usual summer peak. The implementation adds 2 to each sum and bounds their ratio between 0.05 and 8 to stabilize routes with little historical traffic.
3. **Decide how far to pull the route toward its 2019 level.** Compute the same three-month comparison across the whole network. When network traffic is depressed, the model gives greater weight to recovery. A route running below its 2019 level can rise toward it over the forecast horizon. A route above its 2019 level can move down toward it. When network traffic has recovered, the model preserves more of the route’s current level.
4. **Blend the two seasonal estimates.** Scale December 2019’s daily demand by the projected route level, then blend that estimate with the latest December daily rate. For an established route, the recent pattern can receive up to 70% weight. A route with little or no 2019 service receives more weight on its recent history. Multiply the blended daily rate by the 31 days in the target December.
5. **Use the return direction.** Compute LAX → IAH’s own forecast. Translate it into IAH → LAX units using the directional imbalances for the target month in 2019 and in the latest annual cycle. The two ratios are stabilized with the trailing annual imbalance and combined using a geometric mean bounded between 0.25 and 4. The final forecast gives 75% weight to IAH → LAX’s own estimate and 25% to the adjusted reverse estimate. If the reverse route is unavailable, the route keeps its own forecast.

All horizons use the same July 2024 observations. In the saved forecast, IAH → LAX is approximately **39,026 passengers** for both December 2024 and December 2025. Equal forecasts for the same calendar month a year apart are possible when recovery damping is inactive and the two months have the same length.

### Recovery calculation

Let `r` be the route’s recent-to-2019 ratio, `R` the network ratio, and `h` the forecast horizon in months. The selected configuration computes:

```text
recovery_strength = clip((1.06 - R) / 0.20, 0, 1)
pull              = recovery_strength × (1 - exp(-h / 12))
route_level       = (1 - pull) × r + pull × 1
2019_estimate     = daily demand in the target month of 2019 × route_level
```

At or below 86% of 2019 network demand, recovery strength is 1. At or above 106%, it is 0. Between those points it changes linearly. The 12-month damping parameter controls how quickly the projected route level approaches 1. For example, with full recovery strength and an initial route ratio of 0.80, the projected ratio is about 0.93 at month 12 and 0.97 at month 24.

## How the settings were selected

The script evaluates **198 configurations** across two families. One applies a network-wide recovery adjustment. The other lets each route move toward its own 2019 level. At each cutoff, the model computes features from history through that month and scores the next 24 months. Each of the three development folds counts equally when selecting the lowest mean WAPE.

| Forecast cutoff | Selected model’s 24-month WAPE |
|---|---:|
| July 2021 | 20.43% |
| January 2022 | 19.73% |
| July 2022 | 17.07% |
| **Development mean** | **19.08%** |

The selected family is **route recovery**, with a **3-month** demand window, **12-month** damping, **0.70** recent seasonal weight, and **0.25** reverse-route weight. The folds overlap, so their errors are correlated.

## July 2024 holdout

Each row scores part or all of the **same** forecast, across the same directional markets. “Clean” and “pre-anomaly” are project labels. The 17-month window ends with calendar 2025. The 21-month window adds January–April 2026 and ends before the unusual May–June passenger readings flagged in the project. The full window includes those months.

| Window | Target months | WAPE | RMSE (passengers) | Total bias |
|---|---|---:|---:|---:|
| Clean 17 months | Aug 2024–Dec 2025 | 14.73% | 1,189.40 | +1.81% |
| Pre-anomaly 21 months | Aug 2024–Apr 2026 | 15.77% | 1,249.47 | +3.00% |
| Full 24 months | Aug 2024–Jul 2026 | **17.34%** | **1,408.51** | **+4.77%** |

**WAPE** divides total absolute passenger error by total actual passengers. **MAE** is the average absolute miss in passengers per market-month. **RMSE** penalizes larger misses more heavily. **Total bias** measures aggregate overprediction or underprediction. The metrics file also reports pooled R², symmetric percentage error (sMAPE, on a 0–2 scale), and log-scale error (RMSLE).

The principal weakness is the second forecast year. WAPE rose from **13.89%** in year one to **21.02%** in year two, and total bias rose from **+1.42%** to **+8.36%**. The model projects historical patterns for known routes. Schedule changes, future disruptions, and reporting changes can all affect accuracy. This holdout was inspected during the project and measures performance over one historical period.

## Other models in this project

- **Seasonal repeat** copies each market’s last 12 observed calendar slots twice, using the same zero-filled panel. It scored **18.29%** full-holdout WAPE.
- **Abraham’s Model 21** blends a seasonal recovery forecast with three components that use XGBoost. Its saved full-holdout WAPE was **19.30%** on the same 410,856 market-months. [Ensemble code](../../abraham/agent_workspace/21_diverse_family_ensemble/run.py) · [Saved holdout metrics](../../abraham/agent_workspace/21_diverse_family_ensemble/outputs/final_holdout_metrics.json)
- **Follow-up seasonal experiments** compared 13 configurations, including airport reconciliation and pooling seasonal patterns across routes. The pooled model achieved **18.57%** mean development WAPE and **17.89%** full-holdout WAPE. Run `python3 keshav/ml_pipeline/compare_seasonal_methods.py` to reproduce that comparison. [Experiment code](compare_seasonal_methods.py)
- **Earlier XGBoost experiments** predict demand directly and evaluate observed market-months. Their sample differs from the balanced seasonal evaluation. [XGBoost guide](XGBOOST.md)
