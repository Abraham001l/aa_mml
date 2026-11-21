# Earlier XGBoost experiments

Train one XGBoost regressor to predict `OD_PAX` for a directional airport pair at
month `t+h`, for every horizon `h=1..24`. Opposite directions remain separate markets.
The script only prepares features, trains the model, and tests it. It writes
`test_predictions.csv` and `test_metrics.csv` (MAE, RMSE, R², WAPE, and signed bias,
overall and by horizon). `benchmark_metrics.csv` compares XGBoost with seasonal
repeat on the same rows with an available seasonal reference.

```sh
python3 -m pip install -r keshav/ml_pipeline/requirements.txt
python3 keshav/ml_pipeline/ml_pipeline.py
```

Place the pipe-delimited `pax_forecasting.csv` and `airport_info.csv` beside the
script, or supply `--pax PATH --airports PATH`. CSV inputs and generated results
are excluded from Git.

The default cutoff is the latest passenger month minus 24 months. With the current
files, training labels end in July 2024 and test targets run from August 2024 through
July 2026. All test forecasts use the same July 2024 history; later test outcomes
never update their features. Missing target rows are not scored. This evaluates
observed markets with demand at the cutoff, rather than forecasting new routes.

Training uses every third origin month to limit memory and runtime, with all 24
horizons per eligible origin. Labels later than the cutoff are excluded. Use
`--origin-step 1` for every origin month (roughly three times as many examples),
`--cutoff 2024-07` for an explicit cutoff, and `--n-estimators 200` to set tree count.
This is a fixed train/test evaluation with unchanged model hyperparameters.
Because this cutoff was already inspected when proposing the expanded features,
the before/after scores are a development comparison. Use additional historical
cutoffs and a fresh final holdout to assess generalization.

Run the calendar-alignment and leakage checks with:

```sh
python3 -m unittest discover -s keshav/ml_pipeline -p 'test_*.py' -v
```

## Features

| Feature | Definition |
| --- | --- |
| `ORG_AIRPRT`, `DES_AIRPRT`, `market_id` | Categorical airport codes and directional pair identifier, with category mappings fitted on training examples. |
| `distance_km` | Great-circle distance derived from airport coordinates. |
| `is_domestic` | 1 when both airports are in the same country, including non-US markets. |
| `target_month` | Calendar month 1–12 of `t+h`, for seasonality. |
| `forecast_horizon` | Number of months ahead, 1–24. |
| `pax_lag_0`, `pax_lag_1`, `pax_lag_2`, `pax_lag_3`, `pax_lag_6`, `pax_lag_12`, `pax_lag_24` | Demand at the corresponding exact calendar-month offsets from `t`. |
| `pax_rolling_3`, `pax_rolling_12` | Mean demand over the 3 or 12 consecutive months ending at `t`, including `t`. |
| `pax_yoy_growth` | `OD_PAX(t) / OD_PAX(t-12) - 1`; missing if the reference is missing or zero. |
| `pax_rolling_3_yoy_growth` | Three-month mean ending at `t`, divided by the three-month mean ending at `t-12`, minus 1. |
| `origin_pax_total`, `destination_pax_total` | Monthly demand at `t` summed across markets sharing the origin or destination, respectively. |
| `origin_pax_yoy_growth`, `destination_pax_yoy_growth` | Each endpoint's monthly total at `t`, divided by its total at `t-12`, minus 1. |
| `pax_seasonal_reference` | Latest observed demand for this market and target calendar month, known at origin `t`. |
| `pax_seasonal_mean_3` | Mean of the latest up to three observed demand values for that market and target calendar month, known at `t`. |

Missing months remain unknown rather than becoming zero demand. Rolling means
require a complete window; XGBoost handles missing history features natively.
Revenue, airport country identifiers, plots, validation fits, model exports, and
production forecasting are omitted to keep this a single train/test workflow.

All 24 features come from the current datasets. For a July 2024 origin, forecasts
for December 2024 and December 2025 both use December 2023 as their latest possible
seasonal reference. If that market-month is missing, the code looks in earlier
years. Seasonal features can vary by target month; the other history features stay
fixed across the 24 test horizons. Airport aggregates cover the markets present in
this dataset and may change with reporting coverage; they are not necessarily total
airport traffic. Growth denominators of zero produce missing values.

## Evaluation metrics

The model still trains with squared-error loss; WAPE is a reporting metric and
does not determine the training objective. The expanded-feature comparison uses
the same 200 trees, depth, learning rate, origin spacing, and cutoff as the original
model. No parameters are selected using the test results.

| Metric | Meaning and use |
| --- | --- |
| MAE | Mean absolute error in passengers; a directly interpretable typical absolute error. |
| RMSE | Square root of mean squared passenger error; emphasizes large misses and aligns with the training loss. |
| R² (`R2`) | `1 - sum(error²) / sum((actual - mean(actual))²)`; 1 is perfect, 0 matches predicting the test-set mean, and negative values are possible. Undefined for fewer than two observations or constant actuals; saved as missing. |
| WAPE | `100 * sum(abs(error)) / sum(actual)`; total absolute error as a percentage of observed demand, not average per-market percentage error. Undefined when total actual demand is zero. |
| Signed bias | `100 * sum(predicted - actual) / sum(actual)`; positive values indicate overprediction in aggregate. |

RMSE and R² rank models identically when evaluated on exactly the same observations.
A high pooled R² can reflect the large differences in market sizes while masking
poor month-to-month forecasts. WAPE helps express total demand error as a percentage,
but emphasizes higher-volume markets and its denominator changes across test periods.
On identical observations, WAPE and MAE also rank models identically. Use the metrics
together, inspect each horizon, and compare against the seasonal benchmark rather
than interpreting any single score as sufficient evidence of forecasting quality.

## Are the current datasets enough?

Yes, to build and evaluate this feature set. The local passenger file has 1,533,522
observations across 17,119 directional markets from January 2019 through July 2026.
All 150 used airports have country codes and coordinates in the airport file.
There are no duplicate market-months or missing/negative passenger values, but
6,055 markets have gaps within their observed histories. Confirm whether those gaps
represent no demand or missing reporting before treating any of them as zeros.

The main unresolved requirement is data availability: the code assumes demand
through month `t` was available when issuing a forecast. `SF_LOAD_TMS` is a load
timestamp, not evidence of the original monthly publication dates or historical
revisions. Confirm the release delay and whether these are observed demand values
before interpreting the retrospective test as operational forecast performance.
If there is a reporting delay, the feature cutoff must move back accordingly.

Outside data is optional at this stage. Establish accuracy by horizon and compare
with repeating the latest available same-calendar-month demand before adding
features. If long-horizon performance is weak, candidate additions include lagged
metro employment or population and route capacity known at the forecast date.
Never use realized future employment, traffic, or schedules that were unknown at
the forecast origin. This one cutoff does not establish robustness across economic
conditions; additional historical cutoffs are a useful next evaluation.

The full runs used 200 trees, every third training origin, XGBoost 2.1.4, and
7,269,271 training examples. Both models were evaluated on the same 410,271
market-target observations from August 2024 through July 2026:

| Model | MAE | RMSE | R² | WAPE | Signed bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original XGBoost, 12 features | 809.62 | 1,916.91 | 0.9260 | 25.71% | +10.94% |
| Expanded XGBoost, 24 features | 916.82 | 2,055.90 | 0.9149 | 29.11% | +23.35% |
| Seasonal repeat | 575.47 | 1,473.21 | 0.9563 | 18.27% | +4.08% |

The expanded features worsened overall performance under the unchanged settings.
At horizon 24, expanded-model RMSE was 2,811.47 and WAPE was 38.39%, compared with
2,359.70 and 31.52% for the original model. More features did not automatically
improve generalization. Seasonal repeat remains the best of these three methods
on this cutoff, including by RMSE and R². The expanded implementation is retained
as the requested experiment rather than presented as an accuracy improvement.

The seasonal benchmark now falls back to older observed years when the most
recent seasonal anchor is missing, so all 410,271 rows have a reference. The
previous exact-year-only reference covered 410,046 rows. The local run also saved
`results/model_comparison.csv` with the original model, expanded model, and seasonal
reference scores overall and for every horizon. Original predictions were preserved
as `results/original_test_predictions.csv`. These artifacts remain excluded from Git.

References: [XGBoost categorical features](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html),
[time-ordered evaluation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html),
[R² definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html),
[WAPE definition and limitations](https://robjhyndman.com/hyndsight/wape.html),
[BLS metropolitan employment data](https://www.bls.gov/lau/tables.htm).
