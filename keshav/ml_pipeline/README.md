# XGBoost monthly passenger demand

Train one XGBoost regressor to predict `OD_PAX` for a directional airport pair at
month `t+h`, for every horizon `h=1..24`. Opposite directions remain separate markets.
The script only prepares features, trains the model, and tests it. It writes
`test_predictions.csv` and `test_metrics.csv` (MAE, RMSE, and WAPE overall and by horizon).

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
This is a fixed train/test evaluation, with no tuning against the test set.

Run the calendar-alignment and leakage checks with:

```sh
python3 -m unittest discover -s keshav/ml_pipeline -p 'test_*.py' -v
```

## Features

| Feature | Definition |
| --- | --- |
| `ORG_AIRPRT`, `DES_AIRPRT` | Categorical airport codes, with category mappings fitted on training examples. |
| `distance_km` | Great-circle distance derived from airport coordinates. |
| `is_domestic` | 1 when both airports are in the same country, including non-US markets. |
| `target_month` | Calendar month 1–12 of `t+h`, for seasonality. |
| `forecast_horizon` | Number of months ahead, 1–24. |
| `pax_lag_0`, `pax_lag_1`, `pax_lag_12` | Demand at `t`, `t-1`, and `t-12`, using exact calendar months. |
| `pax_rolling_3`, `pax_rolling_12` | Mean demand over the 3 or 12 consecutive months ending at `t`, including `t`. |
| `pax_yoy_growth` | `OD_PAX(t) / OD_PAX(t-12) - 1`; missing if the reference is missing or zero. |

Missing months remain unknown rather than becoming zero demand. Rolling means
require a complete window; XGBoost handles missing history features natively.
Revenue, airport country identifiers, plots, validation fits, model exports, and
production forecasting are omitted to keep this a single train/test workflow.

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

The initial default run (200 trees, every third training origin, XGBoost 2.1.4)
scored 410,271 test examples: MAE **809.62 passengers**, RMSE **1,916.91**, and
WAPE **25.71%**. At horizon 24, WAPE was **31.52%**. Lower errors are better.
A separate seasonal-repeat check on the 410,046 rows with an available historical
anchor scored **18.27%** WAPE, compared with **25.71%** for XGBoost on those same
rows. Thus, this simple model does not yet outperform that reference. This supports
checking model design and performance at additional cutoffs before assuming that
outside data is the missing ingredient. The reference check is not an additional
model fit in the pipeline, and the test scores were not used to tune parameters.

References: [XGBoost categorical features](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html),
[time-ordered evaluation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html),
[BLS metropolitan employment data](https://www.bls.gov/lau/tables.htm).
