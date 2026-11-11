# Passenger demand forecasting experiments

These scripts compare monthly demand forecasts for directional origin–destination markets. They read approved local copies of `AA_PAX_FORECASTING.csv` and `AA_AIRPORT_INFO.csv`. Put the exports in a local data directory outside Git. Keep generated results, source extracts, credentials, and private download links out of commits.

## Run the experiments

From this folder in a PowerShell terminal, set the input and output paths to locations on your computer.

```powershell
python -m pip install -r requirements.txt
$data = 'C:\path\to\team_data'
$out = Join-Path $data 'deep_dive'
$late = Join-Path $out 'late_holdout'

python .\demand_deep_dive.py --data-dir $data --output-dir $out
python .\shock_policy_backtest.py --data-dir $data --output-dir $out
python .\lag_sensitivity.py --data-dir $data --output-dir $out
python .\directional_pooling.py --data-dir $data --output-dir $out
python .\route_smoothing.py --data-dir $data --output-dir $out
python .\demand_deep_dive.py --late-holdout --data-dir $data --output-dir $late
python .\directional_pooling.py --late-holdout --data-dir $data --output-dir $late
python .\route_smoothing.py --late-holdout --data-dir $data --output-dir $late
python .\adaptive_selection.py $out $late
```

The main script supports `--baselines-only` for the five statistical methods and `--workers` to limit model fitting threads. The full run requires historical outcomes through July 2026. Use a fresh output directory for a baselines-only run.

## Optional public proxy inputs

Download the five FAA **All Commercial Service Airports (by Rank)** Excel files for calendar years 2019–2023. Save them as `2019.xlsx` through `2023.xlsx` in `faa_public` under the local output folder. The script gates availability using conservative publication dates from the FAA archive and file pages. [FAA archive](https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/previous_years)

Save the official TSA daily HTML pages for 2019–2025 as `2019.html` through `2025.html` under `tsa_public`. The recent holdout also uses the 2026 page saved as `2026.html`. The code uses the last fully completed month at each forecast date. The current historical pages may incorporate revisions. [TSA passenger volumes](https://www.tsa.gov/travel/passenger-volumes) · [BTS description of the screening release cadence](https://www.bts.gov/browse-statistical-products-and-data/preliminary-estimates/preliminary-estimates-enplanements-tsa)

```powershell
$faa = Join-Path $out 'faa_public'
$tsa = Join-Path $out 'tsa_public'
python .\faa_backtest.py --data-dir $data --faa-dir $faa --output-dir $out
python .\tsa_backtest.py --data-dir $data --tsa-dir $tsa --output-dir $out
python .\public_proxy_holdout.py --data-dir $data --faa-dir $faa --tsa-dir $tsa --output-dir $out
python .\public_proxy_holdout.py --late-holdout --data-dir $data --faa-dir $faa --tsa-dir $tsa --output-dir $late
python .\summarize_deep_dive.py $out
python .\summarize_deep_dive.py $late
```

The report command also works without the optional proxy experiments. It checks that method comparisons use the same scoring samples and that training labels honor the reporting delay.

## Experiment design

- Simulate an 18- or 24-month reporting delay. A forecast can use demand and revenue only through its issue month minus that delay. Model training has the same restriction on observed outcomes. A delay is a scenario until release metadata confirms it.
- Start with the latest eligible same-month demand. Compare a weighted seasonal forecast, a selective shock fallback, dropping flagged anchors, and variants that use local or regional shock rules. The shock threshold was designed from a known historical disruption. A sensitivity test varies the drop required to trigger it.
- Separate seasonal shape from route-level demand, or forecast total network demand using a damped trend and distribute the network prediction using historical route shares. Separate experiments pool opposite directions using an eligible directional share and fit route-level exponential smoothing on log demand. The bounded smoothing variant was added after the unbounded version showed large errors.
- Fit boosted trees and a random forest to learn a bounded correction to the shock fallback. Core inputs are target season, horizon, reference demand, recent demand relative to the reference, and history counts. The context model adds distance, U.S. endpoint flags, historical revenue per passenger, and historical route and airport growth. The contextual feature comparison changes this block together.
- Test FAA boarding growth and TSA screening growth on U.S. domestic routes. These public counts proxy for travel activity. Airport boardings can include connections, and TSA screening totals measure a national level. The FAA holdout uses the available 2019–2023 files. The original 2024 workbook was unavailable at the archived URL.
- Compare WAPE, signed bias, and coverage on observed markets with a common seasonal anchor. Compare lower, middle, and higher volume thirds, assigned using eligible reference demand. The latest holdout supports one- and six-month tests because later outcomes are available for fewer horizons. The adaptive selector tests whether earlier scores can identify a method using only outcomes available at issue time.

Exact package versions used for the initial full run were Python 3.12, pandas 3.0.1, NumPy 2.5.4, scikit-learn 1.9.1, openpyxl 3.1.5, lxml 6.1.1, and threadpoolctl 3.8.0.

[Holt damped trend reference](https://otexts.com/fpp3/holt.html) · [Scikit-learn boosted trees](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) · [Scikit-learn random forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

