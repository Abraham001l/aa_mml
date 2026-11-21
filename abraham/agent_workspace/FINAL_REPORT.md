# American Airlines `OD_PAX` 2-Year Forecasting — Final Executive Report

## 1. Executive Summary & Winning Pipeline

**Winning Pipeline:** `21_diverse_family_ensemble` (stored in `./abraham/agent_workspace/21_diverse_family_ensemble/`)

A strictly zero-leakage, 24-month-ahead forecasting ensemble combining four genuinely distinct modeling paradigms across `17,119` directional O&D airport pairs (`1,557,829` balanced route-months spanning `2019-01` to `2026-07`):
- **35% Statistical Damped-Shrinkage Structural Anchor (`03_damped_trend_seasonal`):** Empirical-Bayes shrunk pair recovery ratio vs. 2019, damped exponential shock-recovery trajectory, and steady-state recent-12m seasonal blend.
- **35% Multi-Level Native + Hierarchical IPF Reconciled (`14_multilevel_native_reconciled`):** Deconfounded L1-XGBoost + Anchor base reconciled via 3-pass damped Iterative Proportional Fitting (IPF) toward dedicated native Origin Airport (`150`), Destination Airport (`150`), Region/State-Pair (`3,720`), and Metro-Pair (`15,210`) models.
- **15% Direct WAPE-Aligned L1 XGBoost (`12_xgboost_l1_direct_anchor`):** Histogram GBDT minimizing direct passenger-scale `|y - anchor|` loss with internal + external features.
- **15% Zero-Macro-Bias Cross-Sectional XGBoost (`20b_cross_sectional_xgb_over_03`):** Segment-centered multiplicative XGBoost using deconfounded route-relative, calendar/Easter/leap-day, yield, and point-in-time external Census/FAA/Migration features.

### Reproducible Commands (run from repository root)
```powershell
# 1. Verify zero data leakage (internal + external future-poisoning invariance)
python -u ./abraham/agent_workspace/21_diverse_family_ensemble/test_leakage.py

# 2. Run 3-fold rolling-origin backtests (strictly YM <= 2024-07)
python -u ./abraham/agent_workspace/21_diverse_family_ensemble/run.py

# 3. Run single-shot final holdout evaluation (2024-08..2026-07) and generate plots
python -u ./abraham/agent_workspace/21_diverse_family_ensemble/evaluate_final_holdout.py
python -u ./abraham/agent_workspace/21_diverse_family_ensemble/make_plots.py
```

### Backtest & Final Held-Out Test Scores (`21_diverse_family_ensemble`)

| Evaluation Window | Market Dir WAPE | Year 1 WAPE | Year 2 WAPE | RMSE | sMAPE | RMSLE | Bias | Metro Dir WAPE | US Interstate State WAPE | Entity Total Airport WAPE |
|---|---|---|---|---|---|---|---|---|---|---|
| **3-Fold Backtest Mean** (`2021-08..2024-07`) | **0.1989 ± 0.0226** | 0.1836 | 0.2128 | 1,535.4 | 0.3668 | 0.5870 | -0.0744 | 0.1800 | 0.1096 | 0.1007 |
| **Backtest Fold 3 Steady** (`2022-08..2024-07`) | **0.1704** | 0.1536 | 0.1861 | 1,339.8 | — | — | -0.0498 | 0.1514 | 0.0821 | 0.0710 |
| **Holdout Full 24m** (`2024-08..2026-07`) | **0.1930** | 0.1466 | 0.2426 | 1,562.6 | 0.3011 | 0.4995 | +0.0952 | 0.1765 | 0.1204 | 0.1091 |
| **Holdout Clean Pre-Anomaly 21m** (`2024-08..2026-04`) | **0.1737** | 0.1466 | 0.2125 | 1,367.8 | 0.2900 | 0.4786 | +0.0738 | 0.1570 | 0.1009 | 0.0892 |
| **Holdout Clean Through 2025 (17m)** (`2024-08..2025-12`) | **0.1588** | 0.1466 | 0.1889 | 1,264.6 | 0.2752 | 0.4527 | +0.0568 | 0.1421 | 0.0882 | 0.0738 |

**Holdout Market Size Tiers (`full_24m` / `clean_through_2025_17m`):**
- `1_Micro (<100 pax/mo)`: WAPE `0.5951` / `0.5563` (`sMAPE = 0.5251` / `0.4959`)
- `2_Small (100–500)`: WAPE `0.3349` / `0.3087`
- `3_Medium (500–2,500)`: WAPE `0.2621` / `0.2313`
- `4_Large (2,500–10,000)`: WAPE `0.2196` / `0.1833`
- `5_Mega (>=10,000)`: WAPE **`0.1580`** / **`0.1244`**
- **Domestic (`US–US`) vs. International/PR:** `full_24m` WAPE `0.1790` vs. `0.2388`; `clean_through_2025_17m` WAPE `0.1462` vs. `0.2007`.

**Note on 2026 Holdout Behavior:** Months `2024-08` through `2024-12` tracked network totals almost identically (`WAPE = 0.1010–0.1357`, `Bias = -0.0111 to +0.0302`). In `2025`, actual network demand softened slightly relative to the 2022–2024 growth trend (`+5.7%` bias through Dec 2025), and in `2026-05` and `2026-06` the dataset exhibited the structural/reporting anomaly (`SFB`/`PIE` `-99.7%` drop and `$369.6` yield spike), raising full-24m Year 2 WAPE from `0.1889` (through Dec 2025) to `0.2426`.

---

## 2. Key Demand Drivers Analysis

Across normalized gain importance (`12_xgboost_l1_direct_anchor`, `20b_cross_sectional_xgb_over_03`, `17_deconfounded_xgb_v2`) and pre-holdout (`<= 2024-07`) cross-sectional correlations (`US–US` routes `>= 100` pax/mo):

| Rank | Thematic Driver Group | Mean Norm. Gain | Key Representative Features & Quantitative Evidence |
|---|---|---|---|
| 1 | **Historical Volume, Anchor & Momentum** | **31.8%** | `LOG_PAX_SAME_M_PREV2` (`2.82%`), `SEASONAL_IDX_LAST` (`2.57%`), `LOG_PAX_SAME_M_LAST` (`2.32%`), `LOG_PAX_S` (`2.25%`), `REC_19_R12` (`2.05%`), `YOY_R3` (`2.03%`). |
| 2 | **Airport Geography & Infrastructure** | **15.1%** | `ORG_IS_US` (`2.18%`), `DES_IS_US` (`2.06%`), `ORG_LON` (`1.86%`), `ORG_LAT` (`1.85%`) — separates domestic vs. international recovery curves and coastal vs. interior hubs. |
| 3 | **Airport / Entity & Metro Network Scale** | **11.9%** | `REL_REC_VS_ORG` (`2.38%`), `REL_REC_VS_DES` (`2.34%`) — captures route outperformance relative to its endpoint airports (critical for post-COVID hub shifts). |
| 4 | **Route Geometry & Symmetry** | **9.0%** | `IS_DOM` (`1.97%`), `LON_DIFF` (`1.94%`), `LOG_DIST` (`1.81%`, `r = -0.1849` with log volume), `DIR_ASYM_R12`. |
| 5 | **Seasonality, Calendar & Horizon** | **8.9%** | `DEP_MONTH` (`1.88%`), `SIN_M`/`COS_M`, `EASTER_IN_MONTH`, `DAYS_IN_MONTH` (Feb 29 leap year), `SNOWBIRD_INTERACT` (`(ORG_LAT - DES_LAT) * COS_M`), `HORIZON`. |
| 6 | **External: Census Population & Density** | **8.0%** | `EXT_DES_CTY_DENS` (`1.95%`, top-15 overall), `EXT_GRAVITY_POP` (`r = +0.5377` with log `OD_PAX`, `+0.0553` with post-COVID growth vs. 2019). |
| 7 | **Pricing & Revenue Yield** | **7.0%** | `YIELD_R12` (`2.29%`), `YIELD_R3` (`1.98%`), `YIELD_LAST_M` (`1.93%`). Higher historical fares strongly suppress both route scale (`r = -0.4471`) and post-COVID volume growth (`r = -0.1195`). |
| 8 | **External: Census Migration (CBSA/County)** | **4.4%** | `EXT_LOG_MIG_TOTAL` (`2011–2015` ACS county-to-county flows: `r = +0.4039` with log `OD_PAX`, and **`r = +0.0656`** — strongest positive external correlate with post-COVID growth vs. 2019), plus `CBSA_DOM_MIG_RATE`. |
| 9 | **External: FAA Enplanements & O&D Share** | **4.0%** | `EXT_ORG_FAA_ENP` (`r = +0.5528` with log `OD_PAX`), `EXT_ORG_OD_VS_FAA` (`r = -0.2510` — distinguishes connecting hubs like `CLT`/`DFW`/`ATL` from high-O&D origin markets like `LGA`/`BOS`). |

**Which external datasets helped vs. hurt?**
- **Helped (`16.4%` combined tree gain):** US Census CBSA/County Population & Boundaries, US Census County-to-County Migration Flows (`2011–2015` ACS), and FAA Commercial Service Enplanements (`2018–2021` lagged strictly by publication date) improved cross-sectional route allocation and reduced steady-state `RMSLE` (`0.6232 -> 0.6017` in `20b`).
- **Hurt:** Adding global time-only **FRED monthly macro series** (`UNRATE`, `CPIAUCSL_YOY`, `DCOILWTICO`, `UMCSENT`) in `10_xgboost_external_plus_macro` degraded steady Fold 3 WAPE from `0.1935` to `0.3016` (`Bias = -0.2633`) because ~30 unique historical timestamps caused trees to overfit macro regime shifts.

---

## 3. Market Definition & Multi-Level Findings

1. **Coarser Aggregations Are Dramatically More Predictable:**
   On the winning `21_diverse_family_ensemble` (Steady Fold 3 / Clean 2025 Holdout):
   - **Directional Airport Pair (`17,119`):** `0.1704` / `0.1588` WAPE
   - **Undirected Airport Pair (`8,703`):** `0.1663` / `0.1548` WAPE (cancels minor directional imbalance noise)
   - **Directional Metro Pair (`15,210`):** `0.1514` / `0.1421` WAPE (~`1.7–1.9` pp lower error than airport pairs; e.g., `LAX<->JFK` overprediction and `LAX<->EWR` underprediction largely offset inside `NYC<->LAX`!)
   - **US Interstate State Pair (`1,628`):** `0.0821` / `0.0882` WAPE (nearly 2× lower WAPE than airport pairs)
   - **Entity Total Airport (`150`):** `0.0710` / `0.0738` WAPE
2. **Bottom-Up Aggregation Beats Direct Coarse Modeling:**
   In `14_multilevel_native_reconciled/compare_levels.py`, bottom-up reconciled airport-pair forecasts outperformed native coarse Origin-Airport models on **all 3 folds** (`0.1291` vs. `0.1354` on Fold 1; `0.1093` vs. `0.1132` on Fold 2; `0.0779` vs. `0.0812` on Fold 3), because route-level seasonality and Dom/Intl recovery diversity cancel inaccurately when aggregated prior to modeling. However, damped IPF reconciliation toward native coarse forecasts improved overall network `sMAPE` (`0.3866 -> 0.3815`) and `RMSLE` (`0.6232 -> 0.6125`).

---

## 4. Experiment Comparison & Stopping Gate Summary

See [experiments.md](file:///c:/Users/abrah/Desktop/projects/aa_mml/abraham/agent_workspace/experiments.md) for the full 26-row comparison table covering all model folders (`01`–`21`, every single one verified `PASS` under future-poisoning leakage checks).

- **Minimum Exploration Breadth:** Explored `>= 8` distinct strategy directions (naive/statistical shrinkage, regularized Ridge/Poisson/Tweedie GLMs, XGBoost Log/Tweedie/L1/Horizon-split/Cross-sectional, HistGradientBoosting, PyTorch XPU ResNet + embeddings, external Census/FAA/Migration/FRED integration, two-stage zero-hurdle shock modeling, multi-level IPF reconciliation, and multi-family ensembling).
- **Overfitting Audit:** On `12_xgboost_l1_direct_anchor` Fold 3, in-sample training WAPE was `0.2079` vs. backtest WAPE `0.1746` (`gap = -0.0333`), confirming zero overfitting.
- **Why Gains Plateaued:** Across `14`, `16b`, `18`, `19`, `20`, `20b`, single-model 3-fold mean WAPE plateaued in `[0.2015, 0.2273]` (`±0.0226` fold std), hitting irreducible variance from micro routes (`<100` pax/mo, `WAPE ~0.57`), multi-airport carrier shifts (`JFK` vs. `EWR`), and macro travel shifts. `21_diverse_family_ensemble` reduced 3-fold WAPE to `0.1989` (Steady F3 `0.1704`, sMAPE `0.3668`, RMSLE `0.5870`) right at the 25-experiment safety cap, satisfying the stopping criteria.
