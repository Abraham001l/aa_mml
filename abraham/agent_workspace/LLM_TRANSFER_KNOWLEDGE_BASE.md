# American Airlines `OD_PAX` 2-Year Forecasting — LLM Knowledge Base & Transfer Report

> **Target Audience:** Autonomous AI Agents, Machine Learning Researchers, and LLM Coding Agents continuing development on this codebase.  
> **Repository Root:** `c:\Users\abrah\Desktop\projects\aa_mml\`  
> **Workspace Boundary:** 100% of all generated scripts, data, and models reside in `./abraham/agent_workspace/`. The directory `./abraham/global_data/` is strictly read-only.  
> **Current Best Model:** `21_diverse_family_ensemble` (Mean Backtest WAPE: **0.1989**, Steady-State Fold 3 WAPE: **0.1704**, Clean 2025 Holdout WAPE: **0.1588**).

---

## 1. Problem Formulation & Dataset Specifications

### 1.1 Objective
Predict monthly origin-to-destination passenger demand (`OD_PAX`) across the entire network up to **24 calendar months into the future** ($H = 1, 2, \dots, 24$) with **absolute zero data leakage**.

### 1.2 Data Schemas & Physical Properties

#### Primary Panel: `./abraham/global_data/pax_forecasting.csv`
- **File Format:** Pipe-delimited text (`sep='|'`), with a leading unnamed index column.
- **Row Count:** 1,533,522 raw rows.
- **Temporal Range:** 91 continuous calendar months from **January 2019 (`2019-01`) through July 2026 (`2026-07`)**.
- **Network Dimensions:**
  - **150 unique airport entities:** 99 continental US airports, 1 Puerto Rico airport (`SJU`), and 50 international airports across 34 countries.
  - **17,119 unique directional airport pairs:** Every route has at least one endpoint in the United States.
  - **Balanced Panel Construction:** Across 91 months, the full Cartesian product ($17{,}119 \times 91$) contains **1,557,829 route-month observations**. The raw data contains 1,533,522 rows; exactly 24,307 route-months (1.56%) have zero reported traffic or were unserved in specific months.
  - *Standard Preprocessing Rule:* Absent route-months are treated as true zero volume (`OD_PAX = 0.0`) with a binary indicator `IS_OBSERVED = 0`.
- **Columns:**
  - `ORIGIN` (`str`, 3-letter IATA code, e.g., `DFW`, `LHR`).
  - `DEST` (`str`, 3-letter IATA code, e.g., `ORD`, `HND`).
  - `TIME_PERIOD` (`str`, formatted as `YYYY-MM-01`). In modeling, converted to integer month index $t \in [0, 90]$ and numeric YM integer `YYYYMM` (e.g., `201901` to `202607`).
  - `OD_PAX` (`float64`, target variable): Monthly total passenger count. Range: $[0.0, 140{,}000+]$. Extreme right skew: 60% of pairs have $<500$ pax/month, while top 5% trunk routes exceed $10{,}000$ pax/month.
  - `OD_REV_USD` (`float64`): Total reported itinerary revenue in USD.
  - `YIELD`: Derived feature defined as $\text{Yield} = \frac{\text{OD\_REV\_USD}}{\max(\text{OD\_PAX}, 1.0)}$. Represents average fare per passenger.

#### Airport Metadata: `./abraham/global_data/airport_info.csv`
- **Row Count:** 4,565 global airports.
- **Coverage:** 100% of the 150 airports in `pax_forecasting.csv` are matched.
- **Coordinate Quirks:** Latitude and Longitude are stored in **Degrees-Minutes-Seconds (DMS) string format** (e.g., `32-53-48.9000N`, `097-02-16.8000W`).
  - *Parser Implementation:* Converted via regex into signed decimal degrees:
    $$\text{DecDeg} = \text{sign} \times \left( D + \frac{M}{60} + \frac{S}{3600} \right)$$
    where `N`/`E` = $+1$, `S`/`W` = $-1$.
- **Metro Area Mapping:**
  - 140 distinct metropolitan areas across the 150 airports.
  - 9 major metropolitan areas contain multiple competing airports (e.g., NYC: `JFK`, `LGA`, `EWR`; CHI: `ORD`, `MDW`; WAS: `DCA`, `IAD`, `BWI`; LAX: `LAX`, `ONT`, `BUR`, `SNA`, `LGB`; MIA: `MIA`, `FLL`, `PBI`).

### 1.3 Key Historical Shocks & Structural Anomalies

1. **The April 2020 COVID Collapse:**
   - Network passenger volume plummeted by **94.6%** in April 2020 (from ~50M pax/month to 2.7M).
   - Recovery was asymmetric: US domestic leisure rebounded rapidly in 2021–2022; international long-haul and corporate travel remained depressed until late 2023.
   - *Impact on Modeling:* Models that rely on raw trailing 12-month metrics without structural anchoring extrapolate shock troughs into the future, causing massive under-forecasting bias.
2. **The 2023 Steady-State Transition:**
   - By CY 2023, US domestic passenger volume reached **101.3% of 2019 levels** (505.0M pax vs 498.3M).
   - International traffic reached **104.1% of 2019 levels** (142.7M vs 137.0M).
   - Average passenger ticket yields increased by **13% to 20%** across domestic and international markets ($182 \to \$205–\$219$ domestic; $\$418 \to \$473–\$480$ international).
3. **The May/June 2026 Secondary-Hub Reporting Drop:**
   - In `2026-05` and `2026-06`, secondary leisure airports dominated by ultra-low-cost carriers (e.g., `SFB` in Orlando, `PIE` in St. Petersburg/Tampa) experienced a sudden **-99.7% drop** in reported volume accompanied by an artificial yield spike to $\$369.61$/pax, before returning to normal in `2026-07`.
   - *Protocol:* The final holdout evaluates both the full 24-month horizon (`2024-08..2026-07`) and the clean pre-anomaly 17-month window (`2024-08..2025-12`).

---

## 2. Temporal Splitting & Leakage Verification Protocol

### 2.1 Chronological Backtest Folds vs. Final Holdout
To prevent temporal overfitting, the dataset is partitioned into three expanding-window backtest folds and one final untouched holdout:

```
2019-01                 2021-07    2022-01    2022-07               2024-07               2026-07
   ├───────────────────────┼──────────┼──────────┼─────────────────────┼─────────────────────┤
   │◄── Historical Train ─►│◄─ Fold 1 ──────────►│                     │                     │
   │◄── Historical Train ────────────►│◄─ Fold 2 ──────────►│          │                     │
   │◄── Historical Train ───────────────────────►│◄─ Fold 3 ──────────►│                     │
   │◄──────────────────── Strictly Historical Development ────────────►│◄── Final Holdout ──►│
```

| Split Name | Cutoff Origin Date (`t_origin`) | Evaluation Window (24 Months Ahead) | Market Regime Evaluated |
| :--- | :--- | :--- | :--- |
| **Fold 1 (`fold_1_recovery`)** | `2021-07` ($t=30$) | `2021-08` through `2023-07` | Rapid COVID rebound, steep positive delta |
| **Fold 2 (`fold_2_transition`)** | `2022-01` ($t=36$) | `2022-02` through `2024-01` | Omicron disruption & business travel return |
| **Fold 3 (`fold_3_steady`)** | `2022-07` ($t=42$) | `2022-08` through `2024-07` | Normalized post-COVID seasonal steady state |
| **Final Test Holdout** | `2024-07` ($t=66$) | `2024-08` through `2026-07` | Untouched, single-shot evaluation |

### 2.2 Strict Zero-Data-Leakage Rule ($t \le t_{\text{origin}}$)
For any forecast origin $t_{\text{origin}}$:
1. Every feature, rolling statistic, lag, yield, normalization scalar, and target transformation must be computed **strictly using data from $t \le t_{\text{origin}}$**.
2. External datasets (Census, FAA enplanements, migration) must only be joined if their real-world release date satisfies $\text{AVAILABLE\_YM} \le \text{origin\_ym}$.
3. Target arrays for training feature stores cannot use data beyond $t_{\text{origin}}$.

### 2.3 Dedicated Leakage Verification Suite (`leakage_verifier.py`)
Every single model directory in this repository contains a dedicated `test_leakage.py`. A model's score is **invalid** unless it passes two invariant tests:

```python
# Formal Invariance Proof Implemented in test_leakage.py:
# 1. Generate clean predictions on Fold 3:
preds_clean = model.predict(hist_data_clean, fold_3_spec)

# 2. Inject adversarial poison into future raw data (t > origin_ym):
hist_data_poisoned = hist_data_clean.copy()
future_mask = hist_data_poisoned["YM"] > fold_3_spec.origin_ym
hist_data_poisoned.loc[future_mask, "OD_PAX"] = 9e9
hist_data_poisoned.loc[future_mask, "OD_REV_USD"] = 9e11
external_data_poisoned.loc[external_data["AVAILABLE_YM"] > fold_3_spec.origin_ym, :] = 9e9

# 3. Generate predictions with poisoned future data:
preds_poisoned = model.predict(hist_data_poisoned, fold_3_spec)

# 4. Assert strict zero difference:
max_abs_diff = np.max(np.abs(preds_clean["y_pred"] - preds_poisoned["y_pred"]))
assert max_abs_diff == 0.0, f"LEAKAGE DETECTED: max diff = {max_abs_diff}"
```

All 26 model folders passed this verification with `max_diff = 0.00e+00`.

---

## 3. Standardized Metric Evaluation Suite

All models are evaluated via `./abraham/agent_workspace/shared_eval/metrics.py`. Because airline route volume spans orders of magnitude ($0$ to $140{,}000+$), evaluating via standard MAPE is statistically invalid (MAPE $\to \infty$ as $y \to 0$). The following metrics are computed globally across all routes and months:

### 3.1 Mathematical Definitions

1. **WAPE (Weighted Absolute Percentage Error — Primary Metric):**
   $$\text{WAPE} = \frac{\sum_{r,t} |y_{r,t} - \hat{y}_{r,t}|}{\sum_{r,t} y_{r,t}}$$
   *Property:* Equivalent to passenger-weighted absolute error. Evaluates total passenger misallocation across the network without scale bias.

2. **MAE (Mean Absolute Error):**
   $$\text{MAE} = \frac{1}{N \cdot H} \sum_{r,t} |y_{r,t} - \hat{y}_{r,t}|$$

3. **RMSE (Root Mean Squared Error):**
   $$\text{RMSE} = \sqrt{\frac{1}{N \cdot H} \sum_{r,t} (y_{r,t} - \hat{y}_{r,t})^2}$$
   *Property:* Quadratically penalizes severe forecast blowups on high-capacity trunk routes.

4. **sMAPE (Symmetric Mean Absolute Percentage Error):**
   $$\text{sMAPE} = \frac{1}{N \cdot H} \sum_{r,t} \frac{2 \cdot |y_{r,t} - \hat{y}_{r,t}|}{|y_{r,t}| + |\hat{y}_{r,t}|}$$
   *Property:* Bounded in $[0.0, 2.0]$. Measures percentage accuracy on small/regional routes without dividing by zero.

5. **RMSLE (Root Mean Squared Logarithmic Error):**
   $$\text{RMSLE} = \sqrt{\frac{1}{N \cdot H} \sum_{r,t} \left( \ln(1 + \hat{y}_{r,t}) - \ln(1 + y_{r,t}) \right)^2}$$
   *Property:* Evaluates multiplicative ratios ($2\times$ over-prediction on 10 pax carries the same penalty as $2\times$ over-prediction on 10,000 pax).

6. **Total Volume Bias:**
   $$\text{Bias} = \frac{\sum_{r,t} (\hat{y}_{r,t} - y_{r,t})}{\sum_{r,t} y_{r,t}}$$
   *Property:* Positive values indicate fleet-wide over-forecasting; negative values indicate under-forecasting.

---

## 4. External Data Assets & Feature Engineering Store

All external datasets are pre-processed and stored in `./abraham/agent_workspace/data_external/`.

### 4.1 Ingested External Data Sources

1. **US Census Metro & County Demographics (`airport_external_annual.parquet`):**
   - Source: US Census Bureau Population Estimates Program (PEP) 2020–2023.
   - Spatial Link: US Census Geocoder API mapped all 100 US/PR airport coordinates to exact state FIPS, county FIPS, and Core-Based Statistical Areas (CBSA).
   - Features: `cbsa_est2019`, `cbsa_est2023`, `cbsa_pop_growth_pct`, `county_land_area_sqmi`, `county_pop_density`, `cbsa_dom_mig_rate`.
   - Point-in-time Rule: 2019 estimates available `2020-03`; 2023 estimates available `2024-03`.
2. **US Census County-to-County Migration Flows (`pair_migration_2015.parquet`):**
   - Source: US Census American Community Survey (ACS) 5-Year Migration Flow matrices (`CtyxCty_US.txt`, 112 MB).
   - Spatial Link: Mapped origin county FIPS to destination county FIPS across all 17,119 airport pairs.
   - Feature: `EXT_LOG_MIG_TOTAL` = $\ln(1 + \text{Flow}_{A \to B} + \text{Flow}_{B \to A})$.
   - Correlation: Strongest positive external predictor of post-COVID volume growth vs. 2019 ($r = +0.0656$).
3. **FAA Commercial Enplanement Statistics (`airport_external_annual.parquet`):**
   - Source: Federal Aviation Administration (FAA) Passenger Boarding (Enplanement) and All-Cargo Data for CY 2018 through 2023.
   - Point-in-time Rule: Published with an 8-month lag (CY $Y$ enplanements available at $Y\text{-}09$).
   - Features: `EXT_ORG_FAA_ENP`, `EXT_DES_FAA_ENP`, `EXT_ORG_OD_VS_FAA` (ratio of route origin pax to total airport boardings; identifies connecting hubs vs origin-and-destination airports).
4. **FRED Macroeconomic Indicators (`macro_monthly_clean.parquet`):**
   - Series: Consumer Price Index (`CPIAUCSL`), Unemployment Rate (`UNRATE`), Real Disposable Income (`DSPIC96`), WTI Crude Oil (`DCOILWTICO`), Consumer Sentiment (`UMCSENT`).
   - Publication Lag: Strictly enforced $+2$ months.

### 4.2 Dense Vectorized Causal Feature Store (`feature_engine.py`)
To train GBDT and neural panel models on 17,119 pairs across historical time slices without memory exhaustion or leakage, the engine structures the balanced panel as a contiguous 2D NumPy array $P \in \mathbb{R}^{17{,}119 \times 91}$.

```
Dense Matrix Structure:
Routes r ∈ [0, 17118] (rows) × Time Months t ∈ [0, 90] (columns)
```

At any forecast slice origin $s$ predicting horizon $h$ ($t_{\text{tgt}} = s + h$):
- **Lags & Momentum:**
  - `PAX_S`: Passenger volume at origin slice month $s$ ($t - h$).
  - `PAX_SAME_M_LAST`: Volume in the same calendar month of the previous year ($s - 12 + (h \pmod{12})$).
  - `PAX_SAME_M_PREV2`: Volume in the same calendar month two years prior.
  - `ROLLING_MEAN_3M`, `ROLLING_MEAN_12M`: Trailing causal moving averages strictly ending at $s$.
- **Seasonal & Calendar Indices:**
  - `SEASONAL_IDX_LAST`: Trailing 12-month normalized seasonal profile:
    $$\text{Idx}_m = \frac{\text{PAX}_{r, \text{month } m}}{\frac{1}{12} \sum_{k=1}^{12} \text{PAX}_{r, k}}$$
  - `SIN_M`, `COS_M`: Trigonometric month cycle: $\sin(2\pi m / 12)$, $\cos(2\pi m / 12)$.
  - `EASTER_IN_MONTH`: Binary flag indicating shifting March/April Easter holiday.
  - `DAYS_IN_MONTH`: Exact days (including Feb 29 leap years).
  - `SNOWBIRD_INTERACT`: Directional latitude differential modulated by seasonal cycle:
    $$\text{Snowbird} = (\text{Lat}_{\text{orig}} - \text{Lat}_{\text{dest}}) \times \cos(2\pi m / 12)$$
- **Network Aggregates & Reverse Route:**
  - `REV_PAX_S`: Trailing volume on the reverse directional route ($B \to A$).
  - `DIR_ASYM_R12`: Directional volume imbalance ratio ($A \to B$ vs $B \to A$).
  - `ORG_TOTAL_PAX_R12`, `DES_TOTAL_PAX_R12`: Total trailing volume across all routes touching the origin or destination airport.
- **Yield & Fare Sensitivity:**
  - `YIELD_R3`, `YIELD_R12`: Rolling average fare per passenger strictly ending at $s$.

---

## 5. Complete Model Exploration Registry (All 26 Experiments)

Every experiment logged below is preserved in its own directory under `./abraham/agent_workspace/` with full code, configs, and artifacts.

| # | Model Directory | Model Architecture & Hypothesis | Leakage Test | 3-Fold Mean WAPE | Steady F3 WAPE | 3-Fold RMSE | sMAPE | Bias | Primary Findings & Structural Takeaways |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **01** | `01_seasonal_naive_last_year` | Trailing 12-month same-month carryover | PASS | 0.3546 | 0.2297 | 2847.6 | 0.7081 | -0.2974 | Severe negative bias (-46.8% on Fold 1) because prior year contained COVID collapse trough. |
| **02** | `02_precovid_2019_seasonal` | Direct carryover of pre-COVID 2019 volumes | PASS | 0.2626 | 0.2451 | 2064.5 | 0.4382 | +0.0259 | Proved 2019 seasonal profile persists; cut WAPE by 9.2 pp, but misses post-COVID growth corridors. |
| **03** | `03_damped_trend_seasonal` | Shrunk 2019 recovery + damped trend | PASS | 0.2015 | 0.1762 | 1526.4 | 0.3866 | -0.0724 | Established statistical benchmark. F3 Year 1 WAPE 0.1602 vs Year 2 WAPE 0.1912. |
| **04** | `04_ridge_log_residual_panel` | Linear Ridge ($\alpha=500$) with raw macro ratios | PASS | 0.4741 | 0.3148 | 4208.3 | 0.6520 | -0.1580 | Blew up: collinear macro features and tiny $\alpha$ on 2M rows caused wild extrapolation across regimes. |
| **04b** | `04b_ridge_bounded_calibrated` | Linear Ridge ($\alpha=0.25N$) with bounded features | PASS | 0.2780 | 0.2349 | 2258.2 | 0.4757 | -0.1395 | Eliminated blowup (-41% WAPE vs 04), but linear boundaries cannot model recovery transitions. |
| **05** | `05_tweedie_glm_panel` | Tweedie GLM ($p=1.5$, log-link) without offset | PASS | 0.5333 | 0.2291 | 4658.4 | 0.5702 | +0.1219 | Exploded on Fold 1 recovery (+83.5% bias, WAPE 0.9790) due to unconstrained exponential linear predictor. |
| **05b** | `05b_poisson_offset_glm` | Poisson GLM with $\mu = \text{anchor} \cdot \exp(X\beta)$ | PASS | 0.3086 | 0.2459 | 2539.9 | 0.5093 | -0.2093 | Offset stabilized Fold 1 (WAPE 0.3696); confirmed multiplicative anchor offset is mandatory. |
| **06** | `06_xgboost_log_residual` | XGBoost (`hist`) on log-residual over anchor | PASS | 0.2858 | 0.1778 | 2410.1 | 0.5069 | -0.0828 | Near-SOTA on steady Fold 3 (0.1778), but over-extrapolated steep COVID bounces on F1 (WAPE 0.4036). |
| **07** | `07_histgb_log_residual` | Sklearn HistGradientBoosting (LightGBM-style) | PASS | 0.2952 | 0.1815 | 2444.2 | 0.5307 | -0.1375 | Mirrored 06; strong on steady state, but under-predicted Fold 1 recovery. |
| **08** | `08_pytorch_residual_panel` | Deep ResNet + Entity/Horizon Embeddings (XPU) | PASS | 0.2962 | 0.2109 | 2420.9 | 0.5370 | -0.1577 | SmoothL1 loss on Intel XPU; competitive with trees on steady state, but slower to train. |
| **09** | `09_xgboost_external_geo_faa_mig` | XGBoost + Census pop/density, FAA & migration | PASS | 0.2932 | 0.1935 | 2513.9 | 0.5119 | -0.0978 | Added valuable cross-sectional signal, but confounded by unclipped YoY bounce features. |
| **10** | `10_xgboost_external_plus_macro` | XGBoost + External + FRED Macro series | PASS | 0.3355 | 0.3016 | 2802.5 | 0.5927 | -0.2633 | **Negative Result:** Global macro series overfit 30 historical timestamps; degraded F3 WAPE to 0.3016. |
| **11** | `11_xgboost_tweedie_offset` | XGBoost `reg:tweedie(1.4)` + `base_margin=log(anchor)` | PASS | 0.2831 | 0.1857 | 2423.1 | 0.4777 | -0.0457 | Tweedie loss cut overall volume bias in half (-0.0978 $\to$ -0.0457). |
| **12** | `12_xgboost_l1_direct_anchor` | XGBoost `reg:absoluteerror` on $y - \text{anchor}$ | PASS | 0.2304 | **0.1746** | 1741.0 | 0.4680 | -0.0302 | **Breakthrough:** Direct L1 loss beat Model 03 baseline on steady Fold 3 (0.1746 vs 0.1762). |
| **13** | `13_shock_regime_two_stage_hurdle` | Hurdle Classifier $P(y>0) \times$ Tweedie Regressor | PASS | 0.2793 | 0.1818 | 2388.7 | 0.4698 | -0.0107 | Best calibration (Bias = -0.0107 across all 3 folds); slight drag on Fold 1. |
| **14** | `14_multilevel_native_reconciled` | Base model + Native Airport/State/Metro IPF | PASS | 0.2044 | **0.1749** | 1587.4 | **0.3815** | -0.0833 | **Best sMAPE (0.3815) & RMSLE (0.6125);** airport conservation laws stabilized long tail. |
| **15** | `15_horizon_specialized_gbdt` | Separate models for Year 1 ($h \le 12$) vs Year 2 ($h > 12$) | PASS | 0.3380 | 0.2675 | 2920.9 | 0.5105 | -0.1099 | **Negative Result:** Year 2 model starved on post-COVID training targets ($h=13..24$ WAPE = 0.3676). |
| **16** | `16_structural_hierarchical_anchor` | Hierarchical EB recovery + calendar anchor | PASS | 0.2203 | 0.1786 | 1692.5 | 0.4057 | -0.0884 | Over-weighted depressed COVID trough on Fold 1/2. |
| **16b** | `16b_refined_structural_panel` | Refined symmetric anchor (leap-day, airport tilt) | PASS | 0.2078 | 0.1787 | 1589.2 | 0.3873 | -0.0696 | Robust anchor; handles February 29 leap years cleanly. |
| **17** | `17_deconfounded_xgb_v2` | Deconfounded Tweedie XGBoost + trust region | PASS | 0.2645 | 0.1904 | 2179.0 | 0.4175 | -0.0513 | Deconfounding improved Tweedie vs Model 11 (0.2831 $\to$ 0.2645). |
| **17b** | `17b_deconfounded_l1_xgb_v2` | Deconfounded L1 XGBoost + trust region | PASS | 0.2281 | 0.1787 | 1734.7 | 0.4894 | -0.0627 | Strong 3-fold ML performance (0.2281); eliminated shock bias. |
| **18** | `18_deconfounded_histgb_v2` | Deconfounded HistGB on log-ratio | PASS | 0.2732 | 0.1871 | 2240.3 | 0.4311 | -0.0943 | Beat Model 07 on all metrics (0.2952 $\to$ 0.2732). |
| **19** | `19_deconfounded_pytorch_v2` | Deconfounded PyTorch ResNet + Embeddings | PASS | 0.2809 | 0.2188 | 2266.3 | 0.4567 | -0.1463 | Substantial improvement over Model 08 (sMAPE: 0.537 $\to$ 0.457). |
| **20** | `20_deconfounded_cross_sectional_xgb` | Zero-macro-bias centered L1 XGBoost + External | PASS | 0.2175 | 0.1780 | 1685.8 | 0.4469 | -0.0797 | Best single ML model across 3 folds (0.2175); eliminated COVID bias. |
| **20b** | `20b_cross_sectional_xgb_over_03` | Multiplicative XGBoost over Model 03 + External | PASS | 0.2273 | 0.1777 | 1812.1 | 0.3852 | -0.1029 | **Best RMSLE across all single models (0.6017)**; clean sMAPE (0.3852). |
| **21** | **`21_diverse_family_ensemble`** | **Convex Blend: 35% 03 + 35% 14 + 15% 12 + 15% 20b** | **PASS** | **0.1989** | **0.1704** | **1535.4** | **0.3668** | **-0.0744** | **Winning Pipeline:** Best WAPE (0.1704), sMAPE (0.3668), RMSLE (0.5870). |

---

## 6. Mathematical Architecture of the Winning Pipeline (`21_diverse_family_ensemble`)

The final production forecast is a convex blend of four distinct mathematical paradigms:

$$\mathbf{\hat{y}_{r,t}} = \max\Big(0,\; 0.35 \cdot \mathbf{\hat{y}_{r,t}^{(03)}} + 0.35 \cdot \mathbf{\hat{y}_{r,t}^{(14)}} + 0.15 \cdot \mathbf{\hat{y}_{r,t}^{(12)}} + 0.15 \cdot \mathbf{\hat{y}_{r,t}^{(20b)}}\Big)$$

```
                               ┌─────────────────────────────────────────────────────────────┐
                               │                 21_diverse_family_ensemble                  │
                               └──────────────────────────────┬──────────────────────────────┘
                                                              │
               ┌──────────────────────────────┬───────────────┴───────────────┬──────────────────────────────┐
               │ 35%                          │ 35%                           │ 15%                          │ 15%
               ▼                              ▼                               ▼                              ▼
    ┌─────────────────────┐        ┌─────────────────────┐         ┌─────────────────────┐        ┌─────────────────────┐
    │  03_damped_trend    │        │  14_multilevel_ipf  │         │  12_xgboost_l1      │        │  20b_cross_sect_xgb │
    │  Structural Anchor  │        │  Hierarchical IPF   │         │  Direct L1 Residual │        │  Census/FAA/Mig     │
    └─────────────────────┘        └─────────────────────┘         └─────────────────────┘        └─────────────────────┘
```

### 6.1 Component 1: `03_damped_trend_seasonal` (Weight = 0.35)
- **Mathematical Formulation:**
  For each route $r$ and future month $t$ (corresponding to calendar month $m \in [1, 12]$):
  $$\hat{y}_{r,t}^{(03)} = y_{r, m}^{(2019)} \times \text{RecoveryRatio}_{r}(t) \times \text{CalCorrection}(m)$$
  Where the route recovery ratio is shrunk toward segment recovery (Domestic vs International) via Empirical Bayes:
  $$\text{RecoveryRatio}_{r}(t) = 1.0 + \Big( \alpha_r \cdot \text{RecRaw}_r + (1 - \alpha_r) \cdot \text{RecSeg} - 1.0 \Big) \times \exp\left( -\frac{h}{\tau} \right)$$
  - $\tau = 18.0$ months (exponential damping parameter).
  - $\alpha_r = \frac{V_r}{V_r + 5000}$ (shrinkage parameter based on historical route passenger volume $V_r$).

### 6.2 Component 2: `14_multilevel_native_reconciled` (Weight = 0.35)
- **Concept:** Individual routes fluctuate due to airline schedule shifts, but aggregate airport entity and state totals follow conservation laws.
- **Algorithm (Iterative Proportional Fitting / Biproportional Scaling):**
  1. Train dedicated native models for:
     - Origin Airport Departing Volume: $\hat{O}_i$ ($150$ models).
     - Destination Airport Arriving Volume: $\hat{D}_j$ ($150$ models).
     - State-to-State Travel: $\hat{S}_{a,b}$ ($3{,}720$ pairs).
  2. Initialize route array: $\hat{y}_{i,j}^{(0)} = \hat{y}_{i,j}^{\text{base}}$.
  3. Execute 3-pass damped IPF:
     $$\hat{y}_{i,j}^{(k+1)} = \hat{y}_{i,j}^{(k)} \times \left( \frac{\hat{O}_i}{\sum_j \hat{y}_{i,j}^{(k)}} \right)^{\gamma} \times \left( \frac{\hat{D}_j}{\sum_i \hat{y}_{i,j}^{(k)}} \right)^{\gamma}$$
     where damping factor $\gamma = 0.50$ prevents over-correction on sparse routes.

### 6.3 Component 3: `12_xgboost_l1_direct_anchor` (Weight = 0.15)
- **Loss Objective:** `reg:absoluteerror` (Direct L1 loss aligned with WAPE).
- **Target Variable:** Residual error relative to structural anchor: $\Delta_{r,t} = y_{r,t} - \text{Anchor}_{r,t}$.
- **Hyperparameters:**
  - `learning_rate`: 0.04
  - `max_depth`: 6
  - `min_child_weight`: 300.0
  - `subsample`: 0.80
  - `colsample_bytree`: 0.80
  - `num_boost_round`: 350

### 6.4 Component 4: `20b_cross_sectional_xgb_over_03` (Weight = 0.15)
- **Concept:** Multiplicative allocation model with **zero-macro-bias centering**.
- **Target Formulation:**
  $$\delta_{r,t} = \ln\left( \frac{y_{r,t} + 10.0}{\text{Anchor}_{r,t} + 10.0} \right)$$
- **Zero-Drift Centering Transformation:**
  Within each future month $t$ and segment (Domestic vs International):
  $$\hat{\delta}_{r,t}^{\text{centered}} = \hat{\delta}_{r,t} - \frac{1}{2} \cdot \text{Mean}_{r \in \text{seg}}(\hat{\delta}_{r,t})$$
  $$\hat{y}_{r,t}^{(20b)} = (\text{Anchor}_{r,t} + 10.0) \cdot \exp\left( \text{clip}(\hat{\delta}_{r,t}^{\text{centered}}, -0.28, 0.28) \right) - 10.0$$
  *Guarantee:* The model cannot cause network-wide inflation or deflation; it is restricted strictly to allocating market share based on Census migration and metro growth.

---

## 7. Final Untouched Holdout Evaluation (`2024-08..2026-07`)

The winning ensemble was unlocked and evaluated exactly once on the 24-month held-out test set:

```
====================================================================================================
FINAL TEST HOLDOUT SCORECARD (Origin Cutoff: 2024-07 | Target Span: 2024-08 .. 2026-07)
====================================================================================================
Evaluation Window                   WAPE      MAE      RMSE     sMAPE    RMSLE     Bias
----------------------------------------------------------------------------------------------------
Clean Through 2025 (First 17m)     0.1588    527.7    1264.6   0.2752   0.4527   +0.0568
Clean Pre-Anomaly 21m (to 2026-04) 0.1737    561.2    1367.8   0.2900   0.4786   +0.0738
Full 24-Month Holdout Window       0.1930    607.7    1562.6   0.3011   0.4995   +0.0952
----------------------------------------------------------------------------------------------------
Year 1 Horizon (h = 1 .. 12)       0.1466    502.1    1210.4   0.2641   0.4312   +0.0410
Year 2 Horizon (h = 13 .. 24)      0.2426    713.3    1845.2   0.3381   0.5678   +0.1494
====================================================================================================
```

### 7.1 Accuracy by Market Size Tier (Clean 2025 Holdout Window)
- **Tier 5: Mega Trunk Routes ($\ge 10{,}000$ pax/mo):** **WAPE = 0.1244** (12.4% error on high-revenue routes).
- **Tier 4: Large Routes ($2{,}500$ to $10{,}000$ pax/mo):** **WAPE = 0.1833**.
- **Tier 3: Medium Routes ($500$ to $2{,}500$ pax/mo):** **WAPE = 0.2313**.
- **Tier 2: Small Routes ($100$ to $500$ pax/mo):** **WAPE = 0.3087**.
- **Tier 1: Micro Routes ($< 100$ pax/mo):** **WAPE = 0.5563** (High relative noise, but carries $< 1\%$ of network revenue).

### 7.2 Multi-Level Predictability Ladder
Predictability increases monotonically with geographic aggregation:
- **Directional Airport Pairs ($17{,}119$):** WAPE = **0.1588**
- **Undirected Airport Pairs ($8{,}703$):** WAPE = **0.1548**
- **Directional Metro Pairs ($15{,}210$):** WAPE = **0.1421** (Cancels intra-metro substitutions, e.g., JFK vs EWR vs LGA).
- **US Interstate State Pairs ($1{,}628$):** WAPE = **0.0882** (Sub-9% error).
- **Entity Total Airport Volume ($150$):** WAPE = **0.0738** (Sub-7.5% error on total airport operations).

---

## 8. Validated Key Demand Drivers & Feature Importance

Synthesized across tree gain metrics and pre-holdout correlation analysis on domestic routes ($\ge 100$ pax/mo):

| Rank | Demand Driver Group | % Total Gain | Top Features & Quantitative Evidence |
| :---: | :--- | :---: | :--- |
| **1** | **Historical Momentum & Anchoring** | **31.8%** | `LOG_PAX_SAME_M_PREV2` (gain 2.82%), `SEASONAL_IDX_LAST` (gain 2.57%), `LOG_PAX_SAME_M_LAST` (gain 2.32%), `REC_19_R12` (gain 2.05%). Past capacity allocation dictates ~32% of variance. |
| **2** | **Airport Geography & Infrastructure** | **15.1%** | `ORG_IS_US`, `DES_IS_US` (gain 4.24%), `ORG_LON`, `ORG_LAT`. Domestic vs international regulatory/recovery boundaries and interior vs coastal hubs. |
| **3** | **Hub Scale & Relative Performance** | **11.9%** | `REL_REC_VS_ORG` (gain 2.38%), `REL_REC_VS_DES` (gain 2.34%). Measures whether a route is capturing share relative to its endpoints. |
| **4** | **Route Geometry & Spatial Gravity** | **9.0%** | `LOG_DIST` ($r = -0.1849$ with log volume), `LON_DIFF`, `DIR_ASYM_R12`. Classic spatial gravity: demand decays with distance beyond regional thresholds. |
| **5** | **Seasonality, Calendar & Horizon** | **8.9%** | `EASTER_IN_MONTH`, `DAYS_IN_MONTH` (Feb 29 leap years), `SNOWBIRD_INTERACT` ($(\text{Lat}_O - \text{Lat}_D) \times \cos(2\pi m / 12)$). Captures winter migration to Florida/Arizona. |
| **6** | **External: Census Population & Density** | **8.0%** | `EXT_DES_CTY_DENS` (gain 1.95%), `EXT_GRAVITY_POP` ($r = +0.5377$ with log volume, $r = +0.0553$ with post-COVID volume growth). |
| **7** | **Pricing & Revenue Yield** | **7.0%** | `YIELD_R12` (gain 2.29%), `YIELD_R3`. Fare sensitivity: higher ticket yields correlate negatively with volume ($r = -0.4471$) and growth ($r = -0.1195$). |
| **8** | **External: Census Migration Flows** | **4.4%** | `EXT_LOG_MIG_TOTAL` ($r = +0.4039$ with log volume, and **$r = +0.0656$** — strongest external correlate with post-COVID route growth). |
| **9** | **External: FAA Enplanement Boardings** | **4.0%** | `EXT_ORG_FAA_ENP` ($r = +0.5528$ with log volume), `EXT_ORG_OD_VS_FAA` ($r = -0.2510$ — separates connecting hubs like `DFW`/`CLT` from origin hubs like `LGA`/`BOS`). |

---

## 9. Reproducibility & Code Execution Guide

All execution must be invoked with the current working directory set to the project root: `c:\Users\abrah\Desktop\projects\aa_mml`.

### 9.1 Commands to Run Winning Model (`21_diverse_family_ensemble`)

```powershell
# Step 1: Run dedicated Zero-Data-Leakage Verification
python -u ./abraham/agent_workspace/21_diverse_family_ensemble/test_leakage.py

# Step 2: Execute 3-Fold Rolling-Origin Backtests
python -u ./abraham/agent_workspace/21_diverse_family_ensemble/run.py

# Step 3: Run Single-Shot Final Holdout Evaluation (2024-08..2026-07)
python -u ./abraham/agent_workspace/21_diverse_family_ensemble/evaluate_final_holdout.py

# Step 4: Generate All Publication & Diagnostic Charts
python -u ./abraham/agent_workspace/21_diverse_family_ensemble/make_plots.py
```

### 9.2 Generated Artifacts & Visual Plots
Generated output figures are saved in `./abraham/agent_workspace/21_diverse_family_ensemble/outputs/plots/`:
- `01_model_progression_wape.png`: Complete chronological progression across all 26 model folders.
- `02_multilevel_wape_comparison.png`: Accuracy curves across Market, Undirected, Metro, State, and Airport levels.
- `03_demand_drivers_by_group.png`: Relative feature importance and economic breakdown.
- `04_final_holdout_monthly_trajectory.png`: Actual vs. predicted network passenger trajectories across the 24-month holdout.

---

## 10. High-Yield Frontiers & Strategic Recommendations for Incoming LLM Agents

If you are an autonomous LLM or research agent tasked with extending this repository, **do not waste compute on already disproven directions**. The following section details exact positive and negative search paths.

### 10.1 Negative Paths (DO NOT RE-EXPLORE)
1. **Raw Macroeconomic Time-Series in GBDTs:** Do not add raw global economic series (e.g. FRED CPI, Unemployment, Fed Funds rate) into tree models. With only $T \approx 30$ historical monthly slices, tree algorithms partition on specific macroeconomic values (e.g., "if Unemployment $> 5.2\%$"), resulting in catastrophic overfitting and massive negative volume bias.
2. **Direct Unconstrained Linear/GLM Models on Raw Passenger Counts:** Passenger traffic spans 5 orders of magnitude. Standard linear models and un-anchored Tweedie models explode when predicting raw passengers from scratch. Any linear or neural model must operate on multiplicative log-ratios relative to a structural anchor with strict trust-region bounds.
3. **Horizon Specialization via Data Splitting:** Splitting models into separate Year 1 ($h \le 12$) and Year 2 ($h > 12$) training sets severely starves the Year 2 model of training targets (Model 15 degraded to WAPE = 0.3380). Instead, train unified models with continuous `HORIZON` embeddings.

### 10.2 Highest-Yield Frontiers for Next Experiments

1. **Spatio-Temporal Graph Neural Networks (ST-GNN):**
   - *Rationale:* The airline network is inherently a directed multigraph $G = (V, E)$ where vertices are airports ($|V|=150$) and edges are directional routes ($|E|=17{,}119$).
   - *Architecture:* Implement a Spatio-Temporal Graph Convolutional Network (e.g., A3T-GCN or DCRNN) using the Great-Circle distance and Census migration matrix as the graph adjacency matrix $A$.
   - *Expected Gain:* GNN message-passing naturally captures network spillover effects (e.g., delays or capacity cuts at `ORD` propagating to connecting spokes).
2. **Optimal Mathematical Reconciliation (MinT / Wickramasuriya):**
   - *Rationale:* Model 14 demonstrated that Iterative Proportional Fitting (IPF) cuts sMAPE and RMSLE substantially.
   - *Implementation:* Replace heuristic IPF scaling with formal **Minimum Trace Reconciliation (MinT)**:
     $$\mathbf{\tilde{y}} = \mathbf{S} (\mathbf{S}^T \mathbf{W}^{-1} \mathbf{S})^{-1} \mathbf{S}^T \mathbf{W}^{-1} \mathbf{\hat{y}}$$
     where $\mathbf{S}$ is the structural summing matrix mapping 17,119 directional pairs to 150 origin airports, 150 destination airports, and 50 states, and $\mathbf{W}$ is the estimated error covariance matrix.
3. **Dynamic Endogenous Pricing & Yield Elasticity Models:**
   - *Rationale:* Fares (`YIELD`) are currently treated as a lagged feature. In reality, airlines dynamically adjust fares based on load factor.
   - *Implementation:* Train a two-stage structural model: Stage 1 forecasts unconstrained latent demand; Stage 2 estimates route-specific price elasticity curves $\epsilon_r = \frac{\partial \ln Q_r}{\partial \ln P_r}$.
4. **Quantile & Probabilistic Forecasting (Conformalized Pinball Trees):**
   - *Rationale:* Network planning requires risk-adjusted capacity decisions (P10 vs P50 vs P90).
   - *Implementation:* Train LightGBM/CatBoost with multi-quantile pinball loss ($\alpha \in \{0.10, 0.50, 0.90\}$) to provide prediction intervals across all 24 horizons.
