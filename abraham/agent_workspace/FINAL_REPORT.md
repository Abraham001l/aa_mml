# American Airlines `OD_PAX` 2-Year Forecasting — Final Executive Report

## 1. Executive Summary & Winning Pipeline

**Winning Pipeline:** `24_spatial_gravity_network_ensemble` (stored in `./abraham/agent_workspace/24_spatial_gravity_network_ensemble/`)

A strictly zero-leakage, 24-month-ahead forecasting ensemble combining three distinct modeling paradigms across `17,119` directional O&D airport pairs (`1,557,829` balanced route-months spanning `2019-01` to `2026-07`):
- **40% Diverse Multi-Family Ensemble (`21_diverse_family_ensemble`):** Structural Damped-Shrinkage Anchor + Multi-Level Native & IPF Reconciled models + Direct L1 WAPE-aligned XGBoost + Zero-macro-bias Cross-Sectional XGBoost with external Census/FAA/Migration data.
- **10% Deep Spatial Gravity Network (`22_deep_spatial_gravity`):** Tier-adaptive empirical-Bayes route shrinkage + Wilson spatial demographic gravity ($\ln(M_O) + \ln(M_D) - 1.2\ln(\text{Dist}) + 0.25\ln(1 + \text{MIG})$) + segment-centered spatial booster.
- **50% Persistent Network Recovery (`compute_persistent_network_recovery`):** Endpoint-pooled recovery prior with persistent route-level relative drift.

### Reproducible Commands (run from repository root)
```powershell
# 1. Verify zero data leakage (internal + external future-poisoning invariance)
python -u ./abraham/agent_workspace/24_spatial_gravity_network_ensemble/test_leakage.py

# 2. Run 3-fold rolling-origin backtests (strictly YM <= 2024-07)
python -u ./abraham/agent_workspace/24_spatial_gravity_network_ensemble/run.py

# 3. Run single-shot final holdout evaluation (2024-08..2026-07) and generate plots
python -u ./abraham/agent_workspace/24_spatial_gravity_network_ensemble/evaluate_final_holdout.py
python -u ./abraham/agent_workspace/24_spatial_gravity_network_ensemble/make_plots.py
```

### Backtest & Final Held-Out Test Scores (`24_spatial_gravity_network_ensemble`)

| Evaluation Window | Market Dir WAPE | Year 1 WAPE | Year 2 WAPE | RMSE | sMAPE | RMSLE | Bias | Metro Dir WAPE | US Interstate State WAPE | Entity Total Airport WAPE |
|---|---|---|---|---|---|---|---|---|---|---|
| **3-Fold Backtest Mean** (`2021-08..2024-07`) | **0.1956 ± 0.0274** | 0.1818 | 0.2080 | 1,523.0 | 0.3653 | 0.5818 | -0.0311 | 0.1742 | 0.1018 | 0.0894 |
| **Backtest Fold 3 Steady** (`2022-08..2024-07`) | **0.1657** | 0.1521 | 0.1783 | 1,269.1 | 0.3485 | 0.5746 | -0.0152 | 0.1466 | 0.0738 | **0.0569** |
| **Backtest Fold 2 Transition** (`2021-08..2023-07`) | **0.1892** | 0.1799 | 0.1977 | 1,514.8 | 0.3602 | 0.5714 | -0.0248 | 0.1678 | 0.0984 | 0.0872 |
| **Backtest Fold 1 Recovery** (`2020-08..2022-07`) | **0.2319** | 0.2134 | 0.2481 | 1,785.1 | 0.3872 | 0.5994 | -0.0533 | 0.2082 | 0.1332 | 0.1241 |
| **Holdout Clean Through 2025 (17m)** (`2024-08..2025-12`) | **0.1708** | 0.1575 | 0.2035 | 1,345.7 | 0.3035 | 0.4856 | +0.0597 | 0.1527 | 0.0925 | 0.0766 |
| **Holdout Clean Pre-Anomaly 21m** (`2024-08..2026-04`) | **0.1858** | 0.1575 | 0.2261 | 1,450.1 | 0.3171 | 0.5093 | +0.0757 | 0.1675 | 0.1051 | 0.0917 |
| **Holdout Full 24m** (`2024-08..2026-07`) | **0.2037** | 0.1575 | 0.2531 | 1,632.1 | 0.3263 | 0.5268 | +0.0959 | 0.1858 | 0.1234 | 0.1105 |

**Holdout Market Size Tiers (`clean_through_2025_17m`):**
- `1_Micro (<100 pax/mo)`: WAPE `0.5923` (`sMAPE = 0.5502`, `Bias = -0.0569`)
- `2_Small (100–500)`: WAPE `0.3307` (`sMAPE = 0.3465`, `Bias = -0.0590`)
- `3_Medium (500–2,500)`: WAPE `0.2467` (`sMAPE = 0.2527`, `Bias = +0.0055`)
- `4_Large (2,500–10,000)`: WAPE `0.1992` (`sMAPE = 0.2139`, `Bias = +0.0606`)
- `5_Mega (>=10,000)`: WAPE **`0.1333`** (`sMAPE = 0.1416`, `Bias = +0.0758`)
- **Domestic (`US–US`) vs. International/PR:** `clean_through_2025_17m` WAPE `0.1577` vs. `0.2141`.

---

## 2. Key Demand Drivers Analysis

Across normalized gain importance (`12_xgboost_l1_direct_anchor`, `20b_cross_sectional_xgb_over_03`, `22_deep_spatial_gravity`) and pre-holdout (`<= 2024-07`) cross-sectional correlations (`US–US` routes `>= 100` pax/mo):

| Rank | Thematic Driver Group | Mean Norm. Gain | Key Representative Features & Quantitative Evidence |
|---|---|---|---|
| 1 | **Historical Volume, Anchor & Momentum** | **31.8%** | `LOG_PAX_SAME_M_PREV2` (`2.82%`), `SEASONAL_IDX_LAST` (`2.57%`), `LOG_PAX_SAME_M_LAST` (`2.32%`), `LOG_PAX_S` (`2.25%`), `REC_19_R12` (`2.05%`), `YOY_R3` (`2.03%`). |
| 2 | **Airport Geography & Infrastructure** | **15.1%** | `ORG_IS_US` (`2.18%`), `DES_IS_US` (`2.06%`), `ORG_LON` (`1.86%`), `ORG_LAT` (`1.85%`) — separates domestic vs. international recovery curves and coastal vs. interior hubs. |
| 3 | **Spatial Gravity & Demographic Mass** | **13.5%** | `DEEP_GRAVITY_INDEX` ($\ln(M_O) + \ln(M_D) - 1.2\ln(\text{Dist}) + 0.25\ln(1 + \text{MIG})$), `POP_ASYM`, `EXT_GRAVITY_POP` ($r = +0.5377$). |
| 4 | **Airport / Entity & Metro Network Scale** | **11.9%** | `REL_REC_VS_ORG` (`2.38%`), `REL_REC_VS_DES` (`2.34%`) — captures route outperformance relative to its endpoint airports (critical for post-COVID hub shifts). |
| 5 | **Route Geometry & Distance Decay** | **9.0%** | `IS_DOM` (`1.97%`), `LON_DIFF` (`1.94%`), `LOG_DIST` (`1.81%`, $r = -0.1849$ with log volume), `DIR_ASYM_R12`. |
| 6 | **Seasonality, Calendar & Horizon** | **8.9%** | `DEP_MONTH` (`1.88%`), `SIN_M`/`COS_M`, `EASTER_IN_MONTH`, `DAYS_IN_MONTH` (Feb 29 leap year), `SNOWBIRD_INTERACT` (`(ORG_LAT - DES_LAT) * COS_M`), `HORIZON`. |
| 7 | **Pricing & Revenue Yield** | **7.0%** | `YIELD_R12` (`2.29%`), `YIELD_R3` (`1.98%`), `YIELD_LAST_M` (`1.93%`). Higher historical fares strongly suppress both route scale ($r = -0.4471$) and post-COVID volume growth ($r = -0.1195$). |
| 8 | **External: Census Migration (CBSA/County)** | **4.4%** | `EXT_LOG_MIG_TOTAL` (`2011–2015` ACS county-to-county flows: $r = +0.4039$ with log `OD_PAX`, and $r = +0.0656$ with post-COVID growth), plus `CBSA_DOM_MIG_RATE`. |
| 9 | **External: FAA Enplanements & O&D Share** | **4.0%** | `EXT_ORG_FAA_ENP` ($r = +0.5528$ with log `OD_PAX`), `EXT_ORG_OD_VS_FAA` ($r = -0.2510$ — distinguishes connecting hubs like `CLT`/`DFW`/`ATL` from high-O&D origin markets like `LGA`/`BOS`). |

**Which external datasets helped vs. hurt?**
- **Helped (`17.9%` combined tree gain):** US Census CBSA/County Population & Boundaries, US Census County-to-County Migration Flows (`2011–2015` ACS), and FAA Commercial Service Enplanements (`2018–2021` lagged strictly by publication date) improved cross-sectional route allocation, drove Entity Airport throughput WAPE down to `5.69%`, and reduced steady-state `RMSLE`.
- **Hurt:** Adding global time-only **FRED monthly macro series** (`UNRATE`, `CPIAUCSL_YOY`, `DCOILWTICO`, `UMCSENT`) degraded steady Fold 3 WAPE from `0.1935` to `0.3016` (`Bias = -0.2633`) because ~30 unique historical timestamps caused trees to overfit macro regime shifts.

---

## 3. Market Definition & Multi-Level Findings

1. **Coarser Aggregations Are Dramatically More Predictable:**
   On `24_spatial_gravity_network_ensemble` (Steady Fold 3 / Clean 2025 Holdout):
   - **Directional Airport Pair (`17,119`):** `0.1657` / `0.1708` WAPE
   - **Undirected Airport Pair (`8,703`):** `0.1612` / `0.1665` WAPE (cancels directional booking imbalance)
   - **Directional Metro Pair (`15,210`):** `0.1466` / `0.1527` WAPE (~`2.0` pp lower error than airport pairs; e.g., JFK vs EWR substitutions offset inside NYC<->LAX)
   - **US Interstate State Pair (`1,628`):** `0.0738` / `0.0925` WAPE (more than 2× lower error than airport pairs)
   - **Entity Total Airport (`150`):** **`0.0569`** / `0.0766` WAPE (airport throughput predicted with <5.7% error!)
2. **Bottom-Up Aggregation Beats Direct Coarse Modeling:**
   Bottom-up aggregation from the spatial gravity ensemble directly achieved 5.69% airport throughput WAPE, outperforming coarser top-down models (7.65%), because route-level seasonality, distance decay, and domestic vs. international recovery curves cancel inaccurately when aggregated prior to modeling.

---

## 4. Multi-Agent Swarm Collaboration & Infrastructure

To explore beyond the current state of the art, complete infrastructure and documentation have been staged for spawning 5 concurrent exploration agents:
- **Operations Manual:** [REMOTE_CLUSTER_MANUAL.md](file:///c:/Users/abrah/Desktop/projects/aa_mml/abraham/agent_workspace/REMOTE_CLUSTER_MANUAL.md) for Alienware NVIDIA GPU nodes (`alien1`, `alien2`) with SLURM scheduler.
- **Mission Directive:** [MULTI_AGENT_MISSION_DIRECTIVE.md](file:///c:/Users/abrah/Desktop/projects/aa_mml/abraham/agent_workspace/MULTI_AGENT_MISSION_DIRECTIVE.md) establishing directory boundaries (`./abraham/agent_<id>/`), zero-leakage verification, and standardized backtesting.
- **Ready-to-Paste Launch Prompts:** [AGENT_LAUNCH_PROMPTS.md](file:///c:/Users/abrah/Desktop/projects/aa_mml/abraham/agent_workspace/AGENT_LAUNCH_PROMPTS.md) assigning each of the 5 agents to a dedicated research frontier (ST-GNN, Structural MinT, Yield Elasticity, Conformal Quantiles, Latent Factor Tensor Panel).
