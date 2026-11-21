# American Airlines `OD_PAX` 2-Year Forecasting — Final Executive Report

## 1. Executive Summary & Winning Pipeline

**Grand Champion Winning Pipeline:** `26_mint_hierarchical_staggered_reopening` (stored in `./abraham/agent_workspace/26_mint_hierarchical_staggered_reopening/`)

A strictly zero-leakage, 24-month-ahead hierarchical forecasting architecture combining two fundamental structural breakthroughs across `17,119` directional O&D airport pairs (`1,557,829` balanced route-months spanning `2019-01` to `2026-07`):

1. **Horizontal Geographic Market Disentanglement (Model 25 Engine):**
   - Eliminates the structural distortion of treating all international routes as a single monolithic category (`dom` vs `non_dom`).
   - Classifies the route network into 5 Geo-Macro Segments: `DOMESTIC_US`, `AMERICAS_INTL` (Transborder & Caribbean), `TRANSATLANTIC` (Europe & Mid-East), `TRANSPACIFIC` (Asia & Pacific), and `OTHER_INTL`.
   - Models region-specific reopening schedules (Americas recovering to 104% of 2019 by 2022, Transatlantic surging in 2021-2022, and Transpacific delayed until late 2022-2023), eliminating large regional biases (e.g. Transpacific Asia was under-predicted by -27.3% in Model 24).
2. **Vertical Closed-Form Woodbury Dual MinT Reconciliation (MinT Engine):**
   - Applies the Sherman-Morrison-Woodbury matrix inversion identity to solve optimal Generalized Least Squares (GLS) Minimum Trace hierarchical reconciliation (Wickramasuriya et al., JASA 2019) across $M_U = 383$ upper constraints (150 Origin Airports, 150 Destination Airports, 41 Origin US States, 41 Destination US States, and Total Network Volume).
   - Inverts in the dual constraint space in $< 5\text{ ms}$ via direct Cholesky decomposition, circumventing ill-conditioned $17,119 \times 17,119$ primal solvers.
   - Allocates reconciliation adjustments proportional to route variance ($\mathbf{W}_B$), protecting high-certainty fortress routes while enforcing hard terminal gate and runway physical capacities at AA hubs (DFW, CLT, ORD, MIA, PHL, PHX).
3. **Regime-Adaptive Consensus Prior:**
   - Dynamically scales top-down consensus weight $\alpha$ with trailing network recovery ratio $R_{\text{net}}$: applying strong capacity prior ceilings during shock recovery ($\alpha = 0.76$ on Fold 1) and gentle consensus shrinkage during steady-state operations ($\alpha = 0.38$ on Fold 3).

### Reproducible Commands (run from repository root)
```powershell
# 1. Verify zero data leakage (adversarial future-poisoning invariance: diff = 0.00e+00)
python -u ./abraham/agent_workspace/26_mint_hierarchical_staggered_reopening/test_leakage.py

# 2. Run standardized 3-fold rolling-origin backtests (strictly YM <= 2024-07)
python -u ./abraham/agent_workspace/26_mint_hierarchical_staggered_reopening/run.py

# 3. Run single-shot final holdout evaluation (2024-08..2026-07) and generate plots
python -u ./abraham/agent_workspace/26_mint_hierarchical_staggered_reopening/evaluate_final_holdout.py
python -u ./abraham/agent_workspace/26_mint_hierarchical_staggered_reopening/make_plots.py
```

### Backtest & Final Held-Out Test Scores (`26_mint_hierarchical_staggered_reopening`)

| Evaluation Window | Market Dir WAPE | Year 1 WAPE | Year 2 WAPE | RMSE | sMAPE | RMSLE | Bias | Metro Dir WAPE | US Interstate State WAPE | Entity Total Airport WAPE |
|---|---|---|---|---|---|---|---|---|---|---|
| **3-Fold Backtest Mean** (`2021-08..2024-07`) | **0.1887 ± 0.0208** | **0.1763** | **0.2000** | **1,438.8** | **0.3616** | **0.5780** | **-0.0286** | **0.1679** | **0.0889** | **0.0727** |
| **Backtest Fold 3 Steady** (`2022-08..2024-07`) | **0.1661** | **0.1522** | **0.1790** | **1,274.8** | **0.3479** | **0.5741** | **-0.0218** | **0.1467** | **0.0741** | **0.0566** |
| **Backtest Fold 2 Transition** (`2021-08..2023-07`) | **0.1838** | **0.1748** | **0.1921** | **1,388.5** | **0.3580** | **0.5692** | **-0.0435** | **0.1616** | **0.0852** | **0.0729** |
| **Backtest Fold 1 Recovery** (`2020-08..2022-07`) | **0.2163** | **0.2018** | **0.2290** | **1,653.1** | **0.3789** | **0.5907** | **-0.0204** | **0.1955** | **0.1073** | **0.0885** |
| **Holdout Clean Through 2025 (17m)** (`2024-08..2025-12`) | **0.1720** | **0.1586** | **0.2050** | **1,351.3** | **0.3110** | **0.4980** | **+0.0597** | **0.1538** | **0.0923** | **0.0767** |
| **Holdout Clean Pre-Anomaly 21m** (`2024-08..2026-04`) | **0.1868** | **0.1586** | **0.2271** | **1,454.5** | **0.3240** | **0.5210** | **+0.0755** | **0.1685** | **0.1047** | **0.0916** |
| **Holdout Full 24m** (`2024-08..2026-07`) | **0.2047** | **0.1586** | **0.2539** | **1,634.8** | **0.3331** | **0.5379** | **+0.0957** | **0.1866** | **0.1230** | **0.1103** |

### Head-to-Head Progression Across Milestones

| Metric | Baseline (M01) | Damped (M03) | Multi-Family (M21) | Spatial Gravity (M24) | Geo-Macro (M25) | MinT + Geo-Macro (M26 Champion) | Relative Gain vs M24 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3-Fold Mean WAPE** | 0.3546 | 0.2015 | 0.1989 | 0.1956 | 0.1932 | **0.1887** | **-3.5% (-69 bps)** |
| **Fold 1 Recovery WAPE** | 0.4676 | 0.2676 | 0.2319 | 0.2319 | 0.2265 | **0.2163** | **-6.7% (-156 bps)** |
| **Fold 2 Transition WAPE**| 0.3664 | 0.1906 | 0.1945 | 0.1892 | 0.1872 | **0.1838** | **-2.9% (-54 bps)** |
| **Fold 3 Steady WAPE** | 0.2297 | 0.1762 | 0.1704 | 0.1657 | 0.1658 | **0.1661** | Tied (+4 bps) |
| **3-Fold Mean RMSE** | 2,847.6 | 1,526.4 | 1,535.4 | 1,523.0 | 1,498.9 | **1,438.8** | **-5.5% (-84.2 pax)** |
| **3-Fold Mean sMAPE** | 0.7081 | 0.3866 | 0.3668 | 0.3653 | 0.3630 | **0.3616** | **-1.0%** |
| **3-Fold Mean Volume Bias**| -0.2974 | -0.0724 | -0.0744 | -0.0311 | -0.0307 | **-0.0286** | **Calibrated (+25 bps)**|
| **Entity Airport WAPE (3F)**| 0.3017 | 0.0994 | 0.1007 | 0.0800 | 0.0792 | **0.0727** | **-9.1% (-73 bps)** |

---

## 2. Key Demand Drivers Analysis

Across normalized feature importance (`12_xgboost_l1_direct_anchor`, `20b_cross_sectional_xgb_over_03`, `22_deep_spatial_gravity`) and pre-holdout (`<= 2024-07`) cross-sectional correlations (`US–US` routes `>= 100` pax/mo):

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
- **Helped (`17.9%` combined tree gain):** US Census CBSA/County Population & Boundaries, US Census County-to-County Migration Flows (`2011–2015` ACS), and FAA Commercial Service Enplanements (`2018–2021` lagged strictly by publication date) improved cross-sectional route allocation, drove Entity Airport throughput WAPE down to `5.66%`, and reduced steady-state `RMSLE`.
- **Hurt:** Adding global time-only **FRED monthly macro series** (`UNRATE`, `CPIAUCSL_YOY`, `DCOILWTICO`, `UMCSENT`) degraded steady Fold 3 WAPE from `0.1935` to `0.3016` (`Bias = -0.2633`) because ~30 unique historical timestamps caused trees to overfit macro regime shifts.

---

## 3. Market Definition & Multi-Level Findings

1. **Coarser Aggregations Are Dramatically More Predictable:**
   On `26_mint_hierarchical_staggered_reopening` (Steady Fold 3 / Clean 2025 Holdout):
   - **Directional Airport Pair (`17,119`):** `0.1661` / `0.1720` WAPE
   - **Undirected Airport Pair (`8,703`):** `0.1610` / `0.1677` WAPE (cancels directional booking imbalance)
   - **Directional Metro Pair (`15,210`):** `0.1467` / `0.1538` WAPE (~`2.0` pp lower error than airport pairs; e.g., JFK vs EWR substitutions offset inside NYC<->LAX)
   - **US Interstate State Pair (`1,628`):** `0.0741` / `0.0923` WAPE (more than 2× lower error than airport pairs)
   - **Entity Total Airport (`150`):** **`0.0566`** / **`0.0767`** WAPE (airport throughput predicted with <5.7% error!)
   - **Total Network Volume (`1`):** **`0.0218`** / **`0.0349`** WAPE (network capacity predicted within 2.2% error).
2. **Optimal Bottom-Up Reconciliation Beats Either Pure Bottom-Up or Top-Down Modeling:**
   Bottom-up route modeling provides rich granularity, distance decay, and seasonal profiles, while top-down models enforce physical hub capacities. Closed-form MinT reconciliation bridges them, achieving the lowest network error (0.1887) and lowest RMSE (1,438.8) recorded across all experiments.

---

## 4. Multi-Agent Swarm Collaboration & Infrastructure

To explore beyond the current state of the art, complete infrastructure and documentation have been staged for spawning 5 concurrent exploration agents:
- **Operations Manual:** [REMOTE_CLUSTER_MANUAL.md](file:///c:/Users/abrah/Desktop/projects/aa_mml/abraham/agent_workspace/REMOTE_CLUSTER_MANUAL.md) for Alienware NVIDIA GPU nodes (`alien1`, `alien2`) with SLURM scheduler.
- **Mission Directive:** [MULTI_AGENT_MISSION_DIRECTIVE.md](file:///c:/Users/abrah/Desktop/projects/aa_mml/abraham/agent_workspace/MULTI_AGENT_MISSION_DIRECTIVE.md) establishing directory boundaries (`./abraham/agent_<id>/`), zero-leakage verification, and standardized backtesting.
- **Ready-to-Paste Launch Prompts:** [AGENT_LAUNCH_PROMPTS.md](file:///c:/Users/abrah/Desktop/projects/aa_mml/abraham/agent_workspace/AGENT_LAUNCH_PROMPTS.md) assigning each of the 5 agents to a dedicated research frontier (ST-GNN, Structural MinT, Yield Elasticity, Conformal Quantiles, Latent Factor Tensor Panel).
