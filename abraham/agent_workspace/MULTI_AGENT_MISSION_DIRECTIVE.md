# American Airlines `OD_PAX` 2-Year Forecasting — Multi-Agent Grand Mission Directive

> **Target Audience:** Autonomous AI Research Agents (Swarm Team: 5 Concurrent Exploration Agents).  
> **Repository Root:** `c:\Users\abrah\Desktop\projects\aa_mml\`  
> **Global Objective:** Forecast monthly passenger demand (`OD_PAX`) across 17,119 directional airline routes up to 24 months into the future ($H = 1 \dots 24$) with absolute zero data leakage.  
> **Current Project SOTA Benchmark:** **Model 24 (`24_spatial_gravity_network_ensemble`)**  
> — 3-Fold Mean WAPE: **`0.1956`** | Steady-State Fold 3 WAPE: **`0.1657`** | Entity Airport WAPE: **`0.0569`** | Volume Bias: **`-0.0311`**

---

## 1. Mandatory Workspace Rules & Directory Isolation

### 1.1 Unique Workspace per Agent
Every agent must create and work **100% exclusively inside its own dedicated directory** located at the exact same folder level as `./abraham/agent_workspace/`:

```
./abraham/
├── global_data/                     <-- READ-ONLY: Raw airline data (NEVER touch or modify)
├── agent_workspace/                 <-- READ-ONLY: Reference prior art (Models 01-24, shared_eval)
├── gpt_agent_workspace/             <-- READ-ONLY: Peer agent work
├── agent_<your_unique_id>/          <-- YOUR DEDICATED WORKSPACE (100% of your work goes here)
│   ├── experiments.md               <-- Your central lab notebook
│   ├── <model_name>/                <-- Each model family in its own folder
│   │   ├── run.py
│   │   ├── test_leakage.py
│   │   ├── model_specific_data/
│   │   └── outputs/
```

- Choose a unique identifier for your folder that does **not** already exist, e.g.:
  - `./abraham/agent_st_gnn/`
  - `./abraham/agent_mint_reconciler/`
  - `./abraham/agent_yield_elasticity/`
  - `./abraham/agent_quantile_dist/`
  - `./abraham/agent_tensor_factor/`
- **Never create or write files outside your designated folder.**

### 1.2 Zero Data Leakage Discipline
- For any forecast origin $t_{\text{origin}}$, all inputs, features, lags, rolling statistics, normalizations, and external demographic releases must strictly satisfy $t \le t_{\text{origin}}$.
- You **must** implement and pass an adversarial future-poisoning leakage check (`test_leakage.py`) before logging any score.
- The final held-out test set (**August 2024 through July 2026**) must remain **strictly sealed** until backtests conclusively beat the benchmark.

---

## 2. Available Compute Infrastructure (Local + Remote GPU Cluster)

You have access to both local resources and remote high-performance GPU nodes:

### Local Machine
- **OS:** Windows
- **Compute:** CPU + Intel XPU
- **Installed Stack:** Python 3.14, PyTorch 2.14.1+xpu, XGBoost 3.4.1, Scikit-Learn, Scipy, Pandas, PyArrow

### Remote Alienware NVIDIA GPU Cluster (`alien1` & `alien2`)
If you are training deep neural networks, GNNs, or large hyperparameter sweeps, you can offload jobs to the remote Alienware cluster:
- **Node 1:** `ssh abraham01cyb@alien1`
- **Node 2:** `ssh abraham01cyb@alien2`
- **Password:** `Cr3#!0Ne8cyb`
- **Hardware:** Alienware workstations with NVIDIA GPUs and SLURM workload manager (`sbatch`, `squeue`, `srun --gres=gpu:1`).
- **Remote Work Rule:** When running on remote nodes, create an isolated folder `~/scratch_<your_agent_id>_aa_mml/` and review the operations guide in `./abraham/agent_workspace/REMOTE_CLUSTER_MANUAL.md`.

---

## 3. Data Assets & Prior-Art Knowledge Base

Before designing new models, review the established data and evaluation harness:
- **Primary Data:**
  - `./abraham/global_data/pax_forecasting.csv` (1,533,522 raw rows, 91 calendar months `2019-01..2026-07`, 150 airports, 17,119 directional pairs).
  - `./abraham/global_data/airport_info.csv` (150 airports: 99 US continental, 1 Puerto Rico `SJU`, 50 international).
- **Curated External Datasets (in `./abraham/agent_workspace/data_external/`):**
  - `airport_external_annual.parquet`: US Census CBSA/County populations, population densities, domestic migration rates, and FAA commercial enplanements with point-in-time publication dates (`AVAILABLE_YM`).
  - `pair_migration_2015.parquet`: Official US Census ACS 5-year county-to-county migration flow matrix between all airport county pairs.
- **Shared Evaluation Harness (in `./abraham/agent_workspace/shared_eval/`):**
  - Import `runner.py`, `splits.py`, `metrics.py`, and `leakage_verifier.py` to ensure your evaluation is 100% standardized and comparable.
- **Full 24-Model Knowledge Base:** Read `./abraham/agent_workspace/LLM_TRANSFER_KNOWLEDGE_BASE.md` and `./abraham/agent_workspace/experiments.md`.

---

## 4. The Benchmarks to Beat

Your primary goal is to exceed the current champion **Model 24 (`24_spatial_gravity_network_ensemble`)**:

```
========================================================================================================
CURRENT BENCHMARK SCORECARD TO BEAT (3-Fold Backtest Mean Across 2021-08 .. 2024-07)
========================================================================================================
Model                              Mean WAPE    Steady F3   Mean RMSE   Mean sMAPE  Mean RMSLE   Mean Bias
--------------------------------------------------------------------------------------------------------
01 Seasonal Naive Last Year         0.3546       0.2297      2,847.6     0.7081      0.7912      -0.2974
03 Damped Structural Baseline       0.2015       0.1762      1,526.4     0.3866      0.6232      -0.0724
21 Diverse Family Ensemble          0.1989       0.1704      1,535.4     0.3668      0.5870      -0.0744
22 Deep Spatial Gravity Network     0.2198       0.1729      1,790.4     0.3890      0.6236      -0.0012
24 Spatial Gravity Network Ensemble 0.1956       0.1657      1,523.0     0.3653      0.5818      -0.0311
========================================================================================================
Key Upper-Level Metrics for Model 24 (Steady Fold 3):
- Entity Total Airport WAPE: 0.0569 (5.69% throughput error)
- State US-Interstate WAPE:  0.0738 (7.38% error)
- Metro-Pair Directional:    0.1466 (14.66% error)
========================================================================================================
```

---

## 5. Negative Results & Hard Lessons (DO NOT REPEAT THESE MISTAKES)

1. **Do NOT inject raw macroeconomic time-series into decision trees:** Adding FRED unemployment, inflation, and oil series into GBDTs caused trees to partition on specific timestamp values, destroying out-of-fold generalization (Fold 3 WAPE worsened from 0.193 to 0.302 in Model 10).
2. **Do NOT train unconstrained linear or Poisson GLMs directly on raw passenger counts:** Passenger traffic spans 5 orders of magnitude (from 0 to 140,000). Direct un-anchored regressions explode (+83% bias in Model 05). Models must predict multiplicative adjustments or bounded residuals relative to a structural anchor.
3. **Do NOT split training sets by horizon:** Splitting into separate Year 1 ($h \le 12$) and Year 2 ($h > 12$) training sets starves the Year 2 model of post-COVID training targets (Model 15 degraded to 0.338 WAPE). Use unified multi-horizon models with continuous horizon features/embeddings.
4. **Do NOT use static volume priors for empirical Bayes shrinkage:** A static prior ($1500$) inflates micro/small routes ($<500$ pax/mo), severely degrading `sMAPE` and `RMSLE`. Prior strength must adapt proportionally to historical route scale ($k(V) \propto V_r$).

---

## 6. The Shared Exploration Mission: 5 High-Yield Frontiers

To achieve maximum scientific diversity, each of the 5 agents should spearhead a distinct, high-impact research frontier:

### Frontier 1: Spatio-Temporal Graph Neural Networks (ST-GNN / Graph WaveNet)
- **Concept:** Routes form a directed multigraph $G = (V, E)$ with 150 airport nodes and 17,119 edges.
- **Action:** Build a spatial adjacency matrix using great-circle distances and Census county-to-county migration flows (`pair_migration_2015.parquet`). Implement Graph Convolutional Networks (GCN) or Diffusion Convolution (DCRNN) to capture network hub spillover and connecting flight bank dynamics between feeder routes and fortress hubs (`DFW`, `CLT`, `ORD`, `MIA`).

### Frontier 2: Closed-Form Structural Minimum Trace (MinT) Reconciliation
- **Concept:** Upper levels (Airports ~5.7% WAPE, States ~7.4% WAPE) are far more predictable than individual routes (~16.6% WAPE).
- **Action:** Construct the structural summing matrix $\mathbf{S} \in \mathbb{R}^{M \times K}$ mapping the 17,119 routes to 150 origin airports, 150 destination airports, 50 states, and 1 network total. Implement optimal closed-form MinT (Wickramasuriya et al., *JASA* 2019) using sample shrinkage covariance $\mathbf{W}$ to project incoherent route forecasts onto exact conservation laws.

### Frontier 3: Dynamic Fare Yield Elasticity & Pricing Feedback
- **Concept:** Ticket yield (`OD_REV_USD / OD_PAX`) has a strong negative correlation with demand ($r = -0.45$). Fares are endogenous (airlines raise fares when demand peaks).
- **Action:** Model route-specific price elasticity curves $\epsilon_r = \frac{\partial \ln Q_r}{\partial \ln P_r}$ to separate organic macroeconomic demand growth from fare-induced demand suppression.

### Frontier 4: Multi-Quantile Distributional Regression & Conformal Risk Bounds
- **Concept:** Network planners require risk-adjusted capacity bounds (P10 vs P50 vs P90) rather than point forecasts alone.
- **Action:** Train gradient boosters or neural networks with multi-quantile pinball loss and conformalized time-series prediction bands. Use two-stage zero-inflation hurdle modeling to cleanly separate route dormancy ($P(\text{pax} = 0)$) from route volume scale.

### Frontier 5: Latent Factor Panel Models & Tensor Decomposition (Bai 2009)
- **Concept:** Extract unobserved macro travel regimes directly from the 17,119-route historical panel without relying on noisy external FRED headlines.
- **Action:** Apply Singular Value Decomposition / Interactive Fixed Effects ($P_{rt} = \text{Anchor}_{rt} + \sum_{k=1}^K \lambda_{rk} F_{kt}$) to extract latent network factors $F_{kt}$ (explaining >96% of residual variance). Forecast the 3–5 latent factors causally with seasonal ARIMA/Holt-Winters.

---

## 7. Execution Protocol & Deliverables

Every agent must follow this 4-step execution discipline:

1. **Initialize Workspace:** Create your dedicated folder `./abraham/agent_<unique_id>/` and start an `experiments.md` log.
2. **Implement & Verify Leakage:** Write your model in a descriptive subfolder and run an adversarial future-poisoning verification (`test_leakage.py`). Ensure `max_abs_diff == 0.0` before computing any evaluation metrics.
3. **Execute 3-Fold Backtest:** Run on `fold_1_recovery`, `fold_2_transition`, and `fold_3_steady` using `shared_eval.runner.run_backtest_suite`.
4. **Compare Against Benchmark:** Measure your WAPE, RMSE, sMAPE, RMSLE, and Volume Bias directly against Model 24.
5. **Document Insights:** Record what worked, what failed, and the quantitative driver importances in your workspace lab notebook.

**You have full authorization. Choose your frontier, initialize your workspace, and begin Phase 1.**
