from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent
PLOTS_DIR = MODEL_DIR / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Backtest Progression across key experiments
    exp_list = [
        ("01_seasonal_naive_last_year", "01 Naive LY"),
        ("02_precovid_2019_seasonal", "02 Naive 2019"),
        ("04_ridge_log_residual_panel", "04 Ridge (Raw)"),
        ("04b_ridge_bounded_calibrated", "04b Ridge (Cal)"),
        ("05b_poisson_offset_glm", "05b Poisson GLM"),
        ("06_xgboost_log_residual", "06 XGB LogRes"),
        ("08_pytorch_residual_panel", "08 PyTorch XPU"),
        ("12_xgboost_l1_direct_anchor", "12 XGB L1"),
        ("14_multilevel_native_reconciled", "14 MultiLvl IPF"),
        ("20_deconfounded_cross_sectional_xgb", "20 CS-XGB+Ext"),
        ("03_damped_trend_seasonal", "03 Stat Anchor"),
        ("21_diverse_family_ensemble", "21 Ensemble (Win)"),
    ]
    rows = []
    for folder, label in exp_list:
        p = WORKSPACE / folder / "outputs" / "backtest_metrics.json"
        if p.exists():
            d = json.load(open(p))
            rows.append({
                "label": label,
                "mean_wape": d["summary"]["market_dir_overall"]["wape_mean"],
                "f3_wape": d["folds"]["fold_3_steady"]["market_dir_overall"]["wape"],
            })
    df_exp = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df_exp))
    w = 0.38
    ax.bar(x - w / 2, df_exp["mean_wape"] * 100, width=w, label="3-Fold Mean WAPE (%)", color="#2b5c8f")
    ax.bar(x + w / 2, df_exp["f3_wape"] * 100, width=w, label="Steady-State Fold 3 WAPE (%)", color="#e07a5f")
    ax.set_xticks(x)
    ax.set_xticklabels(df_exp["label"], rotation=30, ha="right")
    ax.set_ylabel("WAPE (%)")
    ax.set_title("Model Progression: 3-Fold Mean vs. Steady-State Fold 3 Market WAPE")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_model_progression_wape.png", dpi=150)
    plt.close()

    # 2. Multi-Level Predictability (Market vs Metro vs State vs Entity) on Winner
    d21 = json.load(open(MODEL_DIR / "outputs" / "backtest_metrics.json"))
    f3_21 = d21["folds"]["fold_3_steady"]
    s21 = d21["summary"]
    lvls = [
        ("Dir Airport Pair", "market_dir_overall"),
        ("Undir Airport Pair", "market_undir_overall"),
        ("Dir Metro Pair", "market_metro_dir_overall"),
        ("US Interstate State", "state_us_interstate_overall"),
        ("Total Airport Entity", "entity_total_airport_overall"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    lx = np.arange(len(lvls))
    ax.bar(lx - 0.18, [s21[k]["wape_mean"] * 100 for _, k in lvls], width=0.36, label="3-Fold Mean WAPE (%)", color="#3d405b")
    ax.bar(lx + 0.18, [f3_21[k]["wape"] * 100 for _, k in lvls], width=0.36, label="Steady-State Fold 3 WAPE (%)", color="#81b29a")
    ax.set_xticks(lx)
    ax.set_xticklabels([n for n, _ in lvls], rotation=15, ha="right")
    ax.set_ylabel("WAPE (%)")
    ax.set_title("Multi-Level Aggregation Accuracy (Winning Ensemble 21)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_multilevel_wape_comparison.png", dpi=150)
    plt.close()

    # 3. Top Demand Driver Groups
    grp_p = MODEL_DIR / "outputs" / "demand_drivers_by_group.csv"
    if grp_p.exists():
        gdf = pd.read_csv(grp_p).sort_values("mean_norm_gain", ascending=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(gdf["driver_group"], gdf["mean_norm_gain"] * 100, color="#457b9d")
        ax.set_xlabel("Mean Normalized Gain Share Across Tree Families (%)")
        ax.set_title("OD_PAX Demand Driver Importance by Domain Group")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "03_demand_drivers_by_group.png", dpi=150)
        plt.close()

    # 4. Final Holdout Monthly Actual vs Predicted Trajectory (2024-08..2026-07)
    mon_p = MODEL_DIR / "outputs" / "final_holdout_monthly_summary.csv"
    if mon_p.exists():
        mdf = pd.read_csv(mon_p)
        labels = [f"{int(y)//100}-{int(y)%100:02d}" for y in mdf["YM"]]
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(labels, mdf["y_true"] / 1e6, marker="o", label="Actual Network OD_PAX (M)", color="#1d3557", linewidth=2)
        ax.plot(labels, mdf["y_pred"] / 1e6, marker="s", linestyle="--", label="Predicted (21 Ensemble, M)", color="#e63946", linewidth=2)
        ax.axvspan(labels.index("2026-05"), len(labels) - 1, color="#ffb703", alpha=0.2, label="May-Jul 2026 Anomaly Window")
        ax.set_xticks(range(0, len(labels), 2))
        ax.set_xticklabels([labels[i] for i in range(0, len(labels), 2)], rotation=35, ha="right")
        ax.set_ylabel("Monthly Network OD_PAX (Millions)")
        ax.set_title("Untouched Final 2-Year Holdout: Actual vs. Predicted Network Volume (2024-08..2026-07)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "04_final_holdout_monthly_trajectory.png", dpi=150)
        plt.close()

    print("Saved plots to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
