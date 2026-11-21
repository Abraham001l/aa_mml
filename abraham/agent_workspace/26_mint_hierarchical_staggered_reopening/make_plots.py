from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent
PLOTS_DIR = MODEL_DIR / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
})


def plot_model_progression():
    models = [
        ("M01: Seasonal Naive", 0.2676, "#94a3b8"),
        ("M03: Damped Trend", 0.2015, "#94a3b8"),
        ("M14: Multi-Level IPF", 0.2010, "#64748b"),
        ("M20b: Cross-Sec XGB", 0.2012, "#64748b"),
        ("M21: Diverse Ensemble", 0.1989, "#3b82f6"),
        ("M24: Spatial Gravity", 0.1956, "#2563eb"),
        ("M25: Geo-Macro Reopen", 0.1932, "#059669"),
        ("M26: MinT + Geo-Macro", 0.1887, "#047857"),
    ]
    labels, scores, colors = zip(*models)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(labels, scores, color=colors, height=0.62, edgecolor="#1e293b", linewidth=0.8)
    ax.set_xlim(0.16, 0.28)
    ax.set_xlabel("3-Fold Mean Network WAPE (Lower is Better)")
    ax.set_title("American Airlines 2-Year Passenger Demand: SOTA Modeling Progression", pad=14, weight="bold")
    ax.axvline(0.1887, color="#047857", linestyle="--", alpha=0.7, linewidth=1.2)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    for bar, score in zip(bars, scores):
        w = bar.get_width()
        ax.text(w + 0.0015, bar.get_y() + bar.get_height() / 2, f"{score:.4f}",
                va="center", ha="left", fontsize=10, weight="bold", color="#0f172a")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "01_model_progression_wape.png", dpi=250)
    plt.close(fig)
    print("Saved 01_model_progression_wape.png")


def plot_multilevel_waterfall():
    levels = [
        ("Total Network (1)", 0.0218, "#047857"),
        ("Entity Airport (150)", 0.0566, "#059669"),
        ("US Interstate (41)", 0.0741, "#10b981"),
        ("Metro Pair (~3,500)", 0.1467, "#3b82f6"),
        ("Route Pair (17,119)", 0.1661, "#1d4ed8"),
    ]
    labels, scores, colors = zip(*levels)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, [s * 100 for s in scores], color=colors, width=0.55, edgecolor="#1e293b", linewidth=0.8)
    ax.set_ylabel("Steady-State Fold 3 WAPE (%)")
    ax.set_title("Multi-Level Hierarchical Predictability Across Aggregation Levels", pad=14, weight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    for bar, score in zip(bars, scores):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.35, f"{score * 100:.2f}%",
                ha="center", va="bottom", fontsize=10, weight="bold", color="#0f172a")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "02_multilevel_wape_waterfall.png", dpi=250)
    plt.close(fig)
    print("Saved 02_multilevel_wape_waterfall.png")


def plot_monthly_holdout_trajectory():
    csv_file = MODEL_DIR / "outputs" / "final_holdout_monthly_summary.csv"
    if not csv_file.exists():
        return
    df = pd.read_csv(csv_file)
    df["YM_str"] = df["YM"].astype(str).str.slice(0, 4) + "-" + df["YM"].astype(str).str.slice(4, 6)

    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    ax2 = ax1.twinx()

    x = np.arange(len(df))
    ax1.plot(x, df["y_true"] / 1e6, label="Actual Network PAX (M)", color="#0f172a", marker="o", linewidth=2.0)
    ax1.plot(x, df["y_pred"] / 1e6, label="Model 26 Forecast (M)", color="#047857", marker="s", linestyle="--", linewidth=2.0)

    ax2.bar(x, df["wape"] * 100, label="Monthly WAPE (%)", color="#3b82f6", alpha=0.25, width=0.45)

    ax1.set_ylabel("Monthly Network Passengers (Millions)")
    ax2.set_ylabel("Monthly WAPE (%)")
    ax1.set_title("Model 26 Untouched 24-Month Holdout Forecast (2024-08 to 2026-07)", pad=14, weight="bold")
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels(df["YM_str"].iloc[::2], rotation=35, ha="right")
    ax1.grid(axis="x", linestyle=":", alpha=0.5)

    # Annotate Year 1 vs Year 2
    ax1.axvline(11.5, color="#dc2626", linestyle=":", alpha=0.8, linewidth=1.5)
    ax1.text(5.5, ax1.get_ylim()[1] * 0.96, "Year 1 (WAPE = 15.86%)", ha="center", weight="bold", color="#1e3a8a")
    ax1.text(17.5, ax1.get_ylim()[1] * 0.96, "Year 2 (Anomalies in 2026-05/06)", ha="center", weight="bold", color="#991b1b")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "03_monthly_holdout_trajectory.png", dpi=250)
    plt.close(fig)
    print("Saved 03_monthly_holdout_trajectory.png")


if __name__ == "__main__":
    plot_model_progression()
    plot_multilevel_waterfall()
    plot_monthly_holdout_trajectory()
