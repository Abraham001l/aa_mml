from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = MODEL_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # 1. Progression of Mean WAPE across Key Milestones
    models = [
        "01 Seasonal Naive",
        "03 Damped Structural",
        "14 Multi-Level IPF",
        "21 Diverse Ensemble",
        "22 Deep Spatial Gravity",
        "24 Grand Ensemble",
        "25 Geo-Macro SOTA"
    ]
    mean_wapes = [0.3546, 0.2015, 0.2015, 0.1989, 0.2198, 0.1956, 0.1932]
    steady_wapes = [0.2297, 0.1762, 0.1738, 0.1704, 0.1729, 0.1657, 0.1658]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, mean_wapes, width, label="3-Fold Mean WAPE", color="#1f77b4")
    plt.bar(x + width/2, steady_wapes, width, label="Steady Fold 3 WAPE", color="#2ca02c")
    plt.axhline(0.20, color="gray", linestyle="--", alpha=0.7, label="20% Error Threshold")
    plt.ylabel("WAPE")
    plt.title("American Airlines OD_PAX Forecasting Progression (Models 01 -> 25)")
    plt.xticks(x, models, rotation=25, ha="right")
    plt.legend()
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_model_progression_wape.png", dpi=200)
    plt.close()

    # 2. Multi-Level Predictability Waterfall
    levels = ["Directional Route", "Undirected Route", "Metro-Pair", "US State-Pair", "Entity Airport"]
    errors = [0.1658, 0.1610, 0.1468, 0.0738, 0.0566]
    colors = ["#d62728", "#ff7f0e", "#bcbd22", "#17becf", "#1f77b4"]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(levels, [e * 100 for e in errors], color=colors, width=0.55)
    plt.ylabel("WAPE (%)")
    plt.title("Model 25 Demand Predictability Across Aggregation Levels")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold")
    plt.ylim(0, 20)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_multilevel_wape_waterfall.png", dpi=200)
    plt.close()
    print("Plots saved successfully in", PLOTS_DIR)


if __name__ == "__main__":
    main()
