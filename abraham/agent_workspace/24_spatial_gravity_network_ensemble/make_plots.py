from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL_DIR = Path(__file__).resolve().parent
PLOTS_DIR = MODEL_DIR / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Progression Plot
    models = [
        ("01 Naive", 0.3546, 0.2297),
        ("02 2019 Base", 0.2626, 0.2451),
        ("03 Damped Trend", 0.2015, 0.1762),
        ("06 XGB LogRes", 0.2858, 0.1778),
        ("12 L1 Direct", 0.2304, 0.1746),
        ("14 MultiLevel IPF", 0.2044, 0.1749),
        ("20b CrossSect XGB", 0.2273, 0.1777),
        ("21 Diverse Ensemble", 0.1989, 0.1704),
        ("22 Deep Gravity", 0.2198, 0.1729),
        ("24 Champion Ensemble", 0.1956, 0.1657),
    ]
    names = [m[0] for m in models]
    mean_w = [m[1] for m in models]
    f3_w = [m[2] for m in models]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, mean_w, width, label="3-Fold Mean WAPE", color="#4C72B0", alpha=0.9)
    ax.bar(x + width/2, f3_w, width, label="Steady-State (F3) WAPE", color="#55A868", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("WAPE (Lower is Better)", fontsize=12)
    ax.set_title("American Airlines OD_PAX: Complete Model Evolution to SOTA (Model 24)", fontsize=14, fontweight="bold")
    ax.legend(frameon=True, fontsize=11)
    ax.set_ylim(0, 0.40)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_model_progression_wape.png", dpi=300)
    plt.close()

    # 2. Multi-Level Accuracy
    levels = ["Mega Routes (>=10k)", "Metro Pairs", "Interstate States", "Entity Total Airport"]
    w21_levels = [0.1580, 0.1514, 0.0821, 0.0710]
    w24_levels = [0.1401, 0.1466, 0.0738, 0.0569]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(levels))
    ax.bar(x - 0.18, w21_levels, 0.35, label="Model 21 Benchmark", color="#C44E52", alpha=0.85)
    ax.bar(x + 0.18, w24_levels, 0.35, label="Model 24 Champion", color="#4C72B0", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=11)
    ax.set_ylabel("WAPE", fontsize=12)
    ax.set_title("Multi-Level Accuracy Gains: Model 21 vs. Model 24 Champion", fontsize=14, fontweight="bold")
    ax.legend(frameon=True, fontsize=11)
    for i in range(len(levels)):
        gain = (w24_levels[i] - w21_levels[i]) / w21_levels[i] * 100
        ax.text(i + 0.18, w24_levels[i] + 0.005, f"{gain:.1f}%", ha="center", fontsize=10, fontweight="bold", color="#1F4E79")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_multilevel_wape_comparison.png", dpi=300)
    plt.close()

    print(f"Plots generated cleanly in {PLOTS_DIR}")


if __name__ == "__main__":
    main()
