from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

models = {
    "m03": "03_damped_trend_seasonal",
    "m12": "12_xgboost_l1_direct_anchor",
    "m14": "14_multilevel_native_reconciled",
    "m16b": "16b_refined_structural_panel",
    "m18": "18_deconfounded_histgb_v2",
    "m19": "19_deconfounded_pytorch_v2",
    "m20": "20_deconfounded_cross_sectional_xgb",
    "m20b": "20b_cross_sectional_xgb_over_03",
}

base = pd.read_parquet(ROOT / models["m03"] / "outputs" / "backtest_predictions.parquet")[
    ["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON", "fold", "y_true", "y_pred"]
].rename(columns={"y_pred": "m03"})

for k, v in models.items():
    if k == "m03":
        continue
    df = pd.read_parquet(ROOT / v / "outputs" / "backtest_predictions.parquet")[["y_pred"]]
    base[k] = df["y_pred"].values

f3 = base[base["fold"] == "fold_3_steady"]
errs = pd.DataFrame({k: f3[k] - f3["y_true"] for k in models})
print("=== Fold 3 Residual Error Correlation Matrix ===")
print(errs.corr().round(3).to_string())


def eval_blend(weights: dict[str, float], name: str) -> None:
    yp = sum(w * base[k] for k, w in weights.items())
    wapes, rmses, smapes, rmsles = [], [], [], []
    for f in ["fold_1_recovery", "fold_2_transition", "fold_3_steady"]:
        sub_idx = base["fold"] == f
        yt = base.loc[sub_idx, "y_true"].values
        yph = np.clip(yp.loc[sub_idx].values, 0.0, None)
        wapes.append(np.sum(np.abs(yt - yph)) / np.sum(yt))
        rmses.append(np.sqrt(np.mean((yt - yph) ** 2)))
        denom = np.abs(yt) + np.abs(yph)
        smapes.append(np.mean(np.divide(2 * np.abs(yt - yph), denom, out=np.zeros_like(denom), where=denom > 0)))
        rmsles.append(np.sqrt(np.mean((np.log1p(yph) - np.log1p(yt)) ** 2)))
    print(
        f"{name:34s} | Mean={np.mean(wapes):.4f} (F1={wapes[0]:.4f}, F2={wapes[1]:.4f}, F3={wapes[2]:.4f}) "
        f"| RMSE={np.mean(rmses):.1f} | sMAPE={np.mean(smapes):.4f} | RMSLE={np.mean(rmsles):.4f}"
    )


eval_blend({"m03": 1.0}, "Pure 03 (Stat Structural)")
eval_blend({"m14": 1.0}, "Pure 14 (Reconciled IPF)")
eval_blend({"m20": 1.0}, "Pure 20 (CS-XGB)")
eval_blend({"m03": 0.5, "m14": 0.5}, "50/50 03 + 14")
eval_blend({"m03": 0.4, "m14": 0.35, "m20": 0.15, "m12": 0.10}, "03 + 14 + 20 + 12")
eval_blend({"m03": 0.35, "m14": 0.30, "m20": 0.15, "m20b": 0.10, "m18": 0.05, "m19": 0.05}, "Diverse 6-model blend")
eval_blend({"m03": 0.30, "m14": 0.35, "m20": 0.15, "m16b": 0.10, "m18": 0.05, "m19": 0.05}, "Diverse multi-family + 16b")
eval_blend({"m03": 0.35, "m14": 0.35, "m12": 0.15, "m20b": 0.15}, "Core 4-model blend")
