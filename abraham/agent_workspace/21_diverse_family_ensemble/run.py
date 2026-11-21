from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_eval.runner import run_backtest_suite
from shared_eval.splits import FoldSpec, assert_origin_cutoff

WORKSPACE = Path(__file__).resolve().parents[1]


def _load_predict_fn(folder_name: str):
    spec = importlib.util.spec_from_file_location(f"mod_{folder_name}", WORKSPACE / folder_name / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.predict_fold


COMPONENT_WEIGHTS = [
    ("03_damped_trend_seasonal", 0.35),
    ("14_multilevel_native_reconciled", 0.35),
    ("12_xgboost_l1_direct_anchor", 0.15),
    ("20b_cross_sectional_xgb_over_03", 0.15),
]


def predict_fold(
    fold: FoldSpec,
    hist_panel: pd.DataFrame,
    tgt_skeleton: pd.DataFrame,
    airport_df: pd.DataFrame,
    model_dir: Path,
) -> pd.DataFrame:
    """
    Self-contained Diverse Multi-Family Ensemble combining:
    - 35% Statistical Damped-Shrinkage Structural Anchor (03)
    - 35% Multi-Level Native Airport/State/Metro IPF Reconciled (14)
    - 15% Direct L1 WAPE-Aligned XGBoost (12)
    - 15% Zero-Macro-Bias Cross-Sectional Census/FAA/Migration XGBoost (20b)
    """
    assert_origin_cutoff(int(hist_panel["YM"].max()), fold.origin_ym)

    comp_preds = []
    out_base = tgt_skeleton[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"]].copy()
    mdata_dir = model_dir / "model_specific_data" / fold.name
    mdata_dir.mkdir(parents=True, exist_ok=True)

    y_blend = np.zeros(len(out_base), dtype=np.float64)
    comp_summary = []
    for folder_name, weight in COMPONENT_WEIGHTS:
        pred_fn = _load_predict_fn(folder_name)
        sub_dir = model_dir / "model_specific_data" / "_sub_artifacts" / folder_name
        df_c = pred_fn(fold, hist_panel.copy(), tgt_skeleton.copy(), airport_df.copy(), sub_dir)
        merged = out_base.merge(
            df_c[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON", "y_pred"]],
            on=["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"],
            how="left",
            validate="one_to_one",
        )
        vals = merged["y_pred"].to_numpy(dtype=np.float64)
        y_blend += weight * vals
        comp_summary.append({"component": folder_name, "weight": weight, "sum_pred": float(np.sum(vals))})

    pd.DataFrame(comp_summary).to_csv(mdata_dir / "ensemble_weights.csv", index=False)
    out_base["y_pred"] = np.clip(y_blend, 0.0, None)
    return out_base


if __name__ == "__main__":
    run_backtest_suite(Path(__file__).resolve().parent, predict_fold)
