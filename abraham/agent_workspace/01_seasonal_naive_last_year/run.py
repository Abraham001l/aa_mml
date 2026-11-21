from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_eval.runner import run_backtest_suite
from shared_eval.splits import FoldSpec, assert_origin_cutoff


def predict_fold(
    fold: FoldSpec,
    hist_panel: pd.DataFrame,
    tgt_skeleton: pd.DataFrame,
    airport_df: pd.DataFrame,
    model_dir: Path,
) -> pd.DataFrame:
    """Repeat the most recent observed same calendar month from [origin-11m .. origin]."""
    assert_origin_cutoff(int(hist_panel["YM"].max()), fold.origin_ym)

    max_t = int(hist_panel["T_IDX"].max())
    recent_12 = hist_panel.loc[hist_panel["T_IDX"] >= max_t - 11, ["ORG_AIRPRT", "DES_AIRPRT", "DEP_MONTH", "OD_PAX"]].copy()

    # Save model-specific fold lookup table
    mdata_dir = model_dir / "model_specific_data" / fold.name
    mdata_dir.mkdir(parents=True, exist_ok=True)
    recent_12.to_parquet(mdata_dir / "trailing_12m_seasonal_lookup.parquet", index=False)

    pred = tgt_skeleton.merge(
        recent_12.rename(columns={"OD_PAX": "y_pred"}),
        on=["ORG_AIRPRT", "DES_AIRPRT", "DEP_MONTH"],
        how="left",
    )
    pred["y_pred"] = pred["y_pred"].fillna(0.0).clip(lower=0.0)
    return pred[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON", "y_pred"]]


if __name__ == "__main__":
    run_backtest_suite(Path(__file__).resolve().parent, predict_fold)
