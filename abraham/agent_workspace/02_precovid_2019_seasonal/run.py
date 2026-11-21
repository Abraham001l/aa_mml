from __future__ import annotations

from pathlib import Path
import sys

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
    """Predict same-calendar-month 2019 pre-COVID baseline (with trailing 12m fallback if 2019=0)."""
    assert_origin_cutoff(int(hist_panel["YM"].max()), fold.origin_ym)

    base_2019 = hist_panel.loc[
        hist_panel["DEP_YEAR"] == 2019, ["ORG_AIRPRT", "DES_AIRPRT", "DEP_MONTH", "OD_PAX"]
    ].rename(columns={"OD_PAX": "PAX_2019"})

    max_t = int(hist_panel["T_IDX"].max())
    recent_12 = hist_panel.loc[
        hist_panel["T_IDX"] >= max_t - 11, ["ORG_AIRPRT", "DES_AIRPRT", "DEP_MONTH", "OD_PAX"]
    ].rename(columns={"OD_PAX": "PAX_RECENT"})

    lookup = base_2019.merge(recent_12, on=["ORG_AIRPRT", "DES_AIRPRT", "DEP_MONTH"], how="outer")
    # If 2019 was 0 (e.g., BNA-SFB Jan-Apr 2019), fall back to recent same-month
    lookup["y_pred"] = lookup["PAX_2019"].where(lookup["PAX_2019"] > 0.0, lookup["PAX_RECENT"]).fillna(0.0)

    mdata_dir = model_dir / "model_specific_data" / fold.name
    mdata_dir.mkdir(parents=True, exist_ok=True)
    lookup.to_parquet(mdata_dir / "precovid_2019_lookup.parquet", index=False)

    pred = tgt_skeleton.merge(
        lookup[["ORG_AIRPRT", "DES_AIRPRT", "DEP_MONTH", "y_pred"]],
        on=["ORG_AIRPRT", "DES_AIRPRT", "DEP_MONTH"],
        how="left",
    )
    pred["y_pred"] = pred["y_pred"].fillna(0.0).clip(lower=0.0)
    return pred[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON", "y_pred"]]


if __name__ == "__main__":
    run_backtest_suite(Path(__file__).resolve().parent, predict_fold)
