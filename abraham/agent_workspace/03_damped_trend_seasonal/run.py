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
    """
    Shock-aware seasonal trend baseline:
    - Computes seasonal baseline per pair using shrunk recent vs 2019 recovery multiplier.
    - Estimates segment-level (Domestic vs Intl) recent growth velocity over trailing 6m vs prior year,
      damped exponentially across horizons h=1..24 toward pre-COVID 2019 seasonal profile.
    """
    assert_origin_cutoff(int(hist_panel["YM"].max()), fold.origin_ym)

    ap_cntry = airport_df.set_index("AIRPRT_CD")["CNTRY_CD"].to_dict()
    hist = hist_panel.copy()
    hist["IS_DOM"] = (hist["ORG_AIRPRT"].map(ap_cntry) == "US") & (hist["DES_AIRPRT"].map(ap_cntry) == "US")

    max_t = int(hist["T_IDX"].max())
    # Baseline seasonal pattern: 2019 same-month
    p19 = hist.loc[hist["DEP_YEAR"] == 2019, ["ORG_AIRPRT", "DES_AIRPRT", "DEP_MONTH", "OD_PAX"]].rename(
        columns={"OD_PAX": "PAX_19"}
    )
    # Recent 12m same-month
    r12 = hist.loc[hist["T_IDX"] >= max_t - 11, ["ORG_AIRPRT", "DES_AIRPRT", "DEP_MONTH", "OD_PAX"]].rename(
        columns={"OD_PAX": "PAX_R12"}
    )

    # Trailing 3m ratio relative to 2019 same 3 calendar months (segment & pair level)
    last3_t = np.arange(max_t - 2, max_t + 1)
    last3 = hist.loc[hist["T_IDX"].isin(last3_t)]
    last3_months = last3["DEP_MONTH"].unique()
    p19_same3 = hist.loc[(hist["DEP_YEAR"] == 2019) & (hist["DEP_MONTH"].isin(last3_months))]

    pair_l3 = last3.groupby(["ORG_AIRPRT", "DES_AIRPRT", "IS_DOM"], as_index=False)["OD_PAX"].sum().rename(columns={"OD_PAX": "PAIR_L3"})
    pair_19_3 = p19_same3.groupby(["ORG_AIRPRT", "DES_AIRPRT"], as_index=False)["OD_PAX"].sum().rename(columns={"OD_PAX": "PAIR_19_3"})
    pair_stats = pair_l3.merge(pair_19_3, on=["ORG_AIRPRT", "DES_AIRPRT"])

    seg_l3 = pair_stats.groupby("IS_DOM")["PAIR_L3"].sum()
    seg_19_3 = pair_stats.groupby("IS_DOM")["PAIR_19_3"].sum()
    seg_rec_now = (seg_l3 / np.maximum(seg_19_3, 1.0)).to_dict()

    # Empirical Bayes shrinkage of pair recovery ratio toward segment recovery ratio
    prior_strength = 1500.0
    pair_stats["SEG_REC_NOW"] = pair_stats["IS_DOM"].map(seg_rec_now)
    pair_stats["PAIR_REC_NOW"] = (
        pair_stats["PAIR_L3"] + prior_strength * pair_stats["SEG_REC_NOW"]
    ) / np.maximum(pair_stats["PAIR_19_3"] + prior_strength, 1.0)
    pair_stats["PAIR_REC_NOW"] = pair_stats["PAIR_REC_NOW"].clip(0.05, 2.5)

    # Trailing 6m vs prior 6m (12m earlier) segment velocity (only if post-COVID trough > 0)
    l6 = hist.loc[hist["T_IDX"].isin(np.arange(max_t - 5, max_t + 1))].groupby("IS_DOM")["OD_PAX"].sum()
    l6_prev = hist.loc[hist["T_IDX"].isin(np.arange(max_t - 17, max_t - 11))].groupby("IS_DOM")["OD_PAX"].sum()
    # Target equilibrium ratio: if below 1.02 * 2019, converge smoothly toward 1.02; if already recovered, damp toward current level with gentle 2% annual trend
    seg_eq = {dom: max(1.02, float(seg_rec_now[dom])) for dom in seg_rec_now}

    mdata_dir = model_dir / "model_specific_data" / fold.name
    mdata_dir.mkdir(parents=True, exist_ok=True)
    pair_stats.to_parquet(mdata_dir / "pair_recovery_stats.parquet", index=False)

    pred = tgt_skeleton.merge(p19, on=["ORG_AIRPRT", "DES_AIRPRT", "DEP_MONTH"], how="left")
    pred = pred.merge(r12, on=["ORG_AIRPRT", "DES_AIRPRT", "DEP_MONTH"], how="left")
    pred = pred.merge(
        pair_stats[["ORG_AIRPRT", "DES_AIRPRT", "IS_DOM", "PAIR_REC_NOW"]],
        on=["ORG_AIRPRT", "DES_AIRPRT"],
        how="left",
    )

    # Horizon-dependent exponential convergence from current recovery level to equilibrium
    # Half-life ~ 9 months when recovering from shock; near-flat persistence when already >= 0.95
    seg_eq_arr = pred["IS_DOM"].map(seg_eq).to_numpy(dtype=np.float64)
    rec_now_arr = pred["PAIR_REC_NOW"].to_numpy(dtype=np.float64)
    h_arr = pred["HORIZON"].to_numpy(dtype=np.float64)

    # If recovering (rec < 0.95), blend toward max(rec_now, seg_eq) with exponential decay
    decay = np.exp(-h_arr / 12.0)
    target_mult = np.where(rec_now_arr < 0.95, np.maximum(rec_now_arr, seg_eq_arr), rec_now_arr * (1.0 + 0.015 * (h_arr / 12.0)))
    mult = decay * rec_now_arr + (1.0 - decay) * target_mult

    # Blend 2019 seasonal shape with recent 12m seasonal shape (heavier recent weight when recovered)
    pax19 = pred["PAX_19"].fillna(0.0).to_numpy(dtype=np.float64)
    paxr12 = pred["PAX_R12"].fillna(0.0).to_numpy(dtype=np.float64)
    y_from_19 = pax19 * mult
    # If pair is already recovered (>= 0.85), blend 50% recent-12m * (1 + 1.5%/yr) and 50% 2019-scaled
    y_from_r12 = paxr12 * (1.0 + 0.015 * (h_arr / 12.0))
    w_recent = np.where(rec_now_arr >= 0.85, 0.50, 0.15)
    y_pred = np.where(pax19 > 0.0, (1.0 - w_recent) * y_from_19 + w_recent * y_from_r12, y_from_r12)

    pred["y_pred"] = np.clip(y_pred, 0.0, None)
    return pred[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON", "y_pred"]]


if __name__ == "__main__":
    run_backtest_suite(Path(__file__).resolve().parent, predict_fold)
