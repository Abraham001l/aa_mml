from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute canonical evaluation metric suite."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    yp_nonneg = np.clip(yp, 0.0, None)

    abs_err = np.abs(yt - yp)
    sq_err = (yt - yp) ** 2
    sum_abs_y = float(np.sum(np.abs(yt)))
    sum_y = float(np.sum(yt))

    wape = float(np.sum(abs_err) / sum_abs_y) if sum_abs_y > 0 else np.nan
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(sq_err)))

    denom = np.abs(yt) + np.abs(yp)
    smape_terms = np.divide(2.0 * abs_err, denom, out=np.zeros_like(abs_err), where=denom > 0)
    smape = float(np.mean(smape_terms))

    rmsle = float(np.sqrt(np.mean((np.log1p(yp_nonneg) - np.log1p(np.clip(yt, 0.0, None))) ** 2)))
    bias = float(np.sum(yp - yt) / sum_y) if sum_y > 0 else np.nan

    return {
        "n": int(len(yt)),
        "sum_y_true": sum_y,
        "sum_y_pred": float(np.sum(yp)),
        "wape": wape,
        "mae": mae,
        "rmse": rmse,
        "smape": smape,
        "rmsle": rmsle,
        "bias": bias,
    }


def assign_market_tiers(hist_panel: pd.DataFrame, origin_ym: int) -> pd.DataFrame:
    """
    Assign size tier per directional pair strictly using trailing 12 months <= origin_ym.
    """
    max_t = int(hist_panel["T_IDX"].max())
    min_t = max(0, max_t - 11)
    trail = hist_panel.loc[(hist_panel["YM"] <= origin_ym) & (hist_panel["T_IDX"] >= min_t)]
    stats = (
        trail.groupby(["ORG_AIRPRT", "DES_AIRPRT"], as_index=False)["OD_PAX"]
        .mean()
        .rename(columns={"OD_PAX": "HIST_MEAN_PAX"})
    )
    bins = [-np.inf, 100.0, 500.0, 2500.0, 10000.0, np.inf]
    labels = ["1_Micro(<100)", "2_Small(100-500)", "3_Medium(500-2500)", "4_Large(2500-10000)", "5_Mega(>=10000)"]
    stats["SIZE_TIER"] = pd.cut(stats["HIST_MEAN_PAX"], bins=bins, labels=labels, right=False).astype(str)
    return stats


def evaluate_predictions(
    pred_df: pd.DataFrame,
    hist_panel: pd.DataFrame,
    airport_df: pd.DataFrame,
    origin_ym: int,
) -> Dict[str, dict]:
    """
    Evaluate directional airport-pair predictions across horizons, size tiers,
    market definitions, state level, and entity/airport level.

    pred_df columns required: [ORG_AIRPRT, DES_AIRPRT, YM, HORIZON, y_true, y_pred]
    """
    required = {"ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON", "y_true", "y_pred"}
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(f"pred_df missing columns: {missing}")

    ap_meta = airport_df.set_index("AIRPRT_CD")
    org_metro = ap_meta["CITY_METRO_IATA_CD"].to_dict()
    org_cntry = ap_meta["CNTRY_CD"].to_dict()
    org_reg = ap_meta["REGION_CD"].to_dict()

    df = pred_df.copy()
    df["y_pred"] = np.clip(df["y_pred"].to_numpy(dtype=np.float64), 0.0, None)
    df["ORG_METRO"] = df["ORG_AIRPRT"].map(org_metro)
    df["DES_METRO"] = df["DES_AIRPRT"].map(org_metro)
    df["ORG_CNTRY"] = df["ORG_AIRPRT"].map(org_cntry)
    df["DES_CNTRY"] = df["DES_AIRPRT"].map(org_cntry)
    df["ORG_REGION"] = df["ORG_AIRPRT"].map(org_reg)
    df["DES_REGION"] = df["DES_AIRPRT"].map(org_reg)
    df["IS_DOMESTIC"] = (df["ORG_CNTRY"] == "US") & (df["DES_CNTRY"] == "US")

    tiers = assign_market_tiers(hist_panel, origin_ym)
    df = df.merge(tiers[["ORG_AIRPRT", "DES_AIRPRT", "SIZE_TIER"]], on=["ORG_AIRPRT", "DES_AIRPRT"], how="left")

    results: Dict[str, dict] = {}

    # 1. Market level: directional airport pairs
    results["market_dir_overall"] = compute_metrics(df["y_true"], df["y_pred"])
    results["market_dir_year1"] = compute_metrics(
        df.loc[df["HORIZON"] <= 12, "y_true"], df.loc[df["HORIZON"] <= 12, "y_pred"]
    )
    results["market_dir_year2"] = compute_metrics(
        df.loc[df["HORIZON"] >= 13, "y_true"], df.loc[df["HORIZON"] >= 13, "y_pred"]
    )

    # Market segment & size tier breakdowns
    for seg_val, seg_name in [(True, "seg_domestic"), (False, "seg_intl_pr")]:
        sub = df.loc[df["IS_DOMESTIC"] == seg_val]
        results[seg_name] = compute_metrics(sub["y_true"], sub["y_pred"])

    for tier_name, sub in df.groupby("SIZE_TIER", observed=True):
        results[f"tier_{tier_name}"] = compute_metrics(sub["y_true"], sub["y_pred"])

    # 2. Alternative Market definitions (Undirected Airport Pair & Directional Metro Pair)
    df["AP_MIN"] = np.minimum(df["ORG_AIRPRT"], df["DES_AIRPRT"])
    df["AP_MAX"] = np.maximum(df["ORG_AIRPRT"], df["DES_AIRPRT"])
    undir = df.groupby(["AP_MIN", "AP_MAX", "YM", "HORIZON"], as_index=False)[["y_true", "y_pred"]].sum()
    results["market_undir_overall"] = compute_metrics(undir["y_true"], undir["y_pred"])

    metro = df.groupby(["ORG_METRO", "DES_METRO", "YM", "HORIZON"], as_index=False)[["y_true", "y_pred"]].sum()
    results["market_metro_dir_overall"] = compute_metrics(metro["y_true"], metro["y_pred"])
    results["market_metro_dir_year1"] = compute_metrics(
        metro.loc[metro["HORIZON"] <= 12, "y_true"], metro.loc[metro["HORIZON"] <= 12, "y_pred"]
    )
    results["market_metro_dir_year2"] = compute_metrics(
        metro.loc[metro["HORIZON"] >= 13, "y_true"], metro.loc[metro["HORIZON"] >= 13, "y_pred"]
    )

    # 3. State level
    us_inter = df.loc[df["IS_DOMESTIC"] & (df["ORG_REGION"] != df["DES_REGION"])]
    st_inter = us_inter.groupby(["ORG_REGION", "DES_REGION", "YM", "HORIZON"], as_index=False)[["y_true", "y_pred"]].sum()
    results["state_us_interstate_overall"] = compute_metrics(st_inter["y_true"], st_inter["y_pred"])
    results["state_us_interstate_year1"] = compute_metrics(
        st_inter.loc[st_inter["HORIZON"] <= 12, "y_true"], st_inter.loc[st_inter["HORIZON"] <= 12, "y_pred"]
    )
    results["state_us_interstate_year2"] = compute_metrics(
        st_inter.loc[st_inter["HORIZON"] >= 13, "y_true"], st_inter.loc[st_inter["HORIZON"] >= 13, "y_pred"]
    )

    all_reg = df.groupby(["ORG_REGION", "DES_REGION", "YM", "HORIZON"], as_index=False)[["y_true", "y_pred"]].sum()
    results["state_all_region_overall"] = compute_metrics(all_reg["y_true"], all_reg["y_pred"])

    # 4. Entity / Airport level
    org_tot = df.groupby(["ORG_AIRPRT", "YM", "HORIZON"], as_index=False)[["y_true", "y_pred"]].sum()
    des_tot = df.groupby(["DES_AIRPRT", "YM", "HORIZON"], as_index=False)[["y_true", "y_pred"]].sum()
    results["entity_org_airport_overall"] = compute_metrics(org_tot["y_true"], org_tot["y_pred"])
    results["entity_des_airport_overall"] = compute_metrics(des_tot["y_true"], des_tot["y_pred"])

    ap_both = pd.concat([
        org_tot.rename(columns={"ORG_AIRPRT": "AIRPRT_CD"}),
        des_tot.rename(columns={"DES_AIRPRT": "AIRPRT_CD"}),
    ], ignore_index=True).groupby(["AIRPRT_CD", "YM", "HORIZON"], as_index=False)[["y_true", "y_pred"]].sum()
    results["entity_total_airport_overall"] = compute_metrics(ap_both["y_true"], ap_both["y_pred"])
    results["entity_total_airport_year1"] = compute_metrics(
        ap_both.loc[ap_both["HORIZON"] <= 12, "y_true"], ap_both.loc[ap_both["HORIZON"] <= 12, "y_pred"]
    )
    results["entity_total_airport_year2"] = compute_metrics(
        ap_both.loc[ap_both["HORIZON"] >= 13, "y_true"], ap_both.loc[ap_both["HORIZON"] >= 13, "y_pred"]
    )

    return results


def summarize_fold_results(fold_metrics: Dict[str, Dict[str, dict]]) -> Dict[str, dict]:
    """Aggregate metrics across multiple backtest folds (mean & std of key rates)."""
    keys = list(next(iter(fold_metrics.values())).keys())
    summary: Dict[str, dict] = {}
    for level_key in keys:
        metric_names = ["wape", "mae", "rmse", "smape", "rmsle", "bias"]
        agg = {}
        for m in metric_names:
            vals = [fold_metrics[f][level_key][m] for f in fold_metrics]
            agg[f"{m}_mean"] = float(np.mean(vals))
            agg[f"{m}_std"] = float(np.std(vals, ddof=0))
        summary[level_key] = agg
    return summary
