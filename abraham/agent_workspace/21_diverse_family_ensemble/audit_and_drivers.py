from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_eval.data_loader import build_balanced_panel, load_airport_info, load_raw_pax
from shared_eval.external_feature_store import ExternalFoldFeatureStore
from shared_eval.metrics import compute_metrics
from shared_eval.splits import BACKTEST_FOLDS

WORKSPACE = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent


def audit_errors() -> dict:
    """Comprehensive pre-holdout error audit across route, state, horizon, tier, and cutoff."""
    oof = pd.read_parquet(MODEL_DIR / "outputs" / "backtest_predictions.parquet")
    ai = load_airport_info().set_index("AIRPRT_CD")
    oof["ORG_STATE"] = oof["ORG_AIRPRT"].map(ai["REGION_CD"])
    oof["DES_STATE"] = oof["DES_AIRPRT"].map(ai["REGION_CD"])
    oof["ORG_CNTRY"] = oof["ORG_AIRPRT"].map(ai["CNTRY_CD"])
    oof["DES_CNTRY"] = oof["DES_AIRPRT"].map(ai["CNTRY_CD"])
    oof["IS_DOM"] = (oof["ORG_CNTRY"] == "US") & (oof["DES_CNTRY"] == "US")
    oof["ABS_ERR"] = np.abs(oof["y_true"] - oof["y_pred"])
    oof["ERR"] = oof["y_pred"] - oof["y_true"]

    out_dir = MODEL_DIR / "outputs"
    # 1. Horizon curve (h = 1..24) on steady Fold 3 vs overall
    h_rows = []
    for h, grp in oof.groupby("HORIZON"):
        m_all = compute_metrics(grp["y_true"], grp["y_pred"])
        f3 = grp[grp["fold"] == "fold_3_steady"]
        m_f3 = compute_metrics(f3["y_true"], f3["y_pred"])
        h_rows.append({
            "horizon": int(h),
            "mean_wape": m_all["wape"],
            "mean_bias": m_all["bias"],
            "f3_wape": m_f3["wape"],
            "f3_bias": m_f3["bias"],
        })
    h_df = pd.DataFrame(h_rows)
    h_df.to_csv(out_dir / "audit_horizon_curve.csv", index=False)

    # 2. Worst routes on Steady Fold 3 & Overall
    f3 = oof[oof["fold"] == "fold_3_steady"]
    worst_routes_f3 = (
        f3.groupby(["ORG_AIRPRT", "DES_AIRPRT", "IS_DOM"], as_index=False)
        .agg(y_true_mean=("y_true", "mean"), y_pred_mean=("y_pred", "mean"), mae=("ABS_ERR", "mean"), bias_pax=("ERR", "mean"))
        .sort_values("mae", ascending=False)
        .head(15)
    )
    worst_routes_f3.to_csv(out_dir / "audit_worst_routes_f3.csv", index=False)

    # 3. Worst US states on Steady Fold 3
    us_f3 = f3[f3["IS_DOM"] & (f3["ORG_STATE"] != f3["DES_STATE"])]
    st_f3 = (
        us_f3.groupby(["ORG_STATE", "DES_STATE", "YM"], as_index=False)[["y_true", "y_pred"]].sum()
        .assign(abs_err=lambda d: np.abs(d["y_true"] - d["y_pred"]))
        .groupby(["ORG_STATE", "DES_STATE"], as_index=False)
        .agg(y_true_sum=("y_true", "sum"), abs_err_sum=("abs_err", "sum"), pred_sum=("y_pred", "sum"))
    )
    st_f3["wape"] = st_f3["abs_err_sum"] / np.maximum(st_f3["y_true_sum"], 1.0)
    st_f3["bias"] = (st_f3["pred_sum"] - st_f3["y_true_sum"]) / np.maximum(st_f3["y_true_sum"], 1.0)
    st_f3.sort_values("abs_err_sum", ascending=False).head(15).to_csv(out_dir / "audit_worst_states_f3.csv", index=False)

    # 4. Worst airports (Entity level) on Steady Fold 3
    ap_f3 = (
        f3.groupby(["ORG_AIRPRT", "YM"], as_index=False)[["y_true", "y_pred"]].sum()
        .assign(abs_err=lambda d: np.abs(d["y_true"] - d["y_pred"]))
        .groupby("ORG_AIRPRT", as_index=False)
        .agg(y_true_sum=("y_true", "sum"), abs_err_sum=("abs_err", "sum"), pred_sum=("y_pred", "sum"))
    )
    ap_f3["wape"] = ap_f3["abs_err_sum"] / np.maximum(ap_f3["y_true_sum"], 1.0)
    ap_f3["bias"] = (ap_f3["pred_sum"] - ap_f3["y_true_sum"]) / np.maximum(ap_f3["y_true_sum"], 1.0)
    ap_f3.sort_values("abs_err_sum", ascending=False).head(15).to_csv(out_dir / "audit_worst_airports_f3.csv", index=False)

    print("=== Audit: Worst 10 Routes on Steady Fold 3 ===")
    print(worst_routes_f3.head(10).round(1).to_string(index=False))
    print("\n=== Audit: Worst 10 Origin Airports (Entity) on Steady Fold 3 ===")
    print(ap_f3.sort_values("abs_err_sum", ascending=False).head(10).round(4).to_string(index=False))

    return {"horizon_y1_f3": float(h_df.loc[h_df["horizon"] <= 12, "f3_wape"].mean()), "horizon_y2_f3": float(h_df.loc[h_df["horizon"] >= 13, "f3_wape"].mean())}


def analyze_demand_drivers() -> None:
    """Quantify top feature drivers across component models and external feature groups."""
    out_dir = MODEL_DIR / "outputs"
    # Collect normalized gain importances from 12, 20b, and 17 on Fold 3
    sources = [
        ("m12_l1", WORKSPACE / "12_xgboost_l1_direct_anchor" / "model_specific_data" / "fold_3_steady" / "xgb_l1_importance.csv", "importance_gain"),
        ("m20b_cs", WORKSPACE / "20b_cross_sectional_xgb_over_03" / "model_specific_data" / "fold_3_steady" / "xgb20b_importance.csv", "importance_gain"),
        ("m17_tw", WORKSPACE / "17_deconfounded_xgb_v2" / "model_specific_data" / "fold_3_steady" / "deconfounded_xgb_v2_importance.csv", "importance_gain"),
    ]
    merged = None
    for name, path, col in sources:
        if path.exists():
            df = pd.read_csv(path)
            df[name] = df[col] / df[col].sum()
            df = df[["feature", name]]
            merged = df if merged is None else merged.merge(df, on="feature", how="outer")
    merged = merged.fillna(0.0)
    cols = [c for c in merged.columns if c != "feature"]
    merged["mean_norm_gain"] = merged[cols].mean(axis=1)
    merged = merged.sort_values("mean_norm_gain", ascending=False)
    merged.to_csv(out_dir / "demand_drivers_feature_importance.csv", index=False)

    # Group by domain driver family
    def assign_group(feat: str) -> str:
        if feat.startswith("EXT_"):
            if "MIG" in feat:
                return "External: Census Migration (CBSA/County)"
            if "CBSA" in feat or "CTY" in feat or "ST_" in feat or "GRAVITY" in feat:
                return "External: Census Population & Density"
            if "FAA" in feat:
                return "External: FAA Airport Enplanements & O&D Share"
            return "External: Other"
        if "YIELD" in feat:
            return "Pricing & Revenue Yield"
        if feat in {"HORIZON", "LOG_H", "DEP_MONTH", "DAYS_IN_MONTH", "FRI_SUN_DAYS", "EASTER_IN_MONTH", "GOOD_FRI_IN_MONTH", "TGIVING_LATE", "SIN_M", "COS_M", "SIN_2M", "COS_2M", "SNOWBIRD_INTERACT"}:
            return "Seasonality, Calendar & Horizon"
        if feat in {"LON_DIFF", "LAT_DIFF_ABS", "DIST_MI", "LOG_DIST", "IS_DOM", "IS_SAME_STATE", "HAS_REV", "DIR_ASYM_R12"}:
            return "Route Geometry & Symmetry"
        if "ORG_" in feat or "DES_" in feat or "METRO" in feat or "SHARE_" in feat or "REL_REC" in feat:
            if any(k in feat for k in ["LAT", "LON", "RUNWAY", "ELEV", "IS_US", "MULTI"]):
                return "Airport Geography & Infrastructure"
            return "Airport/Entity & Metro Network Scale"
        return "Historical Volume, Anchor & Momentum"

    merged["driver_group"] = merged["feature"].map(assign_group)
    grp_df = merged.groupby("driver_group", as_index=False)[cols + ["mean_norm_gain"]].sum().sort_values("mean_norm_gain", ascending=False)
    grp_df.to_csv(out_dir / "demand_drivers_by_group.csv", index=False)

    print("\n=== Top 20 Demand Driver Features (Normalized Gain across L1 / CS / Tweedie) ===")
    print(merged.head(20).round(4).to_string(index=False))
    print("\n=== Demand Drivers by Thematic Group ===")
    print(grp_df.round(4).to_string(index=False))

    # Concrete pre-holdout examples of external driver correlations with 2023/24 steady growth
    raw_df = load_raw_pax()
    pre_df = raw_df.loc[raw_df["YM"] <= 202407].copy()
    hist, _ = build_balanced_panel(pre_df, 202407, include_targets=False)
    ai = load_airport_info()
    ext_store = ExternalFoldFeatureStore(hist, ai, include_macro=False)
    sl = ext_store.build_features_for_slice(ext_store.t_cut, 1)
    sl["ORG_AIRPRT"] = ext_store.pairs_df["ORG_AIRPRT"]
    sl["DES_AIRPRT"] = ext_store.pairs_df["DES_AIRPRT"]
    sl["PAX_R12"] = ext_store.pax_r12[:, ext_store.t_cut]

    dom_sl = sl[(sl["IS_DOM"] > 0.5) & (sl["PAX_R12"] >= 100.0)].copy()
    corr_cols = [
        "EXT_GRAVITY_POP", "EXT_LOG_MIG_TOTAL", "EXT_ORG_CBSA_MIG", "EXT_DES_CBSA_MIG",
        "EXT_ORG_FAA_ENP", "EXT_ORG_OD_VS_FAA", "LOG_DIST", "YIELD_R12", "DIR_ASYM_R12"
    ]
    corrs = {c: float(dom_sl["REC_19_R12"].corr(dom_sl[c])) for c in corr_cols}
    corr_pax = {c: float(np.log1p(dom_sl["PAX_R12"]).corr(dom_sl[c])) for c in corr_cols}
    pd.DataFrame([
        {"feature": c, "corr_with_log_pax": corr_pax[c], "corr_with_post_covid_growth_vs_19": corrs[c]}
        for c in corr_cols
    ]).to_csv(out_dir / "demand_drivers_correlations.csv", index=False)
    print("\n=== Pre-Holdout Cross-Sectional Correlations (US Domestic Routes >= 100 pax/mo) ===")
    print(pd.DataFrame([{"feature": c, "corr_log_pax": corr_pax[c], "corr_growth_vs_19": corrs[c]} for c in corr_cols]).round(4).to_string(index=False))


def check_train_vs_backtest_overfitting() -> None:
    """Audit train-vs-backtest gap for component GBDT models on Fold 3."""
    import xgboost as xgb
    raw_df = load_raw_pax()
    pre_df = raw_df.loc[raw_df["YM"] <= 202407].copy()
    ai = load_airport_info()
    f3 = BACKTEST_FOLDS[-1]
    hist, tgt = build_balanced_panel(pre_df, f3.origin_ym, f3.target_end_ym, include_targets=True)
    store = ExternalFoldFeatureStore(hist, ai, include_macro=False)

    # Check 12_xgboost_l1_direct_anchor in-sample WAPE vs OOF Fold 3 WAPE
    from shared_eval.feature_engine import assemble_training_dataset
    X_tr, y_tr, w_tr, _ = assemble_training_dataset(store, s_min=6, stride_s=3, exclude_acute_covid_targets=True, recency_halflife=18.0)
    feat_cols = [c for c in X_tr.columns if c != "DAMPED_ANCHOR"]
    anc_tr = X_tr["DAMPED_ANCHOR"].to_numpy(dtype=np.float64)
    y_diff = (y_tr - anc_tr).astype(np.float32)
    dtrain = xgb.DMatrix(X_tr[feat_cols], label=y_diff, weight=w_tr)
    booster = xgb.train(
        {"objective": "reg:absoluteerror", "learning_rate": 0.04, "max_depth": 6, "min_child_weight": 50.0,
         "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 5.0, "reg_lambda": 20.0, "tree_method": "hist", "nthread": -1, "seed": 42},
        dtrain, num_boost_round=500,
    )
    diff_tr_hat = booster.predict(dtrain).astype(np.float64)
    max_shift = np.maximum(anc_tr * 2.0, 200.0)
    y_tr_hat = np.clip(anc_tr + np.clip(diff_tr_hat, -max_shift, max_shift), 0.0, None)
    train_wape = float(np.sum(np.abs(y_tr - y_tr_hat)) / np.sum(y_tr))
    anc_train_wape = float(np.sum(np.abs(y_tr - anc_tr)) / np.sum(y_tr))

    m12_json = json.load(open(WORKSPACE / "12_xgboost_l1_direct_anchor" / "outputs" / "backtest_metrics.json"))
    f3_wape = m12_json["folds"]["fold_3_steady"]["market_dir_overall"]["wape"]
    out = pd.DataFrame([{
        "model": "12_xgboost_l1_direct_anchor",
        "fold": "fold_3_steady",
        "train_anchor_wape": anc_train_wape,
        "train_insample_wape": train_wape,
        "backtest_f3_wape": f3_wape,
        "train_to_backtest_gap": f3_wape - train_wape,
    }])
    out.to_csv(MODEL_DIR / "outputs" / "overfitting_audit.csv", index=False)
    print("\n=== Overfitting Audit (Train vs Backtest Fold 3) ===")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    audit_errors()
    analyze_demand_drivers()
    check_train_vs_backtest_overfitting()
