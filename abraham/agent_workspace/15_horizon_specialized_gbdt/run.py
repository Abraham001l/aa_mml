from __future__ import annotations

import gc
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_eval.external_feature_store import ExternalFoldFeatureStore
from shared_eval.feature_engine import assemble_training_dataset
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
    Horizon-Specialized Tweedie XGBoost:
    Trains separate Year-1 (h=1..12) and Year-2 (h=13..24) models with horizon-appropriate
    regularization (Year-2 uses stronger shrinkage & excludes noisy single-month PAX_S transient).
    """
    assert_origin_cutoff(int(hist_panel["YM"].max()), fold.origin_ym)

    store = ExternalFoldFeatureStore(hist_panel, airport_df, include_macro=False)
    X_train, y_train, w_train, _ = assemble_training_dataset(
        store, s_min=6, stride_s=2, exclude_acute_covid_targets=True, recency_halflife=18.0
    )

    h_tr = X_train["HORIZON"].to_numpy(dtype=np.int32)
    y1_mask = h_tr <= 12
    y2_mask = h_tr >= 13

    feat_y1 = [c for c in X_train.columns if c != "DAMPED_ANCHOR"]
    # Year-2 drops noisy 1-month transient LOG_PAX_S & focuses on structural/smoothed signals
    feat_y2 = [c for c in feat_y1 if c != "LOG_PAX_S"]

    bm_all = np.log(np.maximum(X_train["DAMPED_ANCHOR"].to_numpy(dtype=np.float64), 1.0)).astype(np.float32)

    d_y1 = xgb.DMatrix(
        X_train.loc[y1_mask, feat_y1],
        label=np.clip(y_train[y1_mask], 0.0, None),
        weight=w_train[y1_mask],
        base_margin=bm_all[y1_mask],
    )
    d_y2 = xgb.DMatrix(
        X_train.loc[y2_mask, feat_y2],
        label=np.clip(y_train[y2_mask], 0.0, None),
        weight=w_train[y2_mask],
        base_margin=bm_all[y2_mask],
    )
    del X_train, y_train, w_train, bm_all
    gc.collect()

    params_y1 = {
        "objective": "reg:tweedie",
        "tweedie_variance_power": 1.4,
        "learning_rate": 0.04,
        "max_depth": 6,
        "min_child_weight": 50.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 2.0,
        "reg_lambda": 20.0,
        "tree_method": "hist",
        "nthread": -1,
        "seed": 42,
    }
    params_y2 = {
        "objective": "reg:tweedie",
        "tweedie_variance_power": 1.4,
        "learning_rate": 0.035,
        "max_depth": 5,
        "min_child_weight": 100.0,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_alpha": 5.0,
        "reg_lambda": 40.0,
        "tree_method": "hist",
        "nthread": -1,
        "seed": 43,
    }
    b_y1 = xgb.train(params_y1, d_y1, num_boost_round=500)
    b_y2 = xgb.train(params_y2, d_y2, num_boost_round=420)
    del d_y1, d_y2
    gc.collect()

    mdata_dir = model_dir / "model_specific_data" / fold.name
    mdata_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"feature": k, "gain_y1": v} for k, v in b_y1.get_score(importance_type="gain").items()]).sort_values(
        "gain_y1", ascending=False
    ).to_csv(mdata_dir / "y1_importance.csv", index=False)
    pd.DataFrame([{"feature": k, "gain_y2": v} for k, v in b_y2.get_score(importance_type="gain").items()]).sort_values(
        "gain_y2", ascending=False
    ).to_csv(mdata_dir / "y2_importance.csv", index=False)

    pred_frames = []
    pairs = store.pairs_df[["ORG_AIRPRT", "DES_AIRPRT"]].copy()
    for h in range(1, 25):
        t_tgt = store.t_cut + h
        ym_tgt = int(store.cal_df.loc[t_tgt, "YM"])
        X_te_df = store.build_features_for_slice(store.t_cut, h)
        anc = X_te_df["DAMPED_ANCHOR"].to_numpy(dtype=np.float64)
        bm_te = np.log(np.maximum(anc, 1.0)).astype(np.float32)
        if h <= 12:
            y_hat = b_y1.predict(xgb.DMatrix(X_te_df[feat_y1], base_margin=bm_te))
        else:
            y_hat = b_y2.predict(xgb.DMatrix(X_te_df[feat_y2], base_margin=bm_te))
        y_hat = np.clip(y_hat, 0.0, np.maximum(anc * 5.0 + 500.0, 200000.0))

        df_h = pairs.copy()
        df_h["YM"] = ym_tgt
        df_h["HORIZON"] = h
        df_h["y_pred"] = y_hat
        pred_frames.append(df_h)

    pred_all = pd.concat(pred_frames, ignore_index=True)
    return tgt_skeleton[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"]].merge(
        pred_all, on=["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"], how="left"
    )


if __name__ == "__main__":
    run_backtest_suite(Path(__file__).resolve().parent, predict_fold)
