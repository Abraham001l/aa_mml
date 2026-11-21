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
    L1/MAE-loss XGBoost predicting signed residual (y_true - DAMPED_ANCHOR) directly on passenger scale,
    aligned directly with WAPE = sum(|y - y_hat|) / sum(y).
    """
    assert_origin_cutoff(int(hist_panel["YM"].max()), fold.origin_ym)

    store = ExternalFoldFeatureStore(hist_panel, airport_df, include_macro=False)
    X_train, y_train, w_train, _ = assemble_training_dataset(
        store, s_min=6, stride_s=3, exclude_acute_covid_targets=True, recency_halflife=18.0
    )
    feat_cols = [c for c in X_train.columns if c != "DAMPED_ANCHOR"]
    anchor_tr = X_train["DAMPED_ANCHOR"].to_numpy(dtype=np.float64)
    y_diff = (y_train - anchor_tr).astype(np.float32)

    dtrain = xgb.DMatrix(X_train[feat_cols], label=y_diff, weight=w_train)
    del X_train, y_train, w_train, anchor_tr, y_diff
    gc.collect()

    params = {
        "objective": "reg:absoluteerror",
        "learning_rate": 0.04,
        "max_depth": 6,
        "min_child_weight": 50.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 5.0,
        "reg_lambda": 20.0,
        "tree_method": "hist",
        "nthread": -1,
        "seed": 42,
    }
    booster = xgb.train(params, dtrain, num_boost_round=500)
    del dtrain
    gc.collect()

    mdata_dir = model_dir / "model_specific_data" / fold.name
    mdata_dir.mkdir(parents=True, exist_ok=True)
    imp = booster.get_score(importance_type="gain")
    pd.DataFrame(
        [{"feature": k, "importance_gain": v} for k, v in imp.items()]
    ).sort_values("importance_gain", ascending=False).to_csv(
        mdata_dir / "xgb_l1_importance.csv", index=False
    )

    pred_frames = []
    pairs = store.pairs_df[["ORG_AIRPRT", "DES_AIRPRT"]].copy()
    for h in range(1, 25):
        t_tgt = store.t_cut + h
        ym_tgt = int(store.cal_df.loc[t_tgt, "YM"])
        X_te_df = store.build_features_for_slice(store.t_cut, h)
        anchor = X_te_df["DAMPED_ANCHOR"].to_numpy(dtype=np.float64)
        diff_hat = booster.predict(xgb.DMatrix(X_te_df[feat_cols])).astype(np.float64)
        # Guard extreme L1 residual swings relative to route scale
        max_shift = np.maximum(anchor * 2.0, 200.0)
        diff_hat = np.clip(diff_hat, -max_shift, max_shift)
        y_hat = np.clip(anchor + diff_hat, 0.0, None)

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
