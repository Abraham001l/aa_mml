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
    Natural-scale Tweedie XGBoost (variance_power=1.4) using base_margin = log(max(anchor, 1.0)).
    Directly models multiplicative adjustment exp(f(x)) on original OD_PAX scale with zero
    log-retransformation (Jensen's inequality) bias and native zero/intermittent support.
    """
    assert_origin_cutoff(int(hist_panel["YM"].max()), fold.origin_ym)

    store = ExternalFoldFeatureStore(hist_panel, airport_df, include_macro=False)
    X_train, y_train, w_train, _ = assemble_training_dataset(
        store, s_min=6, stride_s=3, exclude_acute_covid_targets=True, recency_halflife=18.0
    )
    feat_cols = [c for c in X_train.columns if c != "DAMPED_ANCHOR"]
    bm_train = np.log(np.maximum(X_train["DAMPED_ANCHOR"].to_numpy(dtype=np.float64), 1.0)).astype(np.float32)

    dtrain = xgb.DMatrix(
        X_train[feat_cols],
        label=np.clip(y_train, 0.0, None),
        weight=w_train,
        base_margin=bm_train,
    )
    del X_train, y_train, w_train, bm_train
    gc.collect()

    params = {
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
    booster = xgb.train(params, dtrain, num_boost_round=500)
    del dtrain
    gc.collect()

    mdata_dir = model_dir / "model_specific_data" / fold.name
    mdata_dir.mkdir(parents=True, exist_ok=True)
    imp = booster.get_score(importance_type="gain")
    pd.DataFrame(
        [{"feature": k, "importance_gain": v} for k, v in imp.items()]
    ).sort_values("importance_gain", ascending=False).to_csv(
        mdata_dir / "xgb_tweedie_importance.csv", index=False
    )

    pred_frames = []
    pairs = store.pairs_df[["ORG_AIRPRT", "DES_AIRPRT"]].copy()
    for h in range(1, 25):
        t_tgt = store.t_cut + h
        ym_tgt = int(store.cal_df.loc[t_tgt, "YM"])
        X_te_df = store.build_features_for_slice(store.t_cut, h)
        anchor = X_te_df["DAMPED_ANCHOR"].to_numpy(dtype=np.float64)
        bm_te = np.log(np.maximum(anchor, 1.0)).astype(np.float32)
        dtest = xgb.DMatrix(X_te_df[feat_cols], base_margin=bm_te)
        y_hat = booster.predict(dtest)
        cap = np.maximum(anchor * 5.0 + 500.0, 200000.0)
        y_hat = np.clip(y_hat, 0.0, cap)

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
