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
    Two-stage Hurdle + Shock-Conditioned XGBoost:
    - Retains 2020-03..2021-02 training targets conditioned on IS_ACUTE_SHOCK=1 (set to 0 at test time),
      so route seasonality & zero-intermittency are learned across the full pre-cutoff span.
    - Stage 1: Classifier P(OD_PAX > 0)
    - Stage 2: Conditional volume regressor on y > 0 using Tweedie(power=1.35) + base_margin=log(anchor).
    """
    assert_origin_cutoff(int(hist_panel["YM"].max()), fold.origin_ym)

    store = ExternalFoldFeatureStore(hist_panel, airport_df, include_macro=False)
    X_train, y_train, w_train, t_tgts = assemble_training_dataset(
        store, s_min=6, stride_s=3, exclude_acute_covid_targets=False, recency_halflife=18.0
    )
    is_shock = ((t_tgts >= 14) & (t_tgts <= 25)).astype(np.float32)
    X_train["IS_ACUTE_SHOCK_REGIME"] = is_shock
    feat_cols = [c for c in X_train.columns if c != "DAMPED_ANCHOR"]

    # Stage 1: Active-route hurdle probability P(y > 0)
    y_active = (y_train > 0.0).astype(np.float32)
    # Give shock-period zeros mild weight (already 0.05x in assemble_training_dataset)
    clf = xgb.XGBClassifier(
        n_estimators=220,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=50.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train[feat_cols], y_active, sample_weight=w_train)

    # Stage 2: Conditional positive volume Tweedie with base_margin
    pos_mask = y_train > 0.0
    X_pos = X_train.loc[pos_mask, feat_cols]
    y_pos = y_train[pos_mask]
    w_pos = w_train[pos_mask]
    bm_pos = np.log(np.maximum(X_train.loc[pos_mask, "DAMPED_ANCHOR"].to_numpy(dtype=np.float64), 1.0)).astype(np.float32)
    del X_train, y_train, w_train
    gc.collect()

    dtrain = xgb.DMatrix(X_pos, label=y_pos, weight=w_pos, base_margin=bm_pos)
    del X_pos, y_pos, w_pos, bm_pos
    gc.collect()

    params = {
        "objective": "reg:tweedie",
        "tweedie_variance_power": 1.35,
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
    pd.DataFrame([{"feature": k, "gain": v} for k, v in booster.get_score(importance_type="gain").items()]).sort_values(
        "gain", ascending=False
    ).to_csv(mdata_dir / "hurdle_stage2_importance.csv", index=False)

    pred_frames = []
    pairs = store.pairs_df[["ORG_AIRPRT", "DES_AIRPRT"]].copy()
    for h in range(1, 25):
        t_tgt = store.t_cut + h
        ym_tgt = int(store.cal_df.loc[t_tgt, "YM"])
        X_te_df = store.build_features_for_slice(store.t_cut, h)
        X_te_df["IS_ACUTE_SHOCK_REGIME"] = np.float32(0.0)
        p_active = clf.predict_proba(X_te_df[feat_cols])[:, 1]
        anchor = X_te_df["DAMPED_ANCHOR"].to_numpy(dtype=np.float64)
        bm_te = np.log(np.maximum(anchor, 1.0)).astype(np.float32)
        vol_hat = booster.predict(xgb.DMatrix(X_te_df[feat_cols], base_margin=bm_te))
        y_hat = np.clip(p_active * vol_hat, 0.0, np.maximum(anchor * 5.0 + 500.0, 200000.0))

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
