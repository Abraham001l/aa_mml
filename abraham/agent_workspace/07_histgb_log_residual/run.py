from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_eval.feature_engine import FoldFeatureStore, assemble_training_dataset
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
    LightGBM-algorithm Histogram GBDT (sklearn HistGradientBoostingRegressor)
    with L1 / absolute_error objective predicting log-residual over damped anchor.
    """
    assert_origin_cutoff(int(hist_panel["YM"].max()), fold.origin_ym)

    store = FoldFeatureStore(hist_panel, airport_df)
    X_train, y_train, w_train, _ = assemble_training_dataset(
        store,
        s_min=6,
        stride_s=3,
        exclude_acute_covid_targets=True,
        recency_halflife=18.0,
    )
    feat_cols = [c for c in X_train.columns if c != "DAMPED_ANCHOR"]
    y_res = np.log1p(y_train) - X_train["LOG_DAMPED_ANCHOR"].to_numpy(dtype=np.float64)
    w_fit = w_train * np.maximum(X_train["LOG_DAMPED_ANCHOR"].to_numpy(dtype=np.float64), 0.5)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=450,
        max_leaf_nodes=63,
        min_samples_leaf=100,
        l2_regularization=10.0,
        max_bins=255,
        early_stopping=False,
        random_state=42,
    )
    model.fit(X_train[feat_cols], y_res, sample_weight=w_fit)

    mdata_dir = model_dir / "model_specific_data" / fold.name
    mdata_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"n_train_rows": [len(X_train)], "n_features": [len(feat_cols)]}).to_csv(
        mdata_dir / "train_summary.csv", index=False
    )

    pred_frames = []
    pairs = store.pairs_df[["ORG_AIRPRT", "DES_AIRPRT"]].copy()
    for h in range(1, 25):
        t_tgt = store.t_cut + h
        ym_tgt = int(store.cal_df.loc[t_tgt, "YM"])
        X_te_df = store.build_features_for_slice(store.t_cut, h)
        delta_hat = model.predict(X_te_df[feat_cols])
        delta_hat = np.clip(delta_hat, -2.5, 2.5)
        log_pred = X_te_df["LOG_DAMPED_ANCHOR"].to_numpy(dtype=np.float64) + delta_hat
        y_hat = np.clip(np.expm1(log_pred), 0.0, None)

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
