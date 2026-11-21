from __future__ import annotations

import gc
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import TweedieRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_eval.feature_engine import FoldFeatureStore, assemble_training_dataset
from shared_eval.runner import run_backtest_suite
from shared_eval.splits import FoldSpec, assert_origin_cutoff

DROP_FOR_GLM = {"DAMPED_ANCHOR", "SEG_REC_NOW", "DIST_MI"}
CLIP_RATIOS = [
    "REC_19_R3", "REC_19_R12", "YOY_R3", "YOY_R12", "SEASONAL_IDX_19",
    "SEASONAL_IDX_LAST", "R3_TO_R12", "ORG_R3_TO_R12", "DES_R3_TO_R12",
    "ORG_REC_19_R3", "DES_REC_19_R3", "YIELD_YOY_R3",
]


def _prep_matrix(df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    sub = df[feat_cols].copy()
    for c in CLIP_RATIOS:
        if c in sub.columns:
            sub[c] = np.clip(sub[c], 0.0, 5.0)
    return sub.to_numpy(dtype=np.float32)


def predict_fold(
    fold: FoldSpec,
    hist_panel: pd.DataFrame,
    tgt_skeleton: pd.DataFrame,
    airport_df: pd.DataFrame,
    model_dir: Path,
) -> pd.DataFrame:
    """Memory-lean Tweedie GLM (log-link, power=1.5) predicting natural-scale OD_PAX."""
    assert_origin_cutoff(int(hist_panel["YM"].max()), fold.origin_ym)

    store = FoldFeatureStore(hist_panel, airport_df)
    X_train, y_train, w_train, _ = assemble_training_dataset(
        store, s_min=6, stride_s=4, exclude_acute_covid_targets=True, recency_halflife=18.0
    )
    feat_cols = [c for c in X_train.columns if c not in DROP_FOR_GLM]
    X_tr = _prep_matrix(X_train, feat_cols)
    del X_train
    gc.collect()

    medians = np.nanmedian(X_tr, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
    mask = ~np.isfinite(X_tr)
    if mask.any():
        X_tr[mask] = np.take(medians, np.where(mask)[1])
    means = np.mean(X_tr, axis=0, dtype=np.float64).astype(np.float32)
    stds = np.sqrt(np.maximum(np.var(X_tr, axis=0, dtype=np.float64), 1e-8)).astype(np.float32)
    X_tr -= means
    X_tr /= stds

    w_norm = w_train / np.mean(w_train)
    glm = TweedieRegressor(power=1.5, alpha=0.10, link="log", max_iter=200, tol=1e-4)
    glm.fit(X_tr, np.clip(y_train, 0.0, None), sample_weight=w_norm)
    del X_tr, y_train, w_train
    gc.collect()

    mdata_dir = model_dir / "model_specific_data" / fold.name
    mdata_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": feat_cols, "coef_std": glm.coef_}).sort_values(
        "coef_std", key=np.abs, ascending=False
    ).to_csv(mdata_dir / "tweedie_coefficients.csv", index=False)

    pred_frames = []
    pairs = store.pairs_df[["ORG_AIRPRT", "DES_AIRPRT"]].copy()
    for h in range(1, 25):
        t_tgt = store.t_cut + h
        ym_tgt = int(store.cal_df.loc[t_tgt, "YM"])
        X_te_raw = store.build_features_for_slice(store.t_cut, h)
        X_te = _prep_matrix(X_te_raw, feat_cols)
        nmask = ~np.isfinite(X_te)
        if nmask.any():
            X_te[nmask] = np.take(medians, np.where(nmask)[1])
        X_te -= means
        X_te /= stds

        y_hat = glm.predict(X_te)
        anchor = X_te_raw["DAMPED_ANCHOR"].to_numpy(dtype=np.float64)
        cap = np.maximum(anchor * 4.0 + 500.0, 180000.0)
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
