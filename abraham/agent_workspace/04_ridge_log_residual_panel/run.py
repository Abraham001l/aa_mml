from __future__ import annotations

import gc
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_eval.feature_engine import FoldFeatureStore, assemble_training_dataset
from shared_eval.runner import run_backtest_suite
from shared_eval.splits import FoldSpec, assert_origin_cutoff


def _fit_exact_weighted_ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float = 500.0):
    """Exact weighted Ridge matching sklearn Pipeline([SimpleImputer, StandardScaler, Ridge(alpha=500)])."""
    medians = np.nanmedian(X, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
    mask = ~np.isfinite(X)
    if mask.any():
        X[mask] = np.take(medians, np.where(mask)[1])

    # Unweighted StandardScaler mean/std matching sklearn StandardScaler
    n, p = X.shape
    means = np.mean(X, axis=0, dtype=np.float64)
    stds = np.sqrt(np.maximum(np.var(X, axis=0, dtype=np.float64), 1e-12))

    w64 = w.astype(np.float64)
    w_sum = float(np.sum(w64))
    y_mean = float(np.sum(w64 * y) / w_sum)
    y_c = (y - y_mean).astype(np.float64)

    XtWX = np.zeros((p, p), dtype=np.float64)
    XtWy = np.zeros(p, dtype=np.float64)
    # Also handle centered weighted intercept properly via weighted feature mean
    w_means = np.zeros(p, dtype=np.float64)
    chunk = 250000
    for i in range(0, n, chunk):
        Xc = (X[i:i + chunk].astype(np.float64) - means) / stds
        w_means += (Xc * w64[i:i + chunk, None]).sum(axis=0)
    w_means /= w_sum

    for i in range(0, n, chunk):
        Xc = (X[i:i + chunk].astype(np.float64) - means) / stds - w_means
        Xwc = Xc * w64[i:i + chunk, None]
        XtWX += Xwc.T @ Xc
        XtWy += Xwc.T @ y_c[i:i + chunk]

    XtWX[np.diag_indices(p)] += alpha
    coef_std = np.linalg.solve(XtWX, XtWy)
    intercept = y_mean - float(w_means @ coef_std)
    return medians, means, stds, intercept, coef_std


def predict_fold(
    fold: FoldSpec,
    hist_panel: pd.DataFrame,
    tgt_skeleton: pd.DataFrame,
    airport_df: pd.DataFrame,
    model_dir: Path,
) -> pd.DataFrame:
    assert_origin_cutoff(int(hist_panel["YM"].max()), fold.origin_ym)

    store = FoldFeatureStore(hist_panel, airport_df)
    X_train, y_train, w_train, _ = assemble_training_dataset(
        store,
        s_min=6,
        stride_s=2,
        exclude_acute_covid_targets=True,
        recency_halflife=18.0,
    )
    feat_cols = [c for c in X_train.columns if c != "DAMPED_ANCHOR"]
    X_tr = X_train[feat_cols].to_numpy(dtype=np.float32)
    log_anc = X_train["LOG_DAMPED_ANCHOR"].to_numpy(dtype=np.float64)
    y_res = np.log1p(y_train) - log_anc
    w_scaled = w_train * np.sqrt(np.maximum(log_anc, 0.25))
    del X_train, y_train, w_train
    gc.collect()

    medians, means, stds, intercept, coef_std = _fit_exact_weighted_ridge(X_tr, y_res, w_scaled, alpha=500.0)
    del X_tr, y_res, w_scaled
    gc.collect()

    mdata_dir = model_dir / "model_specific_data" / fold.name
    mdata_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": feat_cols, "coef_std": coef_std}).sort_values(
        "coef_std", key=np.abs, ascending=False
    ).to_csv(mdata_dir / "ridge_coefficients.csv", index=False)

    pred_frames = []
    pairs = store.pairs_df[["ORG_AIRPRT", "DES_AIRPRT"]].copy()
    for h in range(1, 25):
        t_tgt = store.t_cut + h
        ym_tgt = int(store.cal_df.loc[t_tgt, "YM"])
        X_te_df = store.build_features_for_slice(store.t_cut, h)
        X_te = X_te_df[feat_cols].to_numpy(dtype=np.float32)
        nmask = ~np.isfinite(X_te)
        if nmask.any():
            X_te[nmask] = np.take(medians, np.where(nmask)[1])
        X_te_std = (X_te.astype(np.float64) - means) / stds
        delta_hat = np.clip(intercept + X_te_std @ coef_std, -2.5, 2.5)
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
