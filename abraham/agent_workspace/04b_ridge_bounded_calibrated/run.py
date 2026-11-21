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

DROP_COLS = {
    "DAMPED_ANCHOR",
    "LOG_DAMPED_ANCHOR",
    "SEG_REC_NOW",
    "TROUGH_RATIO",
    "PAIR_REC_SHRUNK",
    "DIST_MI",
    "LOG_PAX_S",
    "LOG_PAX_R3",
    "LOG_PAX_R6",
    "LOG_PAX_R12",
    "LOG_PAX_SAME_M_LAST",
    "LOG_PAX_SAME_M_PREV2",
    "LOG_PAX_SAME_M_2019",
    "LOG_REV_R12",
    "LOG_REV_SAME_M_LAST",
    "LOG_ORG_R12",
    "LOG_DES_R12",
    "LOG_METRO_PAIR_R12",
}

CLIP_RATIOS = [
    "REC_19_R3",
    "REC_19_R12",
    "YOY_R3",
    "YOY_R12",
    "SEASONAL_IDX_19",
    "SEASONAL_IDX_LAST",
    "R3_TO_R12",
    "ORG_R3_TO_R12",
    "DES_R3_TO_R12",
    "ORG_REC_19_R3",
    "DES_REC_19_R3",
    "YIELD_YOY_R3",
]


def _prep_linear_features(df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    sub = df[feat_cols].copy()
    for c in CLIP_RATIOS:
        if c in sub.columns:
            sub[c] = np.clip(sub[c], 0.2, 3.0)
    return sub.to_numpy(dtype=np.float32)


def _fit_closed_form_ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, alpha_per_row: float = 0.25):
    """Memory-lean exact weighted Ridge via chunked normal equations."""
    medians = np.nanmedian(X, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
    mask = ~np.isfinite(X)
    if mask.any():
        X[mask] = np.take(medians, np.where(mask)[1])

    w_norm = (w / np.mean(w)).astype(np.float64)
    w_sum = float(np.sum(w_norm))
    means = (X.T @ w_norm) / w_sum
    # Compute weighted variance cleanly in chunks
    n, p = X.shape
    var = np.zeros(p, dtype=np.float64)
    XtWX = np.zeros((p, p), dtype=np.float64)
    XtWy = np.zeros(p, dtype=np.float64)
    y_mean = float(np.sum(w_norm * y) / w_sum)
    y_c = (y - y_mean).astype(np.float64)

    chunk = 250000
    for i in range(0, n, chunk):
        Xc = X[i:i + chunk].astype(np.float64) - means
        wc = w_norm[i:i + chunk]
        var += (Xc ** 2).T @ wc
        Xwc = Xc * wc[:, None]
        XtWX += Xwc.T @ Xc
        XtWy += Xwc.T @ y_c[i:i + chunk]

    stds = np.sqrt(np.maximum(var / w_sum, 1e-8))
    inv_s = 1.0 / stds
    A = XtWX * np.outer(inv_s, inv_s)
    b = XtWy * inv_s
    A[np.diag_indices(p)] += alpha_per_row * w_sum
    coef_std = np.linalg.solve(A, b)
    return medians, means, stds, y_mean, coef_std


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
        store, s_min=6, stride_s=2, exclude_acute_covid_targets=True, recency_halflife=18.0
    )
    feat_cols = [c for c in X_train.columns if c not in DROP_COLS]
    X_tr = _prep_linear_features(X_train, feat_cols)
    log_anc_tr = X_train["LOG_DAMPED_ANCHOR"].to_numpy(dtype=np.float64)
    y_res = np.clip(np.log1p(y_train) - log_anc_tr, -2.0, 2.0)
    w_raw = w_train * np.maximum(log_anc_tr, 0.5)
    del X_train, y_train, w_train
    gc.collect()

    medians, means, stds, y_mean, coef_std = _fit_closed_form_ridge(X_tr, y_res, w_raw, alpha_per_row=0.25)
    del X_tr, y_res, w_raw
    gc.collect()

    mdata_dir = model_dir / "model_specific_data" / fold.name
    mdata_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": feat_cols, "coef_std": coef_std}).sort_values(
        "coef_std", key=np.abs, ascending=False
    ).to_csv(mdata_dir / "ridge_bounded_coefficients.csv", index=False)

    pred_frames = []
    pairs = store.pairs_df[["ORG_AIRPRT", "DES_AIRPRT"]].copy()
    for h in range(1, 25):
        t_tgt = store.t_cut + h
        ym_tgt = int(store.cal_df.loc[t_tgt, "YM"])
        X_te_df = store.build_features_for_slice(store.t_cut, h)
        X_te = _prep_linear_features(X_te_df, feat_cols)
        nmask = ~np.isfinite(X_te)
        if nmask.any():
            X_te[nmask] = np.take(medians, np.where(nmask)[1])
        X_te_std = (X_te.astype(np.float64) - means) / stds
        delta_hat = np.clip(y_mean + X_te_std @ coef_std, -0.5, 0.5)
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
