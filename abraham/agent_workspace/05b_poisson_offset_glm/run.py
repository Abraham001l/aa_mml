from __future__ import annotations

import gc
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_eval.feature_engine import FoldFeatureStore, assemble_training_dataset
from shared_eval.runner import run_backtest_suite
from shared_eval.splits import FoldSpec, assert_origin_cutoff

DROP_COLS = {
    "DAMPED_ANCHOR", "LOG_DAMPED_ANCHOR", "SEG_REC_NOW", "TROUGH_RATIO", "PAIR_REC_SHRUNK",
    "DIST_MI", "LOG_PAX_S", "LOG_PAX_R3", "LOG_PAX_R6", "LOG_PAX_R12", "LOG_PAX_SAME_M_LAST",
    "LOG_PAX_SAME_M_PREV2", "LOG_PAX_SAME_M_2019", "LOG_REV_R12", "LOG_REV_SAME_M_LAST",
    "LOG_ORG_R12", "LOG_DES_R12", "LOG_METRO_PAIR_R12",
}
CLIP_RATIOS = [
    "REC_19_R3", "REC_19_R12", "YOY_R3", "YOY_R12", "SEASONAL_IDX_19",
    "SEASONAL_IDX_LAST", "R3_TO_R12", "ORG_R3_TO_R12", "DES_R3_TO_R12",
    "ORG_REC_19_R3", "DES_REC_19_R3", "YIELD_YOY_R3",
]


def _prep(df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    sub = df[feat_cols].copy()
    for c in CLIP_RATIOS:
        if c in sub.columns:
            sub[c] = np.clip(sub[c], 0.2, 3.0)
    return sub.to_numpy(dtype=np.float32)


def predict_fold(
    fold: FoldSpec,
    hist_panel: pd.DataFrame,
    tgt_skeleton: pd.DataFrame,
    airport_df: pd.DataFrame,
    model_dir: Path,
) -> pd.DataFrame:
    """
    Convex L2-regularized Poisson Offset GLM: E[y] = anchor * exp(b0 + X @ w).
    Fits directly on passenger counts (exact WAPE-scale optimization, zero Jensen's bias).
    """
    assert_origin_cutoff(int(hist_panel["YM"].max()), fold.origin_ym)

    store = FoldFeatureStore(hist_panel, airport_df)
    X_train, y_train, w_train, _ = assemble_training_dataset(
        store, s_min=6, stride_s=3, exclude_acute_covid_targets=True, recency_halflife=18.0
    )
    feat_cols = [c for c in X_train.columns if c not in DROP_COLS]
    X_tr = _prep(X_train, feat_cols)
    anc_tr = np.maximum(X_train["DAMPED_ANCHOR"].to_numpy(dtype=np.float64), 1.0)
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

    y64 = np.clip(y_train, 0.0, None).astype(np.float64)
    w64 = (w_train / np.mean(w_train)).astype(np.float64)
    scale_denom = float(np.sum(w64 * anc_tr)) + 1.0
    reg_l2 = 0.02

    def obj_and_grad(theta: np.ndarray):
        b0 = theta[0]
        coef = theta[1:]
        eta = np.clip(b0 + X_tr @ coef, -3.0, 3.0)
        mu = anc_tr * np.exp(eta)
        diff = w64 * (mu - y64)
        loss = float(np.sum(w64 * (mu - y64 * eta)) / scale_denom + 0.5 * reg_l2 * np.sum(coef ** 2))
        grad_b0 = float(np.sum(diff) / scale_denom)
        grad_c = (X_tr.T @ diff) / scale_denom + reg_l2 * coef
        return loss, np.concatenate([[grad_b0], grad_c])

    theta0 = np.zeros(X_tr.shape[1] + 1, dtype=np.float64)
    res = minimize(obj_and_grad, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 60})
    b0_hat = float(res.x[0])
    coef_hat = res.x[1:]
    del X_tr, y_train, w_train, y64, w64, anc_tr
    gc.collect()

    mdata_dir = model_dir / "model_specific_data" / fold.name
    mdata_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": feat_cols, "coef_std": coef_hat}).sort_values(
        "coef_std", key=np.abs, ascending=False
    ).to_csv(mdata_dir / "poisson_offset_coefs.csv", index=False)

    pred_frames = []
    pairs = store.pairs_df[["ORG_AIRPRT", "DES_AIRPRT"]].copy()
    for h in range(1, 25):
        t_tgt = store.t_cut + h
        ym_tgt = int(store.cal_df.loc[t_tgt, "YM"])
        X_te_raw = store.build_features_for_slice(store.t_cut, h)
        X_te = _prep(X_te_raw, feat_cols)
        nmask = ~np.isfinite(X_te)
        if nmask.any():
            X_te[nmask] = np.take(medians, np.where(nmask)[1])
        X_te -= means
        X_te /= stds
        anc_te = X_te_raw["DAMPED_ANCHOR"].to_numpy(dtype=np.float64)
        eta = np.clip(b0_hat + X_te @ coef_hat, -1.0, 1.0)
        y_hat = np.clip(np.maximum(anc_te, 0.5) * np.exp(eta), 0.0, np.maximum(anc_te * 3.0 + 500.0, 180000.0))

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
