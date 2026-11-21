from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_eval.external_feature_store import ExternalFoldFeatureStore
from shared_eval.runner import run_backtest_suite
from shared_eval.splits import FoldSpec, assert_origin_cutoff

_spec17 = importlib.util.spec_from_file_location(
    "deconf17",
    Path(__file__).resolve().parents[1] / "17_deconfounded_xgb_v2" / "run.py",
)
_mod17 = importlib.util.module_from_spec(_spec17)
_spec17.loader.exec_module(_mod17)
compute_structural_anchor_for_slice = _mod17.compute_structural_anchor_for_slice
_clean_deconfounded_features = _mod17._clean_deconfounded_features


def predict_fold(
    fold: FoldSpec,
    hist_panel: pd.DataFrame,
    tgt_skeleton: pd.DataFrame,
    airport_df: pd.DataFrame,
    model_dir: Path,
) -> pd.DataFrame:
    """
    Deconfounded LightGBM-style Histogram GBDT (HistGradientBoostingRegressor, absolute_error)
    predicting bounded log-ratio residual over structural anchor_v2.
    """
    assert_origin_cutoff(int(hist_panel["YM"].max()), fold.origin_ym)

    store = ExternalFoldFeatureStore(hist_panel, airport_df, include_macro=False)
    t_cut = store.t_cut

    frames, ys, ws, ancs = [], [], [], []
    for s in range(6, t_cut, 2):
        for h in range(1, min(24, t_cut - s) + 1):
            t_tgt = s + h
            if 14 <= t_tgt <= 25:
                continue
            df_s = store.build_features_for_slice(s, h)
            anc = compute_structural_anchor_for_slice(store, airport_df, s, h)
            df_clean = _clean_deconfounded_features(df_s, anc)
            shock_w = 0.20 if 26 <= t_tgt <= 28 else 1.0
            rec_w = float(np.exp(-(t_cut - t_tgt) / 18.0)) * shock_w

            frames.append(df_clean)
            ys.append(store.P[:, t_tgt].astype(np.float32))
            ws.append(np.full(store.n_pairs, rec_w, dtype=np.float32))
            ancs.append(anc.astype(np.float32))

    X_tr = pd.concat(frames, ignore_index=True)
    y_tr = np.concatenate(ys)
    w_tr = np.concatenate(ws)
    anc_tr = np.concatenate(ancs)
    del frames, ys, ws, ancs
    gc.collect()

    feat_cols = list(X_tr.columns)
    y_ratio_log = np.clip(np.log((y_tr + 10.0) / (anc_tr + 10.0)), -1.5, 1.5)
    # Weight gently by log1p(anchor) so large/medium routes drive WAPE cleanly
    w_fit = w_tr * np.sqrt(np.maximum(np.log1p(anc_tr), 0.5))

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.04,
        max_iter=350,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        l2_regularization=40.0,
        max_bins=255,
        random_state=42,
    )
    model.fit(X_tr[feat_cols], y_ratio_log, sample_weight=w_fit)
    del X_tr, y_tr, w_tr, anc_tr, y_ratio_log, w_fit
    gc.collect()

    mdata_dir = model_dir / "model_specific_data" / fold.name
    mdata_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"n_features": [len(feat_cols)], "loss": ["absolute_error"]}).to_csv(
        mdata_dir / "histgb_v2_meta.csv", index=False
    )

    pred_frames = []
    pairs = store.pairs_df[["ORG_AIRPRT", "DES_AIRPRT"]].copy()
    for h in range(1, 25):
        t_tgt = t_cut + h
        ym_tgt = int(store.cal_df.loc[t_tgt, "YM"])
        X_te_raw = store.build_features_for_slice(t_cut, h)
        anc_te = compute_structural_anchor_for_slice(store, airport_df, t_cut, h)
        X_te_df = _clean_deconfounded_features(X_te_raw, anc_te)

        delta_hat = np.clip(model.predict(X_te_df[feat_cols]), -0.35, 0.35)
        y_hat = np.clip((anc_te + 10.0) * np.exp(delta_hat) - 10.0, 0.0, None)

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
