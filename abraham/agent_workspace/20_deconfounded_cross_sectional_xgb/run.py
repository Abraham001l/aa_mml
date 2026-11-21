from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_eval.external_feature_store import ExternalFoldFeatureStore
from shared_eval.runner import run_backtest_suite
from shared_eval.splits import FoldSpec, assert_origin_cutoff

_spec16b = importlib.util.spec_from_file_location(
    "anchor16b",
    Path(__file__).resolve().parents[1] / "16b_refined_structural_panel" / "run.py",
)
_mod16b = importlib.util.module_from_spec(_spec16b)
_spec16b.loader.exec_module(_mod16b)
compute_anchor_v3 = _mod16b.compute_anchor_v3

_spec17 = importlib.util.spec_from_file_location(
    "deconf17",
    Path(__file__).resolve().parents[1] / "17_deconfounded_xgb_v2" / "run.py",
)
_mod17 = importlib.util.module_from_spec(_spec17)
_spec17.loader.exec_module(_mod17)
_clean_deconfounded_features = _mod17._clean_deconfounded_features


def predict_fold(
    fold: FoldSpec,
    hist_panel: pd.DataFrame,
    tgt_skeleton: pd.DataFrame,
    airport_df: pd.DataFrame,
    model_dir: Path,
) -> pd.DataFrame:
    """
    Macro-Deconfounded Cross-Sectional L1 XGBoost over refined anchor_v3:
    For each training slice (s, h) with s + h = t_tgt <= t_cut, removes segment-level
    macro COVID shock scale factor (sum_seg y(t_tgt) / sum_seg anchor(s, h)) from
    the training target so trees learn PURELY cross-sectional, calendar, yield, and
    external Census/FAA/Migration route allocation drivers with zero COVID bias!
    """
    assert_origin_cutoff(int(hist_panel["YM"].max()), fold.origin_ym)

    store = ExternalFoldFeatureStore(hist_panel, airport_df, include_macro=False)
    t_cut = store.t_cut
    dom_mask = store.is_dom > 0.5

    frames, ys_res, ws = [], [], []
    for s in range(6, t_cut, 2):
        for h in range(1, min(24, t_cut - s) + 1):
            t_tgt = s + h
            if 14 <= t_tgt <= 25:  # Exclude acute lockdown months
                continue
            df_s = store.build_features_for_slice(s, h)
            anc = compute_anchor_v3(store, s, h)
            df_clean = _clean_deconfounded_features(df_s, anc)

            y_raw = store.P[:, t_tgt].astype(np.float64)
            # Zero-macro-bias segment centering strictly inside t_tgt <= t_cut
            anc_adj = anc.copy()
            for mask_seg in (dom_mask, ~dom_mask):
                s_anc = float(np.sum(anc[mask_seg]))
                s_y = float(np.sum(y_raw[mask_seg]))
                if s_anc > 100.0 and s_y > 100.0:
                    ratio = np.clip(s_y / s_anc, 0.35, 1.65)
                    anc_adj[mask_seg] = anc[mask_seg] * ratio

            y_diff = (y_raw - anc_adj).astype(np.float32)
            shock_w = 0.25 if 26 <= t_tgt <= 28 else 1.0
            rec_w = float(np.exp(-(t_cut - t_tgt) / 18.0)) * shock_w

            frames.append(df_clean)
            ys_res.append(y_diff)
            ws.append(np.full(store.n_pairs, rec_w, dtype=np.float32))

    X_tr = pd.concat(frames, ignore_index=True)
    y_tr = np.concatenate(ys_res)
    w_tr = np.concatenate(ws)
    del frames, ys_res, ws
    gc.collect()

    feat_cols = list(X_tr.columns)
    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
    del X_tr, y_tr, w_tr
    gc.collect()

    params = {
        "objective": "reg:absoluteerror",
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 120.0,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_alpha": 10.0,
        "reg_lambda": 60.0,
        "tree_method": "hist",
        "nthread": -1,
        "seed": 42,
    }
    booster = xgb.train(params, dtrain, num_boost_round=420)
    del dtrain
    gc.collect()

    mdata_dir = model_dir / "model_specific_data" / fold.name
    mdata_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"feature": k, "importance_gain": v} for k, v in booster.get_score(importance_type="gain").items()]).sort_values(
        "importance_gain", ascending=False
    ).to_csv(mdata_dir / "cross_sectional_xgb_importance.csv", index=False)

    pred_frames = []
    pairs = store.pairs_df[["ORG_AIRPRT", "DES_AIRPRT"]].copy()
    for h in range(1, 25):
        t_tgt = t_cut + h
        ym_tgt = int(store.cal_df.loc[t_tgt, "YM"])
        X_te_raw = store.build_features_for_slice(t_cut, h)
        anc_te = compute_anchor_v3(store, t_cut, h)
        X_te_df = _clean_deconfounded_features(X_te_raw, anc_te)

        diff_hat = booster.predict(xgb.DMatrix(X_te_df[feat_cols])).astype(np.float64)
        # Center any residual global drift within each segment so aggregate stays anchored
        for mask_seg in (dom_mask, ~dom_mask):
            diff_hat[mask_seg] -= np.mean(diff_hat[mask_seg]) * 0.5
        max_shift = np.maximum(anc_te * 0.35, 120.0)
        y_hat = np.clip(anc_te + np.clip(diff_hat, -max_shift, max_shift), 0.0, None)

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
