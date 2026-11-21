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

_spec16 = importlib.util.spec_from_file_location(
    "anchor16",
    Path(__file__).resolve().parents[1] / "16_structural_hierarchical_anchor" / "run.py",
)
_mod16 = importlib.util.module_from_spec(_spec16)
_spec16.loader.exec_module(_mod16)
compute_structural_anchor_for_slice = _mod16.compute_structural_anchor_for_slice

# Drop global time-regime variables that cause macro extrapolation across folds
DROP_CONFOUNDED = {
    "DAMPED_ANCHOR",
    "LOG_DAMPED_ANCHOR",
    "SEG_REC_NOW",
    "TROUGH_RATIO",
    "PAIR_REC_SHRUNK",
}

CLIP_COLS = [
    "REC_19_R3", "REC_19_R12", "YOY_R3", "YOY_R12",
    "SEASONAL_IDX_19", "SEASONAL_IDX_LAST", "R3_TO_R12",
    "ORG_R3_TO_R12", "DES_R3_TO_R12", "ORG_REC_19_R3", "DES_REC_19_R3", "YIELD_YOY_R3",
]


def _clean_deconfounded_features(df: pd.DataFrame, anchor_v2: np.ndarray) -> pd.DataFrame:
    out = df[[c for c in df.columns if c not in DROP_CONFOUNDED]].copy()
    out["LOG_ANCHOR_V2"] = np.log1p(anchor_v2).astype(np.float32)
    for c in CLIP_COLS:
        if c in out.columns:
            out[c] = np.clip(out[c], 0.35, 2.25).astype(np.float32)
    # Add cross-sectional relative recovery (pair vs its own origin & dest airports -> regime invariant!)
    if "REC_19_R3" in out.columns and "ORG_REC_19_R3" in out.columns:
        out["REL_REC_VS_ORG"] = np.clip(out["REC_19_R3"] / np.maximum(out["ORG_REC_19_R3"], 0.2), 0.3, 2.5).astype(np.float32)
        out["REL_REC_VS_DES"] = np.clip(out["REC_19_R3"] / np.maximum(out["DES_REC_19_R3"], 0.2), 0.3, 2.5).astype(np.float32)
    return out


def predict_fold(
    fold: FoldSpec,
    hist_panel: pd.DataFrame,
    tgt_skeleton: pd.DataFrame,
    airport_df: pd.DataFrame,
    model_dir: Path,
) -> pd.DataFrame:
    """
    Deconfounded Tweedie-XGBoost v2 over hierarchical structural anchor:
    - Uses anchor_v2 as base_margin (log scale)
    - Removes global macro-regime scalars (SEG_REC_NOW, TROUGH_RATIO) & clips YoY spikes
    - Adds cross-sectional relative recovery & Census/FAA/Migration features
    - Bounds learned residual adjustment to [-0.38, +0.38] trust region around anchor_v2.
    """
    assert_origin_cutoff(int(hist_panel["YM"].max()), fold.origin_ym)

    store = ExternalFoldFeatureStore(hist_panel, airport_df, include_macro=False)
    t_cut = store.t_cut

    frames, ys, ws, bms = [], [], [], []
    for s in range(6, t_cut, 2):
        for h in range(1, min(24, t_cut - s) + 1):
            t_tgt = s + h
            if 14 <= t_tgt <= 25:  # Exclude acute pandemic lockdown targets
                continue
            df_s = store.build_features_for_slice(s, h)
            anc = compute_structural_anchor_for_slice(store, airport_df, s, h)
            df_clean = _clean_deconfounded_features(df_s, anc)
            # Downweight early transition months (2021-03..2021-05: T_IDX 26..28) gently
            shock_w = 0.20 if 26 <= t_tgt <= 28 else 1.0
            rec_w = float(np.exp(-(t_cut - t_tgt) / 18.0)) * shock_w

            frames.append(df_clean)
            ys.append(store.P[:, t_tgt].astype(np.float32))
            ws.append(np.full(store.n_pairs, rec_w, dtype=np.float32))
            bms.append(np.log(np.maximum(anc, 1.0)).astype(np.float32))

    X_tr = pd.concat(frames, ignore_index=True)
    y_tr = np.concatenate(ys)
    w_tr = np.concatenate(ws)
    bm_tr = np.concatenate(bms)
    del frames, ys, ws, bms
    gc.collect()

    feat_cols = list(X_tr.columns)
    dtrain = xgb.DMatrix(X_tr, label=np.clip(y_tr, 0.0, None), weight=w_tr, base_margin=bm_tr)
    del X_tr, y_tr, w_tr, bm_tr
    gc.collect()

    params = {
        "objective": "reg:tweedie",
        "tweedie_variance_power": 1.4,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 150.0,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_alpha": 5.0,
        "reg_lambda": 60.0,
        "tree_method": "hist",
        "nthread": -1,
        "seed": 42,
    }
    booster = xgb.train(params, dtrain, num_boost_round=350)
    del dtrain
    gc.collect()

    mdata_dir = model_dir / "model_specific_data" / fold.name
    mdata_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"feature": k, "importance_gain": v} for k, v in booster.get_score(importance_type="gain").items()]).sort_values(
        "importance_gain", ascending=False
    ).to_csv(mdata_dir / "deconfounded_xgb_v2_importance.csv", index=False)

    pred_frames = []
    pairs = store.pairs_df[["ORG_AIRPRT", "DES_AIRPRT"]].copy()
    for h in range(1, 25):
        t_tgt = t_cut + h
        ym_tgt = int(store.cal_df.loc[t_tgt, "YM"])
        X_te_raw = store.build_features_for_slice(t_cut, h)
        anc_te = compute_structural_anchor_for_slice(store, airport_df, t_cut, h)
        X_te_df = _clean_deconfounded_features(X_te_raw, anc_te)
        bm_te = np.log(np.maximum(anc_te, 1.0)).astype(np.float32)

        raw_pred = booster.predict(xgb.DMatrix(X_te_df[feat_cols], base_margin=bm_te)).astype(np.float64)
        # Trust-region clamp relative to structural anchor_v2: [0.68x, 1.46x]
        low_b = anc_te * np.exp(-0.38)
        high_b = np.maximum(anc_te * np.exp(0.38), 10.0)
        y_hat = np.clip(raw_pred, low_b, high_b)

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
