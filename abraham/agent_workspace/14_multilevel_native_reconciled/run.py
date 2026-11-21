from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_eval.external_feature_store import ExternalFoldFeatureStore
from shared_eval.feature_engine import _rolling_sum_2d, assemble_training_dataset
from shared_eval.runner import run_backtest_suite
from shared_eval.splits import FoldSpec, assert_origin_cutoff


def _fit_and_predict_coarse_level(
    P: np.ndarray,
    R: np.ndarray,
    is_dom: np.ndarray,
    t_cut: int,
    cal_df: pd.DataFrame,
    n_rounds: int = 240,
    max_depth: int = 4,
) -> np.ndarray:
    """
    Train & forecast a native deconfounded L1-residual XGBoost directly on an aggregated level (N_groups, t_cut+1).
    Returns predictions of shape (N_groups, 24) for h = 1..24.
    """
    n_grp, n_t = P.shape
    s3, c3 = _rolling_sum_2d(P, 3)
    s6, c6 = _rolling_sum_2d(P, 6)
    s12, c12 = _rolling_sum_2d(P, 12)
    r3, _ = _rolling_sum_2d(R, 3)
    r12, _ = _rolling_sum_2d(R, 12)

    pax_r3 = s3 / c3
    pax_r6 = s6 / c6
    pax_r12 = s12 / c12
    yield_r3 = r3 / np.maximum(s3, 1.0)
    yield_r12 = r12 / np.maximum(s12, 1.0)

    pax_r3_lag12 = np.full_like(pax_r3, np.nan)
    pax_r12_lag12 = np.full_like(pax_r12, np.nan)
    if n_t > 12:
        pax_r3_lag12[:, 12:] = pax_r3[:, :-12]
        pax_r12_lag12[:, 12:] = pax_r12[:, :-12]

    pax_19_mean = np.zeros_like(P)
    cs19 = np.cumsum(P[:, : min(12, n_t)], axis=1)
    for t in range(n_t):
        k = min(t, 11)
        pax_19_mean[:, t] = cs19[:, k] / float(k + 1)

    pax_19_same3 = np.full_like(P, np.nan)
    for t in range(12, n_t):
        ms = [((t - k) % 12) for k in (0, 1, 2)]
        pax_19_same3[:, t] = np.mean(P[:, ms], axis=1)

    dom_mask = is_dom > 0.5
    seg_p = np.vstack([
        P[~dom_mask].sum(axis=0) if (~dom_mask).any() else np.zeros(n_t),
        P[dom_mask].sum(axis=0) if dom_mask.any() else np.zeros(n_t),
    ])
    seg_r3, seg_c3 = _rolling_sum_2d(seg_p, 3)
    seg_r3_mean = seg_r3 / seg_c3
    seg_19_same3 = np.full((2, n_t), np.nan)
    for t in range(12, n_t):
        ms = [((t - k) % 12) for k in (0, 1, 2)]
        seg_19_same3[:, t] = np.mean(seg_p[:, ms], axis=1)
    seg_rec = seg_r3_mean / np.maximum(seg_19_same3, 1.0)
    dom_int = dom_mask.astype(np.int32)

    def build_slice(s: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        t_tgt = s + h
        m_tgt = t_tgt % 12
        t_last = t_tgt - 12 if h <= 12 else t_tgt - 24
        pax_last = P[:, t_last] if t_last >= 0 else np.full(n_grp, np.nan)
        t_prev2 = t_last - 12
        pax_prev2 = P[:, t_prev2] if t_prev2 >= 0 else np.full(n_grp, np.nan)
        pax_19_m = P[:, m_tgt] if s >= m_tgt else np.full(n_grp, np.nan)

        p_s = P[:, s]
        p_r3 = pax_r3[:, s]
        p_r6 = pax_r6[:, s]
        p_r12 = pax_r12[:, s]
        p19m = pax_19_mean[:, s]

        seg_rec_s = np.where(np.isfinite(seg_rec[dom_int, s]), seg_rec[dom_int, s], 1.0)
        p19_3_safe = np.where(np.isfinite(pax_19_same3[:, s]), pax_19_same3[:, s], p19m)
        prior_s = 3000.0
        pair_rec = np.clip((3.0 * p_r3 + prior_s * seg_rec_s) / np.maximum(3.0 * p19_3_safe + prior_s, 1.0), 0.05, 2.5)
        seg_eq = np.maximum(1.02, seg_rec_s)
        decay = np.exp(-float(h) / 12.0)
        tgt_mult = np.where(pair_rec < 0.95, np.maximum(pair_rec, seg_eq), pair_rec * (1.0 + 0.015 * (h / 12.0)))
        mult = decay * pair_rec + (1.0 - decay) * tgt_mult
        p19_b = np.where(np.isfinite(pax_19_m) & (pax_19_m > 0), pax_19_m, np.where(np.isfinite(pax_last), pax_last, p_r12))
        r12_b = np.where(np.isfinite(pax_last), pax_last, p_r12)
        w_rec = np.where(pair_rec >= 0.85, 0.50, 0.15)
        anchor = np.where(p19_b > 0, (1.0 - w_rec) * (p19_b * mult) + w_rec * (r12_b * (1.0 + 0.015 * (h / 12.0))), r12_b * (1.0 + 0.015 * (h / 12.0)))
        anchor = np.clip(anchor, 0.0, None)

        cal = cal_df.iloc[t_tgt]
        X = np.column_stack([
            np.full(n_grp, float(h), dtype=np.float32),
            np.full(n_grp, float(np.log1p(h)), dtype=np.float32),
            np.full(n_grp, float(cal["DEP_MONTH"]), dtype=np.float32),
            np.full(n_grp, float(cal["DAYS_IN_MONTH"]), dtype=np.float32),
            np.full(n_grp, float(cal["FRI_SUN_DAYS"]), dtype=np.float32),
            np.full(n_grp, float(cal["EASTER_IN_MONTH"]), dtype=np.float32),
            np.full(n_grp, float(cal["SIN_M"]), dtype=np.float32),
            np.full(n_grp, float(cal["COS_M"]), dtype=np.float32),
            is_dom.astype(np.float32),
            np.log1p(p_s).astype(np.float32),
            np.log1p(p_r3).astype(np.float32),
            np.log1p(p_r6).astype(np.float32),
            np.log1p(p_r12).astype(np.float32),
            np.log1p(np.maximum(pax_last, 0.0)).astype(np.float32),
            np.log1p(np.maximum(pax_prev2, 0.0)).astype(np.float32),
            np.log1p(np.maximum(pax_19_m, 0.0)).astype(np.float32),
            np.log1p(anchor).astype(np.float32),
            np.clip(p_r3 / np.maximum(p_r12, 1.0), 0.35, 2.25).astype(np.float32),
            np.clip(p_r3 / np.maximum(pax_19_same3[:, s], 10.0), 0.35, 2.25).astype(np.float32),
            np.clip(p_r12 / np.maximum(p19m, 10.0), 0.35, 2.25).astype(np.float32),
            np.clip(p_r3 / np.maximum(pax_r3_lag12[:, s], 10.0), 0.35, 2.25).astype(np.float32),
            np.clip(p_r12 / np.maximum(pax_r12_lag12[:, s], 10.0), 0.35, 2.25).astype(np.float32),
            np.clip(pax_19_m / np.maximum(p19m, 10.0), 0.35, 2.25).astype(np.float32),
            np.log1p(yield_r3[:, s]).astype(np.float32),
            np.log1p(yield_r12[:, s]).astype(np.float32),
        ])
        return X, anchor

    stride = 2 if n_grp <= 4000 else 3
    Xs, y_diffs, ws = [], [], []
    for s in range(6, t_cut, stride):
        for h in range(1, min(24, t_cut - s) + 1):
            t_tgt = s + h
            if 14 <= t_tgt <= 25:
                continue
            X_s, anc_s = build_slice(s, h)
            y_raw = P[:, t_tgt].astype(np.float64)
            anc_adj = anc_s.copy()
            for mseg in (dom_mask, ~dom_mask):
                if mseg.any():
                    sa = float(np.sum(anc_s[mseg]))
                    sy = float(np.sum(y_raw[mseg]))
                    if sa > 100.0 and sy > 100.0:
                        anc_adj[mseg] = anc_s[mseg] * np.clip(sy / sa, 0.35, 1.65)
            Xs.append(X_s)
            y_diffs.append((y_raw - anc_adj).astype(np.float32))
            sw = 0.25 if 26 <= t_tgt <= 28 else 1.0
            ws.append(np.full(n_grp, np.exp(-(t_cut - t_tgt) / 18.0) * sw, dtype=np.float32))

    X_tr = np.vstack(Xs)
    y_tr = np.concatenate(y_diffs)
    w_tr = np.concatenate(ws)
    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
    del Xs, y_diffs, ws, X_tr, y_tr, w_tr
    gc.collect()

    params = {
        "objective": "reg:absoluteerror",
        "learning_rate": 0.03,
        "max_depth": max_depth,
        "min_child_weight": 20.0 if n_grp < 1000 else 80.0,
        "subsample": 0.75,
        "colsample_bytree": 0.8,
        "reg_alpha": 10.0,
        "reg_lambda": 50.0,
        "tree_method": "hist",
        "nthread": -1,
        "seed": 42,
    }
    booster = xgb.train(params, dtrain, num_boost_round=n_rounds)
    del dtrain
    gc.collect()

    preds = np.zeros((n_grp, 24), dtype=np.float64)
    for h in range(1, 25):
        X_te, anc_te = build_slice(t_cut, h)
        diff_hat = booster.predict(xgb.DMatrix(X_te)).astype(np.float64)
        max_shift = np.maximum(anc_te * 0.30, 120.0)
        preds[:, h - 1] = np.clip(anc_te + np.clip(diff_hat, -max_shift, max_shift), 0.0, None)
    return preds


def predict_fold(
    fold: FoldSpec,
    hist_panel: pd.DataFrame,
    tgt_skeleton: pd.DataFrame,
    airport_df: pd.DataFrame,
    model_dir: Path,
) -> pd.DataFrame:
    """
    Multi-Level Native + Damped Hierarchical IPF Reconciliation:
    1. Base Airport-Pair forecast = 50% Damped Seasonal Anchor (03) + 50% Deconfounded L1-XGBoost.
    2. Native Coarse Models at Origin Airport (150), Dest Airport (150), Region/State Pair (3,720),
       and Metro Pair (15,210) levels.
    3. Saves Native Coarse predictions cleanly for bottom-up vs native evaluation, and reconciles
       Airport-Pair predictions via damped IPF toward low-noise Entity, State, and Metro forecasts.
    """
    assert_origin_cutoff(int(hist_panel["YM"].max()), fold.origin_ym)

    store = ExternalFoldFeatureStore(hist_panel, airport_df, include_macro=False)
    t_cut = store.t_cut
    dom_mask = store.is_dom > 0.5

    # 1. Base Airport-Pair: Deconfounded L1 XGBoost over 03 DAMPED_ANCHOR
    DROP_C = {"DAMPED_ANCHOR", "SEG_REC_NOW", "TROUGH_RATIO", "PAIR_REC_SHRUNK"}
    frames, y_diffs, ws = [], [], []
    for s in range(6, t_cut, 3):
        for h in range(1, min(24, t_cut - s) + 1):
            t_tgt = s + h
            if 14 <= t_tgt <= 25:
                continue
            df_s = store.build_features_for_slice(s, h)
            anc = df_s["DAMPED_ANCHOR"].to_numpy(dtype=np.float64)
            y_raw = store.P[:, t_tgt].astype(np.float64)
            anc_adj = anc.copy()
            for mseg in (dom_mask, ~dom_mask):
                sa, sy = float(np.sum(anc[mseg])), float(np.sum(y_raw[mseg]))
                if sa > 100.0 and sy > 100.0:
                    anc_adj[mseg] = anc[mseg] * np.clip(sy / sa, 0.35, 1.65)
            df_c = df_s[[c for c in df_s.columns if c not in DROP_C]].copy()
            for cc in ["REC_19_R3", "REC_19_R12", "YOY_R3", "YOY_R12"]:
                df_c[cc] = np.clip(df_c[cc], 0.35, 2.25)
            frames.append(df_c)
            y_diffs.append((y_raw - anc_adj).astype(np.float32))
            sw = 0.25 if 26 <= t_tgt <= 28 else 1.0
            ws.append(np.full(store.n_pairs, np.exp(-(t_cut - t_tgt) / 18.0) * sw, dtype=np.float32))

    X_tr = pd.concat(frames, ignore_index=True)
    y_tr = np.concatenate(y_diffs)
    w_tr = np.concatenate(ws)
    feat_cols = list(X_tr.columns)
    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
    del frames, y_diffs, ws, X_tr, y_tr, w_tr
    gc.collect()

    base_booster = xgb.train(
        {
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
        },
        dtrain,
        num_boost_round=380,
    )
    del dtrain
    gc.collect()

    base_preds = np.zeros((store.n_pairs, 24), dtype=np.float64)
    for h in range(1, 25):
        X_te_df = store.build_features_for_slice(t_cut, h)
        anc_te = X_te_df["DAMPED_ANCHOR"].to_numpy(dtype=np.float64)
        X_c = X_te_df[feat_cols].copy()
        for cc in ["REC_19_R3", "REC_19_R12", "YOY_R3", "YOY_R12"]:
            X_c[cc] = np.clip(X_c[cc], 0.35, 2.25)
        diff_hat = base_booster.predict(xgb.DMatrix(X_c)).astype(np.float64)
        for mseg in (dom_mask, ~dom_mask):
            diff_hat[mseg] -= np.mean(diff_hat[mseg]) * 0.5
        max_shift = np.maximum(anc_te * 0.35, 120.0)
        xgb_p = np.clip(anc_te + np.clip(diff_hat, -max_shift, max_shift), 0.0, None)
        base_preds[:, h - 1] = 0.50 * anc_te + 0.50 * xgb_p

    # 2. Native Coarser Models
    ap_meta = airport_df.set_index("AIRPRT_CD")
    org_reg = store.pairs_df["ORG_AIRPRT"].map(ap_meta["REGION_CD"]).to_numpy()
    des_reg = store.pairs_df["DES_AIRPRT"].map(ap_meta["REGION_CD"]).to_numpy()
    reg_keys = sorted(set(zip(org_reg, des_reg)))
    reg_lookup = {k: i for i, k in enumerate(reg_keys)}
    reg_idx = np.array([reg_lookup[(o, d)] for o, d in zip(org_reg, des_reg)], dtype=np.int32)
    n_reg = len(reg_keys)

    P_org = store._group_sum_matrix(store.P, store.org_ap_idx, store.n_aps)
    R_org = store._group_sum_matrix(store.R, store.org_ap_idx, store.n_aps)
    org_is_dom = np.bincount(store.org_ap_idx, weights=store.is_dom, minlength=store.n_aps) > 0
    pred_org_native = _fit_and_predict_coarse_level(P_org, R_org, org_is_dom, t_cut, store.cal_df, n_rounds=220, max_depth=4)

    P_des = store._group_sum_matrix(store.P, store.des_ap_idx, store.n_aps)
    R_des = store._group_sum_matrix(store.R, store.des_ap_idx, store.n_aps)
    des_is_dom = np.bincount(store.des_ap_idx, weights=store.is_dom, minlength=store.n_aps) > 0
    pred_des_native = _fit_and_predict_coarse_level(P_des, R_des, des_is_dom, t_cut, store.cal_df, n_rounds=220, max_depth=4)

    P_reg = store._group_sum_matrix(store.P, reg_idx, n_reg)
    R_reg = store._group_sum_matrix(store.R, reg_idx, n_reg)
    reg_is_dom = np.bincount(reg_idx, weights=store.is_dom, minlength=n_reg) > 0
    pred_reg_native = _fit_and_predict_coarse_level(P_reg, R_reg, reg_is_dom, t_cut, store.cal_df, n_rounds=280, max_depth=5)

    P_mp = store._group_sum_matrix(store.P, store.mp_idx, store.n_mp)
    R_mp = store._group_sum_matrix(store.R, store.mp_idx, store.n_mp)
    mp_is_dom = np.bincount(store.mp_idx, weights=store.is_dom, minlength=store.n_mp) > 0
    pred_mp_native = _fit_and_predict_coarse_level(P_mp, R_mp, mp_is_dom, t_cut, store.cal_df, n_rounds=300, max_depth=5)

    # 3. Damped Hierarchical Reconciliation (3 IPF passes)
    reconciled = base_preds.copy()
    for h_col in range(24):
        y = reconciled[:, h_col].copy()
        for _ in range(3):
            mp_sum = np.bincount(store.mp_idx, weights=y, minlength=store.n_mp)
            mp_ratio = np.where(mp_sum > 10.0, pred_mp_native[:, h_col] / mp_sum, 1.0)
            y *= np.clip(mp_ratio[store.mp_idx], 0.7, 1.4) ** 0.25

            reg_sum = np.bincount(reg_idx, weights=y, minlength=n_reg)
            reg_ratio = np.where(reg_sum > 10.0, pred_reg_native[:, h_col] / reg_sum, 1.0)
            y *= np.clip(reg_ratio[reg_idx], 0.7, 1.4) ** 0.25

            o_sum = np.bincount(store.org_ap_idx, weights=y, minlength=store.n_aps)
            o_ratio = np.where(o_sum > 10.0, pred_org_native[:, h_col] / o_sum, 1.0)
            y *= np.clip(o_ratio[store.org_ap_idx], 0.7, 1.4) ** 0.30

            d_sum = np.bincount(store.des_ap_idx, weights=y, minlength=store.n_aps)
            d_ratio = np.where(d_sum > 10.0, pred_des_native[:, h_col] / d_sum, 1.0)
            y *= np.clip(d_ratio[store.des_ap_idx], 0.7, 1.4) ** 0.30

        reconciled[:, h_col] = y

    # Save native coarse & base bottom-up predictions for post-eval comparison
    mdata_dir = model_dir / "model_specific_data" / fold.name
    mdata_dir.mkdir(parents=True, exist_ok=True)
    yms = [int(store.cal_df.loc[t_cut + h, "YM"]) for h in range(1, 25)]
    aps_sorted = sorted(set(store.pairs_df["ORG_AIRPRT"]).union(set(store.pairs_df["DES_AIRPRT"])))
    pd.DataFrame({
        "AIRPRT_CD": np.repeat(aps_sorted, 24),
        "YM": np.tile(yms, len(aps_sorted)),
        "pred_org_native": pred_org_native.ravel(),
        "pred_des_native": pred_des_native.ravel(),
    }).to_parquet(mdata_dir / "native_airport_preds.parquet", index=False)
    pd.DataFrame({
        "ORG_REG": np.repeat([k[0] for k in reg_keys], 24),
        "DES_REG": np.repeat([k[1] for k in reg_keys], 24),
        "YM": np.tile(yms, len(reg_keys)),
        "pred_reg_native": pred_reg_native.ravel(),
    }).to_parquet(mdata_dir / "native_region_preds.parquet", index=False)

    pred_frames = []
    pairs = store.pairs_df[["ORG_AIRPRT", "DES_AIRPRT"]].copy()
    for h in range(1, 25):
        t_tgt = t_cut + h
        ym_tgt = int(store.cal_df.loc[t_tgt, "YM"])
        df_h = pairs.copy()
        df_h["YM"] = ym_tgt
        df_h["HORIZON"] = h
        df_h["y_pred"] = reconciled[:, h - 1]
        pred_frames.append(df_h)

    pred_all = pd.concat(pred_frames, ignore_index=True)
    return tgt_skeleton[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"]].merge(
        pred_all, on=["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"], how="left"
    )


if __name__ == "__main__":
    run_backtest_suite(Path(__file__).resolve().parent, predict_fold)
