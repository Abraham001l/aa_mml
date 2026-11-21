from __future__ import annotations

import gc
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_eval.data_loader import idx_to_ym
from shared_eval.external_feature_store import ExternalFoldFeatureStore
from shared_eval.runner import run_backtest_suite
from shared_eval.splits import FoldSpec, assert_origin_cutoff


def compute_tier_adaptive_gravity_anchor(
    hist_panel: pd.DataFrame,
    airport_df: pd.DataFrame,
    fold: FoldSpec,
    tau: float = 12.0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Tier-Adaptive Endpoint-Pooled Structural Anchor with Spatial Demographics:
    - Shrinks airport endpoint ratios toward segment recovery (Domestic vs Intl).
    - Adapts route shrinkage prior dynamically to route scale (saving small/micro routes from over-inflation).
    - Retains persistent relative route advantages damped toward long-term equilibrium.
    """
    hp = hist_panel.sort_values(["PAIR_ID", "T_IDX"])
    pairs = hp[["PAIR_ID", "ORG_AIRPRT", "DES_AIRPRT"]].drop_duplicates().reset_index(drop=True)
    n = len(pairs)
    P = hp["OD_PAX"].to_numpy().reshape(n, -1)
    s = P.shape[1] - 1
    assert s >= 23

    countries = airport_df.set_index("AIRPRT_CD")["CNTRY_CD"]
    dom = (
        (pairs["ORG_AIRPRT"].map(countries) == "US")
        & (pairs["DES_AIRPRT"].map(countries) == "US")
    ).to_numpy().astype(int)

    oi = pd.factorize(pairs["ORG_AIRPRT"])[0]
    di = pd.factorize(pairs["DES_AIRPRT"])[0]

    idx3 = np.arange(s - 2, s + 1)
    idx6 = np.arange(s - 5, s + 1)
    ref3 = P[:, idx3 % 12].sum(axis=1)
    cur3 = P[:, idx3].sum(axis=1)
    ref6 = P[:, idx6 % 12].sum(axis=1)
    cur6 = P[:, idx6].sum(axis=1)

    seg_denom = np.maximum(np.bincount(dom, weights=ref3), 1.0)
    seg = np.bincount(dom, weights=cur3) / seg_denom

    # Airport nodes pass segment-specific recovery ratios
    def endpoint_prior(groups: np.ndarray) -> np.ndarray:
        keys = groups * 2 + dom
        numer = np.bincount(keys, weights=cur3)
        denom = np.bincount(keys, weights=ref3)
        priors = np.tile(seg, len(numer) // 2 + 1)[: len(numer)]
        return ((numer + 10000.0 * priors) / (denom + 10000.0))[keys]

    pooled = np.sqrt(np.maximum(endpoint_prior(oi) * endpoint_prior(di), 1e-4))

    # Tier-adaptive route prior: scale-proportional to avoid blowing up micro routes!
    hist_vol = np.maximum(P[:, : s + 1].sum(axis=1), 1.0)
    # Adaptive prior weight: 50 for micro, up to 1500 for large trunk routes
    prior_w3 = np.clip(hist_vol * 0.15, 50.0, 1500.0)
    prior_w6 = np.clip(hist_vol * 0.30, 100.0, 3000.0)

    rec3 = (cur3 + prior_w3 * pooled) / (ref3 + prior_w3)
    rec6 = (cur6 + prior_w6 * pooled) / (ref6 + prior_w6)
    now = np.clip(0.75 * rec3 + 0.25 * rec6, 0.03, 3.0)

    # Relative route factor with partial catchup
    relative = np.clip(now / np.maximum(seg[dom], 0.15), 0.10, 2.5)
    persistent = np.where(relative < 1.0, np.sqrt(relative), relative)
    eq = np.maximum(1.02, seg[dom]) * persistent

    # Precompute anchor array (n_pairs, 24)
    anchor_mat = np.zeros((n, 24), dtype=np.float64)
    for h in range(1, 25):
        t = s + h
        m = t % 12
        decay = np.exp(-h / tau)
        mult = decay * now + (1.0 - decay) * eq * (1.0 + 0.015 * h / 12.0)
        pred = P[:, m] * mult
        last = t - (12 if h <= 12 else 24)
        pred = np.where(P[:, m] > 0.0, pred, P[:, last] * (1.0 + 0.015 * h / 12.0))
        anchor_mat[:, h - 1] = np.maximum(pred, 0.0)

    return anchor_mat, P, pairs


def predict_fold(
    fold: FoldSpec,
    hist_panel: pd.DataFrame,
    tgt_skeleton: pd.DataFrame,
    airport_df: pd.DataFrame,
    model_dir: Path,
) -> pd.DataFrame:
    """
    Model 22: Deep Spatial Gravity with Tier-Adaptive Multi-Level Alignment.
    Combines:
    1. Tier-adaptive endpoint-pooled structural recovery anchor.
    2. DeepGravity spatial features (Wilson mass, ACS migration, FAA enplanements, distance decay).
    3. Segment-centered L1 gradient booster predicting spatial route allocations.
    """
    assert_origin_cutoff(int(hist_panel["YM"].max()), fold.origin_ym)

    store = ExternalFoldFeatureStore(hist_panel, airport_df, include_macro=False)
    t_cut = store.t_cut
    dom_mask = store.is_dom > 0.5
    pairs_df = store.pairs_df[["ORG_AIRPRT", "DES_AIRPRT"]].copy()

    anchor_mat, P_full, pairs_idx = compute_tier_adaptive_gravity_anchor(
        hist_panel, airport_df, fold, tau=12.0
    )

    # Build Training Slices
    frames, ys_res, ws = [], [], []
    for s in range(12, t_cut, 2):
        for h in range(1, min(24, t_cut - s) + 1):
            t_tgt = s + h
            if 14 <= t_tgt <= 25:
                continue

            df_s = store.build_features_for_slice(s, h)
            anc = df_s["DAMPED_ANCHOR"].to_numpy(dtype=np.float64)
            y_raw = store.P[:, t_tgt].astype(np.float64)

            # Spatial Gravity Features
            org_pop = np.nan_to_num(df_s["EXT_ORG_CBSA_POP"].to_numpy(), nan=12.0)
            des_pop = np.nan_to_num(df_s["EXT_DES_CBSA_POP"].to_numpy(), nan=12.0)
            dist_log = df_s["LOG_DIST"].to_numpy(dtype=np.float32)
            mig_log = np.nan_to_num(df_s["EXT_LOG_MIG_TOTAL"].to_numpy(), nan=0.0)

            df_s["DEEP_GRAVITY_INDEX"] = (org_pop + des_pop - 1.2 * dist_log + 0.25 * mig_log).astype(np.float32)
            df_s["POP_ASYM"] = (org_pop - des_pop).astype(np.float32)

            # Target: Multiplicative residual bounded log-ratio
            anc_safe = np.maximum(anc, 1.0)
            y_ratio = np.clip(np.log((y_raw + 5.0) / (anc_safe + 5.0)), -1.0, 1.0).astype(np.float32)

            shock_w = 0.30 if 26 <= t_tgt <= 28 else 1.0
            rec_w = float(np.exp(-(t_cut - t_tgt) / 18.0)) * shock_w
            vol_w = np.sqrt(np.maximum(anc_safe, 2.0)).astype(np.float32)

            # Select features
            feat_cols = [
                c for c in df_s.columns
                if c not in {
                    "ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON", "DAMPED_ANCHOR",
                    "LOG_DAMPED_ANCHOR", "SEG_REC_NOW", "TROUGH_RATIO", "PAIR_REC_SHRUNK"
                }
            ]

            frames.append(df_s[feat_cols])
            ys_res.append(y_ratio)
            ws.append(rec_w * vol_w)

    X_tr = pd.concat(frames, ignore_index=True)
    y_tr = np.concatenate(ys_res)
    w_tr = np.concatenate(ws)
    del frames, ys_res, ws
    gc.collect()

    feat_names = list(X_tr.columns)
    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
    del X_tr, y_tr, w_tr
    gc.collect()

    params = {
        "objective": "reg:absoluteerror",
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 400.0,
        "subsample": 0.80,
        "colsample_bytree": 0.75,
        "reg_alpha": 4.0,
        "reg_lambda": 60.0,
        "tree_method": "hist",
        "nthread": -1,
        "seed": 42,
    }
    booster = xgb.train(params, dtrain, num_boost_round=360)
    del dtrain
    gc.collect()

    # Prediction Loop over 24 Horizons
    pred_frames = []
    for h in range(1, 25):
        t_tgt = t_cut + h
        ym_tgt = int(store.cal_df.loc[t_tgt, "YM"])
        X_te = store.build_features_for_slice(t_cut, h)

        org_pop = np.nan_to_num(X_te["EXT_ORG_CBSA_POP"].to_numpy(), nan=12.0)
        des_pop = np.nan_to_num(X_te["EXT_DES_CBSA_POP"].to_numpy(), nan=12.0)
        dist_log = X_te["LOG_DIST"].to_numpy(dtype=np.float32)
        mig_log = np.nan_to_num(X_te["EXT_LOG_MIG_TOTAL"].to_numpy(), nan=0.0)

        X_te["DEEP_GRAVITY_INDEX"] = (org_pop + des_pop - 1.2 * dist_log + 0.25 * mig_log).astype(np.float32)
        X_te["POP_ASYM"] = (org_pop - des_pop).astype(np.float32)

        delta_hat = booster.predict(xgb.DMatrix(X_te[feat_names])).astype(np.float64)

        # Zero-drift segment centering
        for seg_m in (dom_mask, ~dom_mask):
            delta_hat[seg_m] -= np.mean(delta_hat[seg_m]) * 0.5
        delta_hat = np.clip(delta_hat, -0.25, 0.25)

        # Apply to tier-adaptive anchor
        anc_h = anchor_mat[:, h - 1]
        y_hat = np.clip((anc_h + 5.0) * np.exp(delta_hat) - 5.0, 0.0, None)

        # Blend with pure tier-adaptive anchor (70% anchor + 30% spatial booster)
        y_final = 0.70 * anc_h + 0.30 * y_hat

        df_h = pairs_df.copy()
        df_h["YM"] = ym_tgt
        df_h["HORIZON"] = h
        df_h["y_pred"] = y_final
        pred_frames.append(df_h)

    pred_all = pd.concat(pred_frames, ignore_index=True)
    return tgt_skeleton[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"]].merge(
        pred_all, on=["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"], how="left"
    )


if __name__ == "__main__":
    run_backtest_suite(Path(__file__).resolve().parent, predict_fold)
