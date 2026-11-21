from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_eval.runner import run_backtest_suite
from shared_eval.splits import FoldSpec, assert_origin_cutoff

WORKSPACE = Path(__file__).resolve().parents[1]


def _load_predict_fn(folder_name: str):
    spec = importlib.util.spec_from_file_location(f"mod_{folder_name}", WORKSPACE / folder_name / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.predict_fold


def compute_persistent_network_recovery(
    fold: FoldSpec,
    hist_panel: pd.DataFrame,
    tgt_skeleton: pd.DataFrame,
    airport_df: pd.DataFrame,
    tau: float = 12.0,
) -> pd.DataFrame:
    """
    Self-contained Endpoint-Pooled Network Recovery with persistent route deviations.
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

    def endpoint_prior(groups: np.ndarray) -> np.ndarray:
        keys = groups * 2 + dom
        numer = np.bincount(keys, weights=cur3)
        denom = np.bincount(keys, weights=ref3)
        priors = np.tile(seg, len(numer) // 2 + 1)[: len(numer)]
        return ((numer + 10000.0 * priors) / (denom + 10000.0))[keys]

    pooled = np.sqrt(np.maximum(endpoint_prior(oi) * endpoint_prior(di), 1e-4))
    rec3 = (cur3 + 1500.0 * pooled) / (ref3 + 1500.0)
    rec6 = (cur6 + 3000.0 * pooled) / (ref6 + 3000.0)
    now = np.clip(0.75 * rec3 + 0.25 * rec6, 0.03, 3.0)

    relative = np.clip(now / np.maximum(seg[dom], 0.15), 0.10, 2.5)
    persistent = np.where(relative < 1.0, np.sqrt(relative), relative)
    eq = np.maximum(1.02, seg[dom]) * persistent

    frames = []
    for h in range(1, 25):
        t = s + h
        m = t % 12
        decay = np.exp(-h / tau)
        mult = decay * now + (1.0 - decay) * eq * (1.0 + 0.015 * h / 12.0)
        pred = P[:, m] * mult
        last = t - (12 if h <= 12 else 24)
        pred = np.where(P[:, m] > 0.0, pred, P[:, last] * (1.0 + 0.015 * h / 12.0))

        frame = pairs[["ORG_AIRPRT", "DES_AIRPRT"]].copy()
        frame["HORIZON"] = h
        frame["y_pred"] = np.maximum(pred, 0.0)
        frames.append(frame)

    pred_all = pd.concat(frames, ignore_index=True)
    return tgt_skeleton[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"]].merge(
        pred_all, on=["ORG_AIRPRT", "DES_AIRPRT", "HORIZON"], how="left"
    )


def predict_fold(
    fold: FoldSpec,
    hist_panel: pd.DataFrame,
    tgt_skeleton: pd.DataFrame,
    airport_df: pd.DataFrame,
    model_dir: Path,
) -> pd.DataFrame:
    """
    Model 24: Spatial Gravity & Network Recovery Grand Ensemble.
    Convex combination of:
    - 40% Model 21 (Diverse Family Benchmark: Structural Damped + Multi-Level Reconciled + L1 XGB + Cross-Sectional XGB)
    - 10% Model 22 (Deep Spatial Gravity with Tier-Adaptive Empirical Bayes + Wilson Demographic Gravity + ACS Migration)
    - 50% Persistent Network Recovery (Endpoint-Pooled Reopening + Persistent Relative Route Drift)
    """
    assert_origin_cutoff(int(hist_panel["YM"].max()), fold.origin_ym)

    mdata_dir = model_dir / "model_specific_data" / fold.name
    mdata_dir.mkdir(parents=True, exist_ok=True)

    # 1. Component 21 (Diverse Family Ensemble)
    pred_21_fn = _load_predict_fn("21_diverse_family_ensemble")
    sub_dir_21 = mdata_dir / "_sub_artifacts" / "21_diverse_family_ensemble"
    df_21 = pred_21_fn(fold, hist_panel.copy(), tgt_skeleton.copy(), airport_df.copy(), sub_dir_21)

    # 2. Component 22 (Deep Spatial Gravity)
    pred_22_fn = _load_predict_fn("22_deep_spatial_gravity")
    sub_dir_22 = mdata_dir / "_sub_artifacts" / "22_deep_spatial_gravity"
    df_22 = pred_22_fn(fold, hist_panel.copy(), tgt_skeleton.copy(), airport_df.copy(), sub_dir_22)

    # 3. Component Persistent Network Recovery
    df_persistent = compute_persistent_network_recovery(
        fold, hist_panel.copy(), tgt_skeleton.copy(), airport_df.copy(), tau=12.0
    )

    out = tgt_skeleton[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"]].copy()
    out = out.merge(df_21[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON", "y_pred"]], on=["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"])
    out = out.rename(columns={"y_pred": "pred_21"})

    out = out.merge(df_22[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON", "y_pred"]], on=["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"])
    out = out.rename(columns={"y_pred": "pred_22"})

    out = out.merge(df_persistent[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON", "y_pred"]], on=["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"])
    out = out.rename(columns={"y_pred": "pred_persistent"})

    # Optimal blend: 40% 21 + 10% 22 + 50% persistent
    y_final = 0.40 * out["pred_21"].to_numpy() + 0.10 * out["pred_22"].to_numpy() + 0.50 * out["pred_persistent"].to_numpy()
    out["y_pred"] = np.clip(y_final, 0.0, None)

    summary = pd.DataFrame([
        {"component": "21_diverse_family_ensemble", "weight": 0.40, "sum_pred": float(np.sum(out["pred_21"]))},
        {"component": "22_deep_spatial_gravity", "weight": 0.10, "sum_pred": float(np.sum(out["pred_22"]))},
        {"component": "persistent_network_recovery", "weight": 0.50, "sum_pred": float(np.sum(out["pred_persistent"]))},
    ])
    summary.to_csv(mdata_dir / "ensemble_weights.csv", index=False)

    return out[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON", "y_pred"]]


if __name__ == "__main__":
    run_backtest_suite(Path(__file__).resolve().parent, predict_fold)
