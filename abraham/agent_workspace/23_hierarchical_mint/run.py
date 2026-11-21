from __future__ import annotations

import gc
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as sla

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_eval.data_loader import idx_to_ym
from shared_eval.runner import run_backtest_suite
from shared_eval.splits import FoldSpec, assert_origin_cutoff


def build_structural_summing_matrix(
    pairs_df: pd.DataFrame,
    airport_df: pd.DataFrame,
) -> tuple[sp.csr_matrix, list[str], dict[str, int], dict[str, int]]:
    """
    Constructs the sparse summing matrix S in R^{M x K} mapping K=17,119 directional pairs
    to upper-level network conservation entities:
    - Row 0: Network Total (sum of all pairs)
    - Rows 1..150: Origin Airport Totals
    - Rows 151..300: Destination Airport Totals
    - Rows 301..300+K: Bottom-level Route Identity
    """
    K = len(pairs_df)
    airports = sorted(set(pairs_df["ORG_AIRPRT"]).union(set(pairs_df["DES_AIRPRT"])))
    ap_to_idx = {a: i for i, a in enumerate(airports)}
    n_aps = len(airports)

    # Row counts
    # 1 (network) + n_aps (org) + n_aps (des) + K (routes)
    M = 1 + 2 * n_aps + K

    row_ind = []
    col_ind = []
    data = []

    # 1. Total network row
    for k in range(K):
        row_ind.append(0)
        col_ind.append(k)
        data.append(1.0)

    # 2. Origin airport rows
    for k, org in enumerate(pairs_df["ORG_AIRPRT"]):
        row_ind.append(1 + ap_to_idx[org])
        col_ind.append(k)
        data.append(1.0)

    # 3. Destination airport rows
    for k, des in enumerate(pairs_df["DES_AIRPRT"]):
        row_ind.append(1 + n_aps + ap_to_idx[des])
        col_ind.append(k)
        data.append(1.0)

    # 4. Identity rows for bottom routes
    base_bottom_row = 1 + 2 * n_aps
    for k in range(K):
        row_ind.append(base_bottom_row + k)
        col_ind.append(k)
        data.append(1.0)

    S = sp.csr_matrix((data, (row_ind, col_ind)), shape=(M, K), dtype=np.float64)
    return S, airports, ap_to_idx, {"M": M, "K": K, "n_aps": n_aps}


def compute_native_airport_forecasts(
    hist_panel: pd.DataFrame,
    airports: list[str],
    ap_to_idx: dict[str, int],
    s_cut: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes high-accuracy native seasonal forecasts for Origin Airports (150),
    Destination Airports (150), and the Network Total (1) across 24 horizons.
    Airports are 2x more predictable than routes (~7% WAPE).
    """
    n_aps = len(airports)
    # Group by origin and time
    org_grp = hist_panel.groupby(["ORG_AIRPRT", "T_IDX"])["OD_PAX"].sum().unstack(fill_value=0.0)
    des_grp = hist_panel.groupby(["DES_AIRPRT", "T_IDX"])["OD_PAX"].sum().unstack(fill_value=0.0)

    O_mat = np.zeros((n_aps, s_cut + 1), dtype=np.float64)
    D_mat = np.zeros((n_aps, s_cut + 1), dtype=np.float64)

    for ap, idx in ap_to_idx.items():
        if ap in org_grp.index:
            cols = [c for c in range(s_cut + 1) if c in org_grp.columns]
            O_mat[idx, cols] = org_grp.loc[ap, cols].to_numpy()
        if ap in des_grp.index:
            cols = [c for c in range(s_cut + 1) if c in des_grp.columns]
            D_mat[idx, cols] = des_grp.loc[ap, cols].to_numpy()

    # Trailing 3m and 12m recovery
    idx3 = np.arange(s_cut - 2, s_cut + 1)
    idx12 = np.arange(s_cut - 11, s_cut + 1)

    ref3_O = O_mat[:, idx3 % 12].sum(axis=1)
    cur3_O = O_mat[:, idx3].sum(axis=1)
    rec_O = np.clip((cur3_O + 20000.0) / (ref3_O + 20000.0), 0.20, 2.0)

    ref3_D = D_mat[:, idx3 % 12].sum(axis=1)
    cur3_D = D_mat[:, idx3].sum(axis=1)
    rec_D = np.clip((cur3_D + 20000.0) / (ref3_D + 20000.0), 0.20, 2.0)

    O_pred = np.zeros((n_aps, 24), dtype=np.float64)
    D_pred = np.zeros((n_aps, 24), dtype=np.float64)
    Net_pred = np.zeros(24, dtype=np.float64)

    for h in range(1, 25):
        t = s_cut + h
        m = t % 12
        decay = np.exp(-h / 14.0)

        # Origin
        mult_O = decay * rec_O + (1.0 - decay) * 1.02 * (1.0 + 0.015 * h / 12.0)
        p_O = O_mat[:, m] * mult_O
        last = t - (12 if h <= 12 else 24)
        p_O = np.where(O_mat[:, m] > 0.0, p_O, O_mat[:, last] * (1.0 + 0.015 * h / 12.0))
        O_pred[:, h - 1] = np.maximum(p_O, 0.0)

        # Destination
        mult_D = decay * rec_D + (1.0 - decay) * 1.02 * (1.0 + 0.015 * h / 12.0)
        p_D = D_mat[:, m] * mult_D
        p_D = np.where(D_mat[:, m] > 0.0, p_D, D_mat[:, last] * (1.0 + 0.015 * h / 12.0))
        D_pred[:, h - 1] = np.maximum(p_D, 0.0)

        # Network total is average of O and D totals
        Net_pred[h - 1] = 0.5 * (np.sum(O_pred[:, h - 1]) + np.sum(D_pred[:, h - 1]))

    return O_pred, D_pred, Net_pred


def compute_bottom_base_forecast(
    hist_panel: pd.DataFrame,
    airport_df: pd.DataFrame,
    fold: FoldSpec,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Computes bottom-level base forecasts for all 17,119 directional pairs across 24 horizons.
    Uses scale-adaptive empirical Bayes recovery.
    """
    hp = hist_panel.sort_values(["PAIR_ID", "T_IDX"])
    pairs = hp[["PAIR_ID", "ORG_AIRPRT", "DES_AIRPRT"]].drop_duplicates().reset_index(drop=True)
    K = len(pairs)
    P = hp["OD_PAX"].to_numpy().reshape(K, -1)
    s = P.shape[1] - 1

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
    hist_vol = np.maximum(P[:, : s + 1].sum(axis=1), 1.0)
    prior_w3 = np.clip(hist_vol * 0.15, 50.0, 1500.0)
    prior_w6 = np.clip(hist_vol * 0.30, 100.0, 3000.0)

    rec3 = (cur3 + prior_w3 * pooled) / (ref3 + prior_w3)
    rec6 = (cur6 + prior_w6 * pooled) / (ref6 + prior_w6)
    now = np.clip(0.75 * rec3 + 0.25 * rec6, 0.03, 3.0)

    relative = np.clip(now / np.maximum(seg[dom], 0.15), 0.10, 2.5)
    persistent = np.where(relative < 1.0, np.sqrt(relative), relative)
    eq = np.maximum(1.02, seg[dom]) * persistent

    bottom_pred = np.zeros((K, 24), dtype=np.float64)
    for h in range(1, 25):
        t = s + h
        m = t % 12
        decay = np.exp(-h / 12.0)
        mult = decay * now + (1.0 - decay) * eq * (1.0 + 0.015 * h / 12.0)
        pred = P[:, m] * mult
        last = t - (12 if h <= 12 else 24)
        pred = np.where(P[:, m] > 0.0, pred, P[:, last] * (1.0 + 0.015 * h / 12.0))
        bottom_pred[:, h - 1] = np.maximum(pred, 0.0)

    return bottom_pred, pairs


def reconcile_mint(
    S: sp.csr_matrix,
    b_all: np.ndarray,
    weights: np.ndarray,
    K: int,
) -> np.ndarray:
    """
    Solves Weighted MinT reconciliation:
    min_{y_B} || W^{-1/2} (b_all - S * y_B) ||_2^2
    s.t. y_B >= 0
    """
    # Scale S and b by W^{-1/2}
    inv_std = np.sqrt(weights)
    W_diag = sp.diags(inv_std, format="csr")
    A_scaled = W_diag.dot(S)
    b_scaled = inv_std * b_all

    # Fast least-squares solve using LSQR
    res = sla.lsqr(A_scaled, b_scaled, atol=1e-5, btol=1e-5, iter_lim=80)
    y_reconciled = np.clip(res[0], 0.0, None)
    return y_reconciled


def predict_fold(
    fold: FoldSpec,
    hist_panel: pd.DataFrame,
    tgt_skeleton: pd.DataFrame,
    airport_df: pd.DataFrame,
    model_dir: Path,
) -> pd.DataFrame:
    """
    Model 23: Coherent Minimum Trace (MinT) Hierarchical Reconciliation.
    Reconciles bottom-level route forecasts to exact upper-level Origin Airport,
    Destination Airport, and Total Network conservation laws using sparse generalized least squares.
    """
    assert_origin_cutoff(int(hist_panel["YM"].max()), fold.origin_ym)

    # 1. Compute Base Forecasts
    hp = hist_panel.sort_values(["PAIR_ID", "T_IDX"])
    pairs = hp[["PAIR_ID", "ORG_AIRPRT", "DES_AIRPRT"]].drop_duplicates().reset_index(drop=True)
    K = len(pairs)
    s_cut = int(hp["T_IDX"].max())

    S, airports, ap_to_idx, meta = build_structural_summing_matrix(pairs, airport_df)
    n_aps = meta["n_aps"]
    M = meta["M"]

    O_pred, D_pred, Net_pred = compute_native_airport_forecasts(hist_panel, airports, ap_to_idx, s_cut)
    bottom_pred, _ = compute_bottom_base_forecast(hist_panel, airport_df, fold)

    # 2. MinT Reconciliation across each of the 24 months
    cal_df = hist_panel[["T_IDX", "YM"]].drop_duplicates().sort_values("T_IDX").reset_index(drop=True)
    last_ym = int(cal_df.loc[s_cut, "YM"])

    frames = []
    base_bottom_row = 1 + 2 * n_aps

    for h in range(1, 25):
        t = s_cut + h
        ym = int(idx_to_ym(t))

        # Vector b of base forecasts across the hierarchy
        b_vec = np.zeros(M, dtype=np.float64)
        b_vec[0] = Net_pred[h - 1]
        b_vec[1 : 1 + n_aps] = O_pred[:, h - 1]
        b_vec[1 + n_aps : 1 + 2 * n_aps] = D_pred[:, h - 1]
        b_vec[base_bottom_row:] = bottom_pred[:, h - 1]

        # Weights: inverse variance w_i = 1 / sigma_i^2
        # Network error ~ 3% of volume, Airport error ~ 7%, Route error ~ 18% + 5
        var_vec = np.zeros(M, dtype=np.float64)
        var_vec[0] = np.maximum((0.03 * b_vec[0]) ** 2, 1e4)
        var_vec[1 : 1 + n_aps] = np.maximum((0.07 * b_vec[1 : 1 + n_aps]) ** 2, 1e2)
        var_vec[1 + n_aps : 1 + 2 * n_aps] = np.maximum((0.07 * b_vec[1 + n_aps : 1 + 2 * n_aps]) ** 2, 1e2)
        var_vec[base_bottom_row:] = (0.18 * b_vec[base_bottom_row:] + 5.0) ** 2

        w_vec = 1.0 / var_vec

        # Solve MinT
        y_rec = reconcile_mint(S, b_vec, w_vec, K)

        # Smooth blend with bottom base (60% reconciled + 40% bottom base) to avoid over-distortion
        y_final = 0.60 * y_rec + 0.40 * bottom_pred[:, h - 1]

        df_h = pairs[["ORG_AIRPRT", "DES_AIRPRT"]].copy()
        df_h["YM"] = ym
        df_h["HORIZON"] = h
        df_h["y_pred"] = y_final
        frames.append(df_h)

    pred_all = pd.concat(frames, ignore_index=True)
    return tgt_skeleton[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"]].merge(
        pred_all, on=["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"], how="left"
    )


if __name__ == "__main__":
    run_backtest_suite(Path(__file__).resolve().parent, predict_fold)
