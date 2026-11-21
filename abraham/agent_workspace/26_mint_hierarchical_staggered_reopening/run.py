from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_eval.runner import run_backtest_suite
from shared_eval.splits import FoldSpec, assert_origin_cutoff, FINAL_TEST_FOLD
from mint_engine import build_structural_summing_matrix, reconcile_mint_woodbury

WORKSPACE = Path(__file__).resolve().parents[1]


def _load_predict_fn(folder_name: str):
    spec = importlib.util.spec_from_file_location(f"mod_{folder_name}", WORKSPACE / folder_name / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.predict_fold


def get_base_predictions(
    fold: FoldSpec,
    hist_panel: pd.DataFrame,
    tgt_skeleton: pd.DataFrame,
    airport_df: pd.DataFrame,
    model_dir: Path,
) -> pd.DataFrame:
    """
    Retrieve baseline bottom-up forecasts (hat_y_B) from Champion Model 25
    (5-Segment Geo-Macro Staggered Reopening Ensemble).
    Uses precomputed verified out-of-fold parquet if present, otherwise executes dynamically.
    """
    m25_dir = WORKSPACE / "25_geo_macro_staggered_reopening"
    oof_backtest = m25_dir / "outputs" / "backtest_predictions.parquet"
    oof_holdout = m25_dir / "outputs" / "final_holdout_predictions.parquet"

    # 1. Try loading precomputed backtest predictions
    if fold.name != FINAL_TEST_FOLD.name and oof_backtest.exists():
        df_oof = pd.read_parquet(oof_backtest)
        sub = df_oof[df_oof["fold"] == fold.name]
        if len(sub) == len(tgt_skeleton):
            merged = tgt_skeleton[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"]].merge(
                sub[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON", "y_pred"]],
                on=["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"],
                how="left",
            )
            if not merged["y_pred"].isna().any():
                return merged

    # 2. Try loading precomputed final holdout predictions
    if fold.name == FINAL_TEST_FOLD.name and oof_holdout.exists():
        df_hold = pd.read_parquet(oof_holdout)
        if len(df_hold) == len(tgt_skeleton):
            merged = tgt_skeleton[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"]].merge(
                df_hold[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON", "y_pred"]],
                on=["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"],
                how="left",
            )
            if not merged["y_pred"].isna().any():
                return merged

    # Dynamic fallback: invoke Model 25 predict_fold directly
    pred_25_fn = _load_predict_fn("25_geo_macro_staggered_reopening")
    sub_dir_25 = model_dir / "model_specific_data" / fold.name / "_sub_artifacts" / "25_ensemble"
    return pred_25_fn(fold, hist_panel, tgt_skeleton, airport_df, sub_dir_25)


def predict_fold(
    fold: FoldSpec,
    hist_panel: pd.DataFrame,
    tgt_skeleton: pd.DataFrame,
    airport_df: pd.DataFrame,
    model_dir: Path,
) -> pd.DataFrame:
    """
    Model 26: MinT Hierarchical Staggered Reopening Grand Champion.
    
    Synthesizes:
    1. Base level (K = 17,119 routes): Model 25 Geo-Macro Staggered Reopening Ensemble
       (disentangling Domestic, Americas, Transatlantic Europe, and Transpacific Asia recovery).
    2. Upper hierarchy (M_U = 383 constraints):
       - 150 Origin Airports
       - 150 Destination Airports
       - 41 Origin US States
       - 41 Destination US States
       - Total Network Capacity
    3. Closed-Form Sherman-Morrison-Woodbury Dual MinT Operator:
       Solves optimal GLS reconciliation in < 5 ms in the dual constraint space.
    4. Regime-Adaptive Consensus: Sets shrinkage alpha dynamically based on network recovery ratio.
    """
    assert_origin_cutoff(int(hist_panel["YM"].max()), fold.origin_ym)

    mdata_dir = model_dir / "model_specific_data" / fold.name
    mdata_dir.mkdir(parents=True, exist_ok=True)

    hp = hist_panel.sort_values(["PAIR_ID", "T_IDX"])
    pairs = hp[["PAIR_ID", "ORG_AIRPRT", "DES_AIRPRT"]].drop_duplicates().reset_index(drop=True)
    K = len(pairs)
    s_cut = int(hp["T_IDX"].max())

    # Build structural summing matrix mapping routes to Upper Hierarchy
    S_U, meta = build_structural_summing_matrix(pairs, airport_df, include_states=True, include_network=True)
    M_U = meta["M_U"]
    ap_to_idx = meta["ap_to_idx"]
    n_aps = meta["n_aps"]
    st_to_idx = meta["st_to_idx"]
    n_states = meta["n_states"]

    net_slice = meta["row_slices"]["network"]
    org_slice = meta["row_slices"]["org_airports"]
    des_slice = meta["row_slices"]["des_airports"]
    org_st_slice = meta["row_slices"]["org_states"]
    des_st_slice = meta["row_slices"]["des_states"]

    # Compute network shock recovery ratio at origin cutoff
    net_pax = hist_panel.groupby("T_IDX")["OD_PAX"].sum()
    cur3_net = net_pax.loc[s_cut-2:s_cut].sum()
    ref3_net = net_pax.loc[[(s_cut-k)%12 for k in (0, 1, 2)]].sum()
    rec_net = float(cur3_net / ref3_net)

    # Regime-adaptive consensus parameter alpha:
    # High in shock recovery (~0.76), low in steady-state (~0.35)
    alpha = float(np.clip(1.0 - 0.75 * max(0.0, rec_net - 0.65) / 0.35, 0.15, 1.0))

    # Upper native models (Origin & Destination Airports)
    org_pax = hist_panel.groupby(["ORG_AIRPRT", "T_IDX"])["OD_PAX"].sum().unstack(fill_value=0.0)
    des_pax = hist_panel.groupby(["DES_AIRPRT", "T_IDX"])["OD_PAX"].sum().unstack(fill_value=0.0)

    O_mat = np.zeros((n_aps, s_cut + 1), dtype=np.float64)
    D_mat = np.zeros((n_aps, s_cut + 1), dtype=np.float64)
    for a, i in ap_to_idx.items():
        if a in org_pax.index:
            cols = [c for c in range(s_cut + 1) if c in org_pax.columns]
            O_mat[i, cols] = org_pax.loc[a, cols].to_numpy()
        if a in des_pax.index:
            cols = [c for c in range(s_cut + 1) if c in des_pax.columns]
            D_mat[i, cols] = des_pax.loc[a, cols].to_numpy()

    idx3 = np.arange(s_cut - 2, s_cut + 1)
    ref3_O = O_mat[:, idx3 % 12].sum(axis=1)
    cur3_O = O_mat[:, idx3].sum(axis=1)
    rec_O = np.clip((cur3_O + 10000.0) / (ref3_O + 10000.0), 0.20, 2.5)

    ref3_D = D_mat[:, idx3 % 12].sum(axis=1)
    cur3_D = D_mat[:, idx3].sum(axis=1)
    rec_D = np.clip((cur3_D + 10000.0) / (ref3_D + 10000.0), 0.20, 2.5)

    O_pred = np.zeros((n_aps, 24), dtype=np.float64)
    D_pred = np.zeros((n_aps, 24), dtype=np.float64)
    Net_pred = np.zeros(24, dtype=np.float64)

    for h in range(1, 25):
        t = s_cut + h
        m = t % 12
        decay = np.exp(-h / 14.0)
        mult_O = decay * rec_O + (1.0 - decay) * np.maximum(1.02, rec_O) * (1.0 + 0.015 * h / 12.0)
        p_O = O_mat[:, m] * mult_O
        last = t - (12 if h <= 12 else 24)
        p_O = np.where(O_mat[:, m] > 0.0, p_O, O_mat[:, last] * (1.0 + 0.015 * h / 12.0))
        O_pred[:, h - 1] = np.maximum(p_O, 0.0)

        mult_D = decay * rec_D + (1.0 - decay) * np.maximum(1.02, rec_D) * (1.0 + 0.015 * h / 12.0)
        p_D = D_mat[:, m] * mult_D
        p_D = np.where(D_mat[:, m] > 0.0, p_D, D_mat[:, last] * (1.0 + 0.015 * h / 12.0))
        D_pred[:, h - 1] = np.maximum(p_D, 0.0)

        Net_pred[h - 1] = 0.5 * (np.sum(O_pred[:, h - 1]) + np.sum(D_pred[:, h - 1]))

    # Map airport projections to state projections
    ap_meta = airport_df.set_index("AIRPRT_CD")
    ap_state = ap_meta["STATE_PROVNC_CD"].to_dict()

    St_org_pred = np.zeros((n_states, 24), dtype=np.float64)
    St_des_pred = np.zeros((n_states, 24), dtype=np.float64)
    for a, i in ap_to_idx.items():
        st = ap_state.get(a)
        if st in st_to_idx:
            s_idx = st_to_idx[st]
            St_org_pred[s_idx, :] += O_pred[i, :]
            St_des_pred[s_idx, :] += D_pred[i, :]

    # 1. Base bottom-up forecasts (Model 25: Geo-Macro Staggered Reopening)
    base_df = get_base_predictions(fold, hist_panel, tgt_skeleton, airport_df, model_dir)
    base_mat = np.zeros((K, 24), dtype=np.float64)
    for h in range(1, 25):
        sub = base_df.loc[base_df["HORIZON"] == h].merge(pairs, on=["ORG_AIRPRT", "DES_AIRPRT"])
        base_mat[:, h - 1] = sub["y_pred"].to_numpy()

    # 2. Apply Woodbury Closed-Form MinT Reconciliation across all 24 horizons
    reconciled_mat = np.zeros_like(base_mat)
    for h in range(1, 25):
        hat_y_B = base_mat[:, h - 1]
        bu_U = S_U.dot(hat_y_B)

        native_U = np.zeros(M_U, dtype=np.float64)
        native_U[net_slice[0]:net_slice[1]] = Net_pred[h - 1]
        native_U[org_slice[0]:org_slice[1]] = O_pred[:, h - 1]
        native_U[des_slice[0]:des_slice[1]] = D_pred[:, h - 1]
        native_U[org_st_slice[0]:org_st_slice[1]] = St_org_pred[:, h - 1]
        native_U[des_st_slice[0]:des_st_slice[1]] = St_des_pred[:, h - 1]

        hat_y_U = alpha * native_U + (1.0 - alpha) * bu_U

        var_U = np.zeros(M_U, dtype=np.float64)
        var_U[net_slice[0]:net_slice[1]] = np.maximum((0.02 * hat_y_U[net_slice[0]:net_slice[1]]) ** 2, 1e4)
        var_U[org_slice[0]:org_slice[1]] = np.maximum((0.05 * hat_y_U[org_slice[0]:org_slice[1]]) ** 2, 1e2)
        var_U[des_slice[0]:des_slice[1]] = np.maximum((0.05 * hat_y_U[des_slice[0]:des_slice[1]]) ** 2, 1e2)
        var_U[org_st_slice[0]:org_st_slice[1]] = np.maximum((0.06 * hat_y_U[org_st_slice[0]:org_st_slice[1]]) ** 2, 1e3)
        var_U[des_st_slice[0]:des_st_slice[1]] = np.maximum((0.06 * hat_y_U[des_st_slice[0]:des_st_slice[1]]) ** 2, 1e3)

        var_B = (0.16 * hat_y_B + 10.0) ** 2

        y_rec = reconcile_mint_woodbury(
            S_U, hat_y_U, hat_y_B, var_U, var_B, ridge=1e-4, non_negative=True
        )
        reconciled_mat[:, h - 1] = y_rec

    # Save summary metadata
    reconciled_summary = pd.DataFrame([{
        "fold": fold.name,
        "rec_net": rec_net,
        "alpha": alpha,
        "M_U": M_U,
        "K_routes": K,
        "sum_base": float(np.sum(base_mat)),
        "sum_reconciled": float(np.sum(reconciled_mat)),
        "ratio_rec_to_base": float(np.sum(reconciled_mat) / np.sum(base_mat)),
    }])
    reconciled_summary.to_csv(mdata_dir / "mint_reconciliation_summary.csv", index=False)

    # Format output skeleton
    frames = []
    for h in range(1, 25):
        df_h = pairs[["ORG_AIRPRT", "DES_AIRPRT"]].copy()
        df_h["HORIZON"] = h
        df_h["YM"] = int(tgt_skeleton.loc[tgt_skeleton["HORIZON"] == h, "YM"].iloc[0])
        df_h["y_pred"] = reconciled_mat[:, h - 1]
        frames.append(df_h)

    pred_all = pd.concat(frames, ignore_index=True)
    out = tgt_skeleton[["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"]].merge(
        pred_all, on=["ORG_AIRPRT", "DES_AIRPRT", "YM", "HORIZON"], how="left"
    )
    return out


if __name__ == "__main__":
    run_backtest_suite(Path(__file__).resolve().parent, predict_fold)
