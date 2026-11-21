from __future__ import annotations

import calendar
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_eval.feature_engine import FoldFeatureStore
from shared_eval.runner import run_backtest_suite
from shared_eval.splits import FoldSpec, assert_origin_cutoff


def compute_anchor_v3(store: FoldFeatureStore, s: int, h: int) -> np.ndarray:
    """
    Refined structural seasonal-shrinkage forecaster:
    - Keeps fast post-shock equilibrium reversion toward segment ceiling (>=1.03)
    - Adds undirected (A<->B) route recovery shrinkage for noisy directional routes
    - Applies exact calendar days-in-month scaling (e.g. leap Feb 29 vs 28)
    - Uses recent-12m seasonal shape aggressively ONLY once route recovery >= 0.88.
    """
    t_tgt = s + h
    m_tgt_idx = t_tgt % 12
    ym_tgt = int(store.cal_df.loc[t_tgt, "YM"])
    days_tgt = float(calendar.monthrange(ym_tgt // 100, ym_tgt % 100)[1])

    p_r3 = store.pax_r3[:, s]
    p_r12 = store.pax_r12[:, s]

    t_last = t_tgt - 12 if h <= 12 else t_tgt - 24
    pax_last = store.P[:, t_last] if t_last >= 0 else p_r12
    ym_last = int(store.cal_df.loc[t_last, "YM"]) if t_last >= 0 else 201900 + (ym_tgt % 100)
    days_last = float(calendar.monthrange(ym_last // 100, ym_last % 100)[1])

    pax_19_m = store.P[:, m_tgt_idx] if s >= m_tgt_idx else pax_last
    days_19 = float(calendar.monthrange(2019, ym_tgt % 100)[1])

    # Undirected shrinkage (75% own direction, 25% reverse direction)
    rev_valid = store.rev_idx >= 0
    rev_safe = np.maximum(store.rev_idx, 0)
    p_r3_sym = np.where(rev_valid, 0.75 * p_r3 + 0.25 * store.pax_r3[rev_safe, s], p_r3)
    p19_3_raw = np.where(np.isfinite(store.pax_19_same3[:, s]), store.pax_19_same3[:, s], store.pax_2019_mean[:, s])
    p19_3_rev = np.where(np.isfinite(store.pax_19_same3[rev_safe, s]), store.pax_19_same3[rev_safe, s], store.pax_2019_mean[rev_safe, s])
    p19_3_sym = np.where(rev_valid, 0.75 * p19_3_raw + 0.25 * p19_3_rev, p19_3_raw)

    dom_int = store.is_dom.astype(np.int32)
    seg_rec_s = np.where(np.isfinite(store.seg_rec_now[dom_int, s]), store.seg_rec_now[dom_int, s], 1.0)

    # Empirical Bayes shrunk recovery ratio toward segment mean
    prior_strength = 1200.0
    pair_rec = np.clip(
        (3.0 * p_r3_sym + prior_strength * seg_rec_s) / np.maximum(3.0 * p19_3_sym + prior_strength, 1.0),
        0.05,
        2.5,
    )

    # Post-COVID equilibrium floor (>= 1.03 for domestic, >= 1.02 for intl unless still near-zero)
    seg_eq = np.maximum(1.03, seg_rec_s)
    decay = np.exp(-float(h) / 11.0)
    secular = 1.0 + 0.022 * (float(h) / 12.0)
    tgt_mult = np.where(pair_rec < 0.95, np.maximum(pair_rec, seg_eq), pair_rec * secular)
    mult_19 = decay * pair_rec + (1.0 - decay) * tgt_mult

    # Also apply gentle shrunk recent-12m vs 2019 Airport-Entity growth tilt when steady
    org_19 = np.where(np.isfinite(store.pair_org_19_same3[:, s]), store.pair_org_19_same3[:, s], store.pair_org_r12[:, s])
    des_19 = np.where(np.isfinite(store.pair_des_19_same3[:, s]), store.pair_des_19_same3[:, s], store.pair_des_r12[:, s])
    org_rel = np.clip((store.pair_org_r3[:, s] / np.maximum(org_19, 100.0)) / np.maximum(seg_rec_s, 0.2), 0.75, 1.25)
    des_rel = np.clip((store.pair_des_r3[:, s] / np.maximum(des_19, 100.0)) / np.maximum(seg_rec_s, 0.2), 0.75, 1.25)
    ap_tilt = np.where(seg_rec_s >= 0.85, np.sqrt(org_rel * des_rel) ** 0.15, 1.0)

    y_from_19 = pax_19_m * mult_19 * ap_tilt * (days_tgt / days_19)

    # Recent-12m base with days-in-month & gentle secular trend
    r12_base = np.where(np.isfinite(pax_last), pax_last, p_r12) * (days_tgt / days_last)
    # Apply undirected stabilization on tiny/medium routes for recent-12m as well
    pax_last_rev = np.where(rev_valid, store.P[rev_safe, t_last] if t_last >= 0 else store.pax_r12[rev_safe, s], pax_last)
    r12_sym = np.where(rev_valid & (r12_base < 1500.0), 0.85 * r12_base + 0.15 * pax_last_rev * (days_tgt / days_last), r12_base)
    y_from_r12 = r12_sym * secular

    # Only switch heavily to recent-12m once pair recovery >= 0.88 (avoid dragging COVID troughs!)
    w_recent = np.where(pair_rec >= 0.88, 0.62, np.where(pair_rec >= 0.75, 0.30, 0.12))
    y_pred = np.where(pax_19_m > 0.0, (1.0 - w_recent) * y_from_19 + w_recent * y_from_r12, y_from_r12)
    return np.clip(y_pred, 0.0, None)


def predict_fold(
    fold: FoldSpec,
    hist_panel: pd.DataFrame,
    tgt_skeleton: pd.DataFrame,
    airport_df: pd.DataFrame,
    model_dir: Path,
) -> pd.DataFrame:
    assert_origin_cutoff(int(hist_panel["YM"].max()), fold.origin_ym)
    store = FoldFeatureStore(hist_panel, airport_df)

    mdata_dir = model_dir / "model_specific_data" / fold.name
    mdata_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"origin_ym": [fold.origin_ym], "n_pairs": [store.n_pairs]}).to_csv(
        mdata_dir / "anchor_v3_meta.csv", index=False
    )

    pred_frames = []
    pairs = store.pairs_df[["ORG_AIRPRT", "DES_AIRPRT"]].copy()
    for h in range(1, 25):
        t_tgt = store.t_cut + h
        ym_tgt = int(store.cal_df.loc[t_tgt, "YM"])
        y_hat = compute_anchor_v3(store, store.t_cut, h)
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
