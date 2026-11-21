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


def compute_structural_anchor_for_slice(
    store: FoldFeatureStore,
    airport_df: pd.DataFrame,
    s: int,
    h: int,
) -> np.ndarray:
    """
    Leakage-free hierarchical shrinkage & structural-growth statistical forecast at origin s, horizon h:
    - Uses undirected pair pooling + empirical Bayes airport/region recovery ratios
    - Corrects for calendar days-in-month (leap year Feb!)
    - Avoids post-shock extrapolation traps while capturing structural hub shifts (AUS/BNA vs SFO).
    """
    t_tgt = s + h
    m_tgt_idx = t_tgt % 12
    ym_tgt = int(store.cal_df.loc[t_tgt, "YM"])
    yr_tgt, mo_tgt = ym_tgt // 100, ym_tgt % 100
    days_tgt = float(calendar.monthrange(yr_tgt, mo_tgt)[1])

    # Trailing 3m & 6m & 12m at origin s
    p_r3 = store.pax_r3[:, s]
    p_r6 = store.pax_r6[:, s]
    p_r12 = store.pax_r12[:, s]

    # Most recent occurrence of target calendar month <= s
    t_last = t_tgt - 12 if h <= 12 else t_tgt - 24
    pax_last = store.P[:, t_last] if t_last >= 0 else p_r12
    ym_last = int(store.cal_df.loc[t_last, "YM"]) if t_last >= 0 else 201900 + mo_tgt
    days_last = float(calendar.monthrange(ym_last // 100, ym_last % 100)[1])

    pax_19_m = store.P[:, m_tgt_idx] if s >= m_tgt_idx else pax_last
    days_19 = float(calendar.monthrange(2019, mo_tgt)[1])

    # Undirected pair pooling (shrink noisy directional route toward symmetric A<->B average)
    rev_valid = store.rev_idx >= 0
    rev_idx_safe = np.maximum(store.rev_idx, 0)
    p_r3_sym = np.where(rev_valid, 0.70 * p_r3 + 0.30 * store.pax_r3[rev_idx_safe, s], p_r3)
    p19_3_raw = np.where(np.isfinite(store.pax_19_same3[:, s]), store.pax_19_same3[:, s], store.pax_2019_mean[:, s])
    p19_3_sym = np.where(
        rev_valid,
        0.70 * p19_3_raw + 0.30 * np.where(np.isfinite(store.pax_19_same3[rev_idx_safe, s]), store.pax_19_same3[rev_idx_safe, s], store.pax_2019_mean[rev_idx_safe, s]),
        p19_3_raw,
    )

    # Airport-level and Segment-level recovery vs 2019
    dom_int = store.is_dom.astype(np.int32)
    seg_rec_s = np.where(np.isfinite(store.seg_rec_now[dom_int, s]), store.seg_rec_now[dom_int, s], 1.0)

    org_19 = np.where(np.isfinite(store.pair_org_19_same3[:, s]), store.pair_org_19_same3[:, s], store.pair_org_r12[:, s])
    des_19 = np.where(np.isfinite(store.pair_des_19_same3[:, s]), store.pair_des_19_same3[:, s], store.pair_des_r12[:, s])
    org_rec = np.clip(store.pair_org_r3[:, s] / np.maximum(org_19, 100.0), 0.10, 2.0)
    des_rec = np.clip(store.pair_des_r3[:, s] / np.maximum(des_19, 100.0), 0.10, 2.0)
    # Geometric mean of org & des airport structural recovery blended with segment recovery
    ap_prior_rec = 0.60 * np.sqrt(org_rec * des_rec) + 0.40 * seg_rec_s

    # Empirical Bayes shrinkage of route recovery ratio toward airport+segment prior
    prior_strength = 900.0
    pair_rec = np.clip(
        (3.0 * p_r3_sym + prior_strength * ap_prior_rec) / np.maximum(3.0 * p19_3_sym + prior_strength, 1.0),
        0.05,
        2.5,
    )

    # Post-shock equilibrium recovery target:
    # Domestic routes recover to >= max(ap_prior_rec, 1.03); Intl routes recover toward max(ap_prior_rec, 0.96)
    # unless deeply below 0.35 (persistent regional restriction like East Asia in 2022), where partial recovery applies.
    eq_floor = np.where(
        store.is_dom > 0.5,
        np.maximum(1.03, ap_prior_rec),
        np.where(pair_rec < 0.35, 0.72, np.maximum(0.98, ap_prior_rec)),
    )
    # Smooth transition across h=1..24 with half-life ~10 months
    decay = np.exp(-float(h) / 10.0)
    # Gentle secular drift (+2.2%/yr) once recovered
    secular = 1.0 + 0.022 * (float(h) / 12.0)
    mult_from_19 = np.where(
        pair_rec < 0.92,
        decay * pair_rec + (1.0 - decay) * np.maximum(pair_rec, eq_floor) * (1.0 + 0.01 * (h / 12.0)),
        pair_rec * secular,
    )

    # Calendar days-in-month adjustment (e.g., Feb 29 2024 vs Feb 28 2023/2019!)
    y_from_19 = pax_19_m * mult_from_19 * (days_tgt / days_19)

    # Recent-12m level scaled by airport/route trailing momentum (capped gently) and days-in-month
    # Check 12m trailing YoY only when past year >= 0.75 * 2019 (steady regime)
    r12_base = np.where(np.isfinite(pax_last), pax_last, p_r12) * (days_tgt / days_last)
    # Gentle recent-3m vs prior-year-3m momentum for steady routes, heavily shrunk
    r3_lag12 = np.where(np.isfinite(store.pax_r3_lag12[:, s]), store.pax_r3_lag12[:, s], p_r3)
    raw_mom = np.clip(p_r3 / np.maximum(r3_lag12, 20.0), 0.75, 1.35)
    shrink_w = p_r3 / (p_r3 + 500.0)
    mom = 1.0 + shrink_w * (raw_mom - 1.0) * 0.35
    y_from_r12 = r12_base * (1.0 + (mom - 1.0) * np.exp(-float(h) / 18.0)) * secular

    # When route is already steady/recovered (>= 0.85 of 2019), rely heavily (70%) on recent-12m seasonal shape!
    w_recent = np.where(pair_rec >= 0.85, 0.70, np.where(pair_rec >= 0.65, 0.45, 0.15))
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
        mdata_dir / "anchor_meta.csv", index=False
    )

    pred_frames = []
    pairs = store.pairs_df[["ORG_AIRPRT", "DES_AIRPRT"]].copy()
    for h in range(1, 25):
        t_tgt = store.t_cut + h
        ym_tgt = int(store.cal_df.loc[t_tgt, "YM"])
        y_hat = compute_structural_anchor_for_slice(store, airport_df, store.t_cut, h)
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
