from __future__ import annotations

import calendar
from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from shared_eval.data_loader import haversine_miles, idx_to_ym, ym_to_idx


def _easter_date(year: int) -> date:
    """Anonymous Gregorian algorithm for Western Easter Sunday."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def build_calendar_table(max_t_idx: int = 120) -> pd.DataFrame:
    """Deterministic calendar features for t_idx = 0..max_t_idx (zero leakage)."""
    rows = []
    for t in range(max_t_idx + 1):
        ym = idx_to_ym(t)
        yr = ym // 100
        mo = ym % 100
        _, n_days = calendar.monthrange(yr, mo)
        # Count Fridays + Sundays (peak leisure departure days)
        fri_sun = sum(1 for d in range(1, n_days + 1) if date(yr, mo, d).weekday() in (4, 6))
        eas = _easter_date(yr)
        # Spring break / Good Friday proximity in March vs April
        good_fri = eas - timedelta(days=2)
        easter_in_month = 1.0 if eas.month == mo else 0.0
        good_fri_in_month = 1.0 if good_fri.month == mo else 0.0
        # Thanksgiving day in Nov (4th Thursday)
        nov1 = date(yr, 11, 1)
        first_thu = 1 + (3 - nov1.weekday()) % 7
        tgiving_day = first_thu + 21
        tgiving_late = float(tgiving_day >= 25) if mo == 11 else 0.0

        angle = 2.0 * np.pi * (mo - 1) / 12.0
        rows.append({
            "T_IDX": t,
            "YM": ym,
            "DEP_YEAR": yr,
            "DEP_MONTH": mo,
            "DAYS_IN_MONTH": float(n_days),
            "FRI_SUN_DAYS": float(fri_sun),
            "EASTER_IN_MONTH": easter_in_month,
            "GOOD_FRI_IN_MONTH": good_fri_in_month,
            "TGIVING_LATE": tgiving_late,
            "SIN_M": float(np.sin(angle)),
            "COS_M": float(np.cos(angle)),
            "SIN_2M": float(np.sin(2.0 * angle)),
            "COS_2M": float(np.cos(2.0 * angle)),
        })
    return pd.DataFrame(rows)


def _rolling_sum_2d(arr: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling sum along axis=1 (uses only indices <= t)."""
    cs = np.cumsum(arr, axis=1, dtype=np.float64)
    out = cs.copy()
    if window < arr.shape[1]:
        out[:, window:] = cs[:, window:] - cs[:, :-window]
    counts = np.minimum(np.arange(1, arr.shape[1] + 1, dtype=np.float64), float(window))
    return out, counts


class FoldFeatureStore:
    """
    Leakage-free precomputed state store for a single fold's hist_panel (t = 0..T_cut).
    Every array slice at pseudo-origin s strictly uses data from [0 .. s].
    """

    def __init__(self, hist_panel: pd.DataFrame, airport_df: pd.DataFrame):
        self.origin_ym = int(hist_panel["YM"].max())
        self.t_cut = int(hist_panel["T_IDX"].max())
        assert self.t_cut == ym_to_idx(self.origin_ym)

        # Canonical sorted pair index strictly from hist_panel
        pairs = (
            hist_panel[["PAIR_ID", "ORG_AIRPRT", "DES_AIRPRT"]]
            .drop_duplicates()
            .sort_values("PAIR_ID")
            .reset_index(drop=True)
        )
        self.pairs_df = pairs
        self.n_pairs = len(pairs)
        n_t = self.t_cut + 1

        # Pivot into dense (n_pairs, n_t) matrices sorted by (PAIR_ID, T_IDX)
        hp = hist_panel.sort_values(["PAIR_ID", "T_IDX"])
        assert len(hp) == self.n_pairs * n_t, "hist_panel must be balanced"

        self.P = hp["OD_PAX"].to_numpy(dtype=np.float64).reshape(self.n_pairs, n_t)
        self.R = hp["OD_REV_USD"].to_numpy(dtype=np.float64).reshape(self.n_pairs, n_t)
        self.O = hp["IS_OBSERVED"].to_numpy(dtype=np.float64).reshape(self.n_pairs, n_t)

        # Airport metadata & static pair features
        ap = airport_df.set_index("AIRPRT_CD")
        org = pairs["ORG_AIRPRT"]
        des = pairs["DES_AIRPRT"]

        org_lat = org.map(ap["LAT_DD"]).to_numpy(dtype=np.float64)
        org_lon = org.map(ap["LON_DD"]).to_numpy(dtype=np.float64)
        des_lat = des.map(ap["LAT_DD"]).to_numpy(dtype=np.float64)
        des_lon = des.map(ap["LON_DD"]).to_numpy(dtype=np.float64)
        dist = haversine_miles(org_lat, org_lon, des_lat, des_lon)

        org_cntry = org.map(ap["CNTRY_CD"]).to_numpy()
        des_cntry = des.map(ap["CNTRY_CD"]).to_numpy()
        org_metro = org.map(ap["CITY_METRO_IATA_CD"]).to_numpy()
        des_metro = des.map(ap["CITY_METRO_IATA_CD"]).to_numpy()
        org_reg = org.map(ap["REGION_CD"]).to_numpy()
        des_reg = des.map(ap["REGION_CD"]).to_numpy()

        # Multi-airport metro indicator (computed strictly from static airport count among fold airports)
        fold_aps = sorted(set(org).union(set(des)))
        ap_sub = ap.loc[fold_aps]
        metro_counts = ap_sub["CITY_METRO_IATA_CD"].value_counts().to_dict()

        self.is_dom = ((org_cntry == "US") & (des_cntry == "US")).astype(np.float32)
        self.static_feats = {
            "DIST_MI": dist.astype(np.float32),
            "LOG_DIST": np.log1p(dist).astype(np.float32),
            "IS_DOM": self.is_dom,
            "IS_SAME_STATE": (self.is_dom * (org_reg == des_reg)).astype(np.float32),
            "ORG_IS_US": (org_cntry == "US").astype(np.float32),
            "DES_IS_US": (des_cntry == "US").astype(np.float32),
            "ORG_LAT": org_lat.astype(np.float32),
            "ORG_LON": org_lon.astype(np.float32),
            "DES_LAT": des_lat.astype(np.float32),
            "DES_LON": des_lon.astype(np.float32),
            "LAT_DIFF_ABS": np.abs(org_lat - des_lat).astype(np.float32),
            "LON_DIFF": (des_lon - org_lon).astype(np.float32),
            "ORG_RUNWAY": org.map(ap["LNGST_RUNWAY_FT_QTY"]).to_numpy(dtype=np.float32),
            "DES_RUNWAY": des.map(ap["LNGST_RUNWAY_FT_QTY"]).to_numpy(dtype=np.float32),
            "ORG_ELEV": org.map(ap["ELEVATN_FT_QTY"]).to_numpy(dtype=np.float32),
            "DES_ELEV": des.map(ap["ELEVATN_FT_QTY"]).to_numpy(dtype=np.float32),
            "ORG_MULTI_METRO": np.array([metro_counts.get(m, 1) > 1 for m in org_metro], dtype=np.float32),
            "DES_MULTI_METRO": np.array([metro_counts.get(m, 1) > 1 for m in des_metro], dtype=np.float32),
        }

        # Reverse-pair index mapping (-1 if no reverse pair exists)
        pair_lookup: Dict[Tuple[str, str], int] = {
            (o, d): int(pid) for pid, o, d in zip(pairs["PAIR_ID"], org, des)
        }
        rev_idx = np.array([pair_lookup.get((d, o), -1) for o, d in zip(org, des)], dtype=np.int32)
        self.rev_idx = rev_idx
        self.has_rev = (rev_idx >= 0).astype(np.float32)

        # Group mappings for Entity (Org/Des airport) & Metro & Metro-pair
        unique_aps = {a: i for i, a in enumerate(fold_aps)}
        self.org_ap_idx = np.array([unique_aps[a] for a in org], dtype=np.int32)
        self.des_ap_idx = np.array([unique_aps[a] for a in des], dtype=np.int32)
        self.n_aps = len(fold_aps)

        unique_metros = sorted(set(org_metro).union(set(des_metro)))
        m_lookup = {m: i for i, m in enumerate(unique_metros)}
        self.org_m_idx = np.array([m_lookup[m] for m in org_metro], dtype=np.int32)
        self.des_m_idx = np.array([m_lookup[m] for m in des_metro], dtype=np.int32)
        self.n_metros = len(unique_metros)

        mp_keys = sorted(set(zip(org_metro, des_metro)))
        mp_lookup = {k: i for i, k in enumerate(mp_keys)}
        self.mp_idx = np.array([mp_lookup[(om, dm)] for om, dm in zip(org_metro, des_metro)], dtype=np.int32)
        self.n_mp = len(mp_keys)

        self._precompute_causal_matrices()
        self.cal_df = build_calendar_table(120)

    def _group_sum_matrix(self, mat: np.ndarray, group_idx: np.ndarray, n_groups: int) -> np.ndarray:
        """Sum (n_pairs, n_t) along groups -> (n_groups, n_t)."""
        out = np.zeros((n_groups, mat.shape[1]), dtype=np.float64)
        np.add.at(out, group_idx, mat)
        return out

    def _precompute_causal_matrices(self) -> None:
        P, R, O = self.P, self.R, self.O
        n_t = P.shape[1]

        # Causal rolling sums
        s3, c3 = _rolling_sum_2d(P, 3)
        s6, c6 = _rolling_sum_2d(P, 6)
        s12, c12 = _rolling_sum_2d(P, 12)
        r3, _ = _rolling_sum_2d(R, 3)
        r12, _ = _rolling_sum_2d(R, 12)
        o12, _ = _rolling_sum_2d(O, 12)

        self.pax_r3 = s3 / c3
        self.pax_r6 = s6 / c6
        self.pax_r12 = s12 / c12
        self.obs_r12 = o12 / c12

        # Rolling std over 12m
        sq12, _ = _rolling_sum_2d(P ** 2, 12)
        var12 = np.maximum(sq12 / c12 - self.pax_r12 ** 2, 0.0)
        self.pax_std12 = np.sqrt(var12)

        # Yield (USD / PAX) causal r3 & r12
        self.yield_r3 = r3 / np.maximum(s3, 1.0)
        self.yield_r12 = r12 / np.maximum(s12, 1.0)

        # 12m-lagged rolling means (valid for t >= 12)
        self.pax_r3_lag12 = np.full_like(self.pax_r3, np.nan)
        self.pax_r12_lag12 = np.full_like(self.pax_r12, np.nan)
        self.yield_r3_lag12 = np.full_like(self.yield_r3, np.nan)
        if n_t > 12:
            self.pax_r3_lag12[:, 12:] = self.pax_r3[:, :-12]
            self.pax_r12_lag12[:, 12:] = self.pax_r12[:, :-12]
            self.yield_r3_lag12[:, 12:] = self.yield_r3[:, :-12]

        # 2019 mean (only visible once s >= 11, or partial 2019 mean up to min(s, 11)!)
        # Strictly causal: at t <= 11, only months 0..t are observed; at t >= 11, all 12 months of 2019 are observed
        self.pax_2019_mean = np.zeros_like(P)
        cs19 = np.cumsum(P[:, :min(12, n_t)], axis=1)
        for t in range(n_t):
            k = min(t, 11)
            self.pax_2019_mean[:, t] = cs19[:, k] / float(k + 1)

        # Causal 2019 same-3-month reference for pax_r3 (matches calendar months of [t-2..t] in 2019, ONLY if t >= 11 or those months occurred <= t)
        # Note: for t >= 12, calendar months of t-2..t in 2019 all occurred strictly before t!
        self.pax_19_same3 = np.full_like(P, np.nan)
        for t in range(12, n_t):
            ms = [((t - k) % 12) for k in (0, 1, 2)]
            self.pax_19_same3[:, t] = np.mean(P[:, ms], axis=1)

        # Minimum 3m rolling avg up to t (trough depth)
        self.min_r3_cum = np.minimum.accumulate(self.pax_r3, axis=1)

        # Origin & Destination Airport Entity matrices
        org_mat = self._group_sum_matrix(P, self.org_ap_idx, self.n_aps)
        des_mat = self._group_sum_matrix(P, self.des_ap_idx, self.n_aps)
        org_r3_g, _ = _rolling_sum_2d(org_mat, 3)
        org_r12_g, c12_g = _rolling_sum_2d(org_mat, 12)
        des_r3_g, _ = _rolling_sum_2d(des_mat, 3)
        des_r12_g, _ = _rolling_sum_2d(des_mat, 12)

        org_r3_mean = org_r3_g / np.minimum(np.arange(1, n_t + 1), 3.0)
        org_r12_mean = org_r12_g / c12_g
        des_r3_mean = des_r3_g / np.minimum(np.arange(1, n_t + 1), 3.0)
        des_r12_mean = des_r12_g / c12_g

        self.pair_org_r3 = org_r3_mean[self.org_ap_idx]
        self.pair_org_r12 = org_r12_mean[self.org_ap_idx]
        self.pair_des_r3 = des_r3_mean[self.des_ap_idx]
        self.pair_des_r12 = des_r12_mean[self.des_ap_idx]

        # Org & Des 2019 same-3m reference (for t >= 12)
        self.pair_org_19_same3 = np.full_like(P, np.nan)
        self.pair_des_19_same3 = np.full_like(P, np.nan)
        for t in range(12, n_t):
            ms = [((t - k) % 12) for k in (0, 1, 2)]
            self.pair_org_19_same3[:, t] = np.mean(org_mat[:, ms], axis=1)[self.org_ap_idx]
            self.pair_des_19_same3[:, t] = np.mean(des_mat[:, ms], axis=1)[self.des_ap_idx]

        # Metro & Metro-pair matrices
        mp_mat = self._group_sum_matrix(P, self.mp_idx, self.n_mp)
        mp_r12_g, _ = _rolling_sum_2d(mp_mat, 12)
        self.pair_mp_r12 = (mp_r12_g / c12_g)[self.mp_idx]

        # Segment (Dom vs Intl) totals for causal segment recovery & baseline anchor
        dom_mask = self.is_dom > 0.5
        seg_p = np.vstack([P[~dom_mask].sum(axis=0), P[dom_mask].sum(axis=0)])  # row 0=intl, 1=dom
        seg_r3, seg_c3 = _rolling_sum_2d(seg_p, 3)
        seg_r3_mean = seg_r3 / seg_c3
        self.seg_19_same3 = np.full((2, n_t), np.nan)
        for t in range(12, n_t):
            ms = [((t - k) % 12) for k in (0, 1, 2)]
            self.seg_19_same3[:, t] = np.mean(seg_p[:, ms], axis=1)
        self.seg_rec_now = seg_r3_mean / np.maximum(self.seg_19_same3, 1.0)

    def build_features_for_slice(self, s: int, h: int) -> pd.DataFrame:
        """
        Build strictly zero-leakage feature DataFrame for all n_pairs at origin s predicting s + h.
        Uses ONLY P/R/O[:, :s+1] and deterministic calendar of (s + h).
        """
        if s < 0 or s > self.t_cut:
            raise ValueError(f"Origin s={s} out of [0, {self.t_cut}]")
        if h < 1 or h > 24:
            raise ValueError(f"Horizon h={h} out of [1, 24]")

        t_tgt = s + h
        m_tgt_idx = t_tgt % 12  # 0..11 for Jan..Dec

        # Most recent occurrence of target calendar month m_tgt <= s
        if h <= 12:
            t_last = t_tgt - 12
        else:
            t_last = t_tgt - 24
        pax_last = self.P[:, t_last] if t_last >= 0 else np.full(self.n_pairs, np.nan)
        rev_last = self.R[:, t_last] if t_last >= 0 else np.full(self.n_pairs, np.nan)
        t_prev2 = t_last - 12
        pax_prev2 = self.P[:, t_prev2] if t_prev2 >= 0 else np.full(self.n_pairs, np.nan)

        # 2019 same-month anchor ONLY if that 2019 month <= s
        pax_19_m = self.P[:, m_tgt_idx] if s >= m_tgt_idx else np.full(self.n_pairs, np.nan)

        # Reverse pair trailing stats at s
        rev_valid = self.rev_idx >= 0
        rev_r12 = np.where(rev_valid, self.pax_r12[np.maximum(self.rev_idx, 0), s], 0.0)
        rev_last_m = np.where(rev_valid, pax_last[np.maximum(self.rev_idx, 0)], 0.0)

        # Pair trailing stats at s
        p_s = self.P[:, s]
        p_r3 = self.pax_r3[:, s]
        p_r6 = self.pax_r6[:, s]
        p_r12 = self.pax_r12[:, s]
        p_std12 = self.pax_std12[:, s]
        p_19_mean = self.pax_2019_mean[:, s]

        # Ratios & stabilized logs
        rec_19_r3 = p_r3 / np.maximum(self.pax_19_same3[:, s], 5.0)
        rec_19_r12 = p_r12 / np.maximum(p_19_mean, 5.0)
        yoy_r3 = p_r3 / np.maximum(self.pax_r3_lag12[:, s], 5.0)
        yoy_r12 = p_r12 / np.maximum(self.pax_r12_lag12[:, s], 5.0)

        # Clean baseline anchor (mirrors 03_damped_trend_seasonal strictly at origin s)
        dom_int = self.is_dom.astype(np.int32)
        seg_rec_s = self.seg_rec_now[dom_int, s]
        # fallback if s < 12
        seg_rec_s = np.where(np.isfinite(seg_rec_s), seg_rec_s, 1.0)
        prior_strength = 1500.0
        p19_3_safe = np.where(np.isfinite(self.pax_19_same3[:, s]), self.pax_19_same3[:, s], p_19_mean)
        pair_rec_shrunk = np.clip((3.0 * p_r3 + prior_strength * seg_rec_s) / np.maximum(3.0 * p19_3_safe + prior_strength, 1.0), 0.05, 2.5)
        seg_eq = np.maximum(1.02, seg_rec_s)
        decay = np.exp(-float(h) / 12.0)
        tgt_mult = np.where(pair_rec_shrunk < 0.95, np.maximum(pair_rec_shrunk, seg_eq), pair_rec_shrunk * (1.0 + 0.015 * (h / 12.0)))
        mult = decay * pair_rec_shrunk + (1.0 - decay) * tgt_mult
        p19_base = np.where(np.isfinite(pax_19_m) & (pax_19_m > 0), pax_19_m, np.where(np.isfinite(pax_last), pax_last, p_r12))
        r12_base = np.where(np.isfinite(pax_last), pax_last, p_r12)
        y_from_19 = p19_base * mult
        y_from_r12 = r12_base * (1.0 + 0.015 * (h / 12.0))
        w_rec = np.where(pair_rec_shrunk >= 0.85, 0.50, 0.15)
        damped_anchor = np.where(p19_base > 0, (1.0 - w_rec) * y_from_19 + w_rec * y_from_r12, y_from_r12)
        damped_anchor = np.clip(damped_anchor, 0.0, None)

        cal_row = self.cal_df.iloc[t_tgt]

        data = {
            "HORIZON": np.full(self.n_pairs, float(h), dtype=np.float32),
            "LOG_H": np.full(self.n_pairs, float(np.log1p(h)), dtype=np.float32),
            "DEP_MONTH": np.full(self.n_pairs, float(cal_row["DEP_MONTH"]), dtype=np.float32),
            "DAYS_IN_MONTH": np.full(self.n_pairs, float(cal_row["DAYS_IN_MONTH"]), dtype=np.float32),
            "FRI_SUN_DAYS": np.full(self.n_pairs, float(cal_row["FRI_SUN_DAYS"]), dtype=np.float32),
            "EASTER_IN_MONTH": np.full(self.n_pairs, float(cal_row["EASTER_IN_MONTH"]), dtype=np.float32),
            "GOOD_FRI_IN_MONTH": np.full(self.n_pairs, float(cal_row["GOOD_FRI_IN_MONTH"]), dtype=np.float32),
            "TGIVING_LATE": np.full(self.n_pairs, float(cal_row["TGIVING_LATE"]), dtype=np.float32),
            "SIN_M": np.full(self.n_pairs, float(cal_row["SIN_M"]), dtype=np.float32),
            "COS_M": np.full(self.n_pairs, float(cal_row["COS_M"]), dtype=np.float32),
            "SIN_2M": np.full(self.n_pairs, float(cal_row["SIN_2M"]), dtype=np.float32),
            "COS_2M": np.full(self.n_pairs, float(cal_row["COS_2M"]), dtype=np.float32),
            # Pair level & lags
            "LOG_PAX_S": np.log1p(p_s).astype(np.float32),
            "LOG_PAX_R3": np.log1p(p_r3).astype(np.float32),
            "LOG_PAX_R6": np.log1p(p_r6).astype(np.float32),
            "LOG_PAX_R12": np.log1p(p_r12).astype(np.float32),
            "LOG_PAX_SAME_M_LAST": np.log1p(np.maximum(pax_last, 0.0)).astype(np.float32),
            "LOG_PAX_SAME_M_PREV2": np.log1p(np.maximum(pax_prev2, 0.0)).astype(np.float32),
            "LOG_PAX_SAME_M_2019": np.log1p(np.maximum(pax_19_m, 0.0)).astype(np.float32),
            "LOG_DAMPED_ANCHOR": np.log1p(damped_anchor).astype(np.float32),
            "DAMPED_ANCHOR": damped_anchor.astype(np.float32),
            "SEASONAL_IDX_19": (pax_19_m / np.maximum(p_19_mean, 5.0)).astype(np.float32),
            "SEASONAL_IDX_LAST": (pax_last / np.maximum(p_r12, 5.0)).astype(np.float32),
            "PAX_CV12": (p_std12 / np.maximum(p_r12, 1.0)).astype(np.float32),
            "OBS_RATE_12": self.obs_r12[:, s].astype(np.float32),
            "R3_TO_R12": (p_r3 / np.maximum(p_r12, 1.0)).astype(np.float32),
            "REC_19_R3": rec_19_r3.astype(np.float32),
            "REC_19_R12": rec_19_r12.astype(np.float32),
            "YOY_R3": yoy_r3.astype(np.float32),
            "YOY_R12": yoy_r12.astype(np.float32),
            "TROUGH_RATIO": (self.min_r3_cum[:, s] / np.maximum(p_19_mean, 5.0)).astype(np.float32),
            "PAIR_REC_SHRUNK": pair_rec_shrunk.astype(np.float32),
            "SEG_REC_NOW": seg_rec_s.astype(np.float32),
            # Reverse route & asymmetry
            "HAS_REV": self.has_rev,
            "LOG_REV_R12": np.log1p(rev_r12).astype(np.float32),
            "LOG_REV_SAME_M_LAST": np.log1p(np.maximum(rev_last_m, 0.0)).astype(np.float32),
            "DIR_ASYM_R12": ((p_r12 - rev_r12) / (p_r12 + rev_r12 + 10.0)).astype(np.float32),
            # Yield signals
            "YIELD_R3": np.log1p(self.yield_r3[:, s]).astype(np.float32),
            "YIELD_R12": np.log1p(self.yield_r12[:, s]).astype(np.float32),
            "YIELD_YOY_R3": (self.yield_r3[:, s] / np.maximum(self.yield_r3_lag12[:, s], 10.0)).astype(np.float32),
            "YIELD_LAST_M": np.log1p(np.maximum(rev_last, 0.0) / np.maximum(pax_last, 1.0)).astype(np.float32),
            # Entity (Airport) & Metro context
            "LOG_ORG_R12": np.log1p(self.pair_org_r12[:, s]).astype(np.float32),
            "LOG_DES_R12": np.log1p(self.pair_des_r12[:, s]).astype(np.float32),
            "ORG_R3_TO_R12": (self.pair_org_r3[:, s] / np.maximum(self.pair_org_r12[:, s], 1.0)).astype(np.float32),
            "DES_R3_TO_R12": (self.pair_des_r3[:, s] / np.maximum(self.pair_des_r12[:, s], 1.0)).astype(np.float32),
            "ORG_REC_19_R3": (self.pair_org_r3[:, s] / np.maximum(self.pair_org_19_same3[:, s], 10.0)).astype(np.float32),
            "DES_REC_19_R3": (self.pair_des_r3[:, s] / np.maximum(self.pair_des_19_same3[:, s], 10.0)).astype(np.float32),
            "SHARE_OF_ORG": (p_r12 / np.maximum(self.pair_org_r12[:, s], 1.0)).astype(np.float32),
            "SHARE_OF_DES": (p_r12 / np.maximum(self.pair_des_r12[:, s], 1.0)).astype(np.float32),
            "LOG_METRO_PAIR_R12": np.log1p(self.pair_mp_r12[:, s]).astype(np.float32),
            "SHARE_OF_METRO_PAIR": (p_r12 / np.maximum(self.pair_mp_r12[:, s], 1.0)).astype(np.float32),
        }
        for k, v in self.static_feats.items():
            data[k] = v

        # Interaction: snowbird seasonality (North-South lat diff * winter cosine)
        data["SNOWBIRD_INTERACT"] = (
            (self.static_feats["ORG_LAT"] - self.static_feats["DES_LAT"]) * data["COS_M"]
        ).astype(np.float32)

        return pd.DataFrame(data)


def assemble_training_dataset(
    store: FoldFeatureStore,
    s_min: int = 6,
    stride_s: int = 1,
    exclude_acute_covid_targets: bool = True,
    recency_halflife: float = 18.0,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble supervised (s, h) training samples where s + h <= t_cut.
    Returns (X_df, y_true, sample_weights, t_targets).
    Acute COVID target months (2020-03..2021-02 -> T_IDX 14..25) can be excluded or downweighted
    purely based on historical past target months <= t_cut.
    """
    t_cut = store.t_cut
    frames = []
    y_list = []
    w_list = []
    t_list = []

    for s in range(s_min, t_cut, stride_s):
        max_h = min(24, t_cut - s)
        for h in range(1, max_h + 1):
            t_tgt = s + h
            is_acute_covid = 14 <= t_tgt <= 25  # 2020-03 .. 2021-02
            if exclude_acute_covid_targets and is_acute_covid:
                continue
            df_slice = store.build_features_for_slice(s, h)
            y_slice = store.P[:, t_tgt].astype(np.float64)
            # Recency weight + gentle shock downweight if not excluded
            rec_w = float(np.exp(-(t_cut - t_tgt) / recency_halflife))
            if is_acute_covid:
                rec_w *= 0.05
            w_slice = np.full(store.n_pairs, rec_w, dtype=np.float32)
            frames.append(df_slice)
            y_list.append(y_slice)
            w_list.append(w_slice)
            t_list.append(np.full(store.n_pairs, t_tgt, dtype=np.int32))

    X = pd.concat(frames, ignore_index=True)
    y = np.concatenate(y_list)
    w = np.concatenate(w_list)
    t_tgts = np.concatenate(t_list)
    return X, y, w, t_tgts
