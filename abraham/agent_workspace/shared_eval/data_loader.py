from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
GLOBAL_DATA_DIR = WORKSPACE_ROOT.parent / "global_data"
AIRPORT_INFO_PATH = GLOBAL_DATA_DIR / "airport_info.csv"
PAX_FORECASTING_PATH = GLOBAL_DATA_DIR / "pax_forecasting.csv"

MIN_YM = 201901
MAX_YM = 202607
MAX_PRETEST_YM = 202407  # End of pre-holdout window; 2024-08..2026-07 is untouched test


def ym_to_idx(ym: int | np.ndarray | pd.Series) -> int | np.ndarray | pd.Series:
    """Convert YYYYMM integer to 0-based month index starting at 2019-01 = 0."""
    year = ym // 100
    month = ym % 100
    return (year - 2019) * 12 + (month - 1)


def idx_to_ym(idx: int) -> int:
    """Convert 0-based month index (2019-01 = 0) back to YYYYMM integer."""
    year = 2019 + idx // 12
    month = (idx % 12) + 1
    return int(year * 100 + month)


ALL_YMS = [idx_to_ym(i) for i in range(ym_to_idx(MAX_YM) + 1)]


def load_airport_info() -> pd.DataFrame:
    """Load read-only airport_info.csv and compute clean decimal coordinates & region labels."""
    ai = pd.read_csv(AIRPORT_INFO_PATH, sep="|", index_col=0, keep_default_na=False)
    lat = (
        ai["LAT_DEGREE_QTY"]
        + ai["LAT_MINUTE_QTY"] / 60.0
        + ai["LAT_SECOND_QTY"] / 3600.0
    ) * np.where(ai["LAT_HEMSPHR_CD"] == "N", 1.0, -1.0)
    lon = (
        ai["LNGTD_DEGREE_QTY"]
        + ai["LNGTD_MINUTE_QTY"] / 60.0
        + ai["LNGTD_SECOND_QTY"] / 3600.0
    ) * np.where(ai["LNGTD_HEMSPHR_CD"] == "E", 1.0, -1.0)
    ai["LAT_DD"] = lat.astype(np.float64)
    ai["LON_DD"] = lon.astype(np.float64)
    ai["IS_US"] = ai["CNTRY_CD"] == "US"
    ai["REGION_CD"] = np.where(ai["IS_US"], ai["STATE_PROVNC_CD"], ai["CNTRY_CD"])
    return ai


def load_raw_pax() -> pd.DataFrame:
    """Load read-only pax_forecasting.csv with YYYYMM and month index columns."""
    df = pd.read_csv(
        PAX_FORECASTING_PATH,
        sep="|",
        index_col=0,
        dtype={
            "DEP_MONTH": np.int32,
            "DEP_YEAR": np.int32,
            "ORG_AIRPRT": str,
            "DES_AIRPRT": str,
            "OD_PAX": np.float64,
            "OD_REV_USD": np.float64,
        },
    )
    df["YM"] = (df["DEP_YEAR"] * 100 + df["DEP_MONTH"]).astype(np.int32)
    df["T_IDX"] = ym_to_idx(df["YM"]).astype(np.int32)
    return df


def haversine_miles(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance in statute miles."""
    r_miles = 3958.7613
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r_miles * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_balanced_panel(
    raw_df: pd.DataFrame,
    origin_ym: int,
    target_end_ym: int | None = None,
    include_targets: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a strictly zero-leakage balanced panel.

    - Pair universe is defined ONLY from rows observed at YM <= origin_ym.
    - History table covers 2019-01 .. origin_ym with absent route-months filled as 0.0.
    - If include_targets=True and target_end_ym is provided, returns a SEPARATE
      target_df covering (origin_ym, target_end_ym] for evaluation only, with absent
      route-months filled as 0.0 (and OD_REV_USD stripped from targets to avoid misuse).
    """
    hist_raw = raw_df.loc[raw_df["YM"] <= origin_ym].copy()
    pairs_df = (
        hist_raw[["ORG_AIRPRT", "DES_AIRPRT"]]
        .drop_duplicates()
        .sort_values(["ORG_AIRPRT", "DES_AIRPRT"])
        .reset_index(drop=True)
    )
    pairs_df["PAIR_ID"] = np.arange(len(pairs_df), dtype=np.int32)

    origin_idx = int(ym_to_idx(origin_ym))
    hist_t = np.arange(origin_idx + 1, dtype=np.int32)
    n_pairs = len(pairs_df)

    # Vectorized Cartesian product of pairs x history months
    hist_panel = pd.DataFrame({
        "PAIR_ID": np.repeat(pairs_df["PAIR_ID"].to_numpy(), len(hist_t)),
        "ORG_AIRPRT": np.repeat(pairs_df["ORG_AIRPRT"].to_numpy(), len(hist_t)),
        "DES_AIRPRT": np.repeat(pairs_df["DES_AIRPRT"].to_numpy(), len(hist_t)),
        "T_IDX": np.tile(hist_t, n_pairs),
    })
    hist_panel["YM"] = np.array([idx_to_ym(int(t)) for t in hist_panel["T_IDX"]], dtype=np.int32)
    hist_panel["DEP_YEAR"] = (hist_panel["YM"] // 100).astype(np.int32)
    hist_panel["DEP_MONTH"] = (hist_panel["YM"] % 100).astype(np.int32)

    hist_raw_marked = hist_raw[["ORG_AIRPRT", "DES_AIRPRT", "T_IDX", "OD_PAX", "OD_REV_USD"]].copy()
    hist_raw_marked["IS_OBSERVED"] = np.int8(1)

    hist_panel = hist_panel.merge(
        hist_raw_marked,
        on=["ORG_AIRPRT", "DES_AIRPRT", "T_IDX"],
        how="left",
    )
    hist_panel["OD_PAX"] = hist_panel["OD_PAX"].fillna(0.0).astype(np.float64)
    hist_panel["OD_REV_USD"] = hist_panel["OD_REV_USD"].fillna(0.0).astype(np.float64)
    hist_panel["IS_OBSERVED"] = hist_panel["IS_OBSERVED"].fillna(0).astype(np.int8)

    if not include_targets or target_end_ym is None:
        return hist_panel, pd.DataFrame()

    target_end_idx = int(ym_to_idx(target_end_ym))
    if target_end_idx <= origin_idx:
        raise ValueError(f"target_end_ym ({target_end_ym}) must exceed origin_ym ({origin_ym})")

    tgt_t = np.arange(origin_idx + 1, target_end_idx + 1, dtype=np.int32)
    tgt_panel = pd.DataFrame({
        "PAIR_ID": np.repeat(pairs_df["PAIR_ID"].to_numpy(), len(tgt_t)),
        "ORG_AIRPRT": np.repeat(pairs_df["ORG_AIRPRT"].to_numpy(), len(tgt_t)),
        "DES_AIRPRT": np.repeat(pairs_df["DES_AIRPRT"].to_numpy(), len(tgt_t)),
        "T_IDX": np.tile(tgt_t, n_pairs),
        "HORIZON": np.tile(np.arange(1, len(tgt_t) + 1, dtype=np.int32), n_pairs),
    })
    tgt_panel["YM"] = np.array([idx_to_ym(int(t)) for t in tgt_panel["T_IDX"]], dtype=np.int32)
    tgt_panel["DEP_YEAR"] = (tgt_panel["YM"] // 100).astype(np.int32)
    tgt_panel["DEP_MONTH"] = (tgt_panel["YM"] % 100).astype(np.int32)

    tgt_raw = raw_df.loc[(raw_df["YM"] > origin_ym) & (raw_df["YM"] <= target_end_ym), ["ORG_AIRPRT", "DES_AIRPRT", "T_IDX", "OD_PAX"]].copy()
    tgt_raw["IS_OBSERVED"] = np.int8(1)
    tgt_panel = tgt_panel.merge(tgt_raw, on=["ORG_AIRPRT", "DES_AIRPRT", "T_IDX"], how="left")
    tgt_panel["OD_PAX"] = tgt_panel["OD_PAX"].fillna(0.0).astype(np.float64)
    tgt_panel["IS_OBSERVED"] = tgt_panel["IS_OBSERVED"].fillna(0).astype(np.int8)
    tgt_panel.rename(columns={"OD_PAX": "y_true"}, inplace=True)

    return hist_panel, tgt_panel
