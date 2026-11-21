from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared_eval.data_loader import load_airport_info, load_raw_pax, build_balanced_panel
from shared_eval.splits import BACKTEST_FOLDS


def main() -> None:
    ai = load_airport_info()
    df = load_raw_pax()
    print(f"Airport info rows: {len(ai)}, raw pax rows: {len(df)}")

    ai_sub = ai[ai["AIRPRT_CD"].isin(set(df["ORG_AIRPRT"]).union(set(df["DES_AIRPRT"])))].copy()
    m_map = ai_sub.set_index("AIRPRT_CD")["CITY_METRO_IATA_CD"].to_dict()
    same_metro = df[df["ORG_AIRPRT"].map(m_map) == df["DES_AIRPRT"].map(m_map)]
    print("Same-metro pairs:", same_metro.groupby(["ORG_AIRPRT", "DES_AIRPRT"])["OD_PAX"].sum().to_dict())

    undir_pairs = set((min(a, b), max(a, b)) for a, b in zip(df["ORG_AIRPRT"], df["DES_AIRPRT"]))
    dir_metro = set(zip(df["ORG_AIRPRT"].map(m_map), df["DES_AIRPRT"].map(m_map)))
    print(f"Unique directional pairs: {df.groupby(['ORG_AIRPRT','DES_AIRPRT']).ngroups}")
    print(f"Unique undirected pairs: {len(undir_pairs)}")
    print(f"Unique directional metro pairs: {len(dir_metro)}")

    r_map = ai_sub.set_index("AIRPRT_CD")["REGION_CD"].to_dict()
    us_ap = set(ai_sub[ai_sub["IS_US"]]["AIRPRT_CD"])
    is_dom = df["ORG_AIRPRT"].isin(us_ap) & df["DES_AIRPRT"].isin(us_ap)
    same_reg = df["ORG_AIRPRT"].map(r_map) == df["DES_AIRPRT"].map(r_map)
    print(f"US interstate pax share: {df.loc[is_dom & ~same_reg, 'OD_PAX'].sum() / df['OD_PAX'].sum():.4f}")
    print(f"US intrastate pax share: {df.loc[is_dom & same_reg, 'OD_PAX'].sum() / df['OD_PAX'].sum():.4f}")
    print(f"US-Intl/PR pax share: {df.loc[~is_dom, 'OD_PAX'].sum() / df['OD_PAX'].sum():.4f}")
    print(f"US interstate pairs: {df.loc[is_dom & ~same_reg].groupby([df['ORG_AIRPRT'].map(r_map), df['DES_AIRPRT'].map(r_map)]).ngroups}")
    print(f"All region pairs: {df.groupby([df['ORG_AIRPRT'].map(r_map), df['DES_AIRPRT'].map(r_map)]).ngroups}")

    for fold in BACKTEST_FOLDS:
        hist, tgt = build_balanced_panel(df, fold.origin_ym, fold.target_end_ym, include_targets=True)
        print(
            f"{fold.name}: origin={fold.origin_ym}, target={fold.target_start_ym}..{fold.target_end_ym}, "
            f"hist_rows={len(hist)}, tgt_rows={len(tgt)}, pairs={tgt[['ORG_AIRPRT','DES_AIRPRT']].drop_duplicates().shape[0]}"
        )


if __name__ == "__main__":
    main()
