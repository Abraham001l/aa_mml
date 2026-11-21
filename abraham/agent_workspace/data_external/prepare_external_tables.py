from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_eval.data_loader import load_airport_info, load_raw_pax

EXT_DIR = Path(__file__).resolve().parent
RAW_DIR = EXT_DIR / "raw"


def build_airport_point_in_time_table() -> pd.DataFrame:
    """
    Build annual external features per US/PR airport with strict AVAILABLE_YM publication dates.
    """
    geo = pd.read_csv(EXT_DIR / "us_airport_census_geo.csv", dtype=str).fillna("")
    # Map 5-digit county FIPS and CBSA
    geo["COUNTY_FIPS"] = geo["COUNTY_FIPS"].str.zfill(5)
    geo["STATE_FIPS"] = geo["STATE_FIPS"].str.zfill(2)
    geo["CBSA_CD"] = geo["CBSA_CD"].astype(str).str.strip()

    # 1. County Boundaries land/water area from OpenDataSoft
    cb = pd.read_csv(RAW_DIR / "us_county_boundaries_attrs.csv", sep=";", dtype=str).fillna("")
    cb["COUNTY_FIPS"] = cb["geoid"].str.zfill(5)
    cb["COUNTY_ALAND"] = pd.to_numeric(cb["aland"], errors="coerce")
    cb["COUNTY_AWATER"] = pd.to_numeric(cb["awater"], errors="coerce")
    geo = geo.merge(cb[["COUNTY_FIPS", "COUNTY_ALAND", "COUNTY_AWATER"]].drop_duplicates("COUNTY_FIPS"), on="COUNTY_FIPS", how="left")

    # 2. CBSA Population & Domestic Migration (2018-2019 from cbsa_est2019, 2020-2023 from cbsa_est2023)
    c19 = pd.read_csv(RAW_DIR / "cbsa_est2019.csv", encoding="latin1")
    c19 = c19[c19["STCOU"].isna()].copy()  # Metro/Micro area total rows only
    c19["CBSA_CD"] = c19["CBSA"].astype(str).str.strip()

    c23 = pd.read_csv(RAW_DIR / "cbsa_est2023.csv", encoding="latin1")
    c23 = c23[c23["STCOU"].isna()].copy()
    c23["CBSA_CD"] = c23["CBSA"].astype(str).str.strip()

    # Also state-level domestic migration & pop from nst_est2024 (using 2020..2023 ONLY, pub lag May Y+1)
    nst = pd.read_csv(RAW_DIR / "nst_est2024.csv", encoding="latin1")
    nst["STATE_FIPS"] = nst["STATE"].astype(int).astype(str).str.zfill(2)

    # 3. County population 2020-2023 from co_est2023
    co23 = pd.read_csv(RAW_DIR / "co_est2023.csv", encoding="latin1")
    co23["COUNTY_FIPS"] = co23["STATE"].astype(int).astype(str).str.zfill(2) + co23["COUNTY"].astype(int).astype(str).str.zfill(3)
    co23 = co23[co23["COUNTY"] > 0].drop_duplicates("COUNTY_FIPS")

    # 4. FAA Enplanements (2018..2021 + 2023, using exact AVAILABLE_YM)
    faa = pd.read_csv(EXT_DIR / "faa_enplanements_clean.csv")

    records = []
    # Years 2018..2023 (2024 excluded as pub > 2024-07!)
    for _, g in geo.iterrows():
        ap = g["AIRPRT_CD"]
        cbsa = g["CBSA_CD"]
        fips = g["COUNTY_FIPS"]
        st_fips = g["STATE_FIPS"]
        aland_km2 = float(g["COUNTY_ALAND"]) / 1e6 if pd.notna(g["COUNTY_ALAND"]) and g["COUNTY_ALAND"] > 0 else np.nan

        c19_row = c19[c19["CBSA_CD"] == cbsa]
        c23_row = c23[c23["CBSA_CD"] == cbsa]
        co_row = co23[co23["COUNTY_FIPS"] == fips]
        st_row = nst[nst["STATE_FIPS"] == st_fips]

        for yr in [2018, 2019, 2020, 2021, 2022, 2023]:
            avail_ym = (yr + 1) * 100 + 5  # Conservative May Y+1 release for July-1 estimates
            cbsa_pop = np.nan
            cbsa_mig_rate = np.nan
            cbsa_pop_yoy = np.nan
            if yr <= 2019 and len(c19_row):
                cbsa_pop = float(c19_row.iloc[0][f"POPESTIMATE{yr}"])
                prev_p = float(c19_row.iloc[0][f"POPESTIMATE{yr-1}"])
                cbsa_pop_yoy = cbsa_pop / prev_p - 1.0
                cbsa_mig_rate = float(c19_row.iloc[0][f"DOMESTICMIG{yr}"]) / max(cbsa_pop, 1.0) * 1000.0
            elif yr >= 2020 and len(c23_row):
                cbsa_pop = float(c23_row.iloc[0][f"POPESTIMATE{yr}"])
                if yr == 2020 and len(c19_row):
                    prev_p = float(c19_row.iloc[0]["POPESTIMATE2019"])
                elif yr > 2020:
                    prev_p = float(c23_row.iloc[0][f"POPESTIMATE{yr-1}"])
                else:
                    prev_p = np.nan
                if np.isfinite(prev_p) and prev_p > 0:
                    cbsa_pop_yoy = cbsa_pop / prev_p - 1.0
                if f"RDOMESTICMIG{yr}" in c23_row.columns and yr >= 2021:
                    cbsa_mig_rate = float(c23_row.iloc[0][f"RDOMESTICMIG{yr}"])
                elif f"DOMESTICMIG{yr}" in c23_row.columns:
                    cbsa_mig_rate = float(c23_row.iloc[0][f"DOMESTICMIG{yr}"]) / max(cbsa_pop, 1.0) * 1000.0

            cty_pop = float(co_row.iloc[0][f"POPESTIMATE{yr}"]) if (yr >= 2020 and len(co_row) and f"POPESTIMATE{yr}" in co_row.columns) else np.nan
            cty_dens = cty_pop / aland_km2 if (np.isfinite(cty_pop) and np.isfinite(aland_km2) and aland_km2 > 0) else np.nan
            st_mig = float(st_row.iloc[0][f"RDOMESTICMIG{yr}"]) if (yr >= 2021 and len(st_row) and f"RDOMESTICMIG{yr}" in st_row.columns) else np.nan

            records.append({
                "AIRPRT_CD": ap,
                "YEAR": yr,
                "AVAILABLE_YM": avail_ym,
                "CBSA_CD": cbsa,
                "COUNTY_FIPS": fips,
                "CBSA_POP": cbsa_pop,
                "CBSA_POP_YOY": cbsa_pop_yoy,
                "CBSA_DOM_MIG_RATE": cbsa_mig_rate,
                "COUNTY_POP_DENSITY": cty_dens,
                "STATE_DOM_MIG_RATE": st_mig,
            })

    out = pd.DataFrame(records)
    # Merge FAA enplanements respecting exact FAA AVAILABLE_YM
    faa_sub = faa[["AIRPRT_CD", "YEAR", "FAA_ENPLANEMENTS", "AVAILABLE_YM"]].rename(
        columns={"AVAILABLE_YM": "FAA_AVAILABLE_YM"}
    )
    out = out.merge(faa_sub, on=["AIRPRT_CD", "YEAR"], how="left")
    out.to_parquet(EXT_DIR / "airport_external_annual.parquet", index=False)
    print(f"Saved airport_external_annual.parquet: {len(out)} rows, cols={list(out.columns)}")
    return out


def build_county_migration_pairs() -> pd.DataFrame:
    """Parse 2011-2015 Census CtyxCty_US.txt strictly for our 95 US/PR airport counties."""
    geo = pd.read_csv(EXT_DIR / "us_airport_census_geo.csv", dtype=str).fillna("")
    geo["COUNTY_FIPS"] = geo["COUNTY_FIPS"].str.zfill(5)
    fips_set = set(geo.loc[geo["COUNTY_FIPS"].str.len() == 5, "COUNTY_FIPS"])

    county_flows: dict[tuple[str, str], float] = {}
    with open(RAW_DIR / "CtyxCty_US.txt", "r", encoding="latin1") as f:
        for line in f:
            if len(line) < 30:
                continue
            f_cur = line[1:3] + line[3:6]
            f_prev = line[7:9] + line[9:12]
            if f_cur in fips_set and f_prev in fips_set:
                toks = line.split()
                try:
                    est = float(toks[-2])
                except ValueError:
                    est = 0.0
                # f_prev -> f_cur movers
                county_flows[(f_prev, f_cur)] = county_flows.get((f_prev, f_cur), 0.0) + est

    # Map to all directional airport pairs in global data
    raw_df = load_raw_pax()
    pairs = raw_df[["ORG_AIRPRT", "DES_AIRPRT"]].drop_duplicates().copy()
    ap_to_fips = geo.set_index("AIRPRT_CD")["COUNTY_FIPS"].to_dict()

    mig_fwd = []
    mig_rev = []
    for o, d in zip(pairs["ORG_AIRPRT"], pairs["DES_AIRPRT"]):
        fo = ap_to_fips.get(o, "")
        fd = ap_to_fips.get(d, "")
        mig_fwd.append(county_flows.get((fo, fd), 0.0) if (fo and fd) else 0.0)
        mig_rev.append(county_flows.get((fd, fo), 0.0) if (fo and fd) else 0.0)

    pairs["MIG_ORG_TO_DES"] = mig_fwd
    pairs["MIG_DES_TO_ORG"] = mig_rev
    pairs["MIG_TOTAL"] = pairs["MIG_ORG_TO_DES"] + pairs["MIG_DES_TO_ORG"]
    pairs["AVAILABLE_YM"] = 201801  # 2011-2015 ACS released 2017/18 (strictly pre-2019)
    pairs.to_parquet(EXT_DIR / "pair_migration_2015.parquet", index=False)
    print(f"Saved pair_migration_2015.parquet: {len(pairs)} pairs, non-zero={(pairs['MIG_TOTAL'] > 0).sum()}")
    return pairs


def build_macro_monthly_table() -> pd.DataFrame:
    df = pd.read_csv(EXT_DIR / "fred_macro_monthly.csv")
    df = df.sort_values("YM").reset_index(drop=True)
    # Compute causal trailing 12m YoY changes at observation YM (still released at AVAILABLE_YM = YM + 2m)
    for col in ["CPIAUCSL", "DSPIC96", "DCOILWTICO", "AIRRPMTSI", "ENPLANE"]:
        df[f"{col}_YOY"] = df[col] / df[col].shift(12) - 1.0
    df["UNRATE_DIFF12"] = df["UNRATE"] - df["UNRATE"].shift(12)
    df = df[df["YM"] >= 201801].copy()
    df.to_parquet(EXT_DIR / "macro_monthly_clean.parquet", index=False)
    print(f"Saved macro_monthly_clean.parquet: {len(df)} rows")
    return df


if __name__ == "__main__":
    build_airport_point_in_time_table()
    build_county_migration_pairs()
    build_macro_monthly_table()
