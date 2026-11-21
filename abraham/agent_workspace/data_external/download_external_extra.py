from __future__ import annotations

from pathlib import Path
import re
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_external.download_external import download_file, parse_xlsx_first_sheet
from shared_eval.data_loader import load_airport_info, load_raw_pax

OUT_DIR = Path(__file__).resolve().parent
RAW_DIR = OUT_DIR / "raw"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ResearchForecastingPipeline/1.0)"}


def update_faa_enplanements() -> None:
    extra = {
        2021: ("https://www.faa.gov/sites/faa.gov/files/2022-09/cy21-commercial-service-enplanements.xlsx", 202209),
        2022: ("https://www.faa.gov/sites/faa.gov/files/2023-08/cy22-commercial-service-enplanements.xlsx", 202308),
        2023: ("https://www.faa.gov/sites/faa.gov/files/2024-10/cy23-commercial-service-enplanements.xlsx", 202410),
    }
    frames = []
    clean_path = OUT_DIR / "faa_enplanements_clean.csv"
    if clean_path.exists():
        old = pd.read_csv(clean_path)
        old = old[~old["YEAR"].isin([2021, 2022, 2023])]
        frames.append(old)

    for yr, (url, avail_ym) in extra.items():
        dest = RAW_DIR / f"faa_cy{yr}.xlsx"
        if download_file(url, dest):
            raw = parse_xlsx_first_sheet(dest)
            hdr_idx = None
            for i in range(min(15, len(raw))):
                row_str = " ".join(str(x).upper() for x in raw.iloc[i].tolist())
                if "LOCID" in row_str or ("AIRPORT" in row_str and "RANK" in row_str):
                    hdr_idx = i
                    break
            if hdr_idx is None:
                continue
            cols = [str(c).strip() for c in raw.iloc[hdr_idx].tolist()]
            df_y = raw.iloc[hdr_idx + 1:].copy()
            df_y.columns = cols
            loc_col = next((c for c in cols if "LOCID" in c.upper()), None)
            enp_col = next(
                (c for c in cols if f"CY {str(yr)[-2:]}" in c.upper() or f"CY{str(yr)[-2:]}" in c.upper() or "ENPLANEMENT" in c.upper()),
                None,
            )
            if loc_col and enp_col:
                sub = df_y[[loc_col, enp_col]].copy()
                sub.columns = ["AIRPRT_CD", "FAA_ENPLANEMENTS"]
                sub["AIRPRT_CD"] = sub["AIRPRT_CD"].astype(str).str.strip()
                sub["FAA_ENPLANEMENTS"] = pd.to_numeric(
                    sub["FAA_ENPLANEMENTS"].astype(str).str.replace(",", ""), errors="coerce"
                )
                sub["YEAR"] = yr
                sub["AVAILABLE_YM"] = avail_ym
                frames.append(sub.dropna(subset=["FAA_ENPLANEMENTS"]))

    if frames:
        faa_df = pd.concat(frames, ignore_index=True)
        faa_df.to_csv(clean_path, index=False)
        print("FAA years now:", sorted(faa_df["YEAR"].unique()), "rows:", len(faa_df))


def fetch_census_state_2019() -> None:
    download_file(
        "https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/state/totals/NST-EST2019-ALLDATA.csv",
        RAW_DIR / "nst_est2019.csv",
    )


def fetch_migration_bulk() -> None:
    """Discover and download Census 2011-2015 county-to-county migration flows spreadsheet/txt."""
    mig_out = OUT_DIR / "census_county_migration_2015.csv"
    if mig_out.exists():
        print("Migration file already exists")
        return

    geo_df = pd.read_csv(OUT_DIR / "us_airport_census_geo.csv", dtype=str)
    fips_set = set(geo_df.loc[geo_df["COUNTY_FIPS"].str.len() == 5, "COUNTY_FIPS"])

    # Try official 2011-2015 migration page to locate county-to-county flow file
    page = "https://www.census.gov/data/tables/2015/demo/geographic-mobility/county-to-county-migration-2011-2015.html"
    try:
        html = requests.get(page, headers=HEADERS, timeout=30).text
        links = re.findall(r'href=["\']([^"\']*(?:county-to-county|migration)[^"\']*\.(?:xlsx|xls|zip|csv|txt))["\']', html, flags=re.I)
        print("Discovered 2011-2015 migration links:", links[:8])
        for lnk in links:
            full = lnk if lnk.startswith("http") else f"https:{lnk}" if lnk.startswith("//") else f"https://www.census.gov{lnk}"
            if "by-county" in full.lower() or "all" in full.lower() or "county-to-county" in full.lower():
                fname = Path(full.split("?")[0]).name
                dest = RAW_DIR / fname
                if download_file(full, dest, timeout=90) and fname.lower().endswith(".xlsx"):
                    # Parse xlsx using zipfile if needed or inspect sheet names
                    print("Downloaded migration xlsx:", dest)
                    break
    except Exception as e:
        print("Migration page warn:", e)

    # Also test proper Census API syntax (2015/acs/flows uses STATE1 & COUNTY1 with 3-digit county and 2-digit state; test one sample!)
    try:
        r = requests.get(
            "https://api.census.gov/data/2015/acs/flows?get=STATE1,COUNTY1,STATE2,COUNTY2,MOVEDIN,MOVEDOUT,MOVEDNET&for=county:037&in=state:06",
            headers=HEADERS,
            timeout=20,
        )
        print("Sample Census API 2015 flows status:", r.status_code, r.text[:200])
        if r.status_code == 200:
            rows = []
            # Group unique (state_fips, county_3digit) by state_fips -> only ~40 API calls instead of 100!
            st_to_ctys: dict[str, set[str]] = {}
            for fips in fips_set:
                st_to_ctys.setdefault(fips[:2], set()).add(fips[2:])
            for st, ctys in sorted(st_to_ctys.items()):
                cty_arg = ",".join(sorted(ctys))
                u = (
                    f"https://api.census.gov/data/2015/acs/flows?get=STATE1,COUNTY1,STATE2,COUNTY2,MOVEDIN,MOVEDOUT,MOVEDNET"
                    f"&for=county:{cty_arg}&in=state:{st}"
                )
                try:
                    rr = requests.get(u, headers=HEADERS, timeout=25)
                    if rr.status_code == 200:
                        data = rr.json()
                        hdr = data[0]
                        for row in data[1:]:
                            d = dict(zip(hdr, row))
                            g1 = str(d.get("STATE1", "")).zfill(2) + str(d.get("COUNTY1", "")).zfill(3)
                            g2 = str(d.get("STATE2", "")).zfill(2) + str(d.get("COUNTY2", "")).zfill(3)
                            if g1 in fips_set and g2 in fips_set:
                                rows.append({
                                    "GEOID1": g1,
                                    "GEOID2": g2,
                                    "MOVEDIN": pd.to_numeric(d.get("MOVEDIN"), errors="coerce"),
                                    "MOVEDOUT": pd.to_numeric(d.get("MOVEDOUT"), errors="coerce"),
                                    "MOVEDNET": pd.to_numeric(d.get("MOVEDNET"), errors="coerce"),
                                })
                except Exception:
                    pass
            if rows:
                mig_df = pd.DataFrame(rows).fillna(0.0)
                mig_df["AVAILABLE_YM"] = 201801
                mig_df.to_csv(mig_out, index=False)
                print(f"[ok] Saved {mig_out.name}: {len(mig_df)} airport-county migration pairs!")
    except Exception as e:
        print("Census API flows warn:", e)


def fetch_worldbank_chunked() -> None:
    wb_path = OUT_DIR / "worldbank_country_annual.csv"
    if wb_path.exists():
        print("World Bank already exists")
        return
    ai = load_airport_info()
    pax = load_raw_pax()
    aps = sorted(set(pax["ORG_AIRPRT"]).union(set(pax["DES_AIRPRT"])))
    countries = sorted(set(ai[ai["AIRPRT_CD"].isin(aps)]["CNTRY_CD"]))
    codes = [c for c in countries if len(c) == 2 and c not in ("TW",)]

    rows = []
    for ind, col_name in [("NY.GDP.PCAP.CD", "WB_GDP_PCAP"), ("SP.POP.TOTL", "WB_POP")]:
        for i in range(0, len(codes), 8):
            chunk = ";".join(codes[i:i + 8])
            url = f"https://api.worldbank.org/v2/country/{chunk}/indicator/{ind}?date=2016:2024&format=json&per_page=1000"
            for attempt in range(2):
                try:
                    r = requests.get(url, headers=HEADERS, timeout=35).json()
                    if isinstance(r, list) and len(r) > 1 and r[1]:
                        for item in r[1]:
                            if item.get("value") is not None:
                                rows.append({
                                    "CNTRY_CD": item["country"]["id"],
                                    "YEAR": int(item["date"]),
                                    "IND": col_name,
                                    "VAL": float(item["value"]),
                                })
                    break
                except Exception as e:
                    print(f"  WB chunk warn ({ind}, {chunk}): {e}")
                    time.sleep(1)
    if rows:
        wb = pd.DataFrame(rows).pivot_table(index=["CNTRY_CD", "YEAR"], columns="IND", values="VAL").reset_index()
        wb["AVAILABLE_YM"] = (wb["YEAR"] + 1) * 100 + 7
        wb.to_csv(wb_path, index=False)
        print(f"[ok] Saved worldbank_country_annual.csv: {len(wb)} rows across {wb['CNTRY_CD'].nunique()} countries")


if __name__ == "__main__":
    update_faa_enplanements()
    fetch_census_state_2019()
    fetch_migration_bulk()
    fetch_worldbank_chunked()
