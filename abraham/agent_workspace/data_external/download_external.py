from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_eval.data_loader import load_airport_info, load_raw_pax

OUT_DIR = Path(__file__).resolve().parent
RAW_DIR = OUT_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ResearchForecastingPipeline/1.0)"}


def download_file(url: str, dest: Path, timeout: int = 60) -> bool:
    if dest.exists() and dest.stat().st_size > 200:
        print(f"  [cached] {dest.name} ({dest.stat().st_size:,} bytes)")
        return True
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"  [ok] {dest.name} ({len(r.content):,} bytes)")
        return True
    except Exception as e:
        print(f"  [fail] {dest.name}: {e}")
        return False


def parse_xlsx_first_sheet(xlsx_path: Path) -> pd.DataFrame:
    """Minimal stdlib xlsx reader (no openpyxl required)."""
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(xlsx_path, "r") as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root_s = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root_s.findall("main:si", ns):
                texts = [t.text or "" for t in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")]
                shared.append("".join(texts))
        sheet_names = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
        root_w = ET.fromstring(z.read(sheet_names[0]))
        rows = []
        for row in root_w.findall(".//main:sheetData/main:row", ns):
            vals = {}
            for c in row.findall("main:c", ns):
                ref = c.attrib.get("r", "")
                col_str = re.sub(r"[0-9]", "", ref)
                col_idx = 0
                for ch in col_str:
                    col_idx = col_idx * 26 + (ord(ch) - ord("A") + 1)
                t_type = c.attrib.get("t", "")
                v_el = c.find("main:v", ns)
                is_el = c.find("main:is", ns)
                val = ""
                if t_type == "inlineStr" and is_el is not None:
                    val = "".join(t.text or "" for t in is_el.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"))
                elif v_el is not None and v_el.text is not None:
                    if t_type == "s":
                        val = shared[int(v_el.text)]
                    else:
                        val = v_el.text
                vals[col_idx - 1] = val
            if vals:
                max_c = max(vals.keys())
                rows.append([vals.get(i, "") for i in range(max_c + 1)])
    return pd.DataFrame(rows)


def geocode_us_airports(ai_sub: pd.DataFrame) -> pd.DataFrame:
    """Use official US Census Geocoder API to map 100 US/PR airports -> State/County FIPS + CBSA."""
    cache_path = OUT_DIR / "us_airport_census_geo.csv"
    if cache_path.exists():
        print(f"  [cached] {cache_path.name}")
        return pd.read_csv(cache_path, dtype=str)

    us_pr = ai_sub[ai_sub["CNTRY_CD"].isin(["US", "PR"])].copy()
    records = []
    for _, r in us_pr.iterrows():
        url = (
            "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
            f"?x={r['LON_DD']:.6f}&y={r['LAT_DD']:.6f}"
            "&benchmark=Public_AR_Current&vintage=Current_Current&layers=Counties,Metropolitan%20Statistical%20Areas,Micropolitan%20Statistical%20Areas&format=json"
        )
        rec = {
            "AIRPRT_CD": r["AIRPRT_CD"],
            "CITY_METRO_IATA_CD": r["CITY_METRO_IATA_CD"],
            "STATE_PROVNC_CD": r["STATE_PROVNC_CD"],
            "CNTRY_CD": r["CNTRY_CD"],
            "COUNTY_FIPS": "",
            "COUNTY_NAME": "",
            "STATE_FIPS": "",
            "CBSA_CD": "",
            "CBSA_NAME": "",
        }
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20).json()
            geos = resp.get("result", {}).get("geographies", {})
            counties = geos.get("Counties", [])
            if counties:
                rec["COUNTY_FIPS"] = str(counties[0].get("GEOID", ""))
                rec["COUNTY_NAME"] = str(counties[0].get("NAME", ""))
                rec["STATE_FIPS"] = str(counties[0].get("STATE", ""))
            msas = geos.get("Metropolitan Statistical Areas", []) or geos.get("Micropolitan Statistical Areas", [])
            if msas:
                rec["CBSA_CD"] = str(msas[0].get("CBSA", msas[0].get("GEOID", "")))
                rec["CBSA_NAME"] = str(msas[0].get("NAME", ""))
        except Exception as e:
            print(f"  geocode warn {r['AIRPRT_CD']}: {e}")
        records.append(rec)
    out = pd.DataFrame(records)
    out.to_csv(cache_path, index=False)
    print(f"  [ok] Geocoded {len(out)} US/PR airports; matched county={(out['COUNTY_FIPS']!='').sum()}, CBSA={(out['CBSA_CD']!='').sum()}")
    return out


def fetch_faa_enplanements() -> None:
    """Discover & download FAA commercial service enplanement spreadsheets for CY 2018-2023."""
    pages = {
        2018: "https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/previous_years",
        2019: "https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/previous_years",
        2020: "https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/previous_years",
        2021: "https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger_boarding_previous_years/cy21-commercial-service-enplanements",
        2022: "https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger_boarding_previous_years/cy22-commercial-service-enplanements",
        2023: "https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger_boarding/cy23-commercial-service-enplanements",
    }
    discovered = {}
    for page_url in set(pages.values()):
        try:
            html = requests.get(page_url, headers=HEADERS, timeout=30).text
            for m in re.findall(r'href=["\']([^"\']+\.xlsx?)["\']', html, flags=re.IGNORECASE):
                full = m if m.startswith("http") else f"https://www.faa.gov{m}"
                low = full.lower()
                for yr in [2018, 2019, 2020, 2021, 2022, 2023]:
                    yy = str(yr)[-2:]
                    if f"cy{yy}" in low and ("commercial" in low or "enplanement" in low or "all" in low):
                        if yr not in discovered or "commercial" in low:
                            discovered[yr] = full
        except Exception as e:
            print(f"  FAA page warn {page_url}: {e}")

    print("  FAA discovered links:", discovered)
    all_faa = []
    for yr, url in sorted(discovered.items()):
        if not url.lower().endswith(".xlsx"):
            continue
        dest = RAW_DIR / f"faa_cy{yr}.xlsx"
        if download_file(url, dest):
            try:
                raw = parse_xlsx_first_sheet(dest)
                # Find header row containing Locid / LOCID / CY
                hdr_idx = None
                for i in range(min(15, len(raw))):
                    row_str = " ".join(str(x).upper() for x in raw.iloc[i].tolist())
                    if "LOCID" in row_str or "AIRPORT" in row_str and "RANK" in row_str:
                        hdr_idx = i
                        break
                if hdr_idx is None:
                    continue
                cols = [str(c).strip() for c in raw.iloc[hdr_idx].tolist()]
                df_y = raw.iloc[hdr_idx + 1:].copy()
                df_y.columns = cols
                loc_col = next((c for c in cols if "LOCID" in c.upper()), None)
                enp_col = next((c for c in cols if f"CY {str(yr)[-2:]}" in c.upper() or f"CY{str(yr)[-2:]}" in c.upper() or "ENPLANEMENT" in c.upper()), None)
                if loc_col and enp_col:
                    sub = df_y[[loc_col, enp_col]].copy()
                    sub.columns = ["AIRPRT_CD", "FAA_ENPLANEMENTS"]
                    sub["AIRPRT_CD"] = sub["AIRPRT_CD"].astype(str).str.strip()
                    sub["FAA_ENPLANEMENTS"] = pd.to_numeric(sub["FAA_ENPLANEMENTS"].astype(str).str.replace(",", ""), errors="coerce")
                    sub["YEAR"] = yr
                    # Conservative publication lag: CY Y published ~Oct of Y+1
                    sub["AVAILABLE_YM"] = (yr + 1) * 100 + 10
                    all_faa.append(sub.dropna(subset=["FAA_ENPLANEMENTS"]))
            except Exception as e:
                print(f"  FAA parse warn {yr}: {e}")
    if all_faa:
        faa_df = pd.concat(all_faa, ignore_index=True)
        faa_df.to_csv(OUT_DIR / "faa_enplanements_clean.csv", index=False)
        print(f"  [ok] Saved faa_enplanements_clean.csv: {len(faa_df)} rows across years {sorted(faa_df['YEAR'].unique())}")


def fetch_census_pop_and_migration(geo_df: pd.DataFrame) -> None:
    # 1. Census CBSA & State population estimates
    urls = {
        "cbsa_est2019.csv": "https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/metro/totals/cbsa-est2019-alldata.csv",
        "cbsa_est2023.csv": "https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/metro/totals/cbsa-est2023-alldata.csv",
        "cbsa_est2024.csv": "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/metro/totals/cbsa-est2024-alldata.csv",
        "nst_est2019.csv": "https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/state/totals/nst-est2019-alldata.csv",
        "nst_est2024.csv": "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/state/totals/NST-EST2024-ALLDATA.csv",
        "co_est2023.csv": "https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/counties/totals/co-est2023-alldata.csv",
    }
    for fname, url in urls.items():
        download_file(url, RAW_DIR / fname)

    # 2. OpenDataSoft US County Boundaries summary (no geometry bloat - select fields csv)
    ods_url = (
        "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/us-county-boundaries/"
        "exports/csv?select=geoid,statefp,countyfp,name,stusab,state_name,aland,awater,intptlat,intptlon,cbsafp,csafp"
    )
    download_file(ods_url, RAW_DIR / "us_county_boundaries_attrs.csv", timeout=90)

    # 3. Census County-to-County Migration Flows (2011-2015 ACS - published 2017/18, strictly pre-2019!)
    # Query per unique airport county FIPS via Census API
    mig_path = OUT_DIR / "census_county_migration_2015.csv"
    if not mig_path.exists():
        fips_list = sorted(set(geo_df.loc[geo_df["COUNTY_FIPS"].str.len() == 5, "COUNTY_FIPS"]))
        mig_rows = []
        for fips in fips_list:
            st, cty = fips[:2], fips[2:]
            api_url = (
                f"https://api.census.gov/data/2015/acs/flows?get=GEOID1,GEOID2,MOVEDIN,MOVEDOUT,MOVEDNET"
                f"&STATE1={st}&COUNTY1={cty}"
            )
            try:
                r = requests.get(api_url, headers=HEADERS, timeout=20)
                if r.status_code == 200:
                    data = r.json()
                    cols = data[0]
                    for row in data[1:]:
                        d = dict(zip(cols, row))
                        if d.get("GEOID2") in fips_list:
                            mig_rows.append(d)
            except Exception:
                pass
        if mig_rows:
            mig_df = pd.DataFrame(mig_rows)
            for c in ["MOVEDIN", "MOVEDOUT", "MOVEDNET"]:
                mig_df[c] = pd.to_numeric(mig_df[c], errors="coerce").fillna(0.0)
            mig_df["AVAILABLE_YM"] = 201801  # Published 2017/18, valid for all 2019+ origins
            mig_df.to_csv(mig_path, index=False)
            print(f"  [ok] Saved {mig_path.name}: {len(mig_df)} inter-airport-county migration pairs")
        else:
            print("  [warn] No migration rows returned from 2015 ACS flows API")


def fetch_fred_and_worldbank(ai_sub: pd.DataFrame) -> None:
    # FRED monthly series
    fred_ids = ["CPIAUCSL", "UNRATE", "DSPIC96", "DCOILWTICO", "UMCSENT", "AIRRPMTSI", "ENPLANE"]
    fred_frames = []
    for sid in fred_ids:
        dest = RAW_DIR / f"fred_{sid}.csv"
        if download_file(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", dest):
            try:
                f = pd.read_csv(dest)
                date_col = f.columns[0]
                val_col = f.columns[1]
                f["DATE"] = pd.to_datetime(f[date_col], errors="coerce")
                f["VAL"] = pd.to_numeric(f[val_col], errors="coerce")
                f = f.dropna(subset=["DATE", "VAL"])
                f["YM"] = (f["DATE"].dt.year * 100 + f["DATE"].dt.month).astype(int)
                # Aggregate daily (oil) to monthly mean
                fm = f.groupby("YM", as_index=False)["VAL"].mean().rename(columns={"VAL": sid})
                # Conservative 2-month publication lag for macro releases
                fm["YEAR_M"] = fm["YM"] // 100
                fm["MON_M"] = fm["YM"] % 100
                # add 2 months for AVAILABLE_YM
                idx = (fm["YEAR_M"] - 2000) * 12 + (fm["MON_M"] - 1) + 2
                fm["AVAILABLE_YM"] = (2000 + idx // 12) * 100 + (idx % 12 + 1)
                fred_frames.append(fm[["YM", "AVAILABLE_YM", sid]])
            except Exception as e:
                print(f"  FRED parse warn {sid}: {e}")
    if fred_frames:
        base = fred_frames[0]
        for fm in fred_frames[1:]:
            base = base.merge(fm.drop(columns=["AVAILABLE_YM"]), on="YM", how="outer")
        base = base.sort_values("YM")
        # Recompute AVAILABLE_YM conservatively as +2 months
        y = base["YM"] // 100
        m = base["YM"] % 100
        idx = (y - 2000) * 12 + (m - 1) + 2
        base["AVAILABLE_YM"] = (2000 + idx // 12) * 100 + (idx % 12 + 1)
        base.to_csv(OUT_DIR / "fred_macro_monthly.csv", index=False)
        print(f"  [ok] Saved fred_macro_monthly.csv: {len(base)} months")

    # World Bank API for 35 countries (GDP per capita & Population, 2015-2023)
    wb_path = OUT_DIR / "worldbank_country_annual.csv"
    if not wb_path.exists():
        countries = sorted(set(ai_sub["CNTRY_CD"]))
        # PR is covered by WB as PRI or US; ISO2 works directly for WB v2
        codes_str = ";".join(c for c in countries if len(c) == 2 and c != "TW")
        rows = []
        for ind, col_name in [("NY.GDP.PCAP.CD", "WB_GDP_PCAP"), ("SP.POP.TOTL", "WB_POP")]:
            url = f"https://api.worldbank.org/v2/country/{codes_str}/indicator/{ind}?date=2016:2024&format=json&per_page=2000"
            try:
                r = requests.get(url, headers=HEADERS, timeout=30).json()
                if isinstance(r, list) and len(r) > 1 and r[1]:
                    for item in r[1]:
                        if item.get("value") is not None:
                            rows.append({
                                "CNTRY_CD": item["country"]["id"],
                                "YEAR": int(item["date"]),
                                "IND": col_name,
                                "VAL": float(item["value"]),
                            })
            except Exception as e:
                print(f"  WB warn {ind}: {e}")
        if rows:
            wb = pd.DataFrame(rows).pivot_table(index=["CNTRY_CD", "YEAR"], columns="IND", values="VAL").reset_index()
            # Conservative 12-month publication lag (year Y published July Y+1)
            wb["AVAILABLE_YM"] = (wb["YEAR"] + 1) * 100 + 7
            wb.to_csv(wb_path, index=False)
            print(f"  [ok] Saved worldbank_country_annual.csv: {len(wb)} rows")


def main() -> None:
    ai = load_airport_info()
    pax = load_raw_pax()
    aps = sorted(set(pax["ORG_AIRPRT"]).union(set(pax["DES_AIRPRT"])))
    ai_sub = ai[ai["AIRPRT_CD"].isin(aps)].copy()

    print("1. Geocoding US/PR airports via US Census Geocoder...")
    geo_df = geocode_us_airports(ai_sub)

    print("2. Fetching FAA Airport Enplanements...")
    fetch_faa_enplanements()

    print("3. Fetching US Census Population, County Boundaries & Migration Flows...")
    fetch_census_pop_and_migration(geo_df)

    print("4. Fetching FRED Macro & World Bank Country Series...")
    fetch_fred_and_worldbank(ai_sub)

    print("External download step complete.")


if __name__ == "__main__":
    main()
