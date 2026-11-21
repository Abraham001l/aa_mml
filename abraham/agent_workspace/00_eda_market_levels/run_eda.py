from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_eval.data_loader import build_balanced_panel, haversine_miles, load_airport_info, load_raw_pax


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_data_dir = Path(__file__).resolve().parent / "model_specific_data"
    model_data_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_pax()
    # Strictly pre-holdout EDA only!
    pre_df = raw_df.loc[raw_df["YM"] <= 202407].copy()
    hist_panel, _ = build_balanced_panel(pre_df, origin_ym=202407, include_targets=False)
    ai = load_airport_info()

    ap_meta = ai.set_index("AIRPRT_CD")
    hist_panel["ORG_CNTRY"] = hist_panel["ORG_AIRPRT"].map(ap_meta["CNTRY_CD"])
    hist_panel["DES_CNTRY"] = hist_panel["DES_AIRPRT"].map(ap_meta["CNTRY_CD"])
    hist_panel["ORG_METRO"] = hist_panel["ORG_AIRPRT"].map(ap_meta["CITY_METRO_IATA_CD"])
    hist_panel["DES_METRO"] = hist_panel["DES_AIRPRT"].map(ap_meta["CITY_METRO_IATA_CD"])
    hist_panel["ORG_REG"] = hist_panel["ORG_AIRPRT"].map(ap_meta["REGION_CD"])
    hist_panel["DES_REG"] = hist_panel["DES_AIRPRT"].map(ap_meta["REGION_CD"])
    hist_panel["IS_DOM"] = (hist_panel["ORG_CNTRY"] == "US") & (hist_panel["DES_CNTRY"] == "US")

    # 1. Yearly/seasonal network totals (2019-2024.07)
    ann_seg = (
        hist_panel.groupby(["DEP_YEAR", "IS_DOM"], as_index=False)
        .agg(
            months=("DEP_MONTH", "nunique"),
            pax=("OD_PAX", "sum"),
            rev=("OD_REV_USD", "sum"),
            obs_rate=("IS_OBSERVED", "mean"),
        )
    )
    ann_seg["avg_monthly_pax"] = ann_seg["pax"] / ann_seg["months"]
    ann_seg["yield"] = ann_seg["rev"] / np.maximum(ann_seg["pax"], 1.0)
    ann_seg.to_csv(model_data_dir / "pretest_annual_segment_summary.csv", index=False)

    # 2. Same-month YoY stability across 2022-08..2024-07 vs 2019 at different levels
    # Compare seasonal profile and CoV at Market / Metro / State / Airport levels
    steady = hist_panel.loc[hist_panel["YM"] >= 202208].copy()

    def level_yoy_stats(df_lvl: pd.DataFrame, keys: list[str], name: str) -> dict:
        df_l = df_lvl.groupby(keys + ["DEP_YEAR", "DEP_MONTH", "YM"], as_index=False)["OD_PAX"].sum()
        df_l = df_l.sort_values(keys + ["YM"])
        df_l["PAX_LAG12"] = df_l.groupby(keys)["OD_PAX"].shift(12)
        df_l["PAX_LAG24"] = df_l.groupby(keys)["OD_PAX"].shift(24)
        sub12 = df_l.dropna(subset=["PAX_LAG12"])
        sub24 = df_l.dropna(subset=["PAX_LAG24"])
        wape_12 = float(np.sum(np.abs(sub12["OD_PAX"] - sub12["PAX_LAG12"])) / np.sum(sub12["OD_PAX"]))
        wape_24 = float(np.sum(np.abs(sub24["OD_PAX"] - sub24["PAX_LAG24"])) / np.sum(sub24["OD_PAX"]))
        return {"level": name, "n_entities": int(df_l[keys].drop_duplicates().shape[0]), "yoy_12m_wape": wape_12, "yoy_24m_wape": wape_24}

    lvl_stats = [
        level_yoy_stats(steady, ["ORG_AIRPRT", "DES_AIRPRT"], "market_dir_airport"),
        level_yoy_stats(
            steady.assign(
                AP_MIN=np.minimum(steady["ORG_AIRPRT"], steady["DES_AIRPRT"]),
                AP_MAX=np.maximum(steady["ORG_AIRPRT"], steady["DES_AIRPRT"]),
            ),
            ["AP_MIN", "AP_MAX"],
            "market_undir_airport",
        ),
        level_yoy_stats(steady, ["ORG_METRO", "DES_METRO"], "market_dir_metro"),
        level_yoy_stats(
            steady.loc[steady["IS_DOM"] & (steady["ORG_REG"] != steady["DES_REG"])],
            ["ORG_REG", "DES_REG"],
            "state_us_interstate",
        ),
        level_yoy_stats(steady, ["ORG_AIRPRT"], "entity_org_airport"),
        level_yoy_stats(steady, ["DES_AIRPRT"], "entity_des_airport"),
    ]
    lvl_df = pd.DataFrame(lvl_stats)
    lvl_df.to_csv(out_dir / "pretest_level_predictability.csv", index=False)

    # 3. Distance, directional asymmetry, and 2019->2023/24 structural shift by tier/region
    p19 = hist_panel.loc[hist_panel["DEP_YEAR"] == 2019].groupby(["ORG_AIRPRT", "DES_AIRPRT"], as_index=False)["OD_PAX"].sum().rename(columns={"OD_PAX": "PAX_2019"})
    p23 = hist_panel.loc[hist_panel["DEP_YEAR"] == 2023].groupby(["ORG_AIRPRT", "DES_AIRPRT"], as_index=False)["OD_PAX"].sum().rename(columns={"OD_PAX": "PAX_2023"})
    p_comp = p19.merge(p23, on=["ORG_AIRPRT", "DES_AIRPRT"])
    p_comp["ORG_LAT"] = p_comp["ORG_AIRPRT"].map(ap_meta["LAT_DD"])
    p_comp["ORG_LON"] = p_comp["ORG_AIRPRT"].map(ap_meta["LON_DD"])
    p_comp["DES_LAT"] = p_comp["DES_AIRPRT"].map(ap_meta["LAT_DD"])
    p_comp["DES_LON"] = p_comp["DES_AIRPRT"].map(ap_meta["LON_DD"])
    p_comp["DIST_MI"] = haversine_miles(p_comp["ORG_LAT"].values, p_comp["ORG_LON"].values, p_comp["DES_LAT"].values, p_comp["DES_LON"].values)
    p_comp["IS_DOM"] = (p_comp["ORG_AIRPRT"].map(ap_meta["CNTRY_CD"]) == "US") & (p_comp["DES_AIRPRT"].map(ap_meta["CNTRY_CD"]) == "US")
    p_comp["RATIO_23_19"] = p_comp["PAX_2023"] / np.maximum(p_comp["PAX_2019"], 1.0)
    p_comp.to_csv(model_data_dir / "pretest_2019_vs_2023_pairs.csv", index=False)

    # 4. Zero / non-observed rate by month
    obs_by_ym = hist_panel.groupby("YM", as_index=False).agg(obs_rate=("IS_OBSERVED", "mean"), total_pax=("OD_PAX", "sum"))
    obs_by_ym.to_csv(out_dir / "pretest_monthly_network_totals.csv", index=False)

    summary = {
        "max_ym_analyzed": int(hist_panel["YM"].max()),
        "annual_segment": ann_seg.to_dict(orient="records"),
        "level_yoy_202208_202407": lvl_stats,
        "dist_stats": {
            "min_miles": float(p_comp["DIST_MI"].min()),
            "median_miles": float(p_comp["DIST_MI"].median()),
            "max_miles": float(p_comp["DIST_MI"].max()),
        },
        "recovery_23_vs_19_dom_ratio": float(p_comp.loc[p_comp["IS_DOM"], "PAX_2023"].sum() / p_comp.loc[p_comp["IS_DOM"], "PAX_2019"].sum()),
        "recovery_23_vs_19_intl_ratio": float(p_comp.loc[~p_comp["IS_DOM"], "PAX_2023"].sum() / p_comp.loc[~p_comp["IS_DOM"], "PAX_2019"].sum()),
    }
    with open(out_dir / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Pre-Holdout EDA (<= 2024-07) ===")
    print(ann_seg.to_string(index=False))
    print("\n=== Same-Month Naïve Predictability (2022-08..2024-07) by Level ===")
    print(lvl_df.to_string(index=False))
    print(
        f"\n2023 vs 2019 Volume Ratio: Domestic={summary['recovery_23_vs_19_dom_ratio']:.4f}, "
        f"Intl/PR={summary['recovery_23_vs_19_intl_ratio']:.4f}"
    )


if __name__ == "__main__":
    main()
