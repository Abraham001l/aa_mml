"""Test a release-dated FAA boarding adjustment on U.S. domestic markets."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from demand_deep_dive import DELAYS, HORIZONS, ROUTE, score_records, shock_months


# Conservative availability dates from FAA archive and file pages.
RELEASES = {
    2019: "2020-10-29",
    2020: "2021-11-08",
    2021: "2022-09-16",
    2022: "2023-09-05",
    2023: "2024-10-04",
}


def faa_boardings(directory):
    tables = {}
    for year in RELEASES:
        file = directory / f"{year}.xlsx"
        value = f"CY {year % 100:02d} Enplanements"
        table = pd.read_excel(file, usecols=["ST", "Locid", value])
        table[value] = pd.to_numeric(table[value], errors="coerce")
        table = table.loc[table.Locid.astype(str).str.fullmatch("[A-Z]{3}") & table[value].gt(0)]
        counts = table.set_index(["Locid", "ST"])[value]
        if not counts.index.is_unique:
            raise ValueError(f"Duplicate FAA airport and state keys in {file}.")
        tables[year] = counts
    return tables


def route_growth(airports, routes, boardings, previous, current):
    ratio = boardings[current].div(boardings[previous])
    endpoints = []
    for column in ROUTE:
        codes = routes.get_level_values(column)
        states = codes.map(airports.STATE_PROVNC_CD)
        keys = pd.MultiIndex.from_arrays([codes, states])
        endpoints.append(ratio.reindex(keys).to_numpy())
    return pd.Series(np.sqrt(endpoints[0] * endpoints[1]).clip(.5, 2), index=routes)


def forecasts(eligible, target, shocks, factors, proxy="faa"):
    seasons = eligible.loc[:, eligible.columns.month == target.month].iloc[:, ::-1]
    if not len(seasons.columns):
        return None
    latest = seasons.bfill(axis=1).iloc[:, 0]
    retained = seasons.where(seasons.notna().cumsum(axis=1).le(3))
    count = retained.count(axis=1)
    latest_flagged = pd.Series(False, index=eligible.index)
    for month in seasons.columns[::-1]:
        latest_flagged.loc[seasons[month].notna()] = month in shocks
    switched = latest_flagged & count.ge(2)
    # On an unflagged route, only the first observed month contributes.
    first = retained.notna().cumsum(axis=1).eq(1) & retained.notna()
    switch_mask = np.broadcast_to(switched.to_numpy()[:, None], retained.shape)
    reference = retained.where(first | switch_mask)
    baseline = reference.mean(axis=1)
    adjusted = {}
    for name, power in ((f"{proxy}_half", .5), (f"{proxy}_full", 1)):
        scale = pd.DataFrame({month: factors.get(month, factors.get(month.year, 1.0))
                              for month in reference.columns}, index=reference.index)
        scale = scale.fillna(1.0).pow(power)
        adjusted[name] = reference.mul(scale).mean(axis=1)
    return pd.DataFrame({"seasonal": latest, "shock_fallback": baseline, **adjusted})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--faa-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    airports = pd.read_csv(args.data_dir / "AA_AIRPORT_INFO.csv").set_index("AIRPRT_CD")
    if not airports.index.is_unique:
        raise ValueError("Expected unique airport codes.")
    us = airports.index[airports.CNTRY_CD.eq("US")]
    demand = pd.read_csv(args.data_dir / "AA_PAX_FORECASTING.csv",
                         usecols=ROUTE + ["DEP_YEAR", "DEP_MONTH", "OD_PAX"])
    demand["month"] = pd.PeriodIndex.from_fields(
        year=demand.DEP_YEAR, month=demand.DEP_MONTH, freq="M"
    )
    panel = demand.pivot(index=ROUTE, columns="month", values="OD_PAX").sort_index(axis=1)
    domestic = panel.index.get_level_values(0).isin(us) & panel.index.get_level_values(1).isin(us)
    pax = panel.loc[domestic]
    boardings = faa_boardings(args.faa_dir)
    for year, values in boardings.items():
        print(f"FAA vintage {year}, eligible by {RELEASES[year]}: {len(values):,} commercial airports.")
    scores = []
    windows = (
        ("disruption_recovery", "2022-07", "2023-12", HORIZONS),
        ("full_horizon", "2024-01", "2024-07", HORIZONS),
        ("later_near_term", "2024-08", "2025-07", (1, 6, 12)),
    )
    for delay in DELAYS:
        for window, start, end, horizons in windows:
            for issue in pd.period_range(start, end, freq="M"):
                known = [year for year, date in RELEASES.items()
                         if pd.Timestamp(date) <= issue.end_time]
                newest = max(known)
                factors = {year: route_growth(airports, pax.index, boardings, year, newest)
                           for year in known if year < newest}
                # Flag shocks using all routes, as in the main benchmark.
                shocks = shock_months(panel.loc[:, :issue - delay])
                eligible = pax.loc[:, :issue - delay]
                for horizon in horizons:
                    target = issue + horizon
                    predictions = forecasts(eligible, target, shocks, factors)
                    if predictions is None:
                        continue
                    moved = ~np.isclose(predictions.faa_full, predictions.shock_fallback,
                                       equal_nan=True)
                    groups = pd.DataFrame({"faa_adjustment": np.where(moved, "applied", "same")},
                                          index=pax.index)
                    scores.extend(score_records(issue, horizon, delay, window, pax[target],
                                                predictions, groups))
            print(f"Completed FAA {window}, delay {delay}.", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).to_csv(args.output_dir / "faa_backtest_metrics.csv", index=False)


if __name__ == "__main__":
    main()


