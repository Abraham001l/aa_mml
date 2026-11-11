"""Compare national and airport-level public demand proxies on the same routes."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from demand_deep_dive import (
    DELAYS, EVALUATION_WINDOWS, HOLDOUT_WINDOWS, ROUTE, score_records, shock_months,
)
from faa_backtest import RELEASES, faa_boardings, forecasts, route_growth
from tsa_backtest import monthly_bridge, screening_series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--faa-dir", type=Path, required=True)
    parser.add_argument("--tsa-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--late-holdout", action="store_true")
    args = parser.parse_args()

    airports = pd.read_csv(args.data_dir / "AA_AIRPORT_INFO.csv").set_index("AIRPRT_CD")
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
    screened = screening_series(args.tsa_dir, last_year=2026 if args.late_holdout else 2025)
    windows = HOLDOUT_WINDOWS if args.late_holdout else EVALUATION_WINDOWS
    rows = []
    for delay in DELAYS:
        for window, start, end, horizons in windows:
            for issue in pd.period_range(start, end, freq="M"):
                tsa = monthly_bridge(screened, issue)
                known = [year for year, released in RELEASES.items()
                         if pd.Timestamp(released) <= issue.end_time]
                newest = max(known)
                faa, relative = {}, {}
                for year in known:
                    if year == newest:
                        continue
                    counts = pd.concat([boardings[year], boardings[newest]], axis=1).dropna()
                    national = counts.iloc[:, 1].sum() / counts.iloc[:, 0].sum()
                    faa[year] = route_growth(airports, pax.index, boardings, year, newest)
                    relative[year] = (faa[year] / national).clip(.75, 4 / 3).fillna(1)
                cutoff = issue - delay
                eligible = pax.loc[:, :cutoff]
                shocks = shock_months(panel.loc[:, :cutoff])
                mixed = {month: factor * relative.get(month.year, 1.0)
                         for month, factor in tsa.items()}
                deadband = {month: np.exp(np.sign(np.log(factor)) *
                             max(abs(np.log(factor)) - np.log(1.05), 0))
                            for month, factor in tsa.items()}
                for horizon in horizons:
                    target = issue + horizon
                    predictions = forecasts(eligible, target, shocks, tsa, proxy="tsa")
                    if predictions is None:
                        continue
                    predictions["faa_half"] = forecasts(eligible, target, shocks, faa).faa_half
                    predictions["tsa_faa_local"] = forecasts(
                        eligible, target, shocks, mixed, proxy="mixed"
                    ).mixed_half
                    predictions["tsa_deadband"] = forecasts(
                        eligible, target, shocks, deadband, proxy="deadband"
                    ).deadband_half
                    groups = pd.DataFrame(index=pax.index)
                    rows.extend(score_records(issue, horizon, delay, window, pax[target],
                                              predictions, groups))
            print(f"Completed combined proxy {window}, delay {delay}.", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "public_proxy_metrics.csv", index=False)


if __name__ == "__main__":
    main()
