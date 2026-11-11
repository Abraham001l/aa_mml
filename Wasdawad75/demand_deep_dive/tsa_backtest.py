"""Bridge stale U.S. domestic route history using public TSA screening totals."""
import argparse
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from demand_deep_dive import DELAYS, HORIZONS, ROUTE, score_records, shock_months
from faa_backtest import forecasts


def screening_series(directory, last_year=2025):
    pieces = []
    for year in range(2019, last_year + 1):
        raw = (directory / f"{year}.html").read_text(encoding="utf-8")
        table, = [frame for frame in pd.read_html(StringIO(raw))
                  if {"Date", "Numbers"}.issubset(frame.columns)]
        dates = pd.to_datetime(table.Date, format="%m/%d/%Y")
        if not dates.dt.year.eq(year).all():
            raise ValueError(f"The TSA page for {year} contains dates from another year.")
        screened = pd.to_numeric(table.Numbers.astype(str).str.replace(",", ""), errors="raise")
        pieces.append(pd.Series(screened.to_numpy(), index=pd.DatetimeIndex(dates)))
    daily = pd.concat(pieces).sort_index()
    if not daily.index.is_unique or daily.le(0).any() or daily.isna().any():
        raise ValueError("Expected positive screening counts and unique dates.")
    months = daily.groupby(daily.index.to_period("M"))
    levels = months.mean()
    return levels.loc[months.size().eq(levels.index.days_in_month)]


def monthly_bridge(screened, issue):
    # Use only complete months ending before the issue month.
    available = screened.loc[:issue - 1]
    recent_by_season = available.groupby(available.index.month).last()
    return {month: np.clip(recent_by_season.loc[month.month] / volume, .5, 2.0)
            for month, volume in available.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tsa-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    airports = pd.read_csv(args.data_dir / "AA_AIRPORT_INFO.csv")
    us = airports.loc[airports.CNTRY_CD.eq("US"), "AIRPRT_CD"]
    demand = pd.read_csv(args.data_dir / "AA_PAX_FORECASTING.csv",
                         usecols=ROUTE + ["DEP_YEAR", "DEP_MONTH", "OD_PAX"])
    demand["month"] = pd.PeriodIndex.from_fields(
        year=demand.DEP_YEAR, month=demand.DEP_MONTH, freq="M"
    )
    panel = demand.pivot(index=ROUTE, columns="month", values="OD_PAX").sort_index(axis=1)
    domestic = panel.index.get_level_values(0).isin(us) & panel.index.get_level_values(1).isin(us)
    pax = panel.loc[domestic]
    screened = screening_series(args.tsa_dir)
    print(f"TSA archive has {len(screened)} complete months.")
    scores = []
    windows = (
        ("disruption_recovery", "2022-07", "2023-12", HORIZONS),
        ("full_horizon", "2024-01", "2024-07", HORIZONS),
        ("later_near_term", "2024-08", "2025-07", (1, 6, 12)),
    )
    for delay in DELAYS:
        for window, start, end, horizons in windows:
            for issue in pd.period_range(start, end, freq="M"):
                factors = monthly_bridge(screened, issue)
                eligible = pax.loc[:, :issue - delay]
                shocks = shock_months(panel.loc[:, :issue - delay])
                for horizon in horizons:
                    target = issue + horizon
                    predictions = forecasts(eligible, target, shocks, factors, proxy="tsa")
                    if predictions is None:
                        continue
                    changed = ~np.isclose(predictions.tsa_full, predictions.shock_fallback,
                                          equal_nan=True)
                    groups = pd.DataFrame({"tsa_adjustment": np.where(changed, "applied", "same")},
                                          index=pax.index)
                    scores.extend(score_records(issue, horizon, delay, window, pax[target],
                                                predictions, groups))
            print(f"Completed TSA {window}, delay {delay}.", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).to_csv(args.output_dir / "tsa_backtest_metrics.csv", index=False)


if __name__ == "__main__":
    main()

