"""Measure how the same seasonal rule changes as data becomes more current."""
import argparse
from pathlib import Path

import pandas as pd

from demand_deep_dive import HORIZONS, ROUTE, shock_months


DELAYS = (1, 6, 12, 18, 24)


def anchor(eligible, month, shocks):
    seasons = eligible.loc[:, eligible.columns.month == month].iloc[:, ::-1]
    retained = seasons.where(seasons.notna().cumsum(axis=1).le(3))
    latest = retained.bfill(axis=1).iloc[:, 0]
    flagged = pd.Series(False, index=eligible.index)
    for date in seasons.columns[::-1]:
        flagged.loc[seasons[date].notna()] = date in shocks
    return latest, latest.where(~(flagged & retained.count(axis=1).ge(2)), retained.mean(axis=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    demand = pd.read_csv(args.data_dir / "AA_PAX_FORECASTING.csv", usecols=ROUTE + [
        "DEP_YEAR", "DEP_MONTH", "OD_PAX",
    ])
    demand["month"] = pd.PeriodIndex.from_fields(
        year=demand.DEP_YEAR, month=demand.DEP_MONTH, freq="M"
    )
    pax = demand.pivot(index=ROUTE, columns="month", values="OD_PAX").sort_index(axis=1)
    records = []
    for window, start, end, horizons in (
        ("full_horizon", "2024-01", "2024-07", HORIZONS),
        ("later_near_term", "2024-08", "2025-07", (1, 6, 12)),
    ):
        for issue in pd.period_range(start, end, freq="M"):
            histories = {delay: pax.loc[:, :issue - delay] for delay in DELAYS}
            shocks = {delay: shock_months(eligible) for delay, eligible in histories.items()}
            for horizon in horizons:
                target = issue + horizon
                predictions = {delay: anchor(eligible, target.month, shocks[delay])
                               for delay, eligible in histories.items()}
                actual = pax[target]
                # The oldest eligible forecast sample is shared across every delay.
                matched = actual.notna() & predictions[max(DELAYS)][0].notna()
                observed = actual[matched]
                for delay, (latest, guarded) in predictions.items():
                    for name, forecast in (("seasonal", latest), ("shock_fallback", guarded)):
                        error = forecast[matched] - observed
                        if error.isna().any():
                            raise ValueError("A shorter delay lost a route in the common sample.")
                        records.append({
                            "window": window, "issue": str(issue), "horizon": horizon,
                            "delay": delay, "method": name, "scored": int(matched.sum()),
                            "actual_sum": observed.sum(), "absolute_error": error.abs().sum(),
                            "signed_error": error.sum(),
                        })
        print(f"Completed lag sensitivity for {window}.", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_dir / "lag_sensitivity_metrics.csv", index=False)


if __name__ == "__main__":
    main()
