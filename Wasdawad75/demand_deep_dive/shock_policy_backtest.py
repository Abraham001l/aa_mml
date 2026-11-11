"""Compare fixed policies for low seasonal references without future data."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from demand_deep_dive import DELAYS, HORIZONS, load_data, score_records, shock_months


def policy_predictions(seasons, global_shocks, regional_shocks, geography):
    # Newest first. The first three nonmissing observations are eligible per route.
    retained = seasons.where(seasons.notna().cumsum(axis=1).le(3))
    count = retained.count(axis=1)
    latest = retained.bfill(axis=1).iloc[:, 0]
    mean = retained.mean(axis=1)
    reference_flag = pd.Series(False, index=seasons.index)
    regional_flag = reference_flag.copy()
    for month in seasons.columns[::-1]:
        observed = seasons[month].notna()
        reference_flag.loc[observed] = month in global_shocks
        regional_flag.loc[observed] = geography[observed].map(
            {name: month in flags for name, flags in regional_shocks.items()}
        ).fillna(False)
    switched = reference_flag & count.ge(2)
    baseline = latest.where(~switched, mean)

    clean = seasons.drop(columns=global_shocks, errors="ignore")
    clean = clean.bfill(axis=1).iloc[:, 0] if len(clean.columns) else latest * np.nan
    omit = baseline.where(~(switched & clean.notna()), clean)
    regional = latest.where(~((reference_flag | regional_flag) & count.ge(2)), mean)

    first = retained.notna().cumsum(axis=1).eq(1) & retained.notna()
    older = retained.where(~first)
    old_min, old_max = older.min(axis=1), older.max(axis=1)
    route_flag = (count.eq(3) & old_max.gt(0) & old_min.ge(.70 * old_max)
                  & latest.le(.65 * older.mean(axis=1)))
    local = baseline.where(~(route_flag & ~switched), mean)
    groups = pd.DataFrame({"policy_trigger": np.where(
        switched, "global_shock",
        np.where(route_flag | regional_flag, "local_only", "none"),
    )}, index=seasons.index)
    return pd.DataFrame({
        "seasonal": latest, "shock_fallback": baseline, "omit_shock": omit,
        "regional_shock": regional, "route_low_guard": local,
    }), groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pax, _, airport = load_data(args.data_dir)
    geography = airport.geography
    windows = (
        ("disruption_recovery", "2022-07", "2023-12", HORIZONS),
        ("full_horizon", "2024-01", "2024-07", HORIZONS),
        ("later_near_term", "2024-08", "2025-07", (1, 6, 12)),
    )
    rows = []
    for delay in DELAYS:
        for window, start, end, horizons in windows:
            for issue in pd.period_range(start, end, freq="M"):
                eligible = pax.loc[:, :issue - delay]
                global_shocks = shock_months(eligible)
                stress = {threshold: shock_months(eligible, threshold)
                          for threshold in (.50, .80)}
                regional_shocks = {name: shock_months(eligible.loc[geography.eq(name)])
                                   for name in geography.unique()}
                for horizon in horizons:
                    target = issue + horizon
                    seasons = eligible.loc[:, eligible.columns.month == target.month].iloc[:, ::-1]
                    if seasons.empty:
                        continue
                    predictions, groups = policy_predictions(
                        seasons, global_shocks, regional_shocks, geography
                    )
                    for threshold, alternative in stress.items():
                        candidate, _ = policy_predictions(seasons, alternative, {}, geography)
                        suffix = int(100 * threshold)
                        predictions[f"omit_threshold_{suffix}"] = candidate.omit_shock
                    rows.extend(score_records(issue, horizon, delay, window, pax[target],
                                              predictions, groups))
            print(f"Completed policy comparison {window}, delay {delay}.", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "shock_policy_metrics.csv", index=False)


if __name__ == "__main__":
    main()

