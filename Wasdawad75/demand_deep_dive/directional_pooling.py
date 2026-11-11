"""Test whether pooling opposite directions stabilizes seasonal route forecasts."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from demand_deep_dive import (
    DELAYS, EVALUATION_WINDOWS, HOLDOUT_WINDOWS, load_data, score_records, shock_months,
)
from lag_sensitivity import anchor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--late-holdout", action="store_true")
    args = parser.parse_args()
    pax, _, airport = load_data(args.data_dir)
    reverse = pax.reindex(pax.index.swaplevel()).set_axis(pax.index, axis=0)
    pair = pax + reverse
    windows = HOLDOUT_WINDOWS if args.late_holdout else EVALUATION_WINDOWS
    rows = []
    for delay in DELAYS:
        for window, start, end, horizons in windows:
            for issue in pd.period_range(start, end, freq="M"):
                cutoff = issue - delay
                eligible = pax.loc[:, :cutoff]
                paired = pair.loc[:, :cutoff]
                shocks = shock_months(eligible)
                recent_pair = paired.iloc[:, -12:]
                recent = eligible[recent_pair.columns].where(recent_pair.notna())
                total = recent_pair.sum(axis=1)
                shares = recent.sum(axis=1).div(
                    total.where(total.gt(0) & recent_pair.count(axis=1).ge(6))
                )
                for horizon in horizons:
                    target = issue + horizon
                    latest, baseline = anchor(eligible, target.month, shocks)
                    _, pair_baseline = anchor(paired, target.month, shocks)
                    pooled = (pair_baseline * shares).fillna(baseline)
                    lower, upper = latest.quantile([1 / 3, 2 / 3])
                    groups = airport[["geography"]].copy()
                    groups["route_size"] = np.select(
                        [latest.le(lower), latest.le(upper)], ["lower", "middle"], default="higher"
                    )
                    predictions = pd.DataFrame({
                        "seasonal": latest, "shock_fallback": baseline,
                        "paired_directions": pooled, "paired_blend": .5 * (baseline + pooled),
                    })
                    rows.extend(score_records(issue, horizon, delay, window, pax[target],
                                              predictions, groups))
            print(f"Completed directional pooling {window}, delay {delay}.", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "directional_pooling_metrics.csv", index=False)


if __name__ == "__main__":
    main()
