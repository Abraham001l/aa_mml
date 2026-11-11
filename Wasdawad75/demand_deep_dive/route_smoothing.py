"""Compare route-level exponential smoothing with the seasonal benchmarks."""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from demand_deep_dive import (
    DELAYS, EVALUATION_WINDOWS, HOLDOUT_WINDOWS, load_data, score_records, shock_months,
)
from lag_sensitivity import anchor


@dataclass
class SeasonalState:
    level: np.ndarray
    trend: np.ndarray
    season: np.ndarray
    trained: np.ndarray
    last_slot: int
    damping: float

    def predict(self, steps, index):
        projected = self.level + self.trend * sum(
            self.damping ** step for step in range(1, steps + 1)
        )
        slot = (self.last_slot + steps) % 12
        return pd.Series(np.expm1(projected + self.season[:, slot]).clip(min=0),
                         index=index).where(self.trained)


def fit_state(eligible, damped):
    """Additive seasonal states on log demand, retaining calendar gaps."""
    log_values = np.log1p(eligible.iloc[:, -48:].to_numpy())
    if log_values.shape[1] < 12:
        return None
    seed = log_values[:, :12]
    seed_count = np.isfinite(seed).sum(axis=1)
    level = np.divide(np.nansum(seed, axis=1), seed_count,
                      out=np.zeros(len(seed)), where=seed_count > 0)
    season = np.nan_to_num(seed - level[:, None], nan=0)
    trend = np.zeros(len(seed))
    phi = .9 if damped else 0
    for index in range(12, log_values.shape[1]):
        value = log_values[:, index]
        slot = index % 12
        observed = np.isfinite(value)
        carried = level + phi * trend
        new_level = np.where(observed, .25 * (value - season[:, slot]) + .75 * carried,
                             carried)
        if damped:
            trend = np.where(observed, .10 * (new_level - level) + .90 * phi * trend,
                             phi * trend)
        season[:, slot] = np.where(
            observed, .10 * (value - new_level) + .90 * season[:, slot], season[:, slot]
        )
        level = new_level
    trained = (seed_count >= 9) & (np.isfinite(log_values).sum(axis=1) >= 24)
    return SeasonalState(level, trend, season, trained, (log_values.shape[1] - 1) % 12, phi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--late-holdout", action="store_true")
    args = parser.parse_args()
    pax, _, airport = load_data(args.data_dir)
    windows = HOLDOUT_WINDOWS if args.late_holdout else EVALUATION_WINDOWS
    rows = []
    for delay in DELAYS:
        for window, start, end, horizons in windows:
            for issue in pd.period_range(start, end, freq="M"):
                cutoff = issue - delay
                eligible = pax.loc[:, :cutoff]
                shocks = shock_months(eligible)
                cleaned = eligible.copy()
                cleaned.loc[:, shocks] = np.nan
                states = {"ets_clean": fit_state(cleaned, True),
                          "ets_seasonal": fit_state(eligible, False),
                          "ets_damped": fit_state(eligible, True)}
                for horizon in horizons:
                    target = issue + horizon
                    latest, baseline = anchor(eligible, target.month, shocks)
                    steps = target.ordinal - cutoff.ordinal
                    predictions = pd.DataFrame({"seasonal": latest, "shock_fallback": baseline})
                    for name, state in states.items():
                        predictions[name] = state.predict(steps, pax.index).fillna(
                            baseline
                        ) if state is not None else baseline
                    predictions["ets_damped_blend"] = .5 * (baseline + predictions.ets_damped)
                    bounded = predictions.ets_clean.clip(lower=.5 * baseline, upper=2 * baseline)
                    predictions["ets_clean_bounded_blend"] = .5 * (baseline + bounded)
                    rows.extend(score_records(issue, horizon, delay, window, pax[target],
                                              predictions, airport[["geography"]]))
            print(f"Completed route smoothing {window}, delay {delay}.", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "route_smoothing_metrics.csv", index=False)


if __name__ == "__main__":
    main()


