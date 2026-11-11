"""Evaluate a model-selection rule that only sees already released outcomes."""
import argparse
from pathlib import Path

import pandas as pd

from summarize_deep_dive import load_metrics


BASELINE = "shock_fallback"
CANDIDATES = ("weighted_seasonal", "route_level_shape", "paired_blend", "ets_clean_bounded_blend")


def all_routes(directory):
    pieces = []
    for name, methods in (
        ("backtest_metrics.csv", (BASELINE, "weighted_seasonal", "route_level_shape")),
        ("directional_pooling_metrics.csv", ("paired_blend",)),
        ("route_smoothing_metrics.csv", ("ets_clean_bounded_blend",)),
    ):
        frame = load_metrics(directory / name)
        pieces.append(frame.loc[frame.dimension.eq("all") & frame.method.isin(methods)])
    combined = pd.concat(pieces, ignore_index=True)
    combined["target_month"] = pd.PeriodIndex(combined.target, freq="M")
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("historical_dir", type=Path)
    parser.add_argument("recent_dir", type=Path)
    args = parser.parse_args()

    historic = all_routes(args.historical_dir)
    recent = all_routes(args.recent_dir)
    cases = pd.concat([historic, recent], ignore_index=True)
    output = []
    for (delay, issue, horizon), scored in cases.groupby(["delay", "issue", "horizon"]):
        cutoff = pd.Period(issue, freq="M") - delay
        outcomes = historic.loc[
            historic.delay.eq(delay) & historic.horizon.eq(horizon)
            & historic.target_month.le(cutoff) & historic.target_month.gt(cutoff - 18)
        ]
        history = outcomes.groupby("method").agg(
            actual=("actual_sum", "sum"), error=("absolute_error", "sum"),
            cases=("issue", "nunique"),
        )
        choice = BASELINE
        eligible = history.loc[history.cases.ge(4)]
        if BASELINE in eligible.index:
            wape = 100 * eligible.error / eligible.actual
            ranked = wape.reindex(CANDIDATES).dropna().sort_values()
            if not ranked.empty and ranked.iloc[0] < wape[BASELINE] - .10:
                choice = ranked.index[0]
        by_method = scored.set_index("method")
        if not by_method.index.is_unique:
            raise ValueError("Duplicate model scores for one forecast case.")
        baseline = by_method.loc[BASELINE]
        selected = by_method.loc[choice]
        output.append({
            "window": baseline.window, "delay": delay, "issue": issue, "horizon": horizon,
            "cutoff": str(cutoff), "selection": choice, "eligible_cases": int(
                history.cases.get(choice, 0)
            ),
            "actual_sum": baseline.actual_sum, "baseline_error": baseline.absolute_error,
            "selected_error": selected.absolute_error,
        })
    result = pd.DataFrame(output)
    result.to_csv(args.historical_dir / "adaptive_selection_metrics.csv", index=False)
    summary = result.groupby(["window", "delay"])[
        ["actual_sum", "baseline_error", "selected_error"]
    ].sum()
    print((100 * summary[["baseline_error", "selected_error"]].div(summary.actual_sum, axis=0))
          .round(2).to_string())
    print(result.groupby(["window", "delay", "selection"]).size().to_string())


if __name__ == "__main__":
    main()


