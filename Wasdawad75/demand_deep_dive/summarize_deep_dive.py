"""Summarize local demand backtests without adding source records to Git."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["delay", "issue", "horizon", "window", "dimension", "segment"]
NAMES = {
    "seasonal": "Latest seasonal",
    "shock_fallback": "Shock fallback",
    "weighted_seasonal": "Weighted seasonal",
    "route_level_shape": "Route level and seasonal shape",
    "network_damped": "Damped network forecast",
    "boost_core": "Boosted trees, core",
    "boost_context": "Boosted trees, context",
    "forest_context": "Random forest, context",
    "boost_context_blend": "Half baseline, half boosted",
    "faa_half": "FAA half adjustment",
    "faa_full": "FAA full adjustment",
    "tsa_half": "TSA half adjustment",
    "tsa_full": "TSA full adjustment",
    "tsa_faa_local": "TSA with local FAA growth",
    "tsa_deadband": "TSA with a five-percent deadband",
    "omit_shock": "Use the latest unflagged season",
    "regional_shock": "Regional shock fallback",
    "route_low_guard": "Low-reference guard for individual routes",
    "paired_directions": "Pooled opposite directions",
    "paired_blend": "Half baseline, half pooled directions",
    "ets_seasonal": "Seasonal exponential smoothing",
    "ets_damped": "Damped route exponential smoothing",
    "ets_damped_blend": "Half baseline, half damped smoothing",
    "ets_clean": "Damped smoothing, flagged months omitted",
    "ets_clean_bounded_blend": "Half baseline, half bounded clean smoothing",
    "omit_threshold_50": "Omit shock after a 50% drop",
    "omit_threshold_80": "Omit shock after a 20% drop",
}


def load_metrics(path):
    data = pd.read_csv(path)
    if data.duplicated(KEY + ["method"]).any():
        raise ValueError(f"Duplicate result keys in {path}.")
    same_sample = data.groupby(KEY)[["observed", "scored", "actual_sum"]].nunique()
    if not same_sample.eq(1).all().all() or data.scored.gt(data.observed).any():
        raise ValueError(f"Methods did not use the same scoring sample in {path}.")
    if data[["actual_sum", "absolute_error", "signed_error"]].isna().any().any():
        raise ValueError(f"Missing error sums in {path}.")
    if (data.absolute_error + 1e-5 < data.signed_error.abs()).any():
        raise ValueError(f"Absolute error fell below absolute signed error in {path}.")
    data["case_wape"] = 100 * data.absolute_error / data.actual_sum
    return data


def validate_fits(path):
    if not path.exists():
        return None
    fits = pd.read_csv(path)
    issue = pd.PeriodIndex(fits.fit_issue, freq="M").asi8
    label = pd.PeriodIndex(fits.latest_label, freq="M").asi8
    if np.any(label > issue - fits.delay.to_numpy()):
        raise ValueError("A model used labels beyond its simulated data release.")
    if not fits[["train_h1", "train_h6", "train_h12", "train_h18", "train_h24"]].sum(
        axis=1
    ).eq(fits.examples).all():
        raise ValueError("Training example counts do not reconcile.")
    return fits


def pooled(data, columns):
    result = data.groupby(columns, as_index=False).agg(
        actual=("actual_sum", "sum"), absolute=("absolute_error", "sum"),
        signed=("signed_error", "sum"), average_case=("case_wape", "mean"),
        scored=("scored", "sum"), observed=("observed", "sum"),
        cases=("issue", "size"),
    )
    result["WAPE %"] = 100 * result.absolute / result.actual
    result["Bias %"] = 100 * result.signed / result.actual
    result["Coverage %"] = 100 * result.scored / result.observed
    result["Mean case WAPE %"] = result.average_case
    return result


def paired_stability(data):
    benchmark = data.loc[data.method.eq("shock_fallback"), KEY + ["case_wape", "absolute_error"]]
    pairs = data.merge(benchmark, on=KEY, validate="many_to_one", suffixes=("", "_base"))
    pairs["delta"] = pairs.absolute_error - pairs.absolute_error_base
    pairs["case_win"] = pairs.case_wape + .00001 < pairs.case_wape_base
    output = []
    for key, group in pairs.groupby(["window", "delay", "method"]):
        monthly = group.groupby("issue").delta.sum()
        leave_out = monthly.sum() - monthly
        output.append((*key, int(group.case_win.sum()), len(group),
                       int(leave_out.lt(-.01).sum()), len(leave_out)))
    return pd.DataFrame(output, columns=[
        "window", "delay", "method", "Cases won", "Cases compared",
        "Leave-one-month-out wins", "Months omitted",
    ])


def markdown(data, columns):
    view = data.loc[:, columns].copy()
    if "method" in view:
        view["method"] = view.method.replace(NAMES)
    for column in view:
        if view[column].dtype.kind == "f":
            view[column] = view[column].map(lambda number: f"{number:.2f}")
    rows = [columns, ["---"] * len(columns), *view.astype(str).to_numpy().tolist()]
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def write_summary(directory):
    main = load_metrics(directory / "backtest_metrics.csv")
    fits = validate_fits(directory / "training_cutoffs.csv")
    overall = main.loc[main.dimension.eq("all")]
    case_count = overall.loc[overall.method.eq("seasonal"), ["delay", "issue", "horizon"]]
    sections = [
        "# Passenger demand backtest results",
        "Team-internal. Generated locally from the team data. Review sharing permissions before "
        "copying results elsewhere.",
        f"The seasonal benchmark covers {len(case_count):,} delay, issue-month and horizon "
        "combinations. Each competing method uses the same observed markets with an eligible "
        "seasonal forecast. Pooled WAPE weights by observed passenger volume. Mean case WAPE "
        "weights each forecast date and horizon equally. Bias is signed, with negative values "
        "indicating underprediction.",
        "## Overall by evaluation window",
        markdown(pooled(overall, ["window", "delay", "method"]).sort_values(
            ["window", "delay", "WAPE %"]), [
                "window", "delay", "method", "cases", "WAPE %", "Mean case WAPE %",
                "Bias %", "Coverage %",
            ]),
        "## By horizon",
        markdown(pooled(overall, ["window", "delay", "horizon", "method"]), [
            "window", "delay", "horizon", "method", "WAPE %", "Bias %",
        ]),
        "## Paired stability against the shock fallback",
        "Each leave-one-month-out comparison removes every horizon from one issue month. "
        "This shows sensitivity to single months in this sample. Overlapping forecast targets "
        "and shared disruption periods make these comparisons dependent.",
        markdown(paired_stability(overall), [
            "window", "delay", "method", "Cases won", "Cases compared",
            "Leave-one-month-out wins", "Months omitted",
        ]),
        "## Route-size, geography and historical reference segments",
        "Route-size thirds use the eligible seasonal forecast. Segment assignment never uses "
        "future demand. The full set of horizon-specific segment scores is in the local CSV.",
        markdown(pooled(main.loc[main.dimension.ne("all")],
                        ["window", "delay", "dimension", "segment", "method"]), [
            "window", "delay", "dimension", "segment", "method", "WAPE %", "Bias %",
            "Coverage %",
        ]),
    ]
    if fits is not None:
        sections.extend([
            "## Training availability",
            "Each fit uses labels available at the modeled cutoff. Long-horizon forecasts from "
            "a pooled learner may extrapolate from shorter training horizons.",
            markdown(fits, fits.columns.tolist()),
        ])
    comparisons = (
        ("faa_backtest_metrics.csv", "FAA boarding adjustment", True),
        ("tsa_backtest_metrics.csv", "TSA screening adjustment", True),
        ("public_proxy_metrics.csv", "Combined public demand proxies", True),
        ("shock_policy_metrics.csv", "Alternative shock policies", False),
        ("directional_pooling_metrics.csv", "Pooling opposite directions", False),
        ("route_smoothing_metrics.csv", "Route-level exponential smoothing", False),
    )
    for filename, title, domestic in comparisons:
        path = directory / filename
        if not path.exists():
            continue
        extra = load_metrics(path)
        benchmark = main.loc[main.method.eq("shock_fallback")]
        if domestic:
            benchmark = benchmark.loc[benchmark.dimension.eq("geography")
                                      & benchmark.segment.eq("us_domestic")]
            title += ", U.S. domestic only"
        else:
            benchmark = benchmark.loc[benchmark.dimension.eq("all")]
        matched = extra.loc[extra.dimension.eq("all") & extra.method.eq("shock_fallback")]
        check = matched.merge(benchmark, on=["delay", "issue", "horizon", "window"],
                              validate="one_to_one", suffixes=("", "_main"))
        if len(check) != len(benchmark) or len(check) != len(matched):
            raise ValueError(f"Different forecast cases in {filename}.")
        for column in ["observed", "scored", "actual_sum", "absolute_error", "signed_error"]:
            if not np.allclose(check[column], check[f"{column}_main"]):
                raise ValueError(f"Benchmark mismatch for {column} in {filename}.")
        sections.extend([
            f"## {title}",
            markdown(pooled(extra, ["window", "delay", "dimension", "segment", "method"]), [
                "window", "delay", "dimension", "segment", "method", "cases", "WAPE %", "Bias %",
            ]),
        ])
    lag_path = directory / "lag_sensitivity_metrics.csv"
    if lag_path.exists():
        lag = pd.read_csv(lag_path)
        keys = ["window", "issue", "horizon"]
        common = lag.groupby(keys)[["scored", "actual_sum"]].nunique()
        if not common.eq(1).all().all():
            raise ValueError("Reporting-delay cases used different scoring samples.")
        lag = lag.assign(observed=lag.scored, case_wape=100 * lag.absolute_error / lag.actual_sum)
        sections.extend([
            "## Reporting-delay sensitivity on common markets",
            markdown(pooled(lag, ["window", "delay", "horizon", "method"]), [
                "window", "delay", "horizon", "method", "WAPE %", "Bias %",
            ]),
        ])
    output = directory / "automated_summary.md"
    output.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"Verified comparison samples and saved {output}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    write_summary(args.output_dir)


if __name__ == "__main__":
    main()


