"""Run a release-delay-aware passenger demand backtest on local AA exports."""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROUTE = ["ORG_AIRPRT", "DES_AIRPRT"]
HORIZONS = (1, 6, 12, 18, 24)
DELAYS = (18, 24)
EVALUATION_WINDOWS = (
    ("disruption_recovery", "2022-07", "2023-12", HORIZONS),
    ("full_horizon", "2024-01", "2024-07", HORIZONS),
    ("later_near_term", "2024-08", "2025-07", (1, 6, 12)),
)
HOLDOUT_WINDOWS = (
    ("recent_six_month_holdout", "2025-08", "2026-01", (1, 6)),
    ("recent_one_month_holdout", "2026-02", "2026-06", (1,)),
)


def load_data(data_dir):
    demand = pd.read_csv(data_dir / "AA_PAX_FORECASTING.csv", usecols=ROUTE + [
        "DEP_YEAR", "DEP_MONTH", "OD_PAX", "OD_REV_USD",
    ])
    demand["month"] = pd.PeriodIndex.from_fields(
        year=demand.DEP_YEAR, month=demand.DEP_MONTH, freq="M"
    )
    if demand[ROUTE + ["month"]].isna().any().any() or demand.OD_PAX.lt(0).any():
        raise ValueError("Expected complete route-month keys and nonnegative demand.")
    pax = demand.pivot(index=ROUTE, columns="month", values="OD_PAX")
    months = pd.period_range(pax.columns.min(), pax.columns.max(), freq="M")
    pax = pax.reindex(columns=months)
    revenue = demand.pivot(index=ROUTE, columns="month", values="OD_REV_USD").reindex_like(pax)
    airports = pd.read_csv(data_dir / "AA_AIRPORT_INFO.csv").set_index("AIRPRT_CD")
    if not airports.index.is_unique:
        raise ValueError("Expected unique airport codes.")
    return pax, revenue, airport_context(pax.index, airports)


def airport_context(index, airports):
    def angle(prefix, signs, limit):
        value = (airports[f"{prefix}_DEGREE_QTY"]
                 + airports[f"{prefix}_MINUTE_QTY"] / 60
                 + airports[f"{prefix}_SECOND_QTY"] / 3600)
        return np.deg2rad(value.where(value.between(0, limit))
                          * airports[f"{prefix}_HEMSPHR_CD"].map(signs))

    lat = angle("LAT", {"N": 1, "S": -1}, 90)
    lon = angle("LNGTD", {"E": 1, "W": -1}, 180)
    origin, destination = [index.get_level_values(col) for col in ROUTE]
    lat1, lat2 = origin.map(lat), destination.map(lat)
    lon1, lon2 = origin.map(lon), destination.map(lon)
    arc = (np.sin((lat2 - lat1) / 2) ** 2
           + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    distance = 12742 * np.arcsin(np.sqrt(np.clip(arc, 0, 1)))
    first, second = origin.map(airports.CNTRY_CD), destination.map(airports.CNTRY_CD)
    segment = np.select(
        [first.isna() | second.isna(), (first == "US") & (second == "US"),
         (first == "US") | (second == "US")],
        ["unknown_country", "us_domestic", "us_international"],
        default="both_non_us",
    )
    return pd.DataFrame({
        "geography": segment,
        "log_distance": np.log1p(distance),
        "origin_us": (first == "US").astype(float),
        "destination_us": (second == "US").astype(float),
    }, index=index)


def shock_months(eligible, threshold=.65):
    flagged = []
    for month in eligible.columns[12:]:
        recent, prior = eligible[month], eligible[month - 12]
        pairs = recent.notna() & prior.notna()
        previous = prior[pairs].sum()
        if pairs.sum() >= 100 and previous > 0 and recent[pairs].sum() <= threshold * previous:
            flagged.append(month)
    return flagged


def damped_level(history, steps, alpha=.35, beta=.10, phi=.90):
    """Damped Holt recursion on a log-scale level series."""
    values = history.dropna().to_numpy()
    if not len(values):
        return np.nan
    level, trend = values[0], 0.0
    for value in values[1:]:
        old_level = level
        level = alpha * value + (1 - alpha) * (level + phi * trend)
        trend = beta * (level - old_level) + (1 - beta) * phi * trend
    return np.exp(level + trend * sum(phi ** step for step in range(1, steps + 1)))


def match_windows(eligible):
    months = pd.period_range(end=eligible.columns[-1], periods=12, freq="M")
    recent = eligible.reindex(columns=months)
    prior = eligible.reindex(columns=months - 12).set_axis(months, axis=1)
    paired = recent.notna() & prior.notna()
    enough = paired.sum(axis=1).ge(6)
    return recent, recent.where(paired).sum(axis=1).where(enough), (
        prior.where(paired).sum(axis=1).where(enough)
    )


@dataclass
class ForecastFrame:
    baselines: pd.DataFrame
    core: pd.DataFrame
    context: pd.DataFrame
    groups: pd.DataFrame


class History:
    """Features for one forecast issue. All demand and revenue end at the cutoff."""

    def __init__(self, pax, revenue, airport, cutoff):
        self.pax = pax.loc[:, :cutoff]
        self.airport = airport
        self.cutoff = cutoff
        self.recent, paired_recent, paired_prior = match_windows(self.pax)
        self.route_growth = np.log1p(paired_recent) - np.log1p(paired_prior)
        self.endpoint_growth = {}
        for column in ROUTE:
            totals = pd.DataFrame({"recent": paired_recent, "prior": paired_prior})
            totals = totals.groupby(level=column).sum(min_count=1)
            growth = np.log1p(totals.recent) - np.log1p(totals.prior)
            self.endpoint_growth[column] = self.pax.index.get_level_values(column).map(growth)

        rev = revenue.reindex(columns=self.recent.columns)
        paired = self.recent.gt(0) & rev.ge(0)
        self.revenue_per_pax = rev.where(paired).sum(axis=1).div(
            self.recent.where(paired).sum(axis=1).where(paired.sum(axis=1).ge(6))
        )
        self.shocks = shock_months(self.pax)
        self.route_level = self.pax.T.rolling(12, min_periods=9).mean().T
        network = self.pax.sum(axis=0, min_count=1)
        self.network_level = network.rolling(12, min_periods=12).mean()
        self.network_shape = (network / self.network_level).drop(index=self.shocks)

    def frame(self, target, horizon):
        seasons = self.pax.loc[:, self.pax.columns.month == target.month]
        if seasons.empty:
            return None
        reversed_seasons = seasons.T.iloc[::-1]
        last_three = reversed_seasons.where(reversed_seasons.notna().cumsum().le(3))
        count = last_three.count()
        latest = last_three.bfill().iloc[0]
        median = last_three.median()
        flagged = pd.Series(False, index=self.pax.index)
        for month in seasons:
            flagged.loc[seasons[month].notna()] = month in self.shocks
        switched = flagged & count.ge(2)
        fallback = latest.where(~switched, last_three.mean())

        shapes = (seasons / self.route_level[seasons.columns]).where(last_three.T.notna())
        route_shape = shapes.replace([np.inf, -np.inf], np.nan).median(axis=1)
        shape_forecast = self.recent.mean(axis=1).where(self.recent.count(axis=1).ge(9))
        shape_forecast *= route_shape
        network_shape = self.network_shape.loc[self.network_shape.index.month == target.month]
        seasonal_factor = network_shape.iloc[-4:].median()
        projected = damped_level(np.log(self.network_level.where(self.network_level.gt(0))),
                                 target.ordinal - self.cutoff.ordinal) * seasonal_factor
        anchor_volume = fallback.sum(min_count=1)
        scale = np.clip(projected / anchor_volume, .5, 2) if anchor_volume > 0 else np.nan

        baselines = pd.DataFrame({
            "seasonal": latest,
            "shock_fallback": fallback,
            "weighted_seasonal": .75 * latest + .25 * median,
            "route_level_shape": shape_forecast.fillna(fallback),
            "network_damped": (fallback * scale).fillna(fallback),
        })
        log_base = np.log1p(fallback)
        angle = 2 * np.pi * target.month / 12
        core = pd.DataFrame({
            "target_sin": np.sin(angle), "target_cos": np.cos(angle),
            "horizon": horizon, "log_baseline": log_base,
            "recent_3_relative": np.log1p(self.recent.iloc[:, -3:].mean(axis=1)) - log_base,
            "recent_12_relative": np.log1p(self.recent.mean(axis=1)) - log_base,
            "season_median_relative": np.log1p(median) - log_base,
            "recent_count": self.recent.count(axis=1), "season_count": count,
            "shock_switch": switched.astype(float),
        })
        context = self.airport.drop(columns="geography").assign(
            route_growth=self.route_growth,
            origin_network_growth=self.endpoint_growth[ROUTE[0]],
            destination_network_growth=self.endpoint_growth[ROUTE[1]],
            log_revenue_per_pax=np.log1p(self.revenue_per_pax),
        )
        groups = self.airport[["geography"]].copy()
        lower, upper = latest.quantile([1 / 3, 2 / 3])
        groups["route_size"] = np.select(
            [latest.le(lower), latest.le(upper)], ["lower", "middle"], default="higher"
        )
        groups["reference"] = np.where(switched, "flagged_with_history", "standard")
        return ForecastFrame(baselines, core, context, groups)


def training_data(pax, revenue, airport, delay, latest_label):
    blocks = []
    rng = np.random.default_rng(42)
    for issue in pd.period_range(pax.columns[0] + delay, latest_label - 1, freq="M")[::2]:
        history = History(pax, revenue, airport, issue - delay)
        for horizon in HORIZONS:
            target = issue + horizon
            if target > latest_label:
                continue
            frame = history.frame(target, horizon)
            if frame is None:
                continue
            actual = pax[target]
            base = frame.baselines.shock_fallback
            matched = (actual.notna() & base.notna()).to_numpy().nonzero()[0]
            if not len(matched):
                continue
            rows = rng.choice(matched, size=min(len(matched), 1800), replace=False)
            combined = frame.core.join(frame.context).iloc[rows].astype("float32")
            combined["ratio"] = ((actual.iloc[rows] + 1) / (base.iloc[rows] + 1)).clip(.25, 4)
            combined["weight"] = (base.iloc[rows] + 1).clip(upper=50000) ** .5
            combined["label_month"] = target.ordinal
            combined["training_horizon"] = horizon
            blocks.append(combined)
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def fit_learners(train, core_names, workers):
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from threadpoolctl import threadpool_limits

    varying = train.nunique().gt(1)
    core_names = [name for name in core_names if varying[name]]
    context_names = [c for c in train.columns if varying[c]
                     and c not in ("ratio", "weight", "label_month", "training_horizon")]
    specs = {
        "boost_core": (HistGradientBoostingRegressor(loss="absolute_error", max_iter=65,
                        max_leaf_nodes=15, early_stopping=False, random_state=42), core_names),
        "boost_context": (HistGradientBoostingRegressor(loss="absolute_error", max_iter=65,
                           max_leaf_nodes=15, early_stopping=False, random_state=42), context_names),
        "forest_context": (RandomForestRegressor(n_estimators=48, max_leaf_nodes=24,
                            min_samples_leaf=35, max_features=.75, random_state=42,
                            n_jobs=workers), context_names),
    }
    with threadpool_limits(limits=workers):
        for estimator, columns in specs.values():
            estimator.fit(train[columns], train.ratio, sample_weight=train.weight)
    return specs


def add_learners(predictions, frame, models, workers):
    from threadpoolctl import threadpool_limits

    combined = frame.core.join(frame.context)
    base = predictions.shock_fallback
    with threadpool_limits(limits=workers):
        for name, (model, columns) in models.items():
            ratio = model.predict(combined[columns]).clip(.5, 2)
            predictions[name] = ((base + 1) * ratio - 1).clip(lower=0)
    predictions["boost_context_blend"] = .5 * base + .5 * predictions.boost_context


def score_records(issue, horizon, delay, window, actual, predictions, groups):
    observed = actual.notna()
    covered = observed & predictions.seasonal.notna()
    scores = []
    strata = [("all", "all", pd.Series(True, index=actual.index))]
    for dimension in groups:
        strata.extend((dimension, value, groups[dimension].eq(value))
                      for value in sorted(groups[dimension].unique()))
    for dimension, segment, group in strata:
        matched = covered & group
        volume = actual[matched].sum()
        if not volume:
            continue
        for method in predictions:
            error = predictions.loc[matched, method] - actual[matched]
            if error.isna().any():
                raise ValueError(f"{method} lacks predictions on the common sample.")
            scores.append({
                "delay": delay, "issue": str(issue), "horizon": horizon,
                "target": str(issue + horizon), "window": window,
                "dimension": dimension, "segment": segment, "method": method,
                "observed": int((observed & group).sum()), "scored": int(matched.sum()),
                "actual_sum": volume, "absolute_error": error.abs().sum(),
                "signed_error": error.sum(),
            })
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--late-holdout", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    pax, revenue, airport = load_data(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Windows are fixed so every date within each window supports the same horizons.
    windows = HOLDOUT_WINDOWS if args.late_holdout else EVALUATION_WINDOWS
    if pax.columns[-1] < pd.Period("2026-07", freq="M"):
        raise ValueError("This experiment requires outcomes through July 2026.")
    scores, fits = [], []
    for delay in DELAYS:
        train = None if args.baselines_only else training_data(
            pax, revenue, airport, delay, pd.Period(windows[-1][2], freq="M") - delay
        )
        models = None
        fit_date = None
        for window, start, end, horizons in windows:
            for issue in pd.period_range(start, end, freq="M"):
                cutoff = issue - delay
                history = History(pax, revenue, airport, cutoff)
                if train is not None and window != "disruption_recovery":
                    if fit_date is None or issue.ordinal - fit_date.ordinal >= 6:
                        eligible = train.loc[train.label_month.le(cutoff.ordinal)]
                        if len(eligible) >= 1000:
                            core = history.frame(issue + 1, 1).core.columns.tolist()
                            models = fit_learners(eligible, core, args.workers)
                            fit_date = issue
                            fits.append({
                                "delay": delay, "fit_issue": str(issue),
                                "latest_label": str(pd.Period(
                                    ordinal=int(eligible.label_month.max()), freq="M"
                                )),
                                "examples": len(eligible),
                                **{f"train_h{h}": int(eligible.training_horizon.eq(h).sum())
                                   for h in HORIZONS},
                            })
                            print(f"Fitted three learners for {issue}, delay {delay}, "
                                  f"{len(eligible):,} examples.", flush=True)
                for horizon in horizons:
                    target = issue + horizon
                    frame = history.frame(target, horizon)
                    if frame is None:
                        continue
                    predictions = frame.baselines.copy()
                    if models is not None:
                        add_learners(predictions, frame, models, args.workers)
                    scores.extend(score_records(issue, horizon, delay, window, pax[target],
                                                predictions, frame.groups))
            print(f"Completed {window}, delay {delay}.", flush=True)
    pd.DataFrame(scores).to_csv(args.output_dir / "backtest_metrics.csv", index=False)
    if fits:
        pd.DataFrame(fits).to_csv(args.output_dir / "training_cutoffs.csv", index=False)


if __name__ == "__main__":
    main()



