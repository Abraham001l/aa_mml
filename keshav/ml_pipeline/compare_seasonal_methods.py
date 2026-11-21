"""Evaluate route forecasting methods on the balanced monthly passenger panel."""
import json

import numpy as np
import pandas as pd

from seasonal_challenger import DEV, FINAL, HORIZONS, ROOT, directional_adjustment, forecast_at, get_split, load_panel, scores

BASE = dict(family="route recovery", window=3, recent=.7, recovery=12, pool=.25)
CHECK = ("2021-04", "2021-10", "2022-04")
OUT = ROOT / "results" / "seasonal_methods"


def smooth_seasons(history, prior_strength):
    """Estimate month effects after removing a changing level relative to 2019."""
    dates = history.columns
    months = dates.month.to_numpy() - 1
    daily = history.fillna(0).to_numpy() / dates.days_in_month.to_numpy()
    # Use a small floor for seasonal estimation on sparse routes.
    floor = .05 * daily[:, -12:].mean(axis=1, keepdims=True) + .1
    reference = daily[:, :12] + floor
    ratio = np.log((daily + floor) / reference[:, months])
    season = np.zeros_like(reference)
    mass = np.zeros(12)
    # Center each historical observation on its local, complete 7-month window.
    # Estimate the residual annual cycle with recency weighting. Downweight shock months.
    for t in range(12, len(dates) - 3):
        change = ratio[:, t] - np.median(ratio[:, t-3:t+4], axis=1)
        traffic = daily[:, t].sum() / reference[:, months[t]].sum()
        weight = .5 ** ((len(dates) - 1 - t) / 24) * np.clip(traffic, .1, 1) ** 2
        season[:, months[t]] += weight * np.clip(change, -.8, .8)
        mass[months[t]] += weight
    shape = reference * np.exp(season / (prior_strength + mass))
    shape /= shape.mean(axis=1, keepdims=True)
    return shape, daily


def level_forecast(history, shape, daily):
    """Estimate current route level, then attenuate recovery as network demand normalizes."""
    months = history.columns.month.to_numpy() - 1
    future = pd.period_range(history.columns[-1]+1, periods=24, freq="M")
    def estimate_level(n):
        return daily[:, -n:].sum(axis=1) / shape[:, months[-n:]].sum(axis=1)
    recent = .65 * estimate_level(3) + .35 * estimate_level(6)
    normal = daily[:, :12].mean(axis=1)
    ratio = daily[:, -3:].sum() / daily[:, months[-3:]].sum()
    rebound = np.clip((1.06 - ratio) / .20, 0, 1)
    weight = rebound * (1 - np.exp(-HORIZONS / 12))
    level = recent[:, None] * (1 - weight) + normal[:, None] * weight
    return level * shape[:, future.month - 1] * future.days_in_month.to_numpy()


def airport_reconciliation(history, base):
    """Forecast airport traffic independently, then partially reconcile route totals."""
    result = base.copy()
    for endpoint in (0, 1):
        groups = history.groupby(level=endpoint, sort=True).sum()
        codes = groups.index.get_indexer(history.index.get_level_values(endpoint))
        shape, daily = smooth_seasons(groups, 1)
        target = level_forecast(groups, shape, daily)
        aggregate = np.zeros_like(target)
        np.add.at(aggregate, codes, result)
        factor = np.clip((target + 1) / (aggregate + 1), .75, 1.33)
        result *= np.sqrt(factor[codes])
    return result


def candidates(panel, origin):
    history = panel.loc[:, :pd.Period(origin, freq="M")]
    base = forecast_at(history, origin, **BASE)
    yield "current", base
    yield "recent error calibration", calibrated_forecast(history, origin)
    # Combine short and long demand windows to reduce sensitivity to the cutoff month.
    long_level = forecast_at(history, origin, **dict(BASE, window=9))
    yield "multiple windows", .65 * base + .35 * long_level

    reconciled = airport_reconciliation(history, base)
    yield "airport reconciliation", reconciled
    shape, daily = smooth_seasons(history, 1)
    # Pool month profiles of connected routes via a truncated matrix factorization.
    # Preserve individual profiles by shrinking halfway to the four shared factors.
    log_shape = np.log(shape)
    centered = log_shape - log_shape.mean(axis=0)
    _, directions = np.linalg.eigh(centered.T @ centered)
    shared = directions[:, -4:]
    low_rank = np.exp(log_shape - .5 * (centered - (centered @ shared) @ shared.T))
    low_rank /= low_rank.mean(axis=1, keepdims=True)
    for label, profile in [("smoothed seasons", shape), ("pooled seasons", low_rank)]:
        own = level_forecast(history, profile, daily)
        forecast = .75 * own + .25 * directional_adjustment(history, own)
        yield label, forecast
        yield label + " blend", .5 * base + .5 * forecast
        yield label + " reconciled blend", .5 * reconciled + .5 * forecast

    # When network traffic approaches its 2019 level, use stabilized recent annual
    # seasonality to capture changes in route schedules since the reference year.
    months = history.columns.month.to_numpy() - 1
    recovery_ratio = daily[:, -3:].sum() / daily[:, months[-3:]].sum()
    recent = daily[:, -12:][:, np.argsort(months[-12:])]
    recent_shape = (recent + .1) / (recent.mean(axis=1, keepdims=True) + .1)
    recent_weight = .65 * np.clip((recovery_ratio - .8) / .2, 0, 1)
    adaptive = (1 - recent_weight) * low_rank + recent_weight * recent_shape
    own = level_forecast(history, adaptive, daily)
    adaptive_forecast = .75 * own + .25 * directional_adjustment(history, own)
    yield "adaptive seasons", adaptive_forecast

    # Estimate a seasonal-adjusted local trend, restricting extrapolation to at
    # most four months of change. A longer term growth rate is poorly identified.
    level = daily / np.maximum(adaptive[:, months], .1)
    floor = .05 * level[:, -12:].mean(axis=1) + .2
    monthly_rate = np.log((level[:, -3:].mean(axis=1) + floor) /
                          (level[:, -9:-3].mean(axis=1) + floor)) / 4.5
    trend = np.exp(np.clip(monthly_rate, -.035, .035)[:, None] * 4 * (1 - np.exp(-HORIZONS / 4)))
    yield "adaptive seasons with damped trend", adaptive_forecast * trend

    # Estimate a broad demand change from direction-specific airport totals. Use
    # the geometric mean of the two endpoint rates to moderate route-level noise.
    local_rates = []
    for endpoint in (0, 1):
        codes, names = pd.factorize(history.index.get_level_values(endpoint))
        totals = np.zeros((len(names), 9))
        np.add.at(totals, codes, level[:, -9:])
        rates = np.log((totals[:, -3:].mean(axis=1) + 1) /
                       (totals[:, :6].mean(axis=1) + 1)) / 4.5
        local_rates.append(rates[codes])
    rate = .5 * monthly_rate + .25 * (local_rates[0] + local_rates[1])
    damped = np.exp(np.clip(rate, -.035, .035)[:, None] * 4 * (1 - np.exp(-HORIZONS / 4)))
    yield "adaptive seasons with pooled trend", adaptive_forecast * damped


def calibrated_forecast(panel, origin):
    """Correct persistent recent errors using forecasts whose outcomes are already known."""
    history = panel.loc[:, :pd.Period(origin, freq="M")]
    base = forecast_at(history, origin, **BASE)
    ratios = []
    for lead in (3, 6):
        # Score three months of forecasts from each of two historical origins.
        suborigin = history.columns[-1] - lead
        previous = forecast_at(history, suborigin, **BASE)[:, lead-3:lead]
        actual = history.iloc[:, -3:].fillna(0).to_numpy()
        # Regularize small route counts and cap corrections for abrupt schedule changes.
        ratios.append(np.log((actual.sum(axis=1) + 300) / (previous.sum(axis=1) + 300)))
    # Apply a correction only when both forecast vintages err in the same direction.
    consistent = np.sign(ratios[0]) == np.sign(ratios[1])
    correction = np.where(consistent, np.sign(ratios[0]) * np.minimum(abs(ratios[0]), abs(ratios[1])), 0)
    return base * np.exp(.5 * np.clip(correction, -.25, .25))[:, None]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel(ROOT / "pax_forecasting.csv")
    results = []
    def evaluate(origin, stage, selected=None):
        _, actual = get_split(panel, origin)
        for method, pred in candidates(panel, origin):
            if selected is not None and method not in ("current", selected, "recent error calibration"):
                continue
            if not np.isfinite(pred).all() or (pred < 0).any():
                raise ValueError(f"Invalid forecast for {origin} {method}")
            for period, subset in [("all", slice(None)), ("first17", slice(17)),
                                   ("year1", slice(12)), ("year2", slice(12, None))]:
                results.append(dict(stage=stage, origin=origin, method=method, period=period,
                                    **scores(actual[:, subset], pred[:, subset])))
            if stage == "holdout" and method == selected:
                pd.DataFrame(pred, index=panel.index, columns=pd.Period(origin, freq="M")+HORIZONS).to_csv(OUT / "forecast.csv")
    for origin in DEV:
        evaluate(origin, "development")
    dev = pd.DataFrame(results).query("period == 'all'")
    ranking = dev.groupby("method").WAPE_pct.mean().sort_values()
    winner = ranking.index[0]
    print(ranking.to_string(), flush=True)
    (OUT / "selection.json").write_text(json.dumps({"criterion": "mean development WAPE", "method":winner,
                                                    "development_origins":DEV}, indent=2))
    for origin in CHECK:
        evaluate(origin, "sensitivity", winner)
    evaluate(FINAL, "holdout", winner)
    results = pd.DataFrame(results)
    results.to_csv(OUT / "metrics.csv", index=False)
    print(results.query("stage == 'holdout'").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
