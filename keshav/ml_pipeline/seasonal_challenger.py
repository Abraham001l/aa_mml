"""Compare seasonally adjusted route levels on the transfer report's balanced panel."""
import json
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
HORIZONS = np.arange(1, 25)
DEV = ("2021-07", "2022-01", "2022-07")
FINAL = "2024-07"


def load_panel(path):
    routes = ["ORG_AIRPRT", "DES_AIRPRT"]
    raw = pd.read_csv(path, sep="|", usecols=routes + ["DEP_YEAR", "DEP_MONTH", "OD_PAX"])
    raw["month"] = pd.to_datetime(dict(year=raw.DEP_YEAR, month=raw.DEP_MONTH, day=1)).dt.to_period("M")
    if raw[routes + ["month", "OD_PAX"]].isna().any().any() or (raw.OD_PAX < 0).any():
        raise ValueError("Missing route, month or demand, or negative demand.")
    panel = raw.pivot(index=routes, columns="month", values="OD_PAX").sort_index(axis=1)
    expected = pd.period_range("2019-01", panel.columns[-1], freq="M")
    if not panel.columns.equals(expected):
        raise ValueError("Expected consecutive calendar months beginning in January 2019.")
    return panel


def seasonal_components(history, window, recent_weight, recovery):
    """Forecast using only a routes x months frame ending at the forecast origin."""
    calendar = history.columns
    origin = calendar[-1]
    targets = pd.period_range(origin + 1, periods=24, freq="M")
    # Static route universe is supplied to match the report. Only known volumes enter.
    volume = history.fillna(0).to_numpy()
    daily = volume / calendar.days_in_month.to_numpy()
    reference = daily[:, calendar.year == 2019]
    if reference.shape[1] != 12:
        raise ValueError("A full 2019 seasonal reference is required.")
    # Bound the effect of a sparse or unusually low month on the inferred level.
    def scaled(shape, dates):
        expected = shape[:, dates.month - 1].sum(axis=1)
        observed = daily[:, -len(dates):].sum(axis=1)
        return np.clip((observed + 1) / (expected + 1), 0.1, 6)

    annual = daily[:, -12:][:, np.argsort(calendar[-12:].month)]
    # In a recovery, the last year may include closed borders or suspended routes.
    # Weight recent seasonality more heavily as network volume approaches 2019 levels.
    recent = calendar[-window:]
    network_recovery = daily[:, -window:].sum() / reference[:, recent.month - 1].sum()
    trust_recent = recent_weight * np.clip((network_recovery - .65) / .35, 0, 1)
    ratio = scaled(reference, recent)
    # Keep structural route gains; allow a temporary, network-wide deficit to recover.
    uplift = max(0, 1 / max(network_recovery, .25) - 1)
    growth = 1 + recovery * uplift * (1 - np.exp(-HORIZONS / 8))
    pre = reference[:, targets.month - 1] * ratio[:, None]
    current = annual[:, targets.month - 1] * scaled(annual, recent)[:, None]
    own = ((1 - trust_recent) * pre + trust_recent * current) * growth * targets.days_in_month.to_numpy()
    # Estimate a second forecast using the reverse route, scaled by recent directional
    # volumes after removing the 2019 directional seasonality (important in winter).
    reverse = history.index.get_indexer(history.index.swaplevel())
    valid = reverse >= 0
    backward = own.copy()
    # Smooth the directional seasonal ratio with the annual ratio on thin routes.
    totals = volume[:, -12:].sum(axis=1)
    annual_ratio = (totals + 100) / (totals[reverse] + 100)
    historical_ratio = (100 * annual_ratio[:, None] + reference[:, targets.month - 1] * 30) / (
        100 + reference[reverse][:, targets.month - 1] * 30)
    level_ratio = (ratio + .1) / (ratio[reverse] + .1)
    adjustment = np.clip(historical_ratio * level_ratio[:, None], .25, 4)
    backward[valid] = own[reverse[valid]] * adjustment[valid]
    return own, backward


def recovery_components(history, window, damping, recent_weight):
    """Blend 2019 and recent seasonality, allowing recovery to differ by route."""
    calendar = history.columns
    daily = history.fillna(0).to_numpy() / calendar.days_in_month.to_numpy()
    base = daily[:, calendar.year == 2019]
    recent = calendar[-window:]
    targets = pd.period_range(calendar[-1] + 1, periods=24, freq="M")
    target_months = targets.month - 1
    expected = base[:, recent.month - 1].sum(axis=1)
    observed = daily[:, -window:].sum(axis=1)
    ratio = np.clip((observed + 2) / (expected + 2), .05, 8)
    network_ratio = observed.sum() / expected.sum()
    # A broad traffic deficit provides evidence for a temporary disruption.
    recovery_weight = np.clip((1.06 - network_ratio) / .20, 0, 1)
    # Use route-level shrinkage towards the pre-disruption monthly volume. Smooth
    # towards a current seasonal shape as the network returns to normal.
    pull = (1 - np.exp(-HORIZONS / damping)) * recovery_weight
    level = ratio[:, None] * (1 - pull) + pull
    pre = base[:, target_months] * level
    last = daily[:, -12:][:, np.argsort(calendar[-12:].month)]
    recent_profile = last[:, target_months]
    factor = np.clip((observed + 2) / (last[:, recent.month - 1].sum(axis=1) + 2), .5, 2)
    latest = recent_profile * factor[:, None]
    trust = recent_weight * (1 - recovery_weight)
    # Include the recent profile for routes with little or no 2019 service.
    structural_share = np.minimum(1., base.mean(axis=1) / np.maximum(.25 * daily[:, -12:].mean(axis=1), .1))
    weight = (1 - trust) * structural_share[:, None]
    own = (weight * pre + (1 - weight) * latest) * targets.days_in_month.to_numpy()
    return own


def directional_adjustment(history, own):
    """Use the opposite forecast, translating it to this direction's seasonal share."""
    origin = history.columns[-1]
    months = pd.period_range(origin + 1, periods=24, freq="M").month
    vol = history.fillna(0).to_numpy()
    rev = history.index.get_indexer(history.index.swaplevel())
    both = rev >= 0
    prior = vol[:, :12][:, months - 1]
    last_year = vol[:, -12:][:, np.argsort(history.columns[-12:].month)][:, months - 1]
    # Preserve persistent flow imbalances and average noisy direction-specific shares.
    annual = (vol[:, -12:].sum(axis=1) + 100) / (vol[rev, -12:].sum(axis=1) + 100)
    def share(reference):
        return (reference + 100 * annual[:, None]) / (reference[rev] + 100)
    multiplier = np.sqrt(share(prior) * share(last_year)).clip(.25, 4)
    paired = own.copy()
    paired[both] = own[rev[both]] * multiplier[both]
    return paired


def forecast_at(panel, origin, *, family, window, recent, recovery, pool):
    """Forecast the next 24 months using a fixed configuration and an explicit cutoff."""
    history = panel.loc[:, :pd.Period(origin, freq="M")]
    if family == "route recovery":
        own = recovery_components(history, window, recovery, recent)
        paired = directional_adjustment(history, own)
    elif family == "network recovery":
        own, paired = seasonal_components(history, window, recent, recovery)
    else:
        raise ValueError(f"Unknown forecast family {family!r}")
    return own + pool * (paired - own)


def scores(actual, forecast):
    error = forecast - actual
    square = np.square(error).sum()
    combined = actual + forecast
    smape = np.divide(2*np.abs(error), combined, out=np.zeros_like(error), where=combined>0)
    return dict(sMAPE=smape.mean(), RMSLE=np.sqrt(np.square(np.log1p(forecast)-np.log1p(actual)).mean()),
                count=actual.size, WAPE_pct=100 * np.abs(error).sum()/actual.sum(),
                MAE=np.abs(error).mean(), RMSE=np.sqrt(square/actual.size),
                R2=1-square/np.square(actual-actual.mean()).sum(),
                bias_pct=100*error.sum()/actual.sum())


def get_split(panel, origin):
    t = pd.Period(origin, freq="M")
    hist = panel.loc[:, :t]
    target = panel.loc[:, t + HORIZONS].fillna(0).to_numpy()
    return hist, target


def main():
    panel = load_panel(ROOT / "pax_forecasting.csv")
    out = ROOT / "results" / "seasonal_challenger"
    out.mkdir(exist_ok=True, parents=True)
    rows = []
    # All candidates are selected on the three historical folds before scoring July 2024.
    for origin in DEV:
        history, actual = get_split(panel, origin)
        for window, recent, recovery in product([3, 6, 9], [.35, .7, 1.], [.5, 1.]):
            own, backward = seasonal_components(history, window, recent, recovery)
            for pool in [0., .25, .5]:
                prediction = (1-pool)*own + pool*backward
                rows.append(dict(origin=origin, window=window, recent=recent, recovery=recovery,
                                 pool=pool, WAPE_pct=100*np.abs(prediction-actual).sum()/actual.sum()))
        for window, damping, recent in product([1, 2, 3, 6], [2., 6., 12., 24.], [.35, .7, 1.]):
            prediction = recovery_components(history, window, damping, recent)
            paired = directional_adjustment(history, prediction)
            for pool in [0., .25, .5]:
                blended = prediction + pool * (paired - prediction)
                rows.append(dict(origin=origin, family="route recovery", window=window, recent=recent,
                                 recovery=damping, pool=pool, WAPE_pct=100*np.abs(blended-actual).sum()/actual.sum()))
    trials = pd.DataFrame(rows)
    trials["family"] = trials.family.fillna("network recovery")
    trials.to_csv(out / "development_search.csv", index=False)
    params = ["family", "window", "recent", "recovery", "pool"]
    ranked = trials.groupby(params).WAPE_pct.mean().sort_values()
    config = dict(zip(params, ranked.index[0]))
    config["window"] = int(config["window"])
    (out / "settings.json").write_text(json.dumps(config, indent=2) + "\n")
    print("Selected on development folds:", config, flush=True)
    metrics = []
    for origin in (*DEV, FINAL):
        history, actual = get_split(panel, origin)
        pred = forecast_at(panel, origin, **config)
        repeat = np.tile(history.iloc[:, -12:].fillna(0).to_numpy(), (1, 2))
        for label, forecast in [("challenger", pred), ("seasonal_repeat", repeat)]:
            if not np.isfinite(forecast).all() or (forecast < 0).any():
                raise ValueError("Invalid forecast.")
            for name, months in [("all",slice(None)), ("first17",slice(17)), ("first21",slice(21)),
                                 ("year1",slice(12)), ("year2",slice(12,None))]:
                metrics.append(dict(origin=origin, model=label, period=name, **scores(actual[:,months],forecast[:,months])))
        if origin == FINAL:
            forecasts = pd.DataFrame(pred, index=panel.index, columns=history.columns[-1]+HORIZONS)
            forecasts.to_csv(out / "holdout_forecasts.csv")
    table = pd.DataFrame(metrics)
    table.to_csv(out / "metrics.csv", index=False)
    print(table[(table.period == "all") | ((table.origin == FINAL) & (table.model == "challenger"))].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
