"""Train and test one direct, multi-horizon XGBoost passenger-demand model."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


MARKET = ["ORG_AIRPRT", "DES_AIRPRT"]
CATEGORICAL_FEATURES = MARKET + ["market_id"]
HORIZONS = range(1, 25)
FORECAST_FEATURES = [
    "target_month", "forecast_horizon", "pax_seasonal_reference", "pax_seasonal_mean_3",
]
HISTORY_FEATURES = CATEGORICAL_FEATURES + [
    "distance_km", "is_domestic",
    "pax_lag_0", "pax_lag_1", "pax_lag_2", "pax_lag_3",
    "pax_lag_6", "pax_lag_12", "pax_lag_24",
    "pax_rolling_3", "pax_rolling_12", "pax_yoy_growth", "pax_rolling_3_yoy_growth",
    "origin_pax_total", "origin_pax_yoy_growth",
    "destination_pax_total", "destination_pax_yoy_growth",
]
FEATURES = HISTORY_FEATURES + FORECAST_FEATURES
DATA_DIR = Path(__file__).resolve().parent


def load_data(pax_path, airport_path):
    p = pd.read_csv(
        pax_path, sep="|",
        usecols=["DEP_YEAR", "DEP_MONTH", *MARKET, "OD_PAX"],
    )
    p["date"] = pd.to_datetime(dict(year=p.DEP_YEAR, month=p.DEP_MONTH, day=1))
    p["month_idx"] = p.date.dt.year * 12 + p.date.dt.month - 1
    if p.duplicated(MARKET + ["month_idx"]).any():
        raise ValueError("Passenger data contains duplicate directional market-months.")
    if p[MARKET].isna().any().any() or not np.isfinite(p.OD_PAX).all() or (p.OD_PAX < 0).any():
        raise ValueError("Airport codes and nonnegative, finite OD_PAX are required.")

    a = pd.read_csv(airport_path, sep="|")
    if a.AIRPRT_CD.duplicated().any():
        raise ValueError("Airport data contains duplicate AIRPRT_CD values.")
    for prefix, name, negative in [("LAT", "latitude", "S"), ("LNGTD", "longitude", "W")]:
        a[name] = (
            a[f"{prefix}_DEGREE_QTY"]
            + a[f"{prefix}_MINUTE_QTY"] / 60
            + a[f"{prefix}_SECOND_QTY"] / 3600
        ) * np.where(a[f"{prefix}_HEMSPHR_CD"] == negative, -1, 1)
    a = a.set_index("AIRPRT_CD")
    origin = a.reindex(p.ORG_AIRPRT).reset_index(drop=True)
    destination = a.reindex(p.DES_AIRPRT).reset_index(drop=True)
    for endpoint in [origin, destination]:
        if endpoint[["CNTRY_CD", "latitude", "longitude"]].isna().any().any():
            raise ValueError("Every used airport needs a country and coordinates.")

    lat1, lat2 = np.radians(origin.latitude), np.radians(destination.latitude)
    dlon = np.radians(destination.longitude - origin.longitude)
    haversine = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    p["distance_km"] = (6371 * 2 * np.arcsin(np.sqrt(haversine.clip(0, 1)))).astype("float32")
    # Domestic means both endpoints are in the same country (including non-US markets).
    p["is_domestic"] = (origin.CNTRY_CD == destination.CNTRY_CD).astype("int8")
    return p.drop(columns=["DEP_YEAR", "DEP_MONTH"]).sort_values(MARKET + ["month_idx"]).reset_index(drop=True)


def lookup_market_values(lookup, frame, months):
    keys = pd.MultiIndex.from_arrays([frame.ORG_AIRPRT, frame.DES_AIRPRT, months])
    return lookup.reindex(keys).to_numpy(dtype="float32")


def add_history_features(history):
    """At origin t, use only observed demand through t. Missing months stay unknown."""
    p = history.sort_values(MARKET + ["month_idx"]).reset_index(drop=True).copy()
    lookup = p.set_index(MARKET + ["month_idx"]).OD_PAX
    p["market_id"] = (p.ORG_AIRPRT + "_" + p.DES_AIRPRT).astype("category")
    p["pax_lag_0"] = p.OD_PAX.astype("float32")
    for lag in [1, 2, 3, 6, 12, 24]:
        p[f"pax_lag_{lag}"] = lookup_market_values(lookup, p, p.month_idx - lag)

    groups = p.groupby(MARKET, sort=False, observed=True)
    for window in [3, 12]:
        rolling = groups.OD_PAX.transform(lambda s: s.rolling(window, min_periods=window).mean())
        consecutive = p.month_idx - groups.month_idx.shift(window - 1) == window - 1
        p[f"pax_rolling_{window}"] = rolling.where(consecutive).astype("float32")
    p["pax_yoy_growth"] = (
        p.pax_lag_0 / p.pax_lag_12.replace(0, np.nan) - 1
    ).astype("float32")
    rolling_lookup = p.set_index(MARKET + ["month_idx"]).pax_rolling_3
    prior_year = lookup_market_values(rolling_lookup, p, p.month_idx - 12)
    p["pax_rolling_3_yoy_growth"] = (
        p.pax_rolling_3 / np.where(prior_year == 0, np.nan, prior_year) - 1
    ).astype("float32")

    # Totals describe markets covered by this dataset, rather than all airport traffic.
    for airport_column, prefix in zip(MARKET, ["origin", "destination"]):
        totals = p.groupby([airport_column, "month_idx"]).OD_PAX.sum(min_count=1)
        current_keys = pd.MultiIndex.from_arrays([p[airport_column], p.month_idx])
        prior_keys = pd.MultiIndex.from_arrays([p[airport_column], p.month_idx - 12])
        current = totals.reindex(current_keys).to_numpy(dtype="float32")
        prior = totals.reindex(prior_keys).to_numpy(dtype="float32")
        p[f"{prefix}_pax_total"] = current
        p[f"{prefix}_pax_yoy_growth"] = current / np.where(prior == 0, np.nan, prior) - 1
    return p


def make_examples(history, targets, cutoff_idx, *, training, origin_step=3):
    """Pair features at t with OD_PAX at t+h, keeping training labels <= cutoff."""
    if origin_step < 1:
        raise ValueError("origin_step must be at least 1.")
    if training:
        origins = history.loc[
            (history.month_idx <= cutoff_idx)
            & ((history.month_idx - cutoff_idx) % origin_step == 0)
        ]
    else:
        origins = history.loc[history.month_idx == cutoff_idx]

    lookup = targets.set_index(MARKET + ["month_idx"]).OD_PAX
    history_lookup = history.set_index(MARKET + ["month_idx"]).OD_PAX
    years_available = (cutoff_idx - history.month_idx.min()) // 12 + 1
    frames = []
    for horizon in HORIZONS:
        x = origins[["date", "month_idx", *HISTORY_FEATURES]].copy()
        x["forecast_horizon"] = np.int8(horizon)
        x["target_idx"] = x.month_idx + horizon
        x["target_month"] = (x.target_idx % 12 + 1).astype("int8")
        if training:
            x = x.loc[x.target_idx <= cutoff_idx]
        x["actual_OD_PAX"] = lookup_market_values(lookup, x, x.target_idx)
        x = x.dropna(subset=["actual_OD_PAX"])

        # Latest target-season observation known at t; h>12 must look back >=24 months.
        source_idx = x.target_idx - 12 * ((horizon + 11) // 12)
        reference = np.full(len(x), np.nan, dtype="float32")
        reference_idx = np.full(len(x), np.nan, dtype="float32")
        seasonal_sum = np.zeros(len(x), dtype="float32")
        seasonal_count = np.zeros(len(x), dtype="int8")
        for years_back in range(years_available):
            months = source_idx - 12 * years_back
            values = lookup_market_values(history_lookup, x, months)
            observed = np.isfinite(values)
            latest = np.isnan(reference) & observed
            reference[latest] = values[latest]
            reference_idx[latest] = months.to_numpy()[latest]
            include = observed & (seasonal_count < 3)
            seasonal_sum[include] += values[include]
            seasonal_count[include] += 1
        x["pax_seasonal_reference"] = reference
        x["seasonal_reference_idx"] = reference_idx
        x["pax_seasonal_mean_3"] = seasonal_sum / np.where(seasonal_count == 0, np.nan, seasonal_count)
        frames.append(x)

    examples = pd.concat(frames, ignore_index=True)
    if examples.empty:
        raise ValueError("No examples available; check the cutoff and passenger dates.")
    if not training and examples.forecast_horizon.nunique() != len(HORIZONS):
        raise ValueError("Test data must contain observed targets for every horizon 1–24.")
    return examples


def score_predictions(predictions):
    rows = []
    groups = [("all", predictions), *predictions.groupby("forecast_horizon", sort=True)]
    for horizon, group in groups:
        actual = group.actual_OD_PAX.to_numpy(dtype=float)
        error = group.predicted_OD_PAX.to_numpy(dtype=float) - actual
        total = actual.sum()
        squared_error = np.sum(error ** 2)
        total_variation = np.sum((actual - actual.mean()) ** 2)
        rows.append({
            "forecast_horizon": horizon,
            "count": len(group),
            "MAE": np.abs(error).mean(),
            "RMSE": np.sqrt(squared_error / len(group)),
            "R2": 1 - squared_error / total_variation if len(group) > 1 and total_variation > 0 else np.nan,
            "WAPE_pct": 100 * np.abs(error).sum() / total if total > 0 else np.nan,
            "Bias_pct": 100 * error.sum() / total if total > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pax", type=Path, default=DATA_DIR / "pax_forecasting.csv")
    parser.add_argument("--airports", type=Path, default=DATA_DIR / "airport_info.csv")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "results")
    parser.add_argument("--cutoff", help="Last observed month, YYYY-MM; default: latest data month minus 24.")
    parser.add_argument("--origin-step", type=int, default=3, help="Training origin spacing in months; use 1 for all months.")
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args()
    if args.origin_step < 1 or args.n_estimators < 1:
        parser.error("--origin-step and --n-estimators must be positive.")

    data = load_data(args.pax, args.airports)
    cutoff = pd.Period(args.cutoff, freq="M").to_timestamp() if args.cutoff else data.date.max() - pd.DateOffset(months=24)
    cutoff_idx = cutoff.year * 12 + cutoff.month - 1
    if cutoff + pd.DateOffset(months=24) > data.date.max():
        parser.error("The cutoff must leave 24 months of observed data for testing.")

    # Held-out demand is used only as labels, never to update test-origin features.
    history = add_history_features(data.loc[data.month_idx <= cutoff_idx])
    train = make_examples(history, data, cutoff_idx, training=True, origin_step=args.origin_step)
    test = make_examples(history, data, cutoff_idx, training=False)
    for column in CATEGORICAL_FEATURES:
        categories = pd.CategoricalDtype(sorted(train[column].unique()))
        train[column] = train[column].astype(categories)
        test[column] = test[column].astype(categories)
    if train.forecast_horizon.nunique() != len(HORIZONS):
        parser.error("Training data needs enough history for every horizon 1–24.")

    print(f"Data: {len(data):,} rows, {data[MARKET].drop_duplicates().shape[0]:,} directional markets")
    print(f"Training labels through {cutoff:%Y-%m}; test targets {cutoff + pd.DateOffset(months=1):%Y-%m}–{cutoff + pd.DateOffset(months=24):%Y-%m}")
    print(f"Training examples: {len(train):,}; test examples: {len(test):,}", flush=True)
    print(f"Fitting XGBoost with {len(FEATURES)} features...", flush=True)
    model = XGBRegressor(
        n_estimators=args.n_estimators, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        objective="reg:squarederror", enable_categorical=True,
        n_jobs=4, random_state=42,
    )
    model.fit(train[FEATURES], train.actual_OD_PAX)
    predictions = test[MARKET + [
        "date", "target_idx", "forecast_horizon", "actual_OD_PAX", "pax_seasonal_reference",
    ]].copy()
    predictions = predictions.rename(columns={"date": "forecast_origin"})
    predictions["target_date"] = pd.to_datetime({
        "year": predictions.target_idx // 12,
        "month": predictions.target_idx % 12 + 1,
        "day": 1,
    })
    predictions = predictions.drop(columns="target_idx")
    predictions["predicted_OD_PAX"] = np.maximum(0, model.predict(test[FEATURES]))
    metrics = score_predictions(predictions)
    args.out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.out / "test_predictions.csv", index=False)
    metrics.to_csv(args.out / "test_metrics.csv", index=False)
    common = predictions.dropna(subset=["pax_seasonal_reference"])
    seasonal = common.assign(predicted_OD_PAX=common.pax_seasonal_reference)
    benchmark = pd.concat([
        score_predictions(common).assign(model="XGBoost"),
        score_predictions(seasonal).assign(model="Seasonal repeat"),
    ], ignore_index=True)
    benchmark.to_csv(args.out / "benchmark_metrics.csv", index=False)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print("\nBenchmark comparison (same observed rows):")
    print(benchmark.loc[benchmark.forecast_horizon == "all"].to_string(index=False))
    print(f"Saved test predictions and metrics to {args.out.resolve()}")


if __name__ == "__main__":
    main()
