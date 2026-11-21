"""Checks for calendar alignment and forecast-time leakage."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from ml_pipeline import FEATURES, HISTORY_FEATURES, MARKET, add_history_features, make_examples, score_predictions


class ForecastDataTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2019-01-01", "2024-01-01", freq="MS")
        frames = []
        for origin, destination, scale in [("IAH", "LAX", 1), ("LAX", "IAH", 100)]:
            frames.append(pd.DataFrame({
                "ORG_AIRPRT": origin, "DES_AIRPRT": destination,
                "date": dates, "month_idx": dates.year * 12 + dates.month - 1,
                "OD_PAX": (np.arange(len(dates)) + 10.0) * scale,
                "distance_km": 2215.0, "is_domestic": 1,
            }))
        self.data = pd.concat(frames, ignore_index=True)
        self.cutoff = 2022 * 12  # January 2022, using zero-based month indices.

    def test_calendar_lags_and_missing_windows_keep_directions_separate(self):
        data = self.data.loc[~(
            (self.data.ORG_AIRPRT == "IAH") & (self.data.date == "2020-06-01")
        )]
        history = add_history_features(data)
        july = history.loc[(history.ORG_AIRPRT == "IAH") & (history.date == "2020-07-01")].iloc[0]
        self.assertTrue(np.isnan(july.pax_lag_1))
        self.assertTrue(np.isnan(july.pax_rolling_3))
        self.assertTrue(np.isnan(july.pax_rolling_12))
        january = history.loc[(history.ORG_AIRPRT == "IAH") & (history.date == "2021-01-01")].iloc[0]
        self.assertEqual(january.pax_lag_12, 22.0)
        self.assertAlmostEqual(january.pax_yoy_growth, 34.0 / 22.0 - 1)
        reverse = history.loc[(history.ORG_AIRPRT == "LAX") & (history.date == "2021-01-01")].iloc[0]
        self.assertEqual(reverse.pax_lag_12, 2200.0)
        self.assertEqual(reverse.pax_rolling_3, np.mean([3200, 3300, 3400]))

    def test_zero_year_ago_demand_has_no_infinite_growth(self):
        self.data.loc[self.data.date == "2020-01-01", "OD_PAX"] = 0
        history = add_history_features(self.data)
        growth = history.loc[history.date == "2021-01-01", "pax_yoy_growth"]
        self.assertTrue(growth.isna().all())

    def test_all_horizons_have_correct_targets_and_training_stops_at_cutoff(self):
        history = add_history_features(self.data.loc[self.data.month_idx <= self.cutoff])
        train = make_examples(history, self.data, self.cutoff, training=True, origin_step=1)
        test = make_examples(history, self.data, self.cutoff, training=False)
        self.assertTrue((train.target_idx <= self.cutoff).all())
        self.assertEqual(set(train.forecast_horizon), set(range(1, 25)))
        self.assertEqual(len(test), 48)
        self.assertTrue((test.month_idx == self.cutoff).all())
        self.assertTrue((test.target_idx == self.cutoff + test.forecast_horizon).all())
        self.assertTrue((test.target_month == test.target_idx % 12 + 1).all())
        expected = self.data.set_index(MARKET + ["month_idx"]).OD_PAX
        for row in test.itertuples():
            self.assertEqual(row.actual_OD_PAX, expected.loc[(row.ORG_AIRPRT, row.DES_AIRPRT, row.target_idx)])
        for column in HISTORY_FEATURES:
            self.assertTrue((test.groupby(MARKET)[column].nunique() == 1).all())

    def test_target_season_uses_available_history_even_beyond_twelve_months(self):
        history = add_history_features(self.data.loc[self.data.month_idx <= self.cutoff])
        test = make_examples(history, self.data, self.cutoff, training=False)
        forward = test.loc[test.ORG_AIRPRT == "IAH"].set_index("forecast_horizon")
        self.assertEqual(forward.loc[1, "pax_seasonal_reference"], 35.0)  # February 2021.
        self.assertEqual(forward.loc[13, "pax_seasonal_reference"], 35.0)
        self.assertEqual(forward.loc[1, "pax_seasonal_mean_3"], np.mean([35, 23, 11]))
        self.assertEqual(forward.loc[24, "pax_seasonal_reference"], 46.0)  # January 2022.
        self.assertTrue((test.seasonal_reference_idx <= test.month_idx).all())

    def test_missing_seasonal_anchor_falls_back_to_an_older_observed_year(self):
        data = self.data.loc[~(
            (self.data.ORG_AIRPRT == "IAH") & (self.data.date == "2021-02-01")
        )]
        history = add_history_features(data.loc[data.month_idx <= self.cutoff])
        test = make_examples(history, data, self.cutoff, training=False)
        row = test.loc[(test.ORG_AIRPRT == "IAH") & (test.forecast_horizon == 13)].iloc[0]
        self.assertEqual(row.pax_seasonal_reference, 23.0)  # February 2020.
        self.assertEqual(row.pax_seasonal_mean_3, np.mean([23, 11]))

    def test_new_lags_rolling_growth_and_airport_aggregates(self):
        forward = self.data.loc[self.data.ORG_AIRPRT == "IAH"].copy()
        extra_origin = forward.assign(DES_AIRPRT="JFK", OD_PAX=forward.OD_PAX * 10)
        extra_destination = forward.assign(ORG_AIRPRT="JFK", OD_PAX=forward.OD_PAX * 5)
        history = add_history_features(pd.concat([self.data, extra_origin, extra_destination]))
        row = history.loc[(history.ORG_AIRPRT == "IAH") & (history.DES_AIRPRT == "LAX") & (history.date == "2021-01-01")].iloc[0]
        self.assertEqual(row.market_id, "IAH_LAX")
        for lag in [2, 3, 6, 24]:
            self.assertEqual(row[f"pax_lag_{lag}"], 34 - lag)
        self.assertAlmostEqual(row.pax_rolling_3_yoy_growth, 33 / 21 - 1)
        self.assertEqual(row.origin_pax_total, 34 * 11)
        self.assertEqual(row.destination_pax_total, 34 * 6)
        self.assertAlmostEqual(row.origin_pax_yoy_growth, 34 / 22 - 1)
        self.assertAlmostEqual(row.destination_pax_yoy_growth, 34 / 22 - 1)

    def test_later_training_months_cannot_change_an_earlier_origins_features(self):
        data = self.data.loc[self.data.month_idx <= self.cutoff].copy()
        original = make_examples(add_history_features(data), data, self.cutoff, training=True, origin_step=1)
        data.loc[data.date > "2020-01-01", "OD_PAX"] *= 1000
        modified = make_examples(add_history_features(data), data, self.cutoff, training=True, origin_step=1)
        original = original.loc[original.date == "2020-01-01", FEATURES].reset_index(drop=True)
        modified = modified.loc[modified.date == "2020-01-01", FEATURES].reset_index(drop=True)
        assert_frame_equal(original, modified)

    def test_metric_values_and_undefined_cases(self):
        predictions = pd.DataFrame({
            "actual_OD_PAX": [1.0, 3.0], "predicted_OD_PAX": [2.0, 2.0],
            "forecast_horizon": [1, 1],
        })
        overall = score_predictions(predictions).iloc[0]
        self.assertEqual(overall.MAE, 1.0)
        self.assertEqual(overall.RMSE, 1.0)
        self.assertEqual(overall.R2, 0.0)
        self.assertEqual(overall.WAPE_pct, 50.0)
        self.assertEqual(overall.Bias_pct, 0.0)
        negative_r2 = predictions.assign(predicted_OD_PAX=10.0)
        self.assertLess(score_predictions(negative_r2).iloc[0].R2, 0)
        zero_actuals = predictions.assign(actual_OD_PAX=0.0)
        zero_metrics = score_predictions(zero_actuals).iloc[0]
        self.assertTrue(np.isnan(zero_metrics.R2))
        self.assertTrue(np.isnan(zero_metrics.WAPE_pct))

    def test_changing_future_demand_cannot_change_training_or_test_features(self):
        history = add_history_features(self.data.loc[self.data.month_idx <= self.cutoff])
        original_train = make_examples(history, self.data, self.cutoff, training=True)
        original_test = make_examples(history, self.data, self.cutoff, training=False)
        modified = self.data.copy()
        modified.loc[modified.month_idx > self.cutoff, "OD_PAX"] = 1e9
        modified_history = add_history_features(modified.loc[modified.month_idx <= self.cutoff])
        modified_train = make_examples(modified_history, modified, self.cutoff, training=True)
        modified_test = make_examples(modified_history, modified, self.cutoff, training=False)
        assert_frame_equal(original_train, modified_train)
        assert_frame_equal(original_test[FEATURES], modified_test[FEATURES])
        self.assertFalse(original_test.actual_OD_PAX.equals(modified_test.actual_OD_PAX))


if __name__ == "__main__":
    unittest.main()
