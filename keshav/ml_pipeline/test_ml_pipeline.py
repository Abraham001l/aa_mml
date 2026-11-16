"""Checks for calendar alignment and forecast-time leakage."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from ml_pipeline import FEATURES, MARKET, add_history_features, make_examples


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
        for column in [c for c in FEATURES if c not in {"target_month", "forecast_horizon"}]:
            self.assertTrue((test.groupby(MARKET)[column].nunique() == 1).all())

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
