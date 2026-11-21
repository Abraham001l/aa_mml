import unittest

import numpy as np
import pandas as pd

from seasonal_challenger import forecast_at


class ChallengerTests(unittest.TestCase):
    def test_future_values_and_missing_reverse_route(self):
        months = pd.period_range("2019-01", "2024-12", freq="M")
        routes = pd.MultiIndex.from_tuples([("A", "B"), ("B", "A"), ("A", "C")])
        volume = np.random.default_rng(7).uniform(0, 300, (3, len(months)))
        panel = pd.DataFrame(volume, index=routes, columns=months)
        origin = pd.Period("2022-07", freq="M")
        settings = dict(family="route recovery", window=3, recent=.7, recovery=12, pool=.25)
        forecast = forecast_at(panel, origin, **settings)
        panel.loc[:, panel.columns > origin] = 1e9
        np.testing.assert_array_equal(forecast, forecast_at(panel, origin, **settings))
        unpooled = forecast_at(panel, origin, **dict(settings, pool=0))
        np.testing.assert_array_equal(forecast[2], unpooled[2])
        self.assertEqual(forecast.shape, (3, 24))
        self.assertTrue(np.isfinite(forecast).all() and (forecast >= 0).all())
