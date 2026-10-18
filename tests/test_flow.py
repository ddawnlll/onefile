"""Offline regressions for the DEV direction/continuation flow study (no network)."""
import unittest

import numpy as np
import pandas as pd

import one
from test_one import market_frame


def flow_inputs(days: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = len(days)
    spot = 1e6 * (1 + .3*rng.random(n))
    perp = 2e6 * (1 + .3*rng.random(n))
    return pd.DataFrame({"spot_quote": spot, "spot_taker": spot*.5*(1+.2*rng.standard_normal(n)).clip(.1, .9),
                         "perp_quote": perp, "perp_taker": perp*.5*(1+.2*rng.standard_normal(n)).clip(.1, .9),
                         "oi_btc": 30_000 + np.cumsum(rng.normal(0, 50, n))}, index=days.rename("day"))


class FlowFeatureTests(unittest.TestCase):
    def test_features_are_prefix_causal(self):
        x = market_frame("2020-01-01", "2021-06-01", varying=True)
        days = pd.date_range("2020-01-01", "2021-06-01", freq="D", inclusive="left", tz="UTC")
        flows = flow_inputs(days)
        full = one.flow_panel(x, flows)
        cut = pd.Timestamp("2020-12-01", tz="UTC")
        changed = x.copy()
        changed.loc[changed.time >= cut, ["open", "high", "low", "close", "volume"]] *= 3
        mutated = flows.copy()
        mutated.loc[mutated.index >= cut, ["spot_quote", "perp_quote", "oi_btc"]] *= 5
        after = one.flow_panel(changed, mutated)
        before_cut = full.time < cut - pd.Timedelta(days=1)
        features = one.FLOW_ALL + ["state", "early_dir", "early_cohort"]
        pd.testing.assert_frame_equal(full.loc[before_cut, features].reset_index(drop=True),
                                      after.loc[before_cut, features].reset_index(drop=True))

    def test_causal_age_is_run_length_and_capped(self):
        x = market_frame("2020-01-01", "2020-06-01", varying=True)
        flows = flow_inputs(pd.date_range("2020-01-01", "2020-06-01", freq="D", inclusive="left", tz="UTC"))
        panel = one.flow_panel(x, flows)
        state, age = panel.state.to_numpy(float), panel.causal_age.to_numpy(float)
        self.assertTrue(np.all(age[state == 0] == 0))
        self.assertTrue(np.nanmax(age) <= 21)
        continuing = (state[1:] == state[:-1]) & (state[1:] != 0) & np.isfinite(age[:-1])
        self.assertTrue(np.array_equal(age[1:][continuing], np.minimum(age[:-1][continuing] + 1, 21)))
        started = (state[1:] != state[:-1]) & (state[1:] != 0) & np.isfinite(age[1:])
        self.assertTrue(np.all(age[1:][started] == 1))


class FlowLabelAndSplitTests(unittest.TestCase):
    @staticmethod
    def panel(n=600):
        rng = np.random.default_rng(5)
        time = pd.date_range("2020-01-02", periods=n, tz="UTC")
        frame = pd.DataFrame({"time": time, "day": time - pd.Timedelta(days=1), "paired": True, "state": 0.,
                              "close": 100.})
        for name in one.FLOW_ALL:
            frame[name] = rng.normal(size=n)
        for h in one.FLOW_HORIZONS:
            frame[f"y{h}"] = rng.choice([-1., 0., 1.], size=n)
        return frame

    def test_training_labels_and_embargo_end_before_later_segments(self):
        frame = self.panel()
        start = pd.Timestamp("2020-08-01", tz="UTC")
        end = start + pd.Timedelta(days=90)
        train, stop, cal, test = one.flow_splits(frame, 7, start, end)
        settled = frame.time + pd.Timedelta(days=8)
        stop_start, cal_start = start - pd.Timedelta(days=180), start - pd.Timedelta(days=90)
        self.assertTrue((settled.loc[train.index] <= stop_start).all())
        self.assertTrue((settled.loc[stop.index] <= cal_start).all())
        self.assertTrue((settled.loc[cal.index] <= start).all())
        self.assertTrue(test.time.min() >= start and test.time.max() < end)

    def test_censored_labels_are_excluded_from_training_not_turned_negative(self):
        frame = self.panel()
        frame.loc[frame.index[-20:], "y7"] = np.nan
        train, stop, cal, test = one.flow_splits(frame, 7, pd.Timestamp("2020-08-01", tz="UTC"), pd.Timestamp("2020-10-30", tz="UTC"))
        self.assertTrue(train.y7.notna().all() and stop.y7.notna().all() and cal.y7.notna().all())


class FlowEpisodeAndCostTests(unittest.TestCase):
    def test_episodes_are_maximal_same_sign_runs(self):
        dnum = np.arange(10)
        y = np.array([0, 1, 1, 1, 0, -1, -1, 1, 1, 0])
        self.assertEqual(one._flow_episodes(dnum, y), [(1, 4, 1), (5, 7, -1), (7, 9, 1)])

    def test_episode_breaks_on_calendar_gap(self):
        dnum = np.array([0, 1, 3, 4])
        y = np.array([1, 1, 1, 1])
        self.assertEqual(one._flow_episodes(dnum, y), [(0, 2, 1), (2, 4, 1)])

    def test_trade_cost_and_funding_signs(self):
        times = pd.date_range("2024-01-01", periods=40, freq="4h", tz="UTC")
        bars = pd.DataFrame({"time": times, "open": 100., "funding_rate": 0.})
        # Settlement at 08:00 on decision day 2024-01-02 lies inside the hold [04:00, next-day 00:00).
        bars.loc[bars.time.eq(pd.Timestamp("2024-01-02 08:00", tz="UTC")), "funding_rate"] = .001
        panel = pd.DataFrame({"time": pd.date_range("2024-01-02", periods=5, tz="UTC"), "close": 110.})
        events = pd.DataFrame({"day": [pd.Timestamp("2024-01-01", tz="UTC")],
                               "time": [pd.Timestamp("2024-01-02", tz="UTC")], "direction": [1]})
        long_trade = one.flow_trades(events, panel, bars, horizon=1, entry_bars=1).iloc[0]
        self.assertAlmostEqual(long_trade.gross, 0.1)
        self.assertAlmostEqual(long_trade.funding, .001)   # longs pay positive funding
        self.assertAlmostEqual(long_trade.net, .1 - 2*(one.FEE_BPS+one.SLIPPAGE_BPS)/1e4 - .001)
        short_trade = one.flow_trades(events.assign(direction=-1), panel, bars, horizon=1, entry_bars=1).iloc[0]
        self.assertAlmostEqual(short_trade.gross, -0.1)
        self.assertAlmostEqual(short_trade.funding, -.001)  # shorts receive it
        self.assertAlmostEqual(short_trade.net, -.1 - 2*(one.FEE_BPS+one.SLIPPAGE_BPS)/1e4 + .001)

    def test_block_bootstrap_estimate_is_pooled_ratio(self):
        sums = np.array([2., 0., 4., 6.])
        counts = np.array([1., 0., 2., 2.])
        result = one.flow_block_ci(sums, counts, block=2)
        self.assertAlmostEqual(result["estimate"], 12/5)
        self.assertEqual(result["valid_draws"] > 0, True)


if __name__ == "__main__":
    unittest.main()
