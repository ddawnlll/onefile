"""Offline regressions for the historical XGBoost regime experiment."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import one
from test_one import market_frame


def daily(q):
    return pd.DataFrame({"time": pd.date_range("2020-01-01", periods=len(q), tz="UTC"), "q": np.asarray(q, float)})


class RegimeLabelsTests(unittest.TestCase):
    def test_every_crossing_retains_failures_and_direct_reversal(self):
        q = np.zeros(100)
        q[20:30] = 1
        q[30:40] = -1  # direct reversal, no arbitrary neutral-day requirement
        q[60:62] = 1   # failed persistence, not dropped from the archive
        d, events = one.regime_labels(daily(q))
        self.assertEqual(events.status.tolist(), ["CONFIRMED", "CONFIRMED", "UNCONFIRMED"])
        self.assertEqual(d.y.iloc[13:20].tolist(), [1]*7)
        self.assertEqual(d.y.iloc[23:30].tolist(), [-1]*7)
        self.assertEqual(d.y.iloc[20], 0)  # cannot predict an already-current state
        self.assertTrue(d.y.iloc[53:60].eq(0).all())
        self.assertTrue(d.y.iloc[-12:].isna().all())
        self.assertTrue((d.maturity_time-d.time).eq(pd.Timedelta(days=12)).all())

    def test_missing_future_is_censored_not_negative(self):
        q = np.zeros(80); q[20:30] = 1; q[22] = np.nan
        d, events = one.regime_labels(daily(q))
        self.assertEqual(events.status.iloc[0], "CENSORED")
        self.assertTrue(d.y.iloc[10:23].isna().all())

    def test_new_features_are_prefix_causal(self):
        x = market_frame("2020-01-01", "2021-01-01", varying=True)
        before = one.regime_features(one.daily_market(x))
        changed = x.copy()
        changed.loc[900:, ["open", "high", "low", "close", "volume", "funding_rate"]] *= 4
        after = one.regime_features(one.daily_market(changed))
        assert_frame_equal(before.loc[:149, one.REGIME_FEATURES], after.loc[:149, one.REGIME_FEATURES])


class ChronologyTests(unittest.TestCase):
    @staticmethod
    def model_frame():
        n = 740; rng = np.random.default_rng(8)
        d = pd.DataFrame(rng.normal(size=(n, len(one.REGIME_FEATURES))), columns=one.REGIME_FEATURES)
        d["time"] = pd.date_range("2020-01-01", periods=n, tz="UTC")
        d["y"] = np.where(np.arange(n)%17 < 3, np.where(np.arange(n)//17%2, 1, -1), 0)
        d["target_onset"] = d.time.where(d.y.ne(0))+pd.Timedelta(days=3)
        d["maturity_time"] = d.time+pd.Timedelta(days=12)
        d["forecast_eligible"] = True
        d["q"] = 0.
        return d

    def test_purge_embargo_at_all_boundaries_and_censored_forecasts_retained(self):
        d = self.model_frame()
        start = pd.Timestamp("2021-01-01", tz="UTC")
        d.loc[d.time >= "2021-06-20", "y"] = np.nan
        tr, va, ca, te = one.regime_splits(d, start)
        for left, right in ((tr, va), (va, ca), (ca, te)):
            self.assertLess(left.maturity_time.max()+pd.Timedelta(days=1), right.time.min())
        self.assertTrue(te.y.isna().any())
        self.assertTrue(tr.y.notna().all())

    def test_outer_labels_cannot_change_fitted_predictions_or_tree_count(self):
        d = self.model_frame()
        tr, va, ca, te = one.regime_splits(d, pd.Timestamp("2021-01-01", tz="UTC"))
        a, p, info = one.regime_probabilities(tr, va, ca, te, "xgb")
        changed = te.copy(); changed["y"] = -changed.y
        b, other, detail = one.regime_probabilities(tr, va, ca, changed, "xgb")
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(p, other)
        np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-6)
        for head in ("event", "direction"):
            self.assertEqual(info["heads"][head].get("rounds"), detail["heads"][head].get("rounds"))


class EventMatchingTests(unittest.TestCase):
    def test_one_to_one_cooldown_and_strict_early_warning(self):
        rows = daily(np.zeros(35)); rows["y"] = 0
        p = np.tile([.05, .9, .05], (len(rows), 1))
        events = pd.DataFrame({"onset_i": [4], "onset_time": [rows.time.iloc[4]], "side": [1], "status": ["CONFIRMED"]})
        alarms, opportunities = one.regime_alarm_archive(rows, p, .8, events)
        self.assertEqual(len(opportunities), 1)
        self.assertEqual(sum(a["event_id"] is not None for a in alarms), 1)
        self.assertEqual(alarms[0]["lead_days"], 4)
        self.assertTrue(all(a["event_id"] is None for a in alarms[1:]))
        self.assertEqual(len(one.regime_alarm_archive(rows, p, np.nan, events)[0]), 0)

    def test_same_state_warnings_cannot_claim_persistence_as_new_onset(self):
        rows = daily(np.ones(20))
        p = np.tile([.05, .9, .05], (len(rows), 1))
        e = pd.DataFrame({"onset_i": [4], "onset_time": [rows.time.iloc[4]], "side": [1], "status": ["CONFIRMED"]})
        alarms, opportunities = one.regime_alarm_archive(rows, p, .8, e)
        self.assertEqual(len(alarms), 0)
        self.assertEqual(len(opportunities), 0)

    def test_wrong_first_direction_cannot_be_rescued_by_later_event(self):
        rows = daily(np.zeros(20))
        p = np.tile([.05, .9, .05], (len(rows), 1))
        e = pd.DataFrame({"onset_i": [4, 6], "onset_time": [rows.time.iloc[4], rows.time.iloc[6]],
                          "side": [-1, 1], "status": ["CONFIRMED", "CONFIRMED"]})
        alarms, _ = one.regime_alarm_archive(rows, p, .8, e)
        self.assertIsNone(alarms[0]["event_id"])
        self.assertEqual(sum(a["event_id"] is not None for a in alarms), 0)

    def test_sparse_calibration_cannot_manufacture_eighty_percent_policy(self):
        rows = daily(np.zeros(20)); rows["y"] = 0
        e = pd.DataFrame(columns=["onset_i", "onset_time", "side", "status"])
        p = np.tile([.9, .05, .05], (len(rows), 1))
        self.assertIsNone(one.regime_thresholds(rows, p, e)["precision80"])

    def test_sealed_cli_rejected_before_download(self):
        with patch.object(one, "download") as download, self.assertRaises(SystemExit):
            one.main(["--regime-research", "--unlock-oos", "--download"])
        download.assert_not_called()


class RegimeV3Tests(unittest.TestCase):
    def test_v3_features_are_prefix_causal(self):
        x = market_frame("2020-01-01", "2021-01-01", varying=True)
        before = one.regime3_features(one.daily_market(x))
        changed = x.copy()
        changed.loc[900:, ["open", "high", "low", "close", "volume", "funding_rate"]] *= 4
        after = one.regime3_features(one.daily_market(changed))
        assert_frame_equal(before.loc[:149, one.REGIME3_BASE], after.loc[:149, one.REGIME3_BASE])

    def test_fast_matcher_reproduces_v2_archive(self):
        rng = np.random.default_rng(3)
        q = np.zeros(200); q[30:40] = 1; q[60:66] = -1; q[100:110] = 1; q[112:120] = -1
        rows = daily(q); rows["symbol"] = "BTCUSDT"
        _, e = one.regime_labels(rows)
        e["symbol"] = "BTCUSDT"
        p = rng.dirichlet([4, 1, 1], len(rows))
        for t in (.1, .2, .3):
            old, _ = one.regime_alarm_archive(rows, p, t, e)
            new, _ = one.regime3_alarms(rows, p, t, e)
            self.assertEqual([(a["time"][:10], a["side"], a["event_id"]) for a in old],
                             [(a["time"][:10], a["side"], a["event_id"]) for a in new])

    def test_tune_scope_never_scores_untouched_years(self):
        n = 1400; rng = np.random.default_rng(5)
        d = pd.DataFrame(rng.normal(size=(n, len(one.REGIME3_BASE))), columns=one.REGIME3_BASE)
        d["time"] = pd.date_range("2020-01-01", periods=n, tz="UTC")
        d["symbol"], d["q"] = "BTCUSDT", 0.
        d["y"] = np.where(np.arange(n) % 23 < 2, 1, 0).astype(float)
        d["target_onset"] = d.time.where(d.y.ne(0))+pd.Timedelta(days=2)
        d["event_key"] = np.where(d.y.ne(0), "BTCUSDT|"+d.target_onset.astype(str), "")
        d["maturity_time"] = d.time+pd.Timedelta(days=12)
        e = pd.DataFrame({"onset_i": [], "onset_time": pd.Series([], dtype="datetime64[ns, UTC]"), "side": [],
                          "status": [], "symbol": []})
        f, folds = one.regime3_walk_forward(d, e, ("prior",), "tune")
        self.assertLess(f.time.max(), pd.Timestamp(one.REGIME3_TUNE_END, tz="UTC"))
        self.assertEqual(len(folds), 4)

    def test_v3_cli_rejects_sealed(self):
        with self.assertRaises(SystemExit):
            one.main(["--regime3", "--unlock-oos"])


if __name__ == "__main__":
    unittest.main()
