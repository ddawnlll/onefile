"""H1-A and symmetric execution: synthetic/offline tests, never open holdouts."""
import argparse
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from test_one import TemporaryFiles, market_frame

import one


class DailyCausalityTests(unittest.TestCase):
    def setUp(self):
        self.x = market_frame("2020-01-01", "2021-01-01", varying=True)

    def test_completed_days_and_availability(self):
        d = one.daily_market(self.x.iloc[:-1])
        self.assertEqual(len(d), 365)
        self.assertTrue(d.bars.eq(6).all())
        self.assertEqual(d.time.iloc[0], pd.Timestamp("2020-01-02", tz="UTC"))
        self.assertEqual(d.signal_i.iloc[0], 5)

    def test_log_return_er_matches_monograph(self):
        d = one.daily_market(self.x)
        r = np.log(d.close / d.close.shift(1))
        expected = r.rolling(20).sum().abs() / (r.abs().rolling(20).sum()+1e-12)
        np.testing.assert_allclose(d.er20, expected, equal_nan=True)
        self.assertTrue(d.q.iloc[:20].isna().all())
        self.assertFalse(d.risk.iloc[:29].any())

    def test_per_feature_future_corruption_leaves_all_earlier_features_fixed(self):
        before = one.daily_market(self.x)
        cutoff = 6*120
        cols = one.H1A_MODELS["M2"] + ["time", "risk", "q", "atr", "funding3"]
        for col in ["open", "high", "low", "close", "volume", "funding_rate"]:
            bad = self.x.copy()
            bad.loc[cutoff:, col] = bad.loc[cutoff:, col]*7+1
            after = one.daily_market(bad)
            with self.subTest(col=col):
                assert_frame_equal(before.loc[:119, cols], after.loc[:119, cols])

    def test_warmup_is_not_a_neutral_episode(self):
        d = one.daily_market(self.x)
        first_valid = d.q.first_valid_index()
        self.assertEqual(first_valid, 20)
        self.assertEqual(d.neutral_days.iloc[first_valid-1], 0)
        self.assertFalse(d.risk.iloc[:first_valid+9].any())


class OnsetLabelTests(unittest.TestCase):
    @staticmethod
    def daily(q):
        q = np.asarray(q, float)
        age, n = [], 0
        for v in q:
            n = n+1 if v == 0 else 0
            age.append(n)
        return pd.DataFrame({"time": pd.date_range("2020-01-01", periods=len(q), freq="D", tz="UTC"),
                             "q": q, "risk": (q == 0) & (np.array(age) >= 10)})

    def test_first_crossing_confirmation_and_seven_day_label(self):
        q = np.zeros(100)
        q[30:41] = 1
        q[60] = -1
        q[95:] = 1
        d, e = one.onset_labels(self.daily(q))
        self.assertEqual(e.status.tolist(), ["CONFIRMED", "UNCONFIRMED", "CENSORED"])
        self.assertEqual(e.confirmation_time.iloc[0], d.time.iloc[40])
        self.assertTrue(d.y.iloc[23:30].eq(1).all())
        self.assertEqual(d.y.iloc[22], 0)
        self.assertEqual(d.y.iloc[59], 0)
        self.assertTrue(d.y.iloc[-17:].isna().all())
        self.assertTrue(pd.isna(d.y.iloc[30]))  # onset-day detection is NOT H1-A

    def test_requires_eight_of_next_ten_not_including_onset(self):
        q = np.zeros(70)
        q[30:38] = 1  # onset + seven subsequent: NOT enough
        _, e = one.onset_labels(self.daily(q))
        self.assertEqual(e.status.iloc[0], "UNCONFIRMED")
        q[40] = 1
        _, e = one.onset_labels(self.daily(q))
        self.assertEqual(e.status.iloc[0], "CONFIRMED")

    def test_future_labels_only_change_when_not_mature(self):
        q = np.zeros(100)
        q[50:61] = 1
        before, _ = one.onset_labels(self.daily(q))
        q[65:] = -1
        after, _ = one.onset_labels(self.daily(q))
        assert_frame_equal(before.loc[:47, ["y", "target_onset", "target_confirmation"]],
                           after.loc[:47, ["y", "target_onset", "target_confirmation"]])


class ForecastTests(unittest.TestCase):
    def test_scaler_does_not_fit_test_data(self):
        train = pd.DataFrame({"f": [-2., -1., 0., 1., 2., 3.], "y": [-1, 0, 1, -1, 0, 1]})
        test = pd.DataFrame({"f": [.5]})
        p = one.forecast_prob(train, test, ["f"], 1.)
        contaminated_test = pd.DataFrame({"f": [.5, 1e9]})
        np.testing.assert_allclose(p[0], one.forecast_prob(train, contaminated_test, ["f"], 1.)[0])
        np.testing.assert_allclose(p.sum(axis=1), 1)

    def test_missing_classes_use_logged_prior_fallback(self):
        train = pd.DataFrame({"f": [1., 2., 3.], "y": [0, 0, 1]})
        p = one.forecast_prob(train, train, ["f"], 1.)
        np.testing.assert_allclose(p[0], [3/6, 2/6, 1/6])
        self.assertTrue((p > 0).all())

    def test_brier_class_order_and_log_loss(self):
        bs, ll = one.forecast_losses([0, 1, -1], np.eye(3))
        np.testing.assert_allclose(bs, 0)
        np.testing.assert_allclose(ll, 0)

    def test_calendar_blocks_keep_gaps_and_paired_losses(self):
        frame = pd.DataFrame({"time": pd.to_datetime(["2020-01-01", "2020-01-30"], utc=True),
                              "M1": [.2, .8], "M2": [.2, .8]})
        result = one.block_interval(frame, ["M1", "M2"], ratio=True)
        self.assertEqual(result["estimate"], 0)
        self.assertEqual(result["ci95"], [0, 0])
        self.assertEqual([r["block_days"] for r in result["sensitivity"]], [28, 30, 30])
        self.assertEqual(result, one.block_interval(frame, ["M1", "M2"], ratio=True))

    def test_warning_utility_and_low_power_are_separate_gates(self):
        good_gain = {"estimate": .1, "ci95": [.01, .2]}
        silent = {"detected_events": 0, "false_per_asset_year": 0., "median_lead_days": None}
        self.assertEqual(one.h1a_gate(20, 20, 12, good_gain, silent)["verdict"], "REJECT_EARLY_WARNING_UTILITY")
        result = one.h1a_gate(2, 2, 3, good_gain, silent)
        self.assertIn("INCONCLUSIVE", result["verdict"])
        self.assertFalse(result["heldout_utility_gate"])
        good_alarm = {"detected_events": 2, "false_per_asset_year": 2., "median_lead_days": 3.}
        self.assertTrue(one.h1a_gate(20, 20, 12, good_gain, good_alarm)["heldout_utility_gate"])
        for alarms in ({**good_alarm, "false_per_asset_year": 3.01},
                       {**good_alarm, "median_lead_days": 1.},
                       {**good_alarm, "median_lead_days": None}):
            with self.subTest(alarms=alarms):
                result = one.h1a_gate(20, 20, 12, good_gain, alarms)
                self.assertEqual(result["verdict"], "REJECT_EARLY_WARNING_UTILITY")
                self.assertFalse(result["heldout_utility_gate"])
        boundary = {**good_alarm, "false_per_asset_year": 3., "median_lead_days": 2.}
        self.assertTrue(one.h1a_gate(20, 20, 12, good_gain, boundary)["heldout_utility_gate"])
        rows = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=10, tz="UTC")})
        events = pd.DataFrame(columns=["status", "side", "onset_time", "onset_i"])
        disabled = one.alarm_metrics(rows, np.tile([0., 1., 0.], (10, 1)), 1.01, events, 1.)
        self.assertEqual(one.h1a_gate(20, 20, 12, good_gain, disabled)["verdict"],
                         "REJECT_EARLY_WARNING_UTILITY")

    def test_strict_pre_onset_matching_and_cooldown(self):
        rows = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=15, freq="D", tz="UTC")})
        p = np.tile([.05, .9, .05], (15, 1))
        events = pd.DataFrame({"onset_i": [9], "onset_time": [rows.time.iloc[9]], "side": [1], "status": ["CONFIRMED"]})
        m = one.alarm_metrics(rows, p, .5, events, 1.)
        self.assertEqual(m["alarms"], 3)  # days 0,7,14
        self.assertEqual(m["detected_events"], 1)
        self.assertEqual(m["median_lead_days"], 2)
        self.assertEqual(m["false_alarms"], 2)
        at_onset = one.alarm_metrics(rows.iloc[9:], p[:6], .5, events, 1.)
        self.assertEqual(at_onset["detected_events"], 0)

    def test_array_threshold_ties_preserve_event_order_and_deduplicate_onsets(self):
        origin = pd.Timestamp("2020-01-01", tz="UTC")
        rows = pd.DataFrame({"time": [origin + pd.Timedelta(days=d) for d in (0, 6, 7, 14, 21, 28, 29, 35)]},
                            index=np.arange(101, 109))
        p = np.array([[0., .5, .5], [0., .9, .1], [0., .2, .8], [0., .6, .4],
                      [0., .5, .5], [0., .9, .1], [0., .3, .7], [0., .1, .9]])
        thresholds = np.array([.5, .5, .8, .7, .5, 1.01, .7, .9])
        events = pd.DataFrame({"onset_i": [30, 2, 30, 7],
                               "onset_time": [origin + pd.Timedelta(days=d) for d in (6, 2, 24, 8)],
                               "side": [1, 1, 1, -1], "status": "CONFIRMED"})
        result = one.alarm_metrics(rows, p, thresholds, events, 2.)
        self.assertEqual([a["matched_onset_i"] for a in result["alarm_archive"]], [30, 7, 30, None])
        self.assertEqual([a["lead_days"] for a in result["alarm_archive"]], [6, 1, None, None])
        self.assertEqual(result["alarms"], 4)
        self.assertEqual(result["detected_events"], 2)
        self.assertEqual(result["false_alarms"], 1)
        self.assertEqual(result["false_per_asset_year"], .5)
        self.assertEqual(result["precision"], .75)
        self.assertEqual(result["median_lead_days"], 3.5)

    @staticmethod
    def synthetic_daily():
        rng = np.random.default_rng(42)
        time = pd.date_range("2020-01-01", "2023-08-20", freq="D", tz="UTC")
        n = len(time)
        d = pd.DataFrame({"time": time, "risk": np.arange(n) % 3 == 0,
                          "y": np.resize([0., 1., -1., 0., -1.], n),
                          "target_onset": pd.NaT, "target_confirmation": pd.NaT})
        for col in one.H1A_MODELS["M2"]:
            d[col] = rng.normal(size=n)
        d.loc[~d.risk | (d.time + pd.Timedelta(days=17) > time.max()), "y"] = np.nan
        events = pd.DataFrame(columns=["onset_i", "onset_time", "side", "status"])
        return d, events

    def test_fast_h1a_uses_fixed_c_and_only_inner_m2_alert_fits(self):
        d,events=self.synthetic_daily(); calls=[]
        def predict(train,test,cols,c):
            calls.append((list(cols),c))
            return np.tile([.6,.25,.15],(len(test),1))
        with patch.object(one,"forecast_prob",side_effect=predict):
            forecasts,result=one.h1a_experiment(d,events,fast=True)
        self.assertTrue(calls)
        self.assertTrue(all(c==1. for _,c in calls))
        inner_count=sum(f["models"]["M2"]["inner_folds"] for f in result["folds"] if f.get("models"))
        outer_count=sum(bool(f.get("models")) for f in result["folds"])
        self.assertEqual(sum(cols==one.H1A_MODELS["M2"] for cols,_ in calls),inner_count+outer_count)
        self.assertEqual(sum(cols==one.H1A_MODELS["M0"] for cols,_ in calls),outer_count)
        self.assertTrue(all(f["selection_mode"]=="fixed_C1_inner_M2_alert_only" for f in result["folds"] if f.get("status")!="INSUFFICIENT_ROWS"))
        self.assertTrue(np.isfinite(forecasts.filter(regex="_(none|up|down)$")).all().all())

    def test_archives_censored_predictions_but_scores_only_mature_rows(self):
        d, events = self.synthetic_daily()
        calls = []
        alarm_metrics = one.alarm_metrics

        def predict(train, test, cols, c):
            self.assertTrue(train.y.notna().all())
            return np.tile([.6, .25, .15], (len(test), 1))

        def alarms(rows, p, threshold, events, years):
            self.assertTrue(rows.y.notna().all())
            calls.append((rows.copy(), years))
            return alarm_metrics(rows, p, threshold, events, years)

        with patch.object(one, "forecast_prob", side_effect=predict), patch.object(
                one, "alarm_metrics", side_effect=alarms):
            forecasts, result = one.h1a_experiment(d, events)
        expected = d.loc[d.risk & d.time.ge(pd.Timestamp("2023-01-01", tz="UTC"))]
        self.assertEqual(forecasts.time.tolist(), expected.time.tolist())
        self.assertGreater(result["N_censored"], 0)
        self.assertEqual(result["N_forecasts"], len(expected))
        self.assertEqual(result["N_scored"], int(expected.y.notna().sum()))
        self.assertEqual(result["N_censored"], int(expected.y.isna().sum()))
        self.assertTrue(np.isfinite(forecasts.filter(regex="_(none|up|down)$")).all().all())
        scored = forecasts.loc[forecasts.y.notna()]
        bs, ll = one.forecast_losses(scored.y, scored[["M2_none", "M2_up", "M2_down"]].to_numpy())
        self.assertAlmostEqual(result["metrics"]["M2"]["brier"], bs.mean())
        self.assertAlmostEqual(result["metrics"]["M2"]["log_loss"], ll.mean())
        self.assertEqual(sum(b["N"] for b in result["metrics"]["M2"]["calibration_bins"]), len(scored))
        cap = d.time.max() - pd.Timedelta(days=16)
        expected_days = (cap-pd.Timestamp("2023-01-01", tz="UTC")).days
        self.assertEqual(result["exposure_days"], expected_days)
        self.assertEqual(result["exposure_ranges"][-1]["end_exclusive"], str(cap))
        self.assertTrue(all(years == expected_days/365.25 for _, years in calls[-3:]))
        for fold in result["folds"]:
            if fold.get("status") == "INSUFFICIENT_ROWS":
                continue
            start = pd.Timestamp(fold["start"])
            inner_end = start-pd.Timedelta(days=17)
            expected_inner = (inner_end-pd.Timestamp("2021-07-01", tz="UTC")).days
            self.assertEqual(fold["inner_exposure_days"], expected_inner)
            self.assertEqual(fold["inner_exposure"][-1]["end_exclusive"], str(inner_end))
            self.assertGreater(fold["inner_exposure_days"], fold["train_rows"] // 3)

    def test_all_censored_test_fold_still_archives_forecasts(self):
        d, events = self.synthetic_daily()
        d.loc[d.time.ge(pd.Timestamp("2023-01-01", tz="UTC")), "y"] = np.nan
        with patch.object(one, "forecast_prob", side_effect=lambda tr, te, cols, c:
                          np.tile([.6, .25, .15], (len(te), 1))):
            forecasts, result = one.h1a_experiment(d, events)
        self.assertGreater(len(forecasts), 0)
        self.assertTrue(forecasts.y.isna().all())
        self.assertEqual(result["N_scored"], 0)
        self.assertEqual(result["N_censored"], len(forecasts))
        self.assertIsNone(result["folds"][0]["models"]["M2"]["brier"])
        self.assertNotIn("metrics", result)

    def test_real_forecast_prefix_and_fitted_choices_ignore_future_mutation(self):
        d, events = self.synthetic_daily()
        cutoff = pd.Timestamp("2023-07-01", tz="UTC")
        changed = d.copy(deep=True)
        future = changed.time.ge(cutoff)
        changed.loc[future, one.H1A_MODELS["M2"]] = 1e6
        changed.loc[future & changed.y.notna(), "y"] = 1
        changed.loc[future, "risk"] = True
        first, result = one.h1a_experiment(d, events)
        second, mutated = one.h1a_experiment(changed, events)
        assert_frame_equal(first.loc[first.time.lt(cutoff)].reset_index(drop=True),
                           second.loc[second.time.lt(cutoff)].reset_index(drop=True))
        self.assertEqual(result["folds"][0], mutated["folds"][0])
        self.assertFalse(result["folds"][0]["prior_fallback"])
        self.assertTrue(all(m["inner_folds"] > 0 for m in result["folds"][0]["models"].values()))

    def test_nested_fits_purge_labels_and_ignore_post_fold_data(self):
        rng = np.random.default_rng(42)
        n = 1827
        d = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC"),
                          "risk": True, "y": np.resize([0, 0, 1, 0, -1], n),
                          "target_onset": pd.NaT, "target_confirmation": pd.NaT})
        for col in one.H1A_MODELS["M2"]:
            d[col] = rng.normal(size=n)
        e = pd.DataFrame({"onset_i": [1100], "onset_time": [d.time.iloc[1100]], "side": [1], "status": ["CONFIRMED"]})
        calls = []

        def fake(train, test, cols, c):
            self.assertLess(train.time.max()+pd.Timedelta(days=17), test.time.min())
            calls.append((train.time.max(), test.time.min()))
            return np.tile([.8, .1, .1], (len(test), 1))

        def fake_grid(train, test, cols, c_values):
            prediction = fake(train, test, cols, c_values[0])
            return {c: prediction for c in c_values}

        with (patch.object(one, "forecast_prob", side_effect=fake),
              patch.object(one, "forecast_prob_grid", side_effect=fake_grid)):
            forecasts, result = one.h1a_experiment(d, e)
        self.assertGreater(len(calls), 30)
        self.assertEqual(len(result["folds"]), 4)
        self.assertTrue(forecasts.time.ge(pd.Timestamp("2023-01-01", tz="UTC")).all())
        self.assertFalse(result["minimum_evidence_met"])
        self.assertIn("INCONCLUSIVE", result["verdict"])


class ShortExecutionTests(unittest.TestCase):
    def frame(self):
        x = market_frame("2020-01-01", "2020-02-01").iloc[:50].copy()
        return x.assign(atr=2., high=100., low=100., funding_rate=0.)

    def test_short_entry_bar_tie_and_target(self):
        x = self.frame()
        x.loc[1, ["high", "low"]] = [104., 89.]
        r = one.evaluate_events(x, [0], side=-1)[0]
        self.assertEqual(r["exit"], "STOP")
        self.assertLess(r["net_R"], -1)
        x.loc[1, "high"] = 100.
        r = one.evaluate_events(x, [0], side=-1)[0]
        self.assertEqual(r["exit"], "TARGET")
        self.assertLess(r["net_R"], 3)
        self.assertGreater(r["net_R"], 2.8)

    def test_short_adverse_gap(self):
        x = self.frame()
        x.loc[2, ["open", "high", "low", "close"]] = [110., 111., 109., 110.]
        r = one.evaluate_events(x, [0], side=-1)[0]
        self.assertEqual(r["exit"], "STOP_GAP")
        self.assertAlmostEqual(r["exit_price"], 110*(1+one.SLIPPAGE_BPS/10000))

    def test_short_funding_sign_and_free_cost_identity(self):
        x = self.frame().assign(funding_rate=.001)
        r = one.evaluate_events(x, [0], barriers=False, side=-1, hold_bars=6)[0]
        self.assertEqual(r["funding_usdt_per_btc"], -.5)
        self.assertGreater(r["net_R"], r["net_R_ex_funding"])
        self.assertAlmostEqual(r["net_R"]+r["cost_R"], 0)
        self.assertEqual(r["exit_time"], str(x.time.iloc[6]))


class ResearchSafetyTests(TemporaryFiles):
    def test_empty_conditions_serialize_and_net_targets_are_not_gross_hits(self):
        frame = market_frame("2020-01-01", "2020-07-01", varying=True).assign(volume=1.)
        x = one.features(frame)
        obs, summaries = one.market_observations(x, one.daily_market(x))
        json.dumps(summaries, allow_nan=False)
        empty = next(s for s in summaries if s["condition"] == "high_volume")
        self.assertEqual(empty["N_overlapping_days"], 0)
        self.assertEqual(empty["status"], "INSUFFICIENT_DATA")
        self.assertIsNone(empty["horizons"]["1"]["mean_net"])
        self.assertEqual(empty["horizons"]["1"]["return_quantiles"], [None, None, None])
        self.assertIsNone(empty["barriers"]["3"]["net_R"])
        self.assertIsNone(empty["barriers"]["5"]["net_target_attainment"])
        self.assertEqual(empty["barriers"]["3"]["mean_ci"]["ci95"], [None, None])
        all_momentum = summaries[0]
        self.assertIn("net_target_attainment", all_momentum["barriers"]["3"])
        self.assertIn("gross_barrier_hit", all_momentum["barriers"]["3"])
        self.assertTrue(obs["net_R_3"].notna().all())
        aligned = obs.loc[obs.momentum_aligned]
        for target in (3, 5):
            barrier = all_momentum["barriers"][str(target)]
            self.assertEqual(barrier["exit_policy"], f"gross_{target}R_target_or_1R_stop_or_7d_timeout")
            self.assertEqual(barrier["gross_barrier_hit"], aligned[f"gross_barrier_hit_{target}"].mean())
            self.assertEqual(barrier["net_target_attainment"], aligned[f"net_R_{target}"].ge(target).mean())
            self.assertEqual(barrier["stress_net_target_attainment"], aligned[f"stress_R_{target}"].ge(target).mean())

    def test_gross_target_hits_below_net_threshold_are_not_attainments(self):
        frame = market_frame("2020-01-01", "2020-07-01", varying=True).assign(volume=1.)
        x = one.features(frame)
        evaluate = one.evaluate_event_matrix

        def gross_hits(*args, **kwargs):
            rows = evaluate(*args, **kwargs)
            for row in rows:
                row.update(target_before_stop=1., net_R=row["target_r"]-.03, exit="TARGET")
            return rows

        with patch.object(one, "evaluate_event_matrix", side_effect=gross_hits):
            _, summaries = one.market_observations(x, one.daily_market(x))
        json.dumps(summaries, allow_nan=False)
        for target in (3, 5):
            barrier = summaries[0]["barriers"][str(target)]
            self.assertEqual(barrier["gross_barrier_hit"], 1.)
            self.assertEqual(barrier["net_target_attainment"], 0.)
            self.assertEqual(barrier["stress_net_target_attainment"], 0.)

    def test_research_cannot_unlock_oos_even_with_download(self):
        with patch.object(one, "download") as download:
            code, _, text = self.run_cli(["--research", "--unlock-oos", "--download"])
        self.assertEqual(code, 2)
        self.assertIn("frozen DEV only", text)
        download.assert_not_called()

    def test_all_research_output_paths_are_distinct(self):
        code, _, text = self.run_cli(["--research", "--forecasts", str(self.csv)])
        self.assertEqual(code, 2)
        self.assertIn("distinct", text)

    def test_protocol_written_before_labels_and_mismatch_fails_closed(self):
        args = argparse.Namespace(protocol=str(self.root/"protocol.json"))
        readiness = {"file_sha256": "data", "code_sha256": "code"}
        with (patch.object(one, "daily_market", side_effect=RuntimeError("before any labels")),
              self.assertRaisesRegex(RuntimeError, "before any labels")):
            one.research_run(pd.DataFrame(), readiness, args)
        frozen = json.loads(Path(args.protocol).read_text())
        self.assertEqual(frozen["identity"]["input_sha256"], "data")
        with patch.object(one, "daily_market") as aggregate:
            with self.assertRaisesRegex(ValueError, "mismatch"):
                one.research_run(pd.DataFrame(), {**readiness, "code_sha256": "changed"}, args)
            aggregate.assert_not_called()


class Core5ExtensionTests(unittest.TestCase):
    def test_core5_builds_4h_atr_before_market_execution(self):
        frames = {symbol: market_frame("2020-01-01", "2020-02-01") for symbol in one.CORE5_SYMBOLS}
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(protocol=str(Path(directory)/"protocol.json"), csv="synthetic.csv")
            with (patch.object(one, "validate_core5", return_value=(frames, {})),
                  patch.object(one, "h1a_experiment", return_value=(pd.DataFrame(), {})),
                  patch.object(one, "h1b_experiment", return_value=(pd.DataFrame(), {"event_rows": []})),
                  patch.object(one, "har_rv_experiment", return_value=(pd.DataFrame(), {})),
                  patch.object(one, "funding_increment_experiment", return_value=(pd.DataFrame(), {}))):
                def check_execution(x, daily):
                    self.assertIn("atr", x.columns)
                    self.assertTrue(x.atr.iloc[20:].notna().all())
                    raise RuntimeError("execution reached")
                with patch.object(one, "market_observations", side_effect=check_execution):
                    with self.assertRaisesRegex(RuntimeError, "execution reached"):
                        one.core5_research_run(pd.DataFrame(), {"file_sha256": "synthetic"}, args)

    def test_panel_bootstrap_keeps_same_date_assets_together_and_is_deterministic(self):
        frame = pd.DataFrame({"time": pd.to_datetime(["2020-01-01"]*2+["2020-03-01"]*2, utc=True),
                              "baseline": [.2,.4,.3,.5], "candidate": [.1,.2,.2,.4]})
        first = one.panel_block_interval(frame, ["baseline", "candidate"], ratio=True)
        self.assertEqual(first, one.panel_block_interval(frame, ["baseline", "candidate"], ratio=True))
        self.assertGreater(first["estimate"], 0)
        self.assertEqual([x["block_days"] for x in first["sensitivity"]], [28,56,61])

    def test_bocpd_is_causal_and_probabilities_are_bounded(self):
        rng = np.random.default_rng(71)
        returns = pd.Series(rng.normal(0, .02, 450))
        before = one.bocpd_recent_probability(returns)
        mutated = returns.copy()
        mutated.iloc[350:] += .4
        after = one.bocpd_recent_probability(mutated)
        np.testing.assert_allclose(before[:350], after[:350], equal_nan=True)
        finite = before[np.isfinite(before)]
        self.assertTrue(((finite >= 0) & (finite <= 1)).all())

    def test_warm_started_regularization_grid_matches_independent_fits(self):
        train=pd.DataFrame({"y":np.resize([0,1,-1],90),"f":np.linspace(-2,2,90)})
        test=pd.DataFrame({"f":np.linspace(-1,1,12)})
        grid=one.forecast_prob_grid(train,test,["f"],(.1,1.,10.))
        for c,p in grid.items():
            np.testing.assert_allclose(p,one.forecast_prob(train,test,["f"],c),atol=2e-5)

    def test_archive_cache_reuses_verified_bytes_and_refetches_corruption(self):
        payload=io.BytesIO()
        with zipfile.ZipFile(payload,"w") as archive: archive.writestr("month.csv","a,b\\n1,2\\n")
        response=MagicMock(); response.read.return_value=payload.getvalue(); response.__enter__.return_value=response
        with tempfile.TemporaryDirectory() as directory, patch.dict(one.os.environ,{"ONEFILE_CACHE_DIR":directory}):
            with patch.object(one.urllib.request,"urlopen",return_value=response) as fetch:
                first=one.cached_archive("https://example.test/month.zip")
                second=one.cached_archive("https://example.test/month.zip")
                self.assertEqual(first,second); fetch.assert_called_once()
                cache=next(Path(directory).glob("*.zip")); cache.write_bytes(b"corrupt")
                third=one.cached_archive("https://example.test/month.zip")
                self.assertEqual(first,third); self.assertEqual(fetch.call_count,2)

    def test_event_matrix_builds_independent_native_engine_columns(self):
        x=market_frame("2020-01-01","2020-02-01"); x[["open","high","low","close"]]=100.; x["atr"]=2.
        dtype=[("col","i8"),("status","i8"),("entry_idx","i8"),("exit_idx","i8"),
               ("entry_price","f8"),("exit_price","f8"),("size","f8"),("pnl","f8"),("id","i8")]
        trades=np.array([(0,1,2,85,100.02,99.98,1.,-.12,0),(1,1,2,85,100.02,99.98,1.,-.12,1)],dtype=dtype)
        portfolio=Mock(); portfolio.trades.records_arr=trades; portfolio.orders.records_arr=np.zeros(4,dtype=[("id","i8")])
        with patch.object(one._vectorbt().Portfolio,"from_signals",return_value=portfolio) as simulate:
            rows=one.evaluate_event_matrix(x,[30],(3.,5.),side=1,hold_bars=42)
        kwargs=simulate.call_args.kwargs
        self.assertEqual(kwargs["close"].shape,(86,2)); self.assertEqual(kwargs["entries"].sum(),2)
        self.assertEqual([row["target_r"] for row in rows],[3.,5.])
        self.assertTrue(all(row["exit"]=="TIME" for row in rows))
        self.assertTrue(all(row["net_R"]==-.12/(one.ATR_MULT*2.) for row in rows))

    def test_vectorized_mark_to_market_accounts_for_side_costs_and_settlement_timing(self):
        x=market_frame("2020-01-01","2020-02-01",varying=True)
        x.loc[::3,"funding_rate"]=.0001
        for side in (1,-1):
            index,hold=30,18
            gross,net=one.mark_to_market_returns(x,[index],side,hold)
            entry,exit_=index+1,index+hold
            raw_entry=float(x.open.iloc[entry]); raw_exit=float(x.close.iloc[exit_])
            fill_entry=raw_entry*(1+side*one.SLIPPAGE_BPS/10_000)
            fill_exit=raw_exit*(1-side*one.SLIPPAGE_BPS/10_000)
            fees=one.FEE_BPS/10_000*(fill_entry+fill_exit)
            funding=side*float((x.open.iloc[entry+1:exit_+1]*x.funding_rate.iloc[entry+1:exit_+1]).sum())
            expected=(side*(fill_exit-fill_entry)-fees-funding)/raw_entry
            self.assertAlmostEqual(net[0],expected,places=12)
            self.assertAlmostEqual(gross[0],side*(raw_exit/raw_entry-1),places=12)

    def test_h1b_empty_event_panel_is_a_valid_inconclusive_result(self):
        d = pd.DataFrame({"time":pd.date_range("2020-01-01",periods=100,freq="D",tz="UTC"),
                          "r1":np.random.default_rng(1).normal(0,.01,100),
                          "close":np.exp(np.cumsum(np.random.default_rng(2).normal(0,.01,100)))})
        events = pd.DataFrame(columns=["onset_i","onset_time","side","status"])
        _, result = one.h1b_experiment(d, events, "ETHUSDT")
        self.assertEqual(result["confirmed_events"],0)
        self.assertIsNone(result["recall"])

    def test_har_rv_training_labels_are_mature_before_each_test_fold(self):
        rng = np.random.default_rng(19)
        time = pd.date_range("2019-01-01","2025-01-01",freq="D",inclusive="left",tz="UTC")
        d = pd.DataFrame({"time":time,"r1":rng.normal(0,.02,len(time))})
        forecasts, summary = one.har_rv_experiment(d,"BTCUSDT")
        self.assertFalse(forecasts.empty)
        for row in forecasts.itertuples():
            train_latest = pd.Timestamp(row.train_latest_origin)
            self.assertLess(train_latest+pd.Timedelta(days=row.horizon),pd.Timestamp(row.fold,tz="UTC"))
        self.assertIn("EWMA",summary["metrics"])

    def test_multi_research_rejects_sealed_holdout_before_download(self):
        with patch.object(one,"download_core5_dev") as download:
            with self.assertRaises(SystemExit) as err:
                one.main(["--multi-research","--unlock-oos","--download"])
        self.assertEqual(err.exception.code,2)
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
