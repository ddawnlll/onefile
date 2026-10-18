"""Negative-path and methodology regressions; synthetic/offline, no GitHub or network."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import test_one as fixtures
from test_one import TemporaryFiles, market_frame

import one


class SourceVerificationTests(unittest.TestCase):
    def setUp(self):
        self.frame = market_frame("2020-01-01", "2020-02-01")
        self.frame.loc[[0, 2], "funding_rate"] = [-.001, .002]
        observed = self.frame[self.frame.funding_observed.eq(1)]
        self.source = pd.DataFrame({"time_ms": observed.time.array.as_unit("ms").asi8,
                                    "funding_rate": observed.funding_rate.to_numpy()})

    def verify(self, frame=None):
        with patch.object(one, "fetch_funding_month", return_value=self.source):
            return one.verify_funding_source(self.frame if frame is None else frame, "2020-02-01")

    def test_signed_and_real_zero_rates_match_source(self):
        result = self.verify()
        self.assertEqual(result["status"], "VERIFIED")
        self.assertEqual(result["settlements"], 93)

    def test_zeroed_values_with_all_flags_preserved_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "differ from canonical"):
            self.verify(self.frame.assign(funding_rate=0.))

    def test_single_changed_value_is_rejected(self):
        bad = self.frame.copy()
        bad.loc[4, "funding_rate"] = .00001
        with self.assertRaisesRegex(ValueError, "differ from canonical"):
            self.verify(bad)

    def test_missing_and_duplicate_source_settlements_fail(self):
        for source in [self.source.iloc[1:], pd.concat([self.source, self.source.iloc[:1]])]:
            with (self.subTest(n=len(source)),
                  patch.object(one, "fetch_funding_month", return_value=source),
                  self.assertRaisesRegex(ValueError, "canonical funding history")):
                one.verify_funding_source(self.frame, "2020-02-01")

    def test_nonfinite_source_is_rejected(self):
        self.source.loc[0, "funding_rate"] = np.nan
        with self.assertRaisesRegex(ValueError, "differ from canonical"):
            self.verify()


class SourceCLIContractTests(TemporaryFiles):
    def test_online_source_outage_stops_before_features_execution_or_exports(self):
        market_frame().to_csv(self.csv, index=False)
        with (patch.object(one, "verify_funding_source", side_effect=OSError("source unavailable")),
              patch.object(one, "features") as features,
              patch.object(one, "evaluate_events") as execution):
            code, stdout, stderr = self.run_cli(offline=False)
        self.assertEqual(code, 2)
        self.assertIn("source unavailable", stderr)
        self.assertNotIn("[PASS]", stdout)
        features.assert_not_called()
        execution.assert_not_called()
        self.assertFalse(self.out.exists())
        self.assertFalse(self.evidence.exists())

    def test_online_mismatch_stops_before_features(self):
        market_frame().to_csv(self.csv, index=False)
        with (patch.object(one, "verify_funding_source", side_effect=ValueError("CSV funding values differ")),
              patch.object(one, "features") as features):
            code, stdout, _ = self.run_cli(offline=False)
        self.assertEqual(code, 2)
        features.assert_not_called()
        self.assertNotIn("[PASS]", stdout)
        self.assertFalse(self.evidence.exists())

    def test_offline_mode_is_explicit_unverified_and_has_no_statistical_pass(self):
        market_frame().to_csv(self.csv, index=False)
        with patch.object(one, "verify_funding_source") as verification:
            code, stdout, stderr = self.run_cli()
        self.assertEqual(code, 0, stderr)
        verification.assert_not_called()
        evidence = json.loads(self.evidence.read_text())
        self.assertEqual(evidence["b1_data_readiness"]["funding_source_verification"]["status"], "UNVERIFIED")
        self.assertFalse(evidence["decision_gate"]["edge_proven"])
        self.assertNotIn("EXPLORATORY PASS", stdout)
        self.assertIn("NOT a $500 account", stdout)
        self.assertIn("signal_time,group,matched", self.out.read_text())

    def test_partial_matching_retains_every_setup_and_marks_subset(self):
        market_frame().to_csv(self.csv, index=False)

        def enriched(frame):
            result = frame.assign(atr=2., signal=False, trend=True)
            result.loc[[30, 80, 130], "signal"] = True
            return result

        with patch.object(one, "features", side_effect=enriched):
            code, _, stderr = self.run_cli()
        self.assertEqual(code, 0, stderr)
        rows = pd.read_csv(self.out)
        setup, control = rows[rows.group.eq("setup")], rows[rows.group.eq("control")]
        self.assertEqual(len(setup), 3)
        self.assertEqual(len(control), 1)
        self.assertEqual(int(setup.matched.sum()), 1)
        one.require_disjoint_events(setup.signal_time, control.signal_time)
        evidence = json.loads(self.evidence.read_text())
        self.assertEqual(evidence["setup_events"], 3)
        self.assertEqual(evidence["matched_setup_events"], 1)
        self.assertEqual(evidence["unmatched_setup_events"], 2)
        self.assertEqual(evidence["decision_gate"]["verdict"], "INSUFFICIENT DATA")

    def test_online_success_exports_verification_result(self):
        market_frame().to_csv(self.csv, index=False)
        with patch.object(one, "verify_funding_source", return_value={"status": "VERIFIED", "method": "fixture"}) as verification:
            code, _, stderr = self.run_cli(offline=False)
        self.assertEqual(code, 0, stderr)
        verification.assert_called_once()
        self.assertEqual(json.loads(self.evidence.read_text())["b1_data_readiness"]["funding_source_verification"]["status"], "VERIFIED")

    def test_invalid_trial_count_fails_before_io(self):
        with patch.object(one.pd, "read_csv") as read:
            code, _, _ = self.run_cli(["--trials", "0"])
        self.assertEqual(code, 2)
        read.assert_not_called()


class CalendarStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.times = pd.date_range("2020-01-01", periods=60, freq="MS", tz="UTC")
        self.control_times = self.times + pd.Timedelta(days=14)
        self.values = np.repeat([-1., 2., -2., 3., -1.], 12)

    def stats(self, s=None, c=None, **kwargs):
        return one.bootstrap_stats(self.values if s is None else s, self.values if c is None else c,
                                   n_boot=500, setup_times=self.times, ctrl_times=self.control_times, **kwargs)

    def test_joint_resampling_retains_paired_regime_covariance(self):
        result = self.stats()
        self.assertEqual(result["obs_diff"], 0.)
        self.assertEqual(result["delta_ci_low"], 0.)
        self.assertEqual(result["delta_ci_high"], 0.)
        self.assertEqual(result["p_value"], 1.)
        self.assertGreater(result["ci_high"] - result["ci_low"], 1.)

    def test_deterministic_sensitivity_envelope(self):
        result = self.stats()
        self.assertEqual(result, self.stats())
        self.assertEqual([r["block_months"] for r in result["block_sensitivity"]], [1, 3, 6])
        self.assertEqual(result["p_value"], max(r["p_value"] for r in result["block_sensitivity"]))
        self.assertEqual(result["ci_low"], min(r["ci_low"] for r in result["block_sensitivity"]))
        self.assertIsNone(result["adjusted_p_value"])

    def test_multiplicity_is_applied_and_capped(self):
        result = self.stats(s=self.values + 2., trials=10)
        self.assertEqual(result["adjusted_p_value"], min(1., 10 * result["p_value"]))
        self.assertEqual(self.stats(trials=50)["adjusted_p_value"], 1.)

    def test_missing_calendar_months_are_not_compressed(self):
        times = self.times[::2]
        result = one.bootstrap_stats(self.values[::2], self.values[::2], 500, times, times)
        self.assertEqual(result["calendar_months"], 59)
        self.assertEqual(result["active_months"], 30)

    def test_calendar_order_not_input_row_order_controls_resampling(self):
        order = np.random.default_rng(4).permutation(60)
        result = one.bootstrap_stats(self.values[order], self.values[order], 500,
                                    self.times[order], self.control_times[order])
        self.assertEqual(result, self.stats())

    def test_without_timestamps_inference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Calendar timestamps"):
            one.bootstrap_stats(self.values, self.values, 500)

    def test_invalid_values_counts_and_trials_fail_closed(self):
        for values in [np.array([]), np.array([1.]), np.array([np.nan] * 60), np.ones((2, 30))]:
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                self.stats(s=values)
        for trials in [0, -1, .5, True]:
            with self.subTest(trials=trials), self.assertRaises(ValueError):
                self.stats(trials=trials)
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            self.stats(s=self.values[:-1])
        with self.assertRaisesRegex(ValueError, "n_boot"):
            one.bootstrap_stats(self.values, self.values, 0, self.times, self.control_times)


class AdditionalGateTests(unittest.TestCase):
    def gate(self, **kwargs):
        setup, control = fixtures.ControlGateTests.rows(), fixtures.ControlGateTests.rows(net_r=-1.)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = one.evaluate_gate(setup, control, **kwargs)
        return result, output.getvalue()

    def test_unknown_trial_history_cannot_pass(self):
        result, text = self.gate(source_verified=True)
        self.assertIn("INCONCLUSIVE", result["verdict"])
        self.assertIsNone(result["stats"]["adjusted_p_value"])
        self.assertNotIn("EXPLORATORY PASS", text)

    def test_unverified_source_cannot_pass(self):
        result, text = self.gate(trials=1)
        self.assertIn("INCONCLUSIVE", result["verdict"])
        self.assertNotIn("EXPLORATORY PASS", text)

    def test_positive_complete_case_is_only_exploratory(self):
        result, _ = self.gate(trials=1, source_verified=True)
        self.assertIn("EXPLORATORY PASS", result["verdict"])
        self.assertFalse(result["edge_proven"])

    def test_cross_group_overlap_rejected_even_with_matching_months(self):
        rows = fixtures.ControlGateTests.rows()
        with contextlib.redirect_stdout(io.StringIO()):
            result = one.evaluate_gate(rows, rows, trials=1, source_verified=True)
        self.assertIn("Overlapping setup/control", result["reason"])
        self.assertIsNone(result["stats"])

    def test_datetime_resolution_does_not_change_disjointness(self):
        a = pd.date_range("2020-01-01", periods=30, freq="MS", tz="UTC")
        one.require_disjoint_events(a.as_unit("us"), (a + pd.Timedelta(days=14)).as_unit("ns"))
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            one.require_disjoint_events(a.as_unit("ns"), (a + pd.Timedelta(days=2)).as_unit("us"))


class BehaviorSamplingTests(unittest.TestCase):
    def setUp(self):
        self.times = pd.date_range("2020-01-01", periods=500, freq="4h", tz="UTC")
        self.mask = pd.Series(True, index=range(500))

    def test_calendar_grid_is_common_to_conditions(self):
        subset = self.mask.copy()
        subset.iloc[:17] = False
        all_ids = one.select_non_overlapping(self.mask, 450, self.times)
        subset_ids = one.select_non_overlapping(subset, 450, self.times)
        self.assertEqual(subset_ids, [i for i in all_ids if subset.iloc[i]])
        self.assertTrue(all(np.diff(all_ids) > one.HOLD_BARS))

    def test_all_phases_cover_each_mature_bar_once(self):
        ids = [i for phase in range(43)
               for i in one.select_non_overlapping(self.mask, 450, self.times, phase)]
        self.assertEqual(sorted(ids), list(range(451)))

    def test_calendar_grid_is_slice_and_resolution_invariant(self):
        before = one.select_non_overlapping(self.mask, 450, self.times.as_unit("us"))
        after = one.select_non_overlapping(self.mask.iloc[17:].reset_index(drop=True), 433,
                                          self.times[17:].as_unit("ns"))
        self.assertEqual([i for i in before if i >= 17], [i + 17 for i in after])

    def test_behavior_batches_all_mature_events_and_propagates_target(self):
        frame = one.features(one.validate(market_frame("2020-01-01", "2020-03-01", varying=True)))
        with (patch.object(one, "evaluate_event_matrix", wraps=one.evaluate_event_matrix) as execution,
              contextlib.redirect_stdout(io.StringIO()) as output):
            one.analyze_behavior(frame, target_r=2.)
        self.assertTrue(all(call.args[2] == (2., 2.) and call.kwargs["barriers"] == (True, False)
                            for call in execution.call_args_list))
        sampled = [i for call in execution.call_args_list for i in call.args[1]]
        self.assertEqual(execution.call_count, (len(sampled) + 511) // 512)
        self.assertTrue(all(len(call.args[1]) <= 512 for call in execution.call_args_list))
        self.assertEqual(len(sampled), len(set(sampled)))
        expected = np.flatnonzero(frame.atr.notna().to_numpy())
        expected = expected[expected < len(frame) - one.HOLD_BARS]
        self.assertEqual(sorted(sampled), expected.tolist())
        self.assertIn("PHASE SENSITIVITY", output.getvalue())
        self.assertIn("retrospective labeling only", output.getvalue())


class AdditionalReadinessTests(TemporaryFiles):
    def test_missing_required_columns_return_fail_not_keyerror(self):
        frame = market_frame("2020-01-01", "2020-02-01")
        for col in ["time", "funding_rate", "funding_observed", "symbol"]:
            with self.subTest(col=col):
                result = one.check_data_readiness(frame.drop(columns=col), self.csv,
                                                 "2020-01-01", "2020-02-01")
                self.assertEqual(result["verdict"], "FAIL")
                with self.assertRaisesRegex(ValueError, "Data acceptance FAIL"):
                    one.require_data_readiness(result)


if __name__ == "__main__":
    unittest.main()
