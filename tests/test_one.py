"""Offline regression tests for the six findings in AUDIT_ONE_PY.md.

Run: uv run python -m unittest discover -s tests -v
All files and mocked exchange archives are synthetic; no network is used.
"""
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import data
import market
import one


def market_frame(start=one.DEV_START, end=one.DEV_END, varying=False):
    times = pd.date_range(start, end, freq="4h", inclusive="left", tz="UTC")
    frame = pd.DataFrame({
        "time": times, "open": 100., "high": 101., "low": 99., "close": 100.,
        "volume": 1., "funding_rate": 0.,
        "funding_observed": times.hour.isin([0, 8, 16]).astype("int8"), "symbol": one.SYMBOL,
    })
    if varying:
        i = np.arange(len(frame))
        frame["close"] = 100 + .025*i + 3*np.sin(i/8)
        frame["open"] = frame.close.shift(1).fillna(frame.close.iloc[0])
        frame["high"] = frame[["open", "close"]].max(axis=1) + 1
        frame["low"] = frame[["open", "close"]].min(axis=1) - 1
        frame["volume"] = 100 + 30*np.sin(i/4)
    return frame


def zip_csv(frame):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data.csv", frame.to_csv(index=False))
    return buffer.getvalue()


class TemporaryFiles(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.csv = self.root / "input.csv"
        self.out = self.root / "trades.csv"
        self.evidence = self.root / "evidence.json"

    def run_cli(self, extra=(), offline=True):
        stdout, stderr = io.StringIO(), io.StringIO()
        code = 0
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                one.main(["--csv", str(self.csv), "--out", str(self.out),
                          "--evidence", str(self.evidence), *(["--offline"] if offline else []), *extra])
            except SystemExit as exc:
                code = exc.code
        return code, stdout.getvalue(), stderr.getvalue()


class DataValidationTests(TemporaryFiles):
    def setUp(self):
        super().setUp()
        self.frame = market_frame("2020-01-01", "2020-02-01")

    def readiness(self, frame=None):
        return one.check_data_readiness(
            self.frame if frame is None else frame, self.csv,
            start_date="2020-01-01", end_date="2020-02-01",
        )

    def test_observed_zero_funding_is_valid(self):
        clean = one.validate(self.frame)
        result = self.readiness(clean)
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["actual_8h_settlements"], 93)
        self.assertEqual(result["non_zero_funding_settlements"], 0)

    def test_negative_settlement_funding_is_valid(self):
        self.frame.loc[0, "funding_rate"] = -.001
        self.assertEqual(self.readiness(one.validate(self.frame))["verdict"], "PASS")

    def test_legacy_schema_requires_redownload(self):
        for column in ["funding_observed", "symbol"]:
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "re-downloaded"):
                one.validate(self.frame.drop(columns=column))

    def test_sparse_observations_are_rejected(self):
        self.frame["funding_observed"] = 0
        self.frame.loc[0, ["funding_observed", "funding_rate"]] = [1, .0001]
        with self.assertRaisesRegex(ValueError, "funding observations"):
            one.validate(self.frame)
        result = self.readiness()
        self.assertEqual(result["verdict"], "FAIL")
        self.assertEqual(result["actual_8h_settlements"], 1)

    def test_legacy_sparse_rates_without_observation_evidence_fail_readiness(self):
        self.frame.loc[0, "funding_rate"] = .0001
        result = self.readiness(self.frame.drop(columns="funding_observed"))
        self.assertEqual(result["funding_coverage"], "FAIL")
        self.assertEqual(result["verdict"], "FAIL")

    def test_nonfinite_settlement_rate_is_rejected(self):
        for value in [np.nan, np.inf, -np.inf]:
            bad = self.frame.copy()
            bad.loc[0, "funding_rate"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Non-finite"):
                one.validate(bad)

    def test_off_schedule_rate_is_rejected(self):
        self.frame.loc[1, "funding_rate"] = .0001
        with self.assertRaisesRegex(ValueError, "outside BTC"):
            one.validate(self.frame)
        self.assertEqual(self.readiness()["funding_coverage"], "FAIL")

    def test_off_schedule_observation_is_rejected(self):
        self.frame.loc[1, "funding_observed"] = 1
        with self.assertRaisesRegex(ValueError, "off-schedule"):
            one.validate(self.frame)

    def test_invalid_observation_flags_are_rejected(self):
        for value in [-1, 2, .5, np.nan, "unknown"]:
            bad = self.frame.copy()
            bad["funding_observed"] = bad.funding_observed.astype(object)
            bad.loc[0, "funding_observed"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                one.validate(bad)

    def test_wrong_or_mixed_symbol_is_rejected(self):
        for indices in [[0], self.frame.index]:
            bad = self.frame.copy()
            bad.loc[indices, "symbol"] = "ETHUSDT"
            with self.subTest(indices=str(indices)[:20]), self.assertRaisesRegex(ValueError, "symbol"):
                one.validate(bad)

    def test_missing_nullable_symbol_is_rejected(self):
        self.frame["symbol"] = self.frame.symbol.astype("string")
        self.frame.loc[0, "symbol"] = pd.NA
        with self.assertRaisesRegex(ValueError, "symbol"):
            one.validate(self.frame)

    def test_missing_timestamp_is_rejected_not_silently_dropped(self):
        self.frame.loc[0, "time"] = pd.NaT
        with self.assertRaisesRegex(ValueError, "timestamp"):
            one.validate(self.frame)

    def test_readiness_detects_missing_boundaries(self):
        result = self.readiness(self.frame.iloc[6:].reset_index(drop=True))
        self.assertEqual(result["chronological_bounds"], "FAIL")
        self.assertEqual(result["candle_integrity"], "FAIL")
        with self.assertRaisesRegex(ValueError, "Data acceptance FAIL"):
            one.require_data_readiness(result)

    def test_readiness_detects_duplicate_candle(self):
        bad = self.frame.copy()
        bad.loc[1, "time"] = bad.loc[0, "time"]
        self.assertEqual(self.readiness(bad)["verdict"], "FAIL")


class FileSafetyTests(TemporaryFiles):
    def test_all_path_pairs_must_be_distinct(self):
        paths = [self.csv, self.out, self.evidence]
        for first, second in [(0, 1), (0, 2), (1, 2)]:
            aliases = paths.copy()
            aliases[second] = aliases[first]
            with self.subTest(pair=(first, second)), self.assertRaisesRegex(ValueError, "distinct"):
                one.validate_file_paths(*aliases)

    def test_normalized_alias_is_rejected(self):
        alias = self.root / "unused" / ".." / "input.csv"
        with self.assertRaisesRegex(ValueError, "distinct"):
            one.validate_file_paths(self.csv, alias, self.evidence)

    def test_symlink_alias_is_rejected(self):
        self.csv.write_text("original")
        self.out.symlink_to(self.csv)
        with self.assertRaisesRegex(ValueError, "distinct"):
            one.validate_file_paths(self.csv, self.out, self.evidence)
        self.assertEqual(self.csv.read_text(), "original")

    def test_hardlink_alias_is_rejected(self):
        self.csv.write_text("original")
        os.link(self.csv, self.out)
        with self.assertRaisesRegex(ValueError, "distinct"):
            one.validate_file_paths(self.csv, self.out, self.evidence)

    def test_nonexistent_identical_paths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            one.validate_file_paths(self.csv, self.out, self.out)

    def test_invalid_parent_is_rejected_before_writes(self):
        with self.assertRaisesRegex(ValueError, "Parent directory"):
            one.validate_file_paths(self.csv, self.root / "absent" / "out.csv", self.evidence)

    def test_directory_instead_of_file_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "regular file"):
            one.validate_file_paths(self.root, self.out, self.evidence)

    def test_atomic_writer_failure_preserves_original_and_cleans_temp(self):
        self.out.write_text("original")

        def broken(stream):
            stream.write("partial")
            raise OSError("synthetic disk failure")

        with self.assertRaises(OSError):
            one.atomic_write(self.out, broken)
        self.assertEqual(self.out.read_text(), "original")
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_atomic_replace_failure_preserves_original_and_cleans_temp(self):
        self.out.write_text("original")
        with patch.object(one.os, "replace", side_effect=OSError("replace failed")), self.assertRaises(OSError):
            one.atomic_write(self.out, lambda stream: stream.write("new"))
        self.assertEqual(self.out.read_text(), "original")
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_atomic_write_replaces_complete_file(self):
        self.out.write_text("old")
        one.atomic_write(self.out, lambda stream: stream.write("complete\n"))
        self.assertEqual(self.out.read_text(), "complete\n")
        self.assertEqual(list(self.root.glob(".*.tmp")), [])


class DownloadTests(TemporaryFiles):
    def setUp(self):
        super().setUp()
        cache_env = patch.dict(one.os.environ, {"ONEFILE_CACHE_DIR": str(self.root / "cache")})
        cache_env.start()
        self.addCleanup(cache_env.stop)
        frame = market_frame("2020-01-01", "2020-02-01")
        self.klines = frame[["open", "high", "low", "close", "volume"]].copy()
        self.klines.insert(0, "open_time", frame.time.array.as_unit("ms").asi8)
        times = pd.date_range("2020-01-01", "2020-02-01", freq="8h", inclusive="left", tz="UTC")
        self.funding = pd.DataFrame({"calc_time": times.as_unit("ms").asi8,
                                     "funding_interval": 8, "funding_rate": 0.})
        self.csv.write_text("known-good-original")

    def fetch(self):
        k_zip, f_zip = zip_csv(self.klines), zip_csv(self.funding)

        def response(request, timeout):
            self.assertEqual(timeout, 30)
            self.assertTrue(request.full_url.startswith(one.DATA_SOURCE + "/"))
            return io.BytesIO(f_zip if "/fundingRate/" in request.full_url else k_zip)

        with patch.object(data.urllib.request, "urlopen", side_effect=response):
            one.download(self.csv, end_date="2020-02-01")

    def assert_rejected_without_replacement(self, regex):
        with self.assertRaisesRegex(ValueError, regex):
            self.fetch()
        self.assertEqual(self.csv.read_text(), "known-good-original")
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_valid_zero_rates_keep_observation_evidence(self):
        self.fetch()
        raw = pd.read_csv(self.csv)
        clean = one.validate(raw)
        self.assertEqual(clean.funding_observed.sum(), 93)
        self.assertTrue(clean.funding_rate.eq(0).all())
        self.assertTrue(clean.symbol.eq(one.SYMBOL).all())
        self.assertEqual(one.check_data_readiness(
            clean, self.csv, start_date="2020-01-01", end_date="2020-02-01",
        )["verdict"], "PASS")

    def test_signed_rates_are_preserved(self):
        self.funding.loc[[0, 1], "funding_rate"] = [-.001, .002]
        self.fetch()
        raw = pd.read_csv(self.csv)
        np.testing.assert_allclose(raw.loc[[0, 2], "funding_rate"], [-.001, .002])

    def test_nan_rate_is_not_zero_filled(self):
        self.funding.loc[1:, "funding_rate"] = np.nan
        self.assert_rejected_without_replacement("Non-finite funding")

    def test_infinite_rate_is_rejected(self):
        self.funding.loc[0, "funding_rate"] = np.inf
        self.assert_rejected_without_replacement("Non-finite funding")

    def test_missing_settlement_is_rejected(self):
        self.funding = self.funding.drop(index=10)
        self.assert_rejected_without_replacement("funding history")

    def test_duplicate_settlement_is_rejected(self):
        self.funding = pd.concat([self.funding, self.funding.iloc[[0]]], ignore_index=True)
        self.assert_rejected_without_replacement("funding history")

    def test_off_schedule_settlement_is_rejected(self):
        self.funding.loc[1, "calc_time"] += 4 * 3600_000
        self.assert_rejected_without_replacement("funding history")

    def test_missing_last_candle_does_not_replace_good_file(self):
        self.klines = self.klines.iloc[:-1]
        self.assert_rejected_without_replacement("candle history")

    def test_duplicate_candle_is_rejected(self):
        self.klines = pd.concat([self.klines, self.klines.iloc[[0]]], ignore_index=True)
        self.assert_rejected_without_replacement("candle history")

    def test_invalid_ohlc_does_not_replace_good_file(self):
        self.klines.loc[20, "high"] = 1.
        self.assert_rejected_without_replacement("OHLC")

    def test_timestamp_drift_within_one_minute_is_normalized(self):
        self.funding["calc_time"] += 59_999
        self.fetch()
        self.assertEqual(pd.read_csv(self.csv).funding_observed.sum(), 93)

    def test_excessive_timestamp_drift_is_rejected(self):
        self.funding.loc[0, "calc_time"] += 60_001
        self.assert_rejected_without_replacement("timestamp drift")

    def test_unsupported_symbol_rejected_before_network(self):
        with patch.object(data.urllib.request, "urlopen") as network:
            with self.assertRaisesRegex(ValueError, "Only BTCUSDT"):
                one.download(self.csv, symbol="ETHUSDT")
            network.assert_not_called()


class ControlGateTests(unittest.TestCase):
    @staticmethod
    def rows(n=30, net_r=2.):
        times = pd.date_range("2020-01-01", periods=n, freq="MS", tz="UTC")
        if net_r < 0:
            times += pd.Timedelta(days=14)
        return [{"signal_time": str(time), "net_R": net_r, "path_MFE_R": 3., "path_MAE_R": -1.}
                for time in times]

    def gate(self, setup, control, **kwargs):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            one.evaluate_gate(setup, control, **dict({"trials": 1, "source_verified": True}, **kwargs))
        return output.getvalue()

    def test_single_control_cannot_pass(self):
        with patch.object(one, "bootstrap_stats") as bootstrap:
            text = self.gate(self.rows(), self.rows(1, -1.))
            bootstrap.assert_not_called()
        self.assertIn("INSUFFICIENT DATA", text)
        self.assertNotIn("EXPLORATORY PASS", text)

    def test_both_groups_need_minimum_size(self):
        text = self.gate(self.rows(29), self.rows(29, -1.))
        self.assertIn("INSUFFICIENT DATA", text)

    def test_unequal_counts_cannot_pass(self):
        text = self.gate(self.rows(30), self.rows(31, -1.))
        self.assertIn("monthly counts differ", text)

    def test_equal_sizes_with_wrong_months_cannot_pass(self):
        control = self.rows(net_r=-1.)
        control[0]["signal_time"] = "2021-01-01T00:00:00Z"
        self.assertIn("monthly counts differ", self.gate(self.rows(), control))

    def test_missing_event_identity_cannot_pass(self):
        setup = self.rows()
        for row in setup:
            del row["signal_time"]
        self.assertIn("INSUFFICIENT DATA", self.gate(setup, self.rows(net_r=-1.)))

    def test_nonfinite_metrics_cannot_pass(self):
        setup = self.rows()
        setup[0]["net_R"] = np.nan
        self.assertIn("INSUFFICIENT DATA", self.gate(setup, self.rows(net_r=-1.)))

    def test_valid_balanced_positive_case_still_passes(self):
        self.assertIn("EXPLORATORY PASS", self.gate(self.rows(), self.rows(net_r=-1.)))

    def test_selector_keeps_unmatched_setups_when_no_controls_exist(self):
        frame = market_frame("2023-01-01", "2023-01-31")
        frame["signal"], frame["trend"] = False, False
        frame.loc[[30, 80], "signal"] = True
        frame.loc[[29, 30, 31, 80], "trend"] = True
        self.assertEqual(one.select_events(frame), ([], []))
        self.assertEqual(one.select_setup_events(frame), [30, 80])

    def test_controls_must_be_disjoint_from_setup_horizons(self):
        frame = market_frame("2023-01-01", "2023-01-31")
        frame["signal"], frame["trend"] = False, True
        frame.loc[[30, 80], "signal"] = True
        matched, controls = one.select_events(frame)
        self.assertEqual(len(matched), 1)
        self.assertEqual(len(controls), 1)
        self.assertEqual(one.select_setup_events(frame), [30, 80])
        self.assertTrue(all(abs(i-j) > one.HOLD_BARS for i in [30, 80] for j in controls))

    def test_controls_disjoint_in_longer_window(self):
        frame = market_frame("2023-01-01", "2023-04-01")
        frame["signal"], frame["trend"] = False, True
        frame.loc[[30, 80, 400], "signal"] = True
        signals, controls = one.select_events(frame)
        self.assertEqual((signals, controls), one.select_events(frame))
        self.assertEqual(len(signals), len(controls))
        self.assertTrue(all(abs(i-j) > one.HOLD_BARS for i in signals for j in controls))

    def test_empty_selection_remains_valid_and_gate_is_insufficient(self):
        frame = market_frame("2023-01-01", "2023-01-31")
        frame["signal"], frame["trend"] = False, False
        self.assertEqual(one.select_events(frame), ([], []))
        self.assertIn("INSUFFICIENT DATA", self.gate([], []))


class StudyPreparationTests(TemporaryFiles):
    @classmethod
    def setUpClass(cls):
        cls.history = market_frame(end="2025-02-01", varying=True)

    def test_bad_data_stops_before_features(self):
        with patch.object(one, "features") as features:
            with self.assertRaisesRegex(ValueError, "Data acceptance FAIL"):
                one.prepare_study(self.history.iloc[6:], self.csv, end_date="2025-02-01")
            features.assert_not_called()

    def test_oos_features_equal_continuous_causal_history(self):
        actual, readiness = one.prepare_study(
            self.history, self.csv, start_date=one.DEV_END, end_date="2025-02-01",
        )
        continuous = one.features(one.validate(self.history, end_date="2025-02-01"))
        expected = continuous.loc[continuous.time >= pd.Timestamp(one.DEV_END, tz="UTC")].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected)
        self.assertFalse(pd.isna(actual.atr.iloc[0]))
        self.assertEqual(readiness["feature_history_start"], one.DEV_START)
        self.assertEqual(readiness["feature_history_bars"], len(self.history))
        self.assertTrue(actual.time.ge(pd.Timestamp(one.DEV_END, tz="UTC")).all())

    def test_oos_without_dev_history_is_rejected(self):
        only_oos = self.history[self.history.time >= pd.Timestamp(one.DEV_END, tz="UTC")]
        with self.assertRaisesRegex(ValueError, "Data acceptance FAIL"):
            one.prepare_study(only_oos, self.csv, start_date=one.DEV_END, end_date="2025-02-01")

    def test_future_oos_prices_do_not_change_dev_features(self):
        before, _ = one.prepare_study(self.history, self.csv)
        changed = self.history.copy()
        mask = changed.time >= pd.Timestamp(one.DEV_END, tz="UTC")
        changed.loc[mask, ["open", "high", "low", "close"]] *= 3
        after, _ = one.prepare_study(changed, self.csv)
        pd.testing.assert_frame_equal(before, after)


class CLIContractTests(TemporaryFiles):
    def test_failed_readiness_exits_before_execution_and_outputs(self):
        market_frame("2023-01-01", "2023-02-01").to_csv(self.csv, index=False)
        before = self.csv.read_bytes()
        with patch.object(one, "features") as features, patch.object(one, "evaluate_events") as execution:
            code, stdout, stderr = self.run_cli()
            features.assert_not_called()
            execution.assert_not_called()
        self.assertEqual(code, 2)
        self.assertIn("Data acceptance FAIL", stderr)
        self.assertNotIn("[PASS]", stdout)
        self.assertFalse(self.out.exists())
        self.assertFalse(self.evidence.exists())
        self.assertEqual(self.csv.read_bytes(), before)

    def test_legacy_csv_exits_with_migration_message(self):
        market_frame().drop(columns=["symbol", "funding_observed"]).to_csv(self.csv, index=False)
        code, _, stderr = self.run_cli()
        self.assertEqual(code, 2)
        self.assertIn("--download", stderr)
        self.assertFalse(self.evidence.exists())

    def test_unsupported_cli_symbol_fails_before_read_or_download(self):
        with patch.object(one, "download") as download, patch.object(one.pd, "read_csv") as read:
            code, _, _ = self.run_cli(["--symbol", "ETHUSDT", "--download"])
            download.assert_not_called()
            read.assert_not_called()
        self.assertEqual(code, 2)

    def test_csv_symbol_mismatch_cannot_export_evidence(self):
        market_frame().assign(symbol="ETHUSDT").to_csv(self.csv, index=False)
        code, _, stderr = self.run_cli()
        self.assertEqual(code, 2)
        self.assertIn("symbol", stderr)
        self.assertFalse(self.evidence.exists())

    def test_output_collision_rejected_before_download(self):
        self.csv.write_text("original-data")
        with patch.object(one, "download") as download:
            code, _, stderr = self.run_cli(["--out", str(self.csv), "--download"])
            download.assert_not_called()
        self.assertEqual(code, 2)
        self.assertIn("distinct", stderr)
        self.assertEqual(self.csv.read_text(), "original-data")

    def test_evidence_collision_cannot_destroy_input(self):
        self.csv.write_text("original-data")
        code, _, _ = self.run_cli(["--evidence", str(self.csv)])
        self.assertEqual(code, 2)
        self.assertEqual(self.csv.read_text(), "original-data")

    def test_control_matching_failure_stops_before_execution(self):
        market_frame().to_csv(self.csv, index=False)
        with (
            patch.object(one, "select_events", side_effect=ValueError("insufficient same-month controls")),
            patch.object(one, "evaluate_events") as execution,
        ):
            code, _, stderr = self.run_cli()
            execution.assert_not_called()
        self.assertEqual(code, 2)
        self.assertIn("same-month", stderr)
        self.assertFalse(self.evidence.exists())

    def test_successful_dev_writes_consistent_atomic_evidence(self):
        market_frame().to_csv(self.csv, index=False)
        before = self.csv.read_bytes()
        code, stdout, stderr = self.run_cli()
        self.assertEqual(code, 0, stderr)
        self.assertIn("[PASS] ACTIVE DATA ACCEPTED", stdout)
        metadata = json.loads(self.evidence.read_text())
        readiness = metadata["b1_data_readiness"]
        self.assertEqual(metadata["b1_universe"]["symbol"], readiness["universe_symbol"])
        self.assertEqual(readiness["verdict"], "PASS")
        self.assertEqual(readiness["actual_8h_settlements"], 5481)
        self.assertEqual(readiness["non_zero_funding_settlements"], 0)
        self.assertEqual(metadata["input_sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(self.csv.read_bytes(), before)
        self.assertTrue(self.out.exists())
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_successful_oos_uses_dev_history_but_exports_only_active_window(self):
        raw = market_frame(end="2025-02-01")
        raw.to_csv(self.csv, index=False)
        with patch.object(one, "OOS_END", "2025-02-01"):
            code, stdout, stderr = self.run_cli(["--unlock-oos"])
        self.assertEqual(code, 0, stderr)
        metadata = json.loads(self.evidence.read_text())
        readiness = metadata["b1_data_readiness"]
        self.assertEqual(metadata["bars"], 186)
        self.assertEqual(readiness["feature_history_bars"], len(raw))
        self.assertEqual(readiness["start_date"], one.DEV_END)
        self.assertNotIn("REAL DEV DATA ACCEPTED", stdout)

    def test_real_process_returns_nonzero_for_incomplete_data(self):
        market_frame("2023-01-01", "2023-02-01").to_csv(self.csv, index=False)
        process = subprocess.run(
            [sys.executable, str(Path(one.__file__).resolve()), "--csv", str(self.csv),
             "--out", str(self.out), "--evidence", str(self.evidence)],
            cwd=self.root, capture_output=True, text=True, timeout=60, check=False,
        )
        self.assertEqual(process.returncode, 2)
        self.assertIn("Data acceptance FAIL", process.stderr)
        self.assertNotIn("[PASS]", process.stdout)
        self.assertFalse(self.out.exists())
        self.assertFalse(self.evidence.exists())


if __name__ == "__main__":
    unittest.main()
