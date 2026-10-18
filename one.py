"""ONEFILE Brick 1 — native vectorbt execution, no custom fill/backtest engine.

Install: uv sync --frozen
Check: uv run python one.py --self-test
Run: uv run python one.py --csv btc_4h.csv --behavior
Fetch: uv run python one.py --download --behavior

Every 4h candle has two ordered engine observations: its known OPEN, then its
OHLC RANGE/CLOSE. This allows vectorbt's built-in stops to work on the entry
candle. No price is interpolated and no intrabar high/low ordering is invented:
vectorbt resolves ambiguous OHLC bars stop-first. Time exits execute at CLOSE.
Target gaps use vectorbt's opening-price fill, including favorable gaps; this
replaces the legacy simulator's target-price cap. New entries do not pay the
funding settlement at their entry instant; already open positions do.
The two observations are execution phases, not extra market candles; do not
annualize their returns as if they were equally spaced 4h data.

Native trades/order records supply all fills, quantities, fees and trading PnL.
Funding is an external per-unit cashflow estimate, not a futures margin engine.
Seven-day MFE/MAE include prices after early exits and are path diagnostics.
Results are exploratory event studies, not independent samples, account returns or edge proof.
"""
from __future__ import annotations

import argparse
import ast
import inspect
import sys
import time
import concurrent.futures
import hashlib
import io
import json
import math
import os
import platform
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd
import vectorbt as vbt

DATA_SOURCE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision/data/futures/um/monthly"
SYMBOL = "BTCUSDT"
INTERVAL = "4h"
DEV_START = "2020-01-01"
DEV_END = "2025-01-01"
OOS_END = "2026-04-01"
HOLD_BARS = 42          # 7 days * 6 (4h bars per day)
ATR_LEN = 14
ATR_MULT = 1.5
TARGET_R = 3.0
FEE_BPS = 4.0
SLIPPAGE_BPS = 2.0
SEED = 42
BOOTSTRAP_N = 10_000
DATA_COLUMNS = ["time", "open", "high", "low", "close", "volume", "funding_rate",
                "funding_observed", "symbol"]
CORE5_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT")
CORE5_FIRST_MONTH = {"BTCUSDT": "2020-01", "ETHUSDT": "2020-01", "SOLUSDT": "2020-09",
                     "XRPUSDT": "2020-01", "BNBUSDT": "2020-02"}

# Stage 01 / Brick 1 Kill Criteria thresholds
MIN_EVENTS = 30
MIN_MEAN_NET_R = 0.15
MIN_DELTA_CONTROL_R = 0.15
MIN_P75_MFE_R = 2.50
MAX_P_VALUE = 0.05


def validate_file_paths(csv: Path, out: Path, evidence: Path, **extra: Path) -> None:
    """Reject aliases before any download/write, including symlinks and hardlinks."""
    paths = {"--csv": csv, "--out": out, "--evidence": evidence, **extra}
    for name, path in paths.items():
        if path.exists() and not path.is_file():
            raise ValueError(f"{name} must be a regular file: {path}")
        if not path.resolve().parent.is_dir():
            raise ValueError(f"Parent directory does not exist for {name}: {path}")
    items = list(paths.items())
    for i, (name, path) in enumerate(items):
        for other_name, other in items[i+1:]:
            if path.resolve() == other.resolve() or (
                path.exists() and other.exists() and path.samefile(other)
            ):
                raise ValueError(f"File paths must be distinct: {name} and {other_name}")


def atomic_write(file: Path, writer: Callable[[TextIO], object]) -> None:
    """Replace a single file only after its complete contents have been flushed."""
    file = file.resolve()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=file.parent,
            prefix=f".{file.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, file)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def cached_archive(url: str, user_agent: str = "onefile-brick1/1.0") -> bytes:
    """Reuse immutable Binance ZIPs; verify cached bytes before parsing or refetching."""
    root=Path(os.environ.get("ONEFILE_CACHE_DIR",Path.home()/".cache"/"onefile"/"binance"))
    path=root/(hashlib.sha256(url.encode()).hexdigest()+".zip")
    digest_path=path.with_suffix(".sha256")
    if path.exists() and digest_path.exists():
        payload=path.read_bytes()
        if hashlib.sha256(payload).hexdigest()==digest_path.read_text().strip() and zipfile.is_zipfile(io.BytesIO(payload)):
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                if archive.namelist(): return payload
    req=urllib.request.Request(url,headers={"User-Agent":user_agent})
    with urllib.request.urlopen(req,timeout=30) as resp:
        payload=resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if not archive.namelist(): raise zipfile.BadZipFile(f"Empty Binance archive: {url}")
    digest=hashlib.sha256(payload).hexdigest()
    root.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb",dir=root,prefix=f".{path.name}.",delete=False) as stream:
        temporary=Path(stream.name); stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)
    atomic_write(digest_path,lambda stream:stream.write(digest+"\n"))
    return payload


def fetch_funding_month(ym: str, symbol: str = SYMBOL) -> pd.DataFrame:
    url = f"{DATA_SOURCE}/fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip"
    raw_zip=cached_archive(url)
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf, zf.open(zf.namelist()[0]) as f:
        df = pd.read_csv(f, header=None)
    if str(df.iloc[0, 0]).isalpha() or str(df.iloc[0, 0]) == "calc_time":
        df = df.iloc[1:].reset_index(drop=True)
    # Normalize only settlement timestamp drift of at most one minute.
    snapped_ms = ((df[0].astype("int64") + 1800_000) // 3600_000) * 3600_000
    if ((snapped_ms - df[0].astype("int64")).abs() > 60_000).any():
        raise ValueError(f"Unexpected funding timestamp drift: {ym}")
    return pd.DataFrame({
        "time_ms": snapped_ms,
        "funding_rate": df[2].astype(float),
    })

def verify_funding_source(clean: pd.DataFrame, end_date: str) -> dict:
    """Compare ALL supplied settlements with SHA-256-checked cached/canonical archives.

    Observation flags describe coverage, not authenticity. Cache misses fetch HTTPS; mismatches fail closed.
    --offline explicitly disables this verification.
    A legitimate zero passes only if the source also reports that zero.
    """
    months = pd.date_range(DEV_START, end_date, freq="MS", inclusive="left").strftime("%Y-%m")
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        source = pd.concat(list(pool.map(fetch_funding_month, months)), ignore_index=True).sort_values("time_ms")
    expected = pd.date_range(DEV_START, end_date, freq="8h", inclusive="left", tz="UTC").as_unit("ms").asi8
    if not np.array_equal(source["time_ms"].to_numpy(), expected):
        raise ValueError("Incomplete, duplicate or off-schedule canonical funding history")
    observed = clean.loc[clean["funding_observed"].eq(1)]
    if (not np.array_equal(observed["time"].array.as_unit("ms").asi8, expected)
            or not np.isfinite(source["funding_rate"].to_numpy()).all()
            or not np.allclose(observed["funding_rate"].to_numpy(), source["funding_rate"].to_numpy(),
                               rtol=0, atol=1e-15)):
        raise ValueError("CSV funding values differ from canonical source; use --download")
    return {"status": "VERIFIED", "method": "Binance monthly archives; local ZIP cache SHA-256 checked",
            "settlements": len(expected), "verified_at_utc": pd.Timestamp.now(tz="UTC").isoformat()}


def download(file: Path, symbol: str = SYMBOL, end_date: str = DEV_END) -> None:
    """Validate cached/downloaded canonical BTC candles and funding before replacing CSV."""
    if symbol != SYMBOL:
        raise ValueError(f"Only {SYMBOL} is supported by this research universe")
    months = pd.date_range(DEV_START, end_date, freq="MS", inclusive="left").strftime("%Y-%m")
    if len(months) == 0:
        raise ValueError("Download window must contain complete monthly archives")
    s3_root = DATA_SOURCE

    def fetch_kline_month(ym: str) -> pd.DataFrame:
        url = f"{s3_root}/klines/{symbol}/{INTERVAL}/{symbol}-{INTERVAL}-{ym}.zip"
        raw_zip=cached_archive(url)
        with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf, zf.open(zf.namelist()[0]) as f:
            df = pd.read_csv(f, header=None)
        if str(df.iloc[0, 0]).isalpha() or str(df.iloc[0, 0]) == "open_time":
            df = df.iloc[1:].reset_index(drop=True)
        return pd.DataFrame({
            "time_ms": df[0].astype("int64"),
            "open": df[1].astype(float),
            "high": df[2].astype(float),
            "low": df[3].astype(float),
            "close": df[4].astype(float),
            "volume": df[5].astype(float),
        })

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        jobs = [(pool.submit(fetch_kline_month, ym), pool.submit(fetch_funding_month, ym))
                for ym in months]
        k_frames = [kline.result() for kline, _ in jobs]
        f_frames = [funding.result() for _, funding in jobs]

    klines = pd.concat(k_frames, ignore_index=True).sort_values("time_ms")
    funding = pd.concat(f_frames, ignore_index=True).sort_values("time_ms")
    for label, frame, frequency in [("candle", klines, "4h"), ("funding", funding, "8h")]:
        expected = pd.date_range(DEV_START, end_date, freq=frequency, inclusive="left", tz="UTC")
        if not np.array_equal(frame["time_ms"].to_numpy(), expected.as_unit("ms").asi8):
            raise ValueError(f"Incomplete, duplicate or off-schedule {label} history")
    if not np.isfinite(funding["funding_rate"].to_numpy()).all():
        raise ValueError("Non-finite funding settlement rate; refusing to replace it with zero")
    merged = klines.merge(funding, on="time_ms", how="left", validate="one_to_one", indicator=True)
    merged["funding_observed"] = (merged.pop("_merge") == "both").astype("int8")
    # Only non-settlement candles get zero; missing settlement rates were rejected above.
    merged.loc[merged["funding_observed"] == 0, "funding_rate"] = 0.0
    merged["symbol"] = symbol
    merged["time"] = pd.to_datetime(merged["time_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    clean = validate(merged[DATA_COLUMNS], start_date=DEV_START, end_date=end_date)
    require_data_readiness(check_data_readiness(clean, file, start_date=DEV_START, end_date=end_date))
    atomic_write(file, lambda stream: clean.to_csv(stream, index=False))


def validate(raw: pd.DataFrame, start_date: str = DEV_START, end_date: str = DEV_END) -> pd.DataFrame:
    """Validate OHLCV integrity and strict 4h spacing (fail-closed)."""
    req = DATA_COLUMNS
    missing = set(req) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Legacy CSVs must be re-downloaded with --download")

    x = raw.copy()
    x = x[req].copy()
    x["time"] = pd.to_datetime(x["time"], utc=True)
    if x["time"].isna().any():
        raise ValueError("Missing candle timestamp")
    x = x[(x["time"] >= pd.Timestamp(start_date, tz="UTC")) &
          (x["time"] < pd.Timestamp(end_date, tz="UTC"))].reset_index(drop=True)
    if len(x) < 60:
        raise ValueError("Insufficient candle history")

    diffs = x["time"].diff().iloc[1:]
    if not (diffs == pd.Timedelta(hours=4)).all():
        raise ValueError("Missing/irregular 4h candle spacing")

    numeric_cols = ["open", "high", "low", "close", "volume", "funding_rate"]
    for col in numeric_cols:
        x[col] = pd.to_numeric(x[col], errors="raise").astype(float)
    if not np.isfinite(x[numeric_cols].to_numpy()).all():
        raise ValueError("Non-finite market/funding data")
    if x["symbol"].isna().any() or not x["symbol"].eq(SYMBOL).all():
        raise ValueError(f"CSV symbol must be {SYMBOL}")
    observed = pd.to_numeric(x["funding_observed"], errors="raise")
    if not observed.isin([0, 1]).all():
        raise ValueError("funding_observed must contain only 0 or 1")
    x["funding_observed"] = observed.astype("int8")
    settlement = x["time"].dt.hour.isin([0, 8, 16])
    if not observed.eq(1).eq(settlement).all():
        raise ValueError("Missing or off-schedule funding observations")
    if x.loc[~settlement, "funding_rate"].ne(0.0).any():
        raise ValueError("Non-zero funding outside BTC 8h settlement schedule")
    if not (x["time"] == x["time"].dt.floor("4h")).all():
        raise ValueError("Candles must open on UTC 4h boundaries")
    if (x[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Non-positive price detected")
    if (x["volume"] < 0).any():
        raise ValueError("Negative volume detected")
    if ((x["high"] < x[["open", "close", "low"]].max(axis=1)) |
            (x["low"] > x[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Invalid OHLC bar geometry")
    return x


def check_data_readiness(
    clean: pd.DataFrame,
    file_path: Path,
    start_date: str = DEV_START,
    end_date: str = DEV_END,
    file_hash: str | None = None,
) -> dict:
    """Stage 01 / B1 Data Readiness & Acceptance Gate (fail-closed verification)."""
    candle_times = pd.date_range(start_date, end_date, freq="4h", inclusive="left", tz="UTC")
    funding_times = pd.date_range(start_date, end_date, freq="8h", inclusive="left", tz="UTC")
    expected_candles, expected_8h = len(candle_times), len(funding_times)
    actual_candles = len(clean)
    qc_error = None
    try:
        validate(clean, start_date=start_date, end_date=end_date)
    except (ValueError, TypeError) as exc:
        qc_error = str(exc)
    observed = clean.get("funding_observed", pd.Series(0, index=clean.index)).eq(1)
    times = pd.to_datetime(clean.get("time", pd.Series(pd.NaT, index=clean.index)), utc=True, errors="coerce")
    fnd_times = times.loc[observed]
    rates = pd.to_numeric(clean.get("funding_rate", pd.Series(np.nan, index=clean.index)), errors="coerce")
    actual_8h = int(observed.sum())
    non_zero_fnd = int((rates.loc[observed].notna() & rates.loc[observed].ne(0.0)).sum())

    if file_hash is None:
        file_bytes = file_path.read_bytes() if file_path.exists() else b""
        file_hash = hashlib.sha256(file_bytes).hexdigest() if file_bytes else "n/a"
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    candle_pass = np.array_equal(times.to_numpy(), candle_times.to_numpy())
    funding_pass = (qc_error is None) and np.array_equal(fnd_times.to_numpy(), funding_times.to_numpy())
    boundary_pass = (
        (times.min() == pd.Timestamp(start_date, tz="UTC"))
        and (times.max() == pd.Timestamp(end_date, tz="UTC") - pd.Timedelta(hours=4))
    )

    all_passed = candle_pass and funding_pass and boundary_pass and qc_error is None
    return {
        "universe_symbol": SYMBOL,
        "interval": INTERVAL,
        "data_source": DATA_SOURCE,
        "start_date": start_date,
        "end_date": end_date,
        "expected_candles": expected_candles,
        "actual_candles": actual_candles,
        "missing_candles": expected_candles - actual_candles,
        "expected_8h_settlements": expected_8h,
        "actual_8h_settlements": actual_8h,
        "non_zero_funding_settlements": non_zero_fnd,
        "file_name": file_path.name,
        "file_sha256": file_hash,
        "code_sha256": code_hash,
        "candle_integrity": "PASS" if candle_pass else "FAIL",
        "funding_coverage": "PASS" if funding_pass else "FAIL",
        "chronological_bounds": "PASS" if boundary_pass else "FAIL",
        "qc_integrity": "PASS" if qc_error is None else "FAIL",
        "qc_error": qc_error,
        "verdict": "PASS" if all_passed else "FAIL",
    }


def require_data_readiness(readiness: dict) -> None:
    """A failed acceptance gate must never reach feature calculation or execution."""
    if readiness["verdict"] != "PASS":
        failed = [key for key in ("candle_integrity", "funding_coverage", "chronological_bounds", "qc_integrity")
                  if readiness[key] != "PASS"]
        raise ValueError(f"Data acceptance FAIL: {', '.join(failed)}; {readiness['qc_error'] or 'incomplete date window'}")


def prepare_study(raw: pd.DataFrame, file_path: Path,
                  start_date: str = DEV_START, end_date: str = DEV_END,
                  file_hash: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Validate full causal history before selecting the active DEV/OOS window."""
    history = validate(raw, start_date=DEV_START, end_date=end_date)
    history_readiness = check_data_readiness(history, file_path, start_date=DEV_START, end_date=end_date,
                                             file_hash=file_hash)
    require_data_readiness(history_readiness)
    active = history.loc[history["time"] >= pd.Timestamp(start_date, tz="UTC")].reset_index(drop=True)
    readiness = check_data_readiness(active, file_path, start_date=start_date, end_date=end_date,
                                     file_hash=file_hash)
    require_data_readiness(readiness)
    readiness["feature_history_start"] = DEV_START
    readiness["feature_history_bars"] = len(history)
    enriched = features(history)
    active_features = enriched.loc[enriched["time"] >= pd.Timestamp(start_date, tz="UTC")].reset_index(drop=True)
    return active_features, readiness


def require_matched_controls(signal_times, control_times, min_events: int = 0) -> None:
    """Require both minimum sample sizes and exact per-month event counts."""
    if len(signal_times) < min_events or len(control_times) < min_events:
        raise ValueError(f"Both setup and control require N >= {min_events}: setup={len(signal_times)}, control={len(control_times)}")

    def counts(times) -> dict:
        parsed = pd.to_datetime(pd.Series(times), utc=True, errors="raise", format="mixed")
        if parsed.isna().any():
            raise ValueError("Missing event timestamp")
        return parsed.dt.strftime("%Y-%m").value_counts().to_dict()

    if counts(signal_times) != counts(control_times):
        raise ValueError("Insufficient same-month non-overlapping controls: monthly counts differ")


def features(x: pd.DataFrame) -> pd.DataFrame:
    """Compute strictly causal ATR, regime, and Price+Volume Momentum-Continuation features."""
    df = x.copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_LEN, min_periods=ATR_LEN).mean()
    df["atr_pct"] = df["atr"] / df["close"]
    atr_base = df["atr_pct"].rolling(120, min_periods=30).median()
    df["vol_ratio"] = df["atr_pct"] / atr_base

    ema_fast = df["close"].ewm(span=20, adjust=False).mean()
    ema_slow = df["close"].ewm(span=50, adjust=False).mean()
    ema_macro = df["close"].ewm(span=200, adjust=False).mean()
    vol_sma = df["volume"].rolling(20, min_periods=20).mean()

    # 1. Trend regimes (micro vs macro)
    df["trend"] = (ema_fast > ema_slow) & (df["close"] > ema_slow) & df["atr"].notna()
    df["macro_trend"] = df["trend"] & (df["close"] > ema_macro) & (ema_slow > ema_macro)

    # 2. Volume-backed impulse bar & recent 6-bar (24h) impulse memory
    df["impulse_bar"] = ((df["close"] - df["open"]) >= 1.0 * df["atr"]) & (df["volume"] >= 1.2 * vol_sma)
    df["recent_impulse"] = df["impulse_bar"].shift(1).rolling(6, min_periods=1).max() > 0

    # 3. Controlled pullback: prior 1-3 bars dipped near EMA20 without breaking EMA50
    pullback_low = df["low"].shift(1).rolling(3, min_periods=1).min()
    touched_pullback = pullback_low <= (ema_fast.shift(1) * 1.008)
    held_structure = (
        (pullback_low > ema_slow.shift(1))
        & (df["close"].shift(1) < df["high"].shift(1).rolling(6, min_periods=1).max())
    )
    df["pullback"] = (touched_pullback & held_structure).fillna(False)

    # 4. Continuation trigger: bullish bar reclaiming prior bar high and fast EMA
    df["continuation"] = (
        (df["close"] > df["open"])
        & (df["close"] > df["high"].shift(1))
        & (df["close"] > ema_fast)
    ).fillna(False)

    df["signal"] = (
        df["trend"] & df["recent_impulse"] & df["pullback"] & df["continuation"]
    ).fillna(False)

    # 5. Additional market mechanism observables (strictly causal at bar t close)
    df["funding_3d_bps"] = df["funding_rate"].rolling(18, min_periods=1).sum() * 10_000.0
    df["uncrowded_funding"] = df["funding_3d_bps"] <= 10.0
    df["donchian_7d_breakout"] = (df["close"] > df["high"].shift(1).rolling(42, min_periods=42).max()).fillna(False)
    df["squeeze_breakout"] = ((df["vol_ratio"] < 0.85) & df["donchian_7d_breakout"]).fillna(False)
    flush_bar = ((df["open"] - df["close"]) >= 1.5 * df["atr"]) & (df["volume"] >= 1.5 * vol_sma)
    recent_flush = flush_bar.shift(1).rolling(6, min_periods=1).max() > 0
    df["flush_reclaim"] = (
        recent_flush & (df["close"] > df["open"]) & (df["close"] > df["high"].shift(1))
    ).fillna(False)
    return df


def select_setup_events(x: pd.DataFrame) -> list[int]:
    """All mature setup events; never discard them because controls are unavailable."""
    signal = x["signal"].to_numpy(dtype=bool)
    signals: list[int] = []
    for i in range(max(0, len(x) - HOLD_BARS)):
        if signal[i] and (not signals or i - signals[-1] > HOLD_BARS):
            signals.append(i)
    return signals


def select_events(x: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Outcome-blind matched subset; unmatched setups remain in the full study.

    Controls exclude EVERY original setup horizon, including unmatched setups.
    Earliest eligible control first maximizes disjoint equal-length intervals
    within each month. Seeded setup subsampling uses timestamps only, not returns.
    Lack of common support is reported, never repaired by overlapping windows.
    """
    signals = select_setup_events(x)
    month = x["time"].dt.strftime("%Y-%m").to_numpy()
    trend = x["trend"].to_numpy(dtype=bool)
    signal = x["signal"].to_numpy(dtype=bool)
    controls: list[int] = []
    matched: list[int] = []
    blocked = np.zeros(len(x), dtype=bool)
    for event in signals:
        blocked[max(0, event-HOLD_BARS):event+HOLD_BARS+1] = True
    rng = np.random.default_rng(SEED)
    for m in sorted({month[i] for i in signals}):
        setups = [i for i in signals if month[i] == m]
        chosen: list[int] = []
        candidates = np.flatnonzero((month == m) & trend & ~signal)
        candidates = candidates[candidates < max(0, len(x)-HOLD_BARS)]
        for candidate in candidates:
            i = int(candidate)
            if blocked[i]:
                continue
            chosen.append(i)
            blocked[max(0, i-HOLD_BARS):i+HOLD_BARS+1] = True
            if len(chosen) == len(setups):
                break
        if chosen:
            controls.extend(chosen)
            matched.extend(rng.choice(setups, size=len(chosen), replace=False).tolist())
    matched.sort()
    require_matched_controls(x["time"].iloc[matched], x["time"].iloc[controls])
    return matched, sorted(controls)


def evaluate_events(x: pd.DataFrame, indices: list[int], target_r: float = TARGET_R,
                    barriers: bool = True, *, side: int = 1,
                    hold_bars: int = HOLD_BARS) -> list[dict]:
    """Batch disjoint events through vectorbt; adapt native trade records only.

    Signal t close -> t+1 OPEN; last allowed bar is entry+HOLD_BARS-1 CLOSE.
    1 BTC fixed size and unlimited research capital avoid accidental cash-based
    order rejection. There is no account sizing, compounding or leverage claim.
    Funding at a bar OPEN is charged only if the position already existed before
    that OPEN. An exiting position pays that timestamp; a new entry does not.
    """
    _function_started = time.perf_counter()
    if side not in (-1, 1) or isinstance(side, bool):
        raise ValueError("side must be +1 or -1")
    if not isinstance(hold_bars, int) or hold_bars < 1:
        raise ValueError("hold_bars must be a positive integer")
    if not indices:
        return []
    ids = np.asarray(indices)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Event indices must be integers")
    if len(set(indices)) != len(indices) or np.any(np.diff(ids) <= hold_bars):
        raise ValueError("Events must be sorted, unique and non-overlapping")
    if ids[0] < 0 or ids[-1] + hold_bars >= len(x):
        raise ValueError("Unmatured horizon or invalid event index")
    if not np.isfinite(target_r) or target_r <= 0:
        raise ValueError("target_r must be finite and positive")
    n = len(x)
    open_values = x["open"].to_numpy()
    high_values = x["high"].to_numpy()
    low_values = x["low"].to_numpy()
    close_values = x["close"].to_numpy()
    funding_values = x["funding_rate"].to_numpy()
    time_values = x["time"].array
    funding_prefix = np.r_[0.0, np.cumsum(open_values * funding_values)]
    entry_bars = ids + 1
    exit_bars = ids + hold_bars
    risk = ATR_MULT * x["atr"].to_numpy()[ids]
    expected_entry = open_values[entry_bars] * (1 + side * SLIPPAGE_BPS / 10_000)
    if not np.isfinite(risk).all() or np.any(risk <= 0) or np.any(risk >= expected_entry):
        raise ValueError("Invalid execution geometry/ATR")

    # Pure data adapter: only OPEN is observable on phase 0. Original OHLC is
    # supplied on phase 1. All order matching and stop priority belong to vbt.
    opens = np.repeat(open_values, 2)
    highs, lows, closes = opens.copy(), opens.copy(), opens.copy()
    highs[1::2] = high_values
    lows[1::2] = low_values
    closes[1::2] = close_values
    entries, exits = np.zeros(2*n, bool), np.zeros(2*n, bool)
    entries[2*entry_bars] = True
    exits[2*exit_bars+1] = True
    sl, tp = np.full(2*n, np.nan), np.full(2*n, np.nan)
    if barriers:
        sl[2*entry_bars] = risk / expected_entry
        tp[2*entry_bars] = target_r * risk / expected_entry
    _vectorbt_started = time.perf_counter()
    pf = vbt.Portfolio.from_signals(
        close=pd.Series(closes), entries=entries, exits=exits, price=closes,
        open=opens, high=highs, low=lows, size=1.0, size_type="amount",
        direction="longonly" if side == 1 else "shortonly", accumulate=False, init_cash=np.inf,
        fees=FEE_BPS/10_000, slippage=SLIPPAGE_BPS/10_000,
        sl_stop=sl, tp_stop=tp, stop_entry_price="fillprice",
        stop_exit_price="stopmarket", use_stops=barriers,
        allow_partial=False, raise_reject=True,
    )
    _perf_time("evaluate_events.vectorbt_seconds", time.perf_counter() - _vectorbt_started)
    trades = pf.trades.records_arr
    orders = pf.orders.records_arr
    if len(trades) != len(ids) or len(orders) != 2*len(ids):
        raise RuntimeError("vectorbt did not produce exactly one round trip per event")
    rows = []
    for j, trade in enumerate(trades):
        signal_i = int(ids[j])
        entry_i, exit_i = int(trade["entry_idx"]), int(trade["exit_idx"])
        if trade["status"] != 1 or entry_i != 2*int(entry_bars[j]):
            raise RuntimeError("Unexpected vectorbt open trade/entry phase")
        entry_bar, exit_bar = entry_i//2, exit_i//2
        entry, exit_price = float(trade["entry_price"]), float(trade["exit_price"])
        quantity = float(trade["size"])
        stop, target = entry-side*risk[j], entry+side*target_r*risk[j]
        fees = float(trade["entry_fees"] + trade["exit_fees"])
        native_pnl = float(trade["pnl"])
        # Classify the already executed native fill; never select a fill here.
        unslipped_exit = exit_price / (1-side*SLIPPAGE_BPS/10_000)
        reason = "TIME"
        if barriers:
            if side*(unslipped_exit-stop) <= 1e-9:
                reason = "STOP_GAP" if side*(unslipped_exit-stop) < -1e-9 else "STOP"
            elif side*(unslipped_exit-target) >= -1e-9:
                reason = "TARGET"
        funding = side * float(np.round(funding_prefix[exit_bar+1] - funding_prefix[entry_bar+1], 12)) * quantity
        horizon = slice(entry_bar, entry_bar+hold_bars)
        raw_entry, raw_exit = float(open_values[entry_bar]), float(closes[exit_i])
        # Cost_R is intended for barrier-free path comparison only.
        cost_r = (side*(raw_exit-raw_entry)*quantity-native_pnl+funding)/(risk[j]*quantity) if not barriers else np.nan
        rows.append({
            "signal_time": str(time_values[signal_i]),
            "entry_time": str(time_values[entry_bar]),
            "exit_time": str(time_values[exit_bar]),
            "entry_phase": "OPEN", "exit_phase": "OPEN" if exit_i%2 == 0 else "RANGE_CLOSE",
            "exit": reason, "entry_price": entry, "exit_price": exit_price,
            "stop_price": stop if barriers else np.nan,
            "target_price": target if barriers else np.nan,
            "risk_usdt": float(risk[j]), "size_btc": quantity,
            "fee_usdt_per_btc": fees/quantity,
            "funding_usdt_per_btc": funding/quantity,
            "vectorbt_trade_id": int(trade["id"]), "native_pnl_usdt": native_pnl,
            "net_R_ex_funding": native_pnl/(risk[j]*quantity),
            "net_R": (native_pnl-funding)/(risk[j]*quantity), "cost_R": cost_r,
            "path_MFE_R": ((float(np.max(high_values[horizon]))-entry) if side == 1 else
                           (entry-float(np.min(low_values[horizon]))))/risk[j],
            "path_MAE_R": ((float(np.min(low_values[horizon]))-entry) if side == 1 else
                           (entry-float(np.max(high_values[horizon]))))/risk[j],

            "target_before_stop": int(reason == "TARGET"),
        })
    _perf_time("evaluate_events.total_seconds", time.perf_counter() - _function_started)
    return rows


def evaluate_event_matrix(x: pd.DataFrame, indices: list[int], target_rs: tuple[float,...], *,
                          side: int, hold_bars: int,
                          barriers: bool | tuple[bool,...] = True) -> list[dict]:

    """Run independent event paths as vectorbt columns instead of replaying full history per phase."""
    _function_started = time.perf_counter()
    if side not in (-1,1) or isinstance(side,bool): raise ValueError("side must be +1 or -1")
    if not isinstance(hold_bars,int) or hold_bars<1: raise ValueError("hold_bars must be positive")
    if not indices: return []
    ids=np.asarray(indices)
    if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or np.any(ids<0) or np.any(ids+hold_bars>=len(x)):
        raise ValueError("Invalid or unmatured event indices")
    if len(set(ids.tolist()))!=len(ids) or not target_rs or any(not np.isfinite(r) or r<=0 for r in target_rs):
        raise ValueError("Events must be unique and targets positive")
    event_ids=np.repeat(ids,len(target_rs)); targets=np.tile(np.asarray(target_rs,float),len(ids))
    if isinstance(barriers, (bool, np.bool_)):
        barrier_cols=np.full(len(event_ids), barriers, dtype=bool)
    else:
        modes=np.asarray(barriers)
        if modes.ndim!=1 or len(modes)!=len(target_rs) or not np.isin(modes,[True,False]).all():
            raise ValueError("barriers must be a boolean or one boolean per target")
        barrier_cols=np.tile(modes.astype(bool),len(ids))
    bars=event_ids[:,None]+np.arange(hold_bars+1)[None,:]
    open_values=x.open.to_numpy(); funding_values=x.funding_rate.to_numpy()
    op=open_values[bars].T; hi=x.high.to_numpy()[bars].T
    lo=x.low.to_numpy()[bars].T; cl=x.close.to_numpy()[bars].T
    risk=ATR_MULT*x.atr.to_numpy()[event_ids]
    entry=op[1]*(1+side*SLIPPAGE_BPS/10_000)
    if not np.isfinite(risk).all() or np.any(risk<=0) or np.any(risk>=entry):
        raise ValueError("Invalid execution geometry/ATR")
    opens=np.repeat(op,2,axis=0); highs=opens.copy(); lows=opens.copy(); closes=opens.copy()
    highs[1::2]=hi; lows[1::2]=lo; closes[1::2]=cl
    nrows,ncols=closes.shape
    entries=np.zeros((nrows,ncols),bool); exits=np.zeros_like(entries)
    entries[2,:]=True; exits[2*hold_bars+1,:]=True
    sl=np.full((nrows,ncols),np.nan); tp=np.full((nrows,ncols),np.nan)
    sl[2,barrier_cols]=risk[barrier_cols]/entry[barrier_cols]
    tp[2,barrier_cols]=targets[barrier_cols]*risk[barrier_cols]/entry[barrier_cols]
    _vectorbt_started = time.perf_counter()
    pf=vbt.Portfolio.from_signals(
        close=pd.DataFrame(closes),entries=entries,exits=exits,price=closes,
        open=opens,high=highs,low=lows,size=1.,size_type="amount",
        direction="longonly" if side==1 else "shortonly",accumulate=False,init_cash=np.inf,
        fees=FEE_BPS/10_000,slippage=SLIPPAGE_BPS/10_000,sl_stop=sl,tp_stop=tp,
        stop_entry_price="fillprice",stop_exit_price="stopmarket",use_stops=True,
        allow_partial=False,raise_reject=True)
    _perf_time("evaluate_event_matrix.vectorbt_seconds", time.perf_counter() - _vectorbt_started)
    trades=pf.trades.records_arr
    if len(trades)!=ncols or len(pf.orders.records_arr)!=2*ncols:
        raise RuntimeError("vectorbt matrix did not produce one round trip per event")
    by_col={int(trade["col"]):trade for trade in trades}; rows=[]
    for col in range(ncols):
        trade=by_col[col]; entry_i,exit_i=int(trade["entry_idx"]),int(trade["exit_idx"])
        if trade["status"]!=1 or entry_i!=2:
            raise RuntimeError("Unexpected vectorbt matrix trade/entry phase")
        entry_bar,exit_bar=entry_i//2,exit_i//2
        fill_entry,fill_exit=float(trade["entry_price"]),float(trade["exit_price"])
        reason="TIME"
        if barrier_cols[col]:
            unslipped_exit=fill_exit/(1-side*SLIPPAGE_BPS/10_000)
            stop,target=fill_entry-side*risk[col],fill_entry+side*targets[col]*risk[col]
            if side*(unslipped_exit-stop)<=1e-9:
                reason="STOP_GAP" if side*(unslipped_exit-stop)<-1e-9 else "STOP"
            elif side*(unslipped_exit-target)>=-1e-9:
                reason="TARGET"
        path=bars[col]
        held_idx=path[entry_bar+1:exit_bar+1]
        funding=side*float(np.sum(open_values[held_idx]*funding_values[held_idx]))
        horizon=slice(1,hold_bars+1)
        mfe=((float(np.max(hi[horizon,col]))-fill_entry) if side==1 else
             (fill_entry-float(np.min(lo[horizon,col]))))/risk[col]
        mae=((float(np.min(lo[horizon,col]))-fill_entry) if side==1 else
             (fill_entry-float(np.max(hi[horizon,col]))))/risk[col]
        pnl=float(trade["pnl"]); quantity=float(trade["size"])
        raw_entry=float(op[entry_bar,col]); raw_exit=float(cl[exit_bar,col])
        cost_r=((side*(raw_exit-raw_entry)*quantity-pnl+funding*quantity)/(risk[col]*quantity)
                if not barrier_cols[col] else np.nan)
        rows.append({"signal_i":int(event_ids[col]),"target_r":float(targets[col]),"exit":reason,
                     "entry_price":fill_entry,"exit_price":fill_exit,"risk_usdt":float(risk[col]),
                     "net_R_ex_funding":pnl/(risk[col]*quantity),"net_R":(pnl-funding*quantity)/(risk[col]*quantity),
                     "target_before_stop":int(reason=="TARGET"),"funding_usdt_per_btc":funding,
                     "native_pnl_usdt":pnl,"vectorbt_trade_id":int(trade["id"]),
                     "barriers":bool(barrier_cols[col]),"path_MFE_R":mfe,"path_MAE_R":mae,"cost_R":cost_r})
    _perf_time("evaluate_event_matrix.total_seconds", time.perf_counter() - _function_started)
    return rows


def select_non_overlapping(mask: pd.Series, max_idx: int, times=None, phase: int = 0) -> list[int]:
    """Select non-overlapping indices spaced by > HOLD_BARS from any boolean condition mask."""
    # Fixed calendar grid avoids condition-specific greedy start-date selection.
    spacing = HOLD_BARS + 1
    if not isinstance(phase, int) or not 0 <= phase < spacing:
        raise ValueError("Invalid sampling phase")
    arr = mask.fillna(False).to_numpy(dtype=bool)
    clock = (pd.DatetimeIndex(times).as_unit("ns").asi8 // pd.Timedelta(hours=4).value
             if times is not None else np.arange(len(mask)))
    return [i for i in range(min(len(mask), max_idx + 1)) if arr[i] and clock[i] % spacing == phase]


def analyze_behavior(x: pd.DataFrame, target_r: float = TARGET_R) -> None:
    """Stage 02: Measure conditional trend, momentum, pullback, volatility, funding & Monograph H1-A behavior."""
    max_idx = len(x) - HOLD_BARS - 1
    valid = x["atr"].notna()
    # Cache each eligible event path once. Independent vectorbt columns can be
    # batched across phases; phase subsets are still NOT independent trials.
    barrier_rows, free_rows = {}, {}
    # Each mature bar belongs to exactly one UTC phase. Events are independent
    # vectorbt columns, so bounded batches avoid 43 separate Portfolio builds.
    eligible_ids = np.flatnonzero(valid.to_numpy()[:max_idx + 1])
    for first in range(0, len(eligible_ids), 512):
        ids = eligible_ids[first:first + 512].tolist()
        rows = evaluate_event_matrix(x, ids, (target_r, target_r), side=1, hold_bars=HOLD_BARS,
                                      barriers=(True, False))
        for row in rows:
            (barrier_rows if row["barriers"] else free_rows)[row["signal_i"]] = row

    conditions: list[tuple[str, pd.Series]] = [
        ("C0: All Bars (Unconditional)", valid),
        ("C1: Bear/Counter-Trend (~Trend)", valid & ~x["trend"]),
        ("C2: Micro Uptrend (EMA20>EMA50)", valid & x["trend"]),
        ("C3: Macro Uptrend (+Close>EMA200)", valid & x["macro_trend"]),
        ("C4: Impulse Chase (Enter on Impulse)", valid & x["trend"] & x["impulse_bar"]),
        ("C5: Pullback Only (No Prior Impulse)", valid & x["trend"] & ~x["recent_impulse"] & x["pullback"] & x["continuation"]),
        ("C6: Brick 1 Setup (Imp+PB+Cont)", valid & x["signal"]),
        ("C7: Macro Brick 1 (Setup + >EMA200)", valid & x["signal"] & x["macro_trend"]),
        ("C8: Donchian 7d Breakout", valid & x["donchian_7d_breakout"]),
        ("C9: Vol Squeeze Breakout (<0.85x+7dH)", valid & x["squeeze_breakout"]),
        ("C10: Capitulation Flush Reclaim", valid & x["flush_reclaim"]),
        ("C11: Brick 1 + Uncrowded (3dF<=10bp)", valid & x["signal"] & x["uncrowded_funding"]),
        ("C12: Brick 1 + Crowded (3dF>10bp)", valid & x["signal"] & ~x["uncrowded_funding"]),
        ("C13: Macro Brick 1 + Uncrowded", valid & x["signal"] & x["macro_trend"] & x["uncrowded_funding"]),
    ]

    print("\n=== STAGE 02: CONDITIONAL MARKET BEHAVIOR & CONTINUATION MATRIX ===")
    print(f"Non-overlapping 7d ({HOLD_BARS}-bar) episodes | 1R={ATR_MULT}xATR | Barrier-Free (BF_7d_R) vs Barrier ({target_r}R TP / -1R SL)")
    header = (
        f"  {'Condition':<36} {'N':>4} {'BF_7d_R':>8} {'Cost_R':>7} {'TP%':>6} {'Mean_R':>8} "
        f"{'Med_MFE':>8} {'P75_MFE':>8} {'Med_MAE':>8} {'Skew':>7}"
    )
    print("Fixed UTC phase 0; all-phase sensitivity below is descriptive, NOT a condition ranking or significance test.")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for label, mask in conditions:
        idxs = select_non_overlapping(mask, max_idx, x["time"])
        if not idxs:
            print(f"  {label:<36} {0:>4d}      n/a     n/a    n/a      n/a      n/a      n/a      n/a     n/a")
            continue
        rows = [barrier_rows[i] for i in idxs]
        bf_rows = [free_rows[i] for i in idxs]
        a = pd.DataFrame(rows)
        bf = pd.DataFrame(bf_rows)
        win_pct = float(a.target_before_stop.mean()) * 100
        mean_r = float(a.net_R.mean())
        bf_mean_r = float(bf.net_R.mean())
        cost_r = float(bf.cost_R.mean())
        med_mfe = float(a.path_MFE_R.median())
        p75_mfe = float(np.percentile(a.path_MFE_R, 75))
        med_mae = float(a.path_MAE_R.median())
        skew = med_mfe / max(abs(med_mae), 1e-9)
        print(
            f"  {label:<36} {len(a):>4d} {bf_mean_r:>+7.2f}R {cost_r:>6.2f}R {win_pct:>5.1f}% {mean_r:>+7.3f}R "
            f"{med_mfe:>+7.2f}R {p75_mfe:>+7.2f}R {med_mae:>+7.2f}R {skew:>6.2f}x"
        )

    print("\nPHASE SENSITIVITY: same UTC grids for all conditions; overlapping phases, no independent-N claim")
    for label, mask in conditions:
        means = []
        eligible = [i for i in barrier_rows if bool(mask.iloc[i])]
        for phase in range(HOLD_BARS + 1):
            ids = select_non_overlapping(mask, max_idx, x["time"], phase)
            if ids:
                means.append(float(np.mean([barrier_rows[i]["net_R"] for i in ids])))
        if means:
            print(f"  {label:<36} eligible_windows={len(eligible):4d} populated_phases={len(means):2d}/43 "
                  f"phase_mean_range=[{min(means):+.3f}, {max(means):+.3f}]R")
        else:
            print(f"  {label:<36} no mature windows")

    # Volatility regime conditioning on Setup vs Micro Uptrend
    print("\n=== VOLATILITY REGIME CONDITIONING (ATR% Relative to 20d Rolling Median) ===")
    vol_regimes = [
        ("Low Vol / Compression (< 0.85x)", x["vol_ratio"] < 0.85),
        ("Normal Volatility (0.85x - 1.25x)", (x["vol_ratio"] >= 0.85) & (x["vol_ratio"] <= 1.25)),
        ("High Vol / Expansion (> 1.25x)", x["vol_ratio"] > 1.25),
    ]
    print(f"  {'Volatility Regime':<36} {'Group':<10} {'N':>5} {'TP%':>6} {'Mean_R':>8} {'Med_MFE':>8} {'P75_MFE':>8} {'Med_MAE':>8}")
    print("  " + "-" * 96)
    for v_label, v_mask in vol_regimes:
        for g_label, g_mask in [("Trend", valid & x["trend"] & v_mask), ("Setup", valid & x["signal"] & v_mask)]:
            idxs = select_non_overlapping(g_mask, max_idx, x["time"])
            if not idxs:
                continue
            a = pd.DataFrame([barrier_rows[i] for i in idxs])
            print(
                f"  {v_label:<36} {g_label:<10} {len(a):>5d} {a.target_before_stop.mean()*100:>5.1f}% "
                f"{a.net_R.mean():>+7.3f}R {a.path_MFE_R.median():>+7.2f}R "
                f"{np.percentile(a.path_MFE_R, 75):>+7.2f}R {a.path_MAE_R.median():>+7.2f}R"
            )

    # Monograph Section 06 (H1-A) 1D Regime Onset & Threshold Proximity Empirical Audit
    d1 = x.assign(date=x["time"].dt.floor("D")).groupby("date").agg(
        close=("close", "last"),
    ).reset_index()
    d1["ret"] = np.log(d1["close"] / d1["close"].shift(1))
    r_sum = d1["ret"].rolling(20).sum()
    r_std = np.sqrt((d1["ret"] ** 2).rolling(20).sum() + 1e-12)
    d1["z20"] = r_sum / r_std
    d1["er20"] = r_sum.abs() / (d1["ret"].abs().rolling(20).sum() + 1e-12)
    d1["q"] = np.where((d1["z20"] >= 1.25) & (d1["er20"] >= 0.35), 1,
              np.where((d1["z20"] <= -1.25) & (d1["er20"] >= 0.35), -1, 0))
    q_values = d1["q"].to_numpy()
    run_len = np.zeros(len(d1), dtype=int)
    for i in range(1, len(d1)):
        if q_values[i] == 0:
            run_len[i] = run_len[i - 1] + 1
    d1["in_risk_set"] = (d1["q"] == 0) & (run_len >= 10) & (d1.index >= 30)
    onsets_up, onsets_dn, unconfirmed = [], [], []
    risk_set = d1["in_risk_set"].to_numpy()
    for i in range(31, len(d1) - 10):
        if q_values[i] == 0:
            continue
        if bool(risk_set[i - 1]) and int(q_values[i]) != 0:
            sign = int(q_values[i])
            if int((q_values[i + 1:i + 11] == sign).sum()) >= 8:

                (onsets_up if sign == 1 else onsets_dn).append(i)
            else:
                unconfirmed.append(i)
    print("H1-A confirmed onsets use ten FUTURE days for retrospective labeling only; not a tradable signal.")
    print("\n=== MONOGRAPH H1-A (1D REGIME ONSET & THRESHOLD PROXIMITY) EMPIRICAL AUDIT ===")
    print(f"  1D UTC sample: {len(d1)} days | Risk Set (q=0 for >=10d): {int(d1.in_risk_set.sum())} days ({d1.in_risk_set.mean():.1%})")
    print(f"  Confirmed +1 Onsets: {len(onsets_up)} (Monograph min floor: >=20) | Confirmed -1 Onsets: {len(onsets_dn)} (floor: >=20) | False starts: {len(unconfirmed)}")
    if onsets_up:
        pre_10d = np.median([(d1.close.iloc[i] / d1.close.iloc[i - 10] - 1) * 100 for i in onsets_up])
        post_7d = np.median([(d1.close.iloc[min(len(d1) - 1, i + 7)] / d1.close.iloc[i] - 1) * 100 for i in onsets_up])
        z_t1 = np.median([d1.z20.iloc[i - 1] for i in onsets_up])
        er_t1 = np.median([d1.er20.iloc[i - 1] for i in onsets_up])
        print(f"  Threshold Proximity at t0-1: median Z20={z_t1:.2f} (thresh 1.25), median ER20={er_t1:.2f} (thresh 0.35)")
        print(f"  Price Move Before vs After t0: median 10d pre-onset move = {pre_10d:+.1f}% vs 7d post-onset move = {post_7d:+.1f}%")


def bootstrap_stats(setup_r: np.ndarray, ctrl_r: np.ndarray, n_boot: int = BOOTSTRAP_N,
                    setup_times=None, ctrl_times=None, trials: int | None = None) -> dict:
    """Paired calendar moving-block bootstrap, NOT an IID permutation test.

    Both groups must have the same month counts. Empty calendar months are kept.
    Joint resampling of monthly sums/counts preserves within-month dependence and
    contiguous 1/3/6-month regimes. We report the widest mean CI and largest
    centered-null p across those predeclared block lengths (no best-block search).
    This is an approximate stationary-block model, not a proof of market edge.
    Without a declared total trial count, adjusted significance is unavailable.
    """
    setup_r, ctrl_r = np.asarray(setup_r, float), np.asarray(ctrl_r, float)
    if (setup_r.ndim != 1 or ctrl_r.ndim != 1 or min(len(setup_r), len(ctrl_r)) < 2
            or not np.isfinite(setup_r).all() or not np.isfinite(ctrl_r).all()):
        raise ValueError("Finite one-dimensional samples with at least two observations required")
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 100:
        raise ValueError("n_boot must be an integer >= 100")
    if trials is not None and (not isinstance(trials, (int, np.integer)) or isinstance(trials, bool) or trials < 1):
        raise ValueError("trials must be a positive integer or unknown")
    if setup_times is None or ctrl_times is None:
        raise ValueError("Calendar timestamps are required for dependence-aware inference")
    if len(setup_times) != len(setup_r) or len(ctrl_times) != len(ctrl_r):
        raise ValueError("Timestamp/sample lengths differ")
    require_matched_controls(setup_times, ctrl_times)
    st = pd.to_datetime(pd.Series(list(setup_times)), utc=True, format="mixed")
    ct = pd.to_datetime(pd.Series(list(ctrl_times)), utc=True, format="mixed")
    sm, cm = st.dt.strftime("%Y-%m"), ct.dt.strftime("%Y-%m")
    months = pd.period_range(sm.min(), sm.max(), freq="M").astype(str)
    s_sum = pd.Series(setup_r).groupby(sm).sum().reindex(months, fill_value=0).to_numpy()
    c_sum = pd.Series(ctrl_r).groupby(cm).sum().reindex(months, fill_value=0).to_numpy()
    counts = sm.value_counts().reindex(months, fill_value=0).to_numpy()
    obs_mean, obs_diff = float(setup_r.mean()), float(setup_r.mean() - ctrl_r.mean())
    rng = np.random.default_rng(SEED)
    details = []
    for block in (1, 3, 6):
        length = min(block, len(months))
        starts = rng.integers(0, len(months), size=(n_boot, int(np.ceil(len(months) / length))))
        ids = ((starts[:, :, None] + np.arange(length)) % len(months)).reshape(n_boot, -1)[:, :len(months)]
        ns = counts[ids].sum(axis=1)
        nonempty = ns > 0
        # A missing bootstrap sample is not a zero-return observation.
        if nonempty.sum() < n_boot * .9:
            raise ValueError("Too few nonempty calendar-block resamples")
        boot_mean = s_sum[ids[nonempty]].sum(axis=1) / ns[nonempty]
        boot_diff = (s_sum - c_sum)[ids[nonempty]].sum(axis=1) / ns[nonempty]
        # Null-centered bootstrap tests H0: mean difference <= 0, one sided.
        p = float((1 + np.sum(boot_diff - obs_diff >= obs_diff)) / (len(boot_diff) + 1))
        lo, hi = np.percentile(boot_mean, [2.5, 97.5])
        dlo, dhi = np.percentile(boot_diff, [2.5, 97.5])
        details.append({"block_months": length, "ci_low": float(lo), "ci_high": float(hi),
                        "delta_ci_low": float(dlo), "delta_ci_high": float(dhi), "p_value": p})
    raw_p = max(d["p_value"] for d in details)
    return {"ci_low": min(d["ci_low"] for d in details),
            "ci_high": max(d["ci_high"] for d in details),
            "delta_ci_low": min(d["delta_ci_low"] for d in details),
            "delta_ci_high": max(d["delta_ci_high"] for d in details),
            "obs_mean": obs_mean, "obs_diff": obs_diff, "p_value": raw_p,
            "adjusted_p_value": min(1., raw_p * trials) if trials is not None else None,
            "trials": trials, "calendar_months": len(months),
            "active_months": int((counts > 0).sum()), "block_sensitivity": details,
            "method": "paired calendar moving-block bootstrap; centered-null test; 1/3/6 months"}


def summarize(label: str, rows: list[dict]) -> None:
    print(f"\n{label}: {len(rows)} non-overlapping entries")
    if not rows:
        print("  Insufficient events — NO CONCLUSION")
        return
    a = pd.DataFrame(rows)
    p25, p50, p75, p90 = np.percentile(a.path_MFE_R, [25, 50, 75, 90])
    funding_drag = a.net_R_ex_funding.mean() - a.net_R.mean()
    print(f"  Target first:        {a.target_before_stop.mean():.1%}")
    print(f"  Mean net R (ex-fnd): {a.net_R_ex_funding.mean():+.3f} R")
    print(f"  Mean net R (w/ fnd): {a.net_R.mean():+.3f} R  (funding drag: {-funding_drag:+.3f} R)")
    print(f"  Median net R:        {a.net_R.median():+.3f} R")
    print(f"  7d path MFE (P25/50/75/90): {p25:+.2f}R / {p50:+.2f}R / {p75:+.2f}R / {p90:+.2f}R")
    print(f"  Median 7d path MAE:  {a.path_MAE_R.median():+.2f} R")
    print(f"  Stops / targets / time exits: {a['exit'].str.startswith('STOP').sum()} / {(a['exit'] == 'TARGET').sum()} / {(a['exit'] == 'TIME').sum()}")


def summarize_by_year(signal_rows: list[dict], control_rows: list[dict]) -> None:
    if not signal_rows:
        return
    s = pd.DataFrame(signal_rows)
    c = pd.DataFrame(control_rows) if control_rows else pd.DataFrame()
    s["year"] = pd.to_datetime(s["signal_time"]).dt.year
    if not c.empty:
        c["year"] = pd.to_datetime(c["signal_time"]).dt.year

    print("\nYEARLY / REGIME BREAKDOWN (Full Cost incl. Funding):")
    print(f"  {'Year':<6} {'N_setup':>7} {'TP%':>7} {'Setup_R':>9} {'Ctrl_R':>9} {'Delta_R':>9}")
    for yr in sorted(s["year"].unique()):
        sy = s[s["year"] == yr]
        cy = c[c["year"] == yr] if not c.empty else pd.DataFrame()
        s_mean = float(sy.net_R.mean())
        c_mean = float(cy.net_R.mean()) if not cy.empty else float("nan")
        delta = s_mean - c_mean if not np.isnan(c_mean) else float("nan")
        win_pct = float(sy.target_before_stop.mean()) * 100
        print(f"  {yr:<6} {len(sy):>7d} {win_pct:>6.1f}% {s_mean:>+8.3f}R {c_mean:>+8.3f}R {delta:>+8.3f}R")


def require_disjoint_events(signal_times, control_times) -> None:
    """Validate conservative 7-day research windows within and between groups."""
    groups = [pd.DatetimeIndex(pd.to_datetime(list(t), utc=True, format="mixed")).sort_values().as_unit("ns").asi8
              for t in (signal_times, control_times)]
    horizon = pd.Timedelta(hours=4 * HOLD_BARS).value
    for group in groups:
        if np.any(group == pd.NaT.value):
            raise ValueError("Missing event timestamp")
        if np.any(np.diff(group) <= horizon):
            raise ValueError("Overlapping/duplicate event horizons within a group")
    if groups[0].size and groups[1].size and np.any(np.abs(groups[0][:, None] - groups[1][None, :]) <= horizon):
        raise ValueError("Overlapping setup/control event horizons")


def evaluate_gate(signal_rows: list[dict], control_rows: list[dict],
                  trials: int | None = None, source_verified: bool = False) -> dict:
    """Exploratory screening; never label a sample-selected result as proven edge."""
    print("\n--- STAGE 01 / BRICK 1 EXPLORATORY DECISION GATE ---")
    print("Paired calendar blocks; approximate stationarity assumed; independent validation still required.")
    result = {"verdict": "INSUFFICIENT DATA", "edge_proven": False, "stats": None}
    try:
        s, c = pd.DataFrame(signal_rows), pd.DataFrame(control_rows)
        if s.empty or c.empty:
            raise ValueError("No matched setup or control events")
        require_matched_controls(s["signal_time"], c["signal_time"], min_events=MIN_EVENTS)
        require_disjoint_events(s["signal_time"], c["signal_time"])
        for frame, columns in [(s, ["net_R", "path_MFE_R", "path_MAE_R"]), (c, ["net_R"])]:
            frame[columns] = frame[columns].astype(float)
            if not np.isfinite(frame[columns].to_numpy()).all():
                raise ValueError("Non-finite event metrics")
        months = pd.to_datetime(s["signal_time"], utc=True, format="mixed").dt.strftime("%Y-%m").nunique()
        if months < 12:
            raise ValueError("At least 12 active matched months required for 6-month block sensitivity")
        stats = bootstrap_stats(s.net_R.to_numpy(), c.net_R.to_numpy(),
                                setup_times=s.signal_time, ctrl_times=c.signal_time, trials=trials)
    except (KeyError, ValueError, TypeError) as exc:
        result["reason"] = str(exc)
        print(f"GATE VERDICT: INSUFFICIENT DATA ({exc}); not evidence of hypothesis rejection")
        return result
    result["stats"] = stats
    mean_r, delta_r = float(s.net_R.mean()), stats["obs_diff"]
    p75_mfe = float(np.percentile(s.path_MFE_R, 75))
    med_mfe, med_mae_abs = float(s.path_MFE_R.median()), abs(float(s.path_MAE_R.median()))
    print(f"  Calendar-block mean CI envelope: [{stats['ci_low']:+.3f}, {stats['ci_high']:+.3f}]R")
    print(f"  Paired delta: {delta_r:+.3f}R; delta CI envelope: "
          f"[{stats['delta_ci_low']:+.3f}, {stats['delta_ci_high']:+.3f}]R")
    print(f"  Worst 1/3/6-month centered-null p: {stats['p_value']:.4f}")
    adjusted = stats["adjusted_p_value"]
    print(f"  Bonferroni p: {adjusted:.4f} over {trials} declared trials" if adjusted is not None
          else "  Multiple-testing correction UNAVAILABLE: total trial count unknown; no statistical PASS")
    checks = {
        "Mean net R > +0.15R": mean_r > MIN_MEAN_NET_R,
        "Matched delta >= +0.15R": delta_r >= MIN_DELTA_CONTROL_R,
        "Mean and paired-delta CI lower bounds > 0": stats["ci_low"] > 0 and stats["delta_ci_low"] > 0,
        "Right-tail MFE asymmetry": p75_mfe >= MIN_P75_MFE_R and med_mfe > med_mae_abs,
    }
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if not all(checks.values()):
        verdict = "SCREEN FAIL (not confirmed; do not tune to this sample)"
    elif adjusted is None or not source_verified:
        verdict = "INCONCLUSIVE (trial history unknown or funding source unverified)"
    elif adjusted > MAX_P_VALUE:
        verdict = "SCREEN FAIL (multiplicity-adjusted evidence insufficient)"
    else:
        verdict = "EXPLORATORY PASS -> INDEPENDENT VALIDATION REQUIRED"
    result.update(verdict=verdict, checks=checks, source_verified=source_verified)
    print(f"FINAL GATE VERDICT: {verdict}")
    return result


def self_test() -> None:
    """Executed native-vectorbt regression tests; synthetic, not market evidence."""
    count = 0

    def check(condition: bool, label: str) -> None:
        nonlocal count
        if not condition:
            raise AssertionError(label)
        count += 1
        print(f"  PASS {label}")

    times = pd.date_range("2023-01-01", periods=180, freq="4h", tz="UTC")
    base = pd.DataFrame({"time": times, "open": 100., "high": 101.,
                         "low": 99., "close": 100., "volume": 1., "funding_rate": 0.,
                         "funding_observed": times.hour.isin([0, 8, 16]).astype("int8"), "symbol": SYMBOL})
    base = features(validate(base))
    base.loc[:, "atr"] = 2.  # 1R = 3 USDT per BTC for explicit expectations.
    entry = 100.*(1+SLIPPAGE_BPS/10_000)
    stop, target = entry-3., entry+9.

    def run(frame: pd.DataFrame, **kwargs) -> dict:
        return evaluate_events(frame, [60], **kwargs)[0]

    both = base.copy()
    both.loc[61, ["high", "low"]] = [115., 90.]
    r = run(both)
    check(r["exit"] == "STOP" and r["entry_time"] == r["exit_time"], "entry-candle SL/TP tie: native stop first")
    check(np.isclose(r["exit_price"], stop*(1-SLIPPAGE_BPS/10_000)), "native stop fill with exit slippage")
    check(r["net_R_ex_funding"] < -1., "native fees and slippage reduce stop R below -1")
    expected_fee = (r["entry_price"]+r["exit_price"])*FEE_BPS/10_000
    check(np.isclose(r["fee_usdt_per_btc"], expected_fee), "both native order fees accounted")

    winner = base.copy()
    winner.loc[61, "high"] = 115.
    r = run(winner)
    check(r["exit"] == "TARGET" and r["exit_time"] == str(times[61]), "entry-candle target executes")
    check(np.isclose(r["exit_price"], target*(1-SLIPPAGE_BPS/10_000)), "native TP market slippage")

    gap = base.copy()
    gap.loc[62, ["open", "high", "low", "close"]] = [90., 91., 89., 90.]
    r = run(gap)
    check(r["exit"] == "STOP_GAP" and r["exit_phase"] == "OPEN", "gap through stop fills at next actual open")
    check(np.isclose(r["exit_price"], 90.*(1-SLIPPAGE_BPS/10_000)), "native adverse gap fill")
    gap.loc[62, ["open", "high", "low", "close"]] = [115., 116., 114., 115.]
    r = run(gap)
    check(r["exit"] == "TARGET" and np.isclose(r["exit_price"], 115.*(1-SLIPPAGE_BPS/10_000)), "native favorable target gap at open")

    r = run(base)
    check(r["exit"] == "TIME" and r["exit_time"] == str(times[102]), "42-bar timeout uses final bar close")
    check(r["exit_phase"] == "RANGE_CLOSE", "time exit executes in close phase")
    later_entry = base.copy()
    later_entry.loc[61, ["open", "high", "low", "close"]] = [102., 103., 101., 102.]
    r = run(later_entry, barriers=False)
    check(np.isclose(r["entry_price"], 102.*(1+SLIPPAGE_BPS/10_000)), "entry uses t+1 open, not signal close")
    r = run(both, barriers=False)
    check(r["exit"] == "TIME" and r["exit_time"] == str(times[102]), "barrier-free path uses same native engine")
    check(np.isclose(r["net_R"] + r["cost_R"], 0.), "flat barrier-free gross R equals net R plus cost R")

    funded = base.copy()
    funded.loc[[61, 62, 102, 103], "funding_rate"] = 0.001
    r = run(funded)
    check(np.isclose(r["funding_usdt_per_btc"], .2), "funding: exclude entry instant and post-exit, include held and exit instant")
    check(np.isclose(r["net_R_ex_funding"]-r["net_R"], .2/3.), "positive funding reduces long result")
    funded.loc[:, "funding_rate"] *= -1
    check(run(funded)["net_R"] > run(funded)["net_R_ex_funding"], "negative funding benefits long")
    check(len(evaluate_events(base, [60, 103])) == 2, "batch retains exactly two native closed trades")
    check(evaluate_events(base, []) == [], "empty event batch")
    check(evaluate_events(base, [60, 103]) == evaluate_events(base, [60, 103]), "deterministic native execution")

    def rejects(frame: pd.DataFrame, indices: list[int], text: str) -> None:
        try:
            evaluate_events(frame, indices)
        except ValueError:
            check(True, text)
        else:
            raise AssertionError(text)

    rejects(base, [60, 61], "overlapping events rejected")
    rejects(base, [179], "unmatured event rejected")
    bad_atr = base.copy()
    bad_atr.loc[60, "atr"] = np.nan
    rejects(bad_atr, [60], "invalid ATR rejected")
    for label, frame in [
        ("missing candle", base.drop(index=80)),
        ("missing funding", base.drop(columns="funding_rate")),
        ("NaN price", base.assign(close=base.close.mask(base.index == 80))),
        ("infinite funding", base.assign(funding_rate=np.inf)),
        ("negative volume", base.assign(volume=-1.0)),
        ("non-positive price", base.assign(open=0.0)),
        ("invalid high geometry", base.assign(high=95.0)),
        ("invalid low geometry", base.assign(low=105.0)),
        ("irregular spacing", base.assign(time=base.time + pd.Timedelta(hours=1))),
        ("insufficient history", base.iloc[:10]),
    ]:
        try:
            validate(frame)
        except ValueError:
            check(True, label+" rejected")
        else:
            raise AssertionError(label)

    t_start = str(times[0])
    t_end = str(times[-1] + pd.Timedelta(hours=4))
    readiness_test = check_data_readiness(base, Path("synthetic.csv"), start_date=t_start, end_date=t_end)
    check(readiness_test["actual_candles"] == len(base), "data readiness calculates candle counts")
    check(readiness_test["missing_candles"] == 0, "data readiness confirms zero missing candles")

    a = features(base)
    changed = base.copy()
    changed.loc[151:, ["open", "high", "low", "close"]] *= 2
    b = features(changed)
    columns = ["signal", "atr", "macro_trend", "funding_3d_bps", "squeeze_breakout"]
    pd.testing.assert_frame_equal(a.loc[:150, columns], b.loc[:150, columns])
    check(True, "future prices do not alter earlier features")
    check(select_events(a) == select_events(a), "deterministic event/control selection")
    sr, cr = np.array([2., -1., 2., 2., -1.]), np.array([-1., -1., 2., -1., -1.])
    stamps = pd.date_range("2020-01-01", periods=len(sr), freq="MS", tz="UTC")
    check(bootstrap_stats(sr, cr, 500, stamps, stamps) == bootstrap_stats(sr, cr, 500, stamps, stamps),
          "deterministic paired calendar-block bootstrap")
    print(f"SELF-TEST PASSED: {count} checks; actual vectorbt {vbt.__version__}; synthetic fixtures only")


# H1-A is a forecast experiment, not a trading strategy. All research remains
# here; sklearn is already a vectorbt dependency (now declared explicitly).
H1A_MODELS = {
    "M0": ["age"],
    "M1": ["age", "r5", "r20", "r63", "z20", "er20", "distance", "dz5", "rv5", "rv20"],
}
H1A_MODELS["M2"] = H1A_MODELS["M1"] + ["compression", "volume_surprise"]
H1A_CLASSES = np.array([0, 1, -1])


def daily_market(x: pd.DataFrame) -> pd.DataFrame:
    """Only six completed 4h bars form a UTC day; time is availability, not open."""
    g = x.assign(day=x.time.dt.floor("D")).groupby("day", sort=True)
    d = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
              close=("close", "last"), volume=("volume", "sum"),
              funding=("funding_rate", "sum"), bars=("close", "size"),
              signal_i=("close", lambda s: int(s.index[-1]))).reset_index()
    complete_days = d.loc[d.bars.eq(6), "day"]
    if complete_days.empty:
        raise ValueError("No complete UTC day can be formed from 4h data")
    calendar = pd.date_range(complete_days.iloc[0], complete_days.iloc[-1], freq="D")
    d = d.set_index("day").reindex(calendar)
    d.index.name = "day"
    d = d.reset_index()
    d["bars"] = d.bars.fillna(0).astype(int)
    incomplete = ~d.bars.eq(6)
    d.loc[incomplete, ["open", "high", "low", "close", "volume", "funding", "signal_i"]] = np.nan
    d["time"] = d.day + pd.Timedelta(days=1)
    r = np.log(d.close / d.close.shift(1))
    d["r1"] = r
    for n in (5, 20, 63):
        d[f"r{n}"] = r.rolling(n).sum()
    for n in (5, 20):
        d[f"rv{n}"] = np.sqrt(r.pow(2).rolling(n).mean())
    d["z20"] = d.r20 / np.sqrt(r.pow(2).rolling(20).sum() + 1e-12)
    d["er20"] = d.r20.abs() / (r.abs().rolling(20).sum() + 1e-12)
    d["distance"] = (1.25 - d.z20.abs()).abs()
    d["dz5"] = d.z20.diff(5)
    d["rv20_prior"] = d.rv20.shift(1)
    d["compression"] = d.rv5 / d.rv20.clip(lower=1e-12)
    d["volume_surprise"] = np.log(d.volume.clip(lower=1e-12) /
                                         d.volume.shift(1).rolling(20).median().clip(lower=1e-12))
    d["funding3"] = d.funding.rolling(3).sum()
    d["q"] = np.select([(d.z20 >= 1.25) & (d.er20 >= .35),
                         (d.z20 <= -1.25) & (d.er20 >= .35)], [1., -1.], default=0.)
    d.loc[d.z20.isna(), "q"] = np.nan
    d["crossing"] = d.q.ne(0) & d.q.notna() & d.q.shift(1).eq(0)
    age, run = [], 0
    for q in d.q:
        run = run + 1 if q == 0 else 0
        age.append(run)
    d["neutral_days"] = age
    d["age"] = np.log1p(d.neutral_days)
    d["risk"] = d.q.eq(0) & d.neutral_days.ge(10)
    prev = d.close.shift(1)
    d["atr"] = pd.concat([d.high-d.low, (d.high-prev).abs(), (d.low-prev).abs()], axis=1).max(axis=1).rolling(14).mean()
    return d


def onset_labels(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Future confirmation changes labels only; censored tails are never negatives."""
    d = d.copy()
    records = []
    q = d.q.to_numpy()
    for i in range(1, len(d)):
        if bool(d.risk.iloc[i-1]) and np.isfinite(q[i]) and q[i] != 0:
            mature = i + 10 < len(d)
            confirmed = bool(np.sum(q[i+1:i+11] == q[i]) >= 8) if mature else None
            records.append({"onset_i": i, "onset_time": d.time.iloc[i], "side": int(q[i]),
                            "confirmation_time": d.time.iloc[i+10] if mature else pd.NaT,
                            "confirmed": confirmed, "status": "CONFIRMED" if confirmed else
                            ("UNCONFIRMED" if mature else "CENSORED")})
    events = pd.DataFrame(records, columns=["onset_i", "onset_time", "side", "confirmation_time", "confirmed", "status"])
    d["y"] = np.nan
    d["target_onset"] = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns, UTC]")
    d["target_confirmation"] = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns, UTC]")
    confirmed = [r for r in records if r["confirmed"]]
    for i in np.flatnonzero(d.risk.to_numpy()):
        if i + 17 >= len(d):
            continue
        d.loc[i, "y"] = 0
        future = [r for r in confirmed if i < r["onset_i"] <= i+7]
        if future:
            e = future[0]
            d.loc[i, ["y", "target_onset", "target_confirmation"]] = [e["side"], e["onset_time"], e["confirmation_time"]]
    return d, events


def forecast_prob(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], c: float) -> np.ndarray:
    """Scaler and ridge see training only; absent classes trigger explicit prior fallback."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = train.y.to_numpy(int)
    if len(np.unique(y)) < 3:
        prior = (np.array([(y == k).sum() for k in H1A_CLASSES]) + 1) / (len(y)+3)
        return np.tile(prior, (len(test), 1))
    from threadpoolctl import threadpool_limits
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=100, solver="newton-cholesky"))
    with threadpool_limits(limits=1):
        started = time.perf_counter()
        model.fit(train[cols], y)
        _perf_time("h1a.training_seconds", time.perf_counter()-started)
        started = time.perf_counter()
        p = model.predict_proba(test[cols])
        _perf_time("h1a.inference_seconds", time.perf_counter()-started)
    return p[:, [list(model[-1].classes_).index(k) for k in H1A_CLASSES]]


def forecast_prob_grid(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], c_values: tuple[float,...]) -> dict[float,np.ndarray]:
    """Share train-only scaling and warm-start the fixed C grid within each fold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    y=train.y.to_numpy(int)
    if len(np.unique(y))<3:
        prior=(np.array([(y==k).sum() for k in H1A_CLASSES])+1)/(len(y)+3)
        return {c:np.tile(prior,(len(test),1)) for c in c_values}
    started = time.perf_counter()
    scaler=StandardScaler().fit(train[cols])
    xt=scaler.transform(train[cols]); xv=scaler.transform(test[cols])
    _perf_time("h1a.training_seconds", time.perf_counter()-started)
    model=LogisticRegression(C=c_values[0],max_iter=100,solver="newton-cholesky",warm_start=True)
    result={}
    for c in c_values:
        started = time.perf_counter()
        model.set_params(C=c).fit(xt,y)
        _perf_time("h1a.training_seconds", time.perf_counter()-started)
        started = time.perf_counter()
        predicted=model.predict_proba(xv)
        _perf_time("h1a.inference_seconds", time.perf_counter()-started)
        result[c]=predicted[:,[list(model.classes_).index(k) for k in H1A_CLASSES]]
    return result


def temperature(p: np.ndarray, t: float) -> np.ndarray:
    p = np.maximum(p, 1e-12) ** (1/t)
    return p / p.sum(axis=1, keepdims=True)


def forecast_losses(y, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    onehot = np.asarray(y)[:, None] == H1A_CLASSES[None, :]
    return ((p-onehot)**2).sum(axis=1), -(onehot*np.log(np.maximum(p, 1e-12))).sum(axis=1)


def alarm_metrics(rows: pd.DataFrame, p: np.ndarray, threshold, events: pd.DataFrame,
                  years: float) -> dict:
    """One alarm / 7 days; strict signed pre-onset matching, deduplicated events."""
    _started = time.perf_counter()
    detected, leads, alarms = set(), [], []
    # Confirmed onsets are chronological in onset_labels(). Index them once per
    # alarm scan rather than creating a full events DataFrame mask per candidate.
    confirmed = events.loc[events.status.eq("CONFIRMED")]
    by_side = {}
    for event_side in (-1, 1):
        subset = confirmed.loc[confirmed.side.eq(event_side)]
        if subset.onset_time.is_monotonic_increasing and subset.onset_time.notna().all():
            by_side[event_side] = (pd.DatetimeIndex(subset.onset_time).as_unit("ns").asi8,
                                   subset.onset_i.to_numpy(dtype=int), subset.onset_time.tolist())
        else:
            by_side[event_side] = None  # Preserve original order for unusual inputs.
    last = None
    for j, row in enumerate(rows.itertuples()):
        limit = float(threshold[j]) if np.ndim(threshold) else float(threshold)
        sign_i = 1 if p[j, 1] >= p[j, 2] else 2
        if p[j, sign_i] < limit or (last is not None and row.time < last + pd.Timedelta(days=7)):
            continue
        last = row.time
        sign = int(H1A_CLASSES[sign_i])
        indexed = by_side[sign]
        if indexed is None:
            matching = confirmed.loc[confirmed.side.eq(sign) &
                                     confirmed.onset_time.gt(row.time) &
                                     confirmed.onset_time.le(row.time + pd.Timedelta(days=7))]
            key = None if matching.empty else int(matching.iloc[0].onset_i)
            onset_time = None if matching.empty else matching.iloc[0].onset_time
        else:
            times, onset_ids, onset_times = indexed
            next_i = int(np.searchsorted(times, row.time.value, side="right"))
            matched = next_i < len(times) and times[next_i] <= (row.time + pd.Timedelta(days=7)).value
            key = int(onset_ids[next_i]) if matched else None
            onset_time = onset_times[next_i] if matched else None
        lead_days = None
        if key is not None and key not in detected:
            detected.add(key)
            lead_days = int((onset_time-row.time).days)
            leads.append(lead_days)
        alarms.append({"time": str(row.time), "side": sign, "matched_onset_i": key, "lead_days": lead_days})
    false = sum(a["matched_onset_i"] is None for a in alarms)
    _perf_time("h1a.alarm_metrics_seconds", time.perf_counter() - _started)
    return {"alarms": len(alarms), "detected_events": len(detected), "false_alarms": false,
            "false_per_asset_year": false/max(years, 1e-12),
            "precision": (len(alarms)-false)/len(alarms) if alarms else None,
            "median_lead_days": float(np.median(leads)) if leads else None,
            "alarm_archive": alarms}


@lru_cache(maxsize=32)
def _calendar_block_starts(n: int) -> tuple[np.ndarray,...]:
    rng=np.random.default_rng(SEED); draws=[]
    for requested in (28,56,84):
        length=min(requested,n)
        draws.append(rng.integers(0,n,(1000,int(np.ceil(n/length)))))
    return tuple(draws)


def _calendar_block_interval(values: np.ndarray, *, ratio: bool = False) -> dict:
    """Circular calendar-block CI from block prefix sums; avoids materializing 1000xN indices."""
    _started = time.perf_counter()
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=0)
    estimate = 1-mean[1]/mean[0] if ratio else mean[0]
    n, details = len(values), []
    finite = np.isfinite(values)
    doubled_sum = np.nan_to_num(np.concatenate((values, values)), nan=0.0)
    doubled_count = np.concatenate((finite, finite)).astype(float)
    sum_prefix = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(doubled_sum, axis=0)))
    count_prefix = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(doubled_count, axis=0)))
    starts_by_block = _calendar_block_starts(n)
    for block_i, requested in enumerate((28,56,84)):
        length = min(requested, n)
        nblocks = int(np.ceil(n/length))
        starts = starts_by_block[block_i]
        lengths = np.full(nblocks, length)
        lengths[-1] = n-(nblocks-1)*length
        ends = starts+lengths
        sums = (sum_prefix[ends]-sum_prefix[starts]).sum(axis=1)
        counts = (count_prefix[ends]-count_prefix[starts]).sum(axis=1)
        means = np.divide(sums, counts, out=np.full(sums.shape, np.nan), where=counts>0)
        stat = 1-means[:,1]/means[:,0] if ratio else means[:,0]
        stat = stat[np.isfinite(stat)]
        ci = np.quantile(stat, [.025,.975]).tolist() if len(stat)>=900 else [None,None]
        details.append({"block_days":length,"ci95":ci,"nonempty_draws":len(stat)})
    valid = [row["ci95"] for row in details if row["ci95"][0] is not None]
    ci = [min(row[0] for row in valid),max(row[1] for row in valid)] if len(valid)==3 else [None,None]
    _perf_time("statistics.calendar_block_seconds", time.perf_counter() - _started)
    return {"estimate":float(estimate),"ci95":ci,"sensitivity":details}


def block_interval(frame: pd.DataFrame, columns: list[str], *, ratio: bool = False) -> dict:
    """Paired daily calendar blocks, including ineligible days, NOT IID forecast rows."""
    if frame.empty:
        return {"estimate": None, "ci95": [None, None], "sensitivity": []}
    daily = frame.set_index("time")[columns].astype(float)
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
    return _calendar_block_interval(daily.to_numpy(), ratio=ratio)


def h1a_gate(up: int, down: int, clusters: int, gain: dict, alarms: dict, *, core5: bool = False) -> dict:
    """Evidence, incremental forecast score and held-out warning utility are separate."""
    adequate = up >= 20 and down >= 20 and clusters >= 12
    score_pass = gain["estimate"] >= .05 and gain["ci95"][0] is not None and gain["ci95"][0] > 0
    utility_pass = (alarms["detected_events"] > 0 and alarms["false_per_asset_year"] <= 3 and
                    alarms["median_lead_days"] is not None and alarms["median_lead_days"] >= 2)
    if not adequate:
        verdict = "INCONCLUSIVE / low event power; core-5 exploratory" if core5 else "INCONCLUSIVE / low event power; BTC-only feasibility"
    elif not score_pass:
        verdict = "REJECT_INCREMENTAL_H1A"
    elif not utility_pass:
        verdict = "REJECT_EARLY_WARNING_UTILITY"
    else:
        verdict = "INCONCLUSIVE / requires independent validation" if core5 else "INCONCLUSIVE / requires ETH and independent validation"
    return {"verdict": verdict, "minimum_evidence_met": adequate,
            "forecast_score_gate": bool(score_pass), "heldout_utility_gate": bool(utility_pass)}


def h1a_experiment(d: pd.DataFrame, events: pd.DataFrame, *, fast: bool = False,
                   evaluation_start: str = "2023-01-01",
                   inner_start: str = "2021-07-01") -> tuple[pd.DataFrame, dict]:
    """Three-year expanding train; 6mo tests; fast mode fixes C=1 and nests only alert selection."""
    from sklearn.metrics import average_precision_score
    usable = d.risk & np.isfinite(d[H1A_MODELS["M2"]]).all(axis=1)
    allrows = d.loc[usable].copy()
    # Historical lag placebo, never a circular roll bringing future values back.
    for col in ("compression", "volume_surprise"):
        allrows[f"lag_{col}"] = d[col].shift(60).reindex(allrows.index).fillna(1. if col == "compression" else 0.)
    output, folds, exposure = [], [], []
    mature_calendar = d.loc[d.time + pd.Timedelta(days=17) <= d.time.max(), "time"]
    score_end = mature_calendar.max() + pd.Timedelta(days=1)

    def exposure_range(start, end, cap):
        stop = min(end, cap) if pd.notna(cap) else start
        days = max(0, (stop-start).days)
        return {"start": str(start), "end_exclusive": str(max(start, stop)), "days": days}
    for start in pd.date_range(evaluation_start, DEV_END, freq="6MS", inclusive="left", tz="UTC"):
        end = start + pd.DateOffset(months=6)
        train = allrows.loc[allrows.y.notna() & (allrows.time + pd.Timedelta(days=17) < start)]
        test = allrows.loc[(allrows.time >= start) & (allrows.time < end)]
        if len(train) < 30 or test.empty:
            folds.append({"start": str(start), "status": "INSUFFICIENT_ROWS"})
            continue
        fold_exposure = exposure_range(start, end, score_end)
        exposure.append(fold_exposure)
        train_calendar = mature_calendar.loc[mature_calendar + pd.Timedelta(days=17) < start]
        inner_cap = train_calendar.max() + pd.Timedelta(days=1)
        record = {"start": str(start), "end": str(end),
                  "selection_mode":"fixed_C1_inner_M2_alert_only" if fast else "nested_C_temperature_alert",
                  "train_rows": len(train), "test_rows": len(test),
                  "scored_rows": int(test.y.notna().sum()), "censored_rows": int(test.y.isna().sum()),
                  "exposure": fold_exposure,
                  "train_latest_origin": str(train.time.max()), "test_first_origin": str(test.time.min()),
                  "prior_fallback": len(train.y.unique()) < 3,
                  "train_class_counts": {str(k): int(train.y.eq(k).sum()) for k in H1A_CLASSES}, "models": {}}
        pred = test[["time", "y", "target_onset", "target_confirmation", "z20", "er20", "distance"]].copy()
        pred["fold"] = str(start.date())
        pred["label_maturity_time"] = pred.time + pd.Timedelta(days=17)
        inner_folds, inner_exposure = [], []
        for inner_date in pd.date_range(pd.Timestamp(inner_start, tz="UTC"), start, freq="6MS", inclusive="left"):
            tr = train.loc[train.time + pd.Timedelta(days=17) < inner_date]
            va = train.loc[(train.time >= inner_date) & (train.time < inner_date + pd.DateOffset(months=6))]
            if len(tr) >= 30 and not va.empty:
                inner_folds.append((tr, va))
                inner_exposure.append(exposure_range(inner_date, inner_date + pd.DateOffset(months=6), inner_cap))
        record["inner_exposure"] = inner_exposure
        record["inner_exposure_days"] = sum(r["days"] for r in inner_exposure)
        fast_inner=[(va,forecast_prob(tr,va,H1A_MODELS["M2"],1.)) for tr,va in inner_folds] if fast else []
        for name, cols in H1A_MODELS.items():
            if fast:
                c,t=1.,1.
                inner=fast_inner if name=="M2" else []
                threshold=1.01
                if inner:
                    ir=pd.concat([va for va,_ in inner]).sort_values("time")
                    ip=np.concatenate([p for _,p in inner])
                    years=record["inner_exposure_days"]/365.25
                    options=[]
                    for candidate in (.1,.2,.3,.4,.5,.7,.9):
                        alarms=alarm_metrics(ir,ip,candidate,events,years)
                        if alarms["false_per_asset_year"]<=3 and alarms["median_lead_days"] is not None and alarms["median_lead_days"]>=2:
                            options.append((alarms["detected_events"],candidate))
                    threshold=max(options)[1] if options else 1.01
            else:
                c_values=(.1,1.,10.)
                inner_by_c={value:[] for value in c_values}
                for tr,va in inner_folds:
                    for value,prediction in forecast_prob_grid(tr,va,cols,c_values).items():
                        inner_by_c[value].append((va,prediction))
                candidates=[]
                for value,inner_rows in inner_by_c.items():
                    loss=np.mean(np.concatenate([forecast_losses(va.y,prediction)[0] for va,prediction in inner_rows])) if inner_rows else np.inf
                    candidates.append((loss,value,inner_rows))
                _,c,inner=min(candidates,key=lambda candidate:(candidate[0],candidate[1]))
                if inner:
                    ir=pd.concat([va for va,_ in inner]).sort_values("time")
                    ip=np.concatenate([prediction for _,prediction in inner])
                    t=min((.75,1.,1.5,2.),key=lambda value:np.mean(forecast_losses(ir.y,temperature(ip,value))[1]))
                    ip=temperature(ip,t)
                    years=record["inner_exposure_days"]/365.25
                    options=[]
                    for threshold in (.1,.2,.3,.4,.5,.7,.9):
                        alarms=alarm_metrics(ir,ip,threshold,events,years)
                        if alarms["false_per_asset_year"]<=3 and alarms["median_lead_days"] is not None and alarms["median_lead_days"]>=2:
                            options.append((alarms["detected_events"],threshold))
                    threshold=max(options)[1] if options else 1.01
                else:
                    t,threshold=1.,1.01
            p = temperature(forecast_prob(train, test, cols, c), t)
            for j, label in enumerate(("none", "up", "down")):
                pred[f"{name}_{label}"] = p[:, j]
            pred[f"{name}_threshold"] = threshold
            mature = test.y.notna().to_numpy()
            bs, _ = forecast_losses(test.y[mature], p[mature])
            record["models"][name] = {"C": c, "temperature": t, "alert_threshold": threshold,
                                      "inner_folds": len(inner), "brier": float(bs.mean()) if len(bs) else None}
            if name == "M2":
                pc = H1A_MODELS["M1"] + ["lag_compression", "lag_volume_surprise"]
                pp = temperature(forecast_prob(train, test, pc, c), t)
                for j, label in enumerate(("none", "up", "down")):
                    pred[f"placebo_{label}"] = pp[:, j]
        output.append(pred)
        folds.append(record)
    if not output:
        return pd.DataFrame(), {"verdict": "INCONCLUSIVE / no usable folds", "folds": folds,
                                "N_forecasts": 0, "N_scored": 0, "N_censored": 0,
                                "exposure_ranges": exposure, "exposure_days": 0}
    forecasts = pd.concat(output, ignore_index=True).sort_values("time").reset_index(drop=True)
    scored = forecasts.loc[forecasts.y.notna()].copy()
    exposure_days = sum(r["days"] for r in exposure)
    counts = {"N_forecasts": len(forecasts), "N_scored": len(scored),
              "N_censored": int(forecasts.y.isna().sum()),
              "exposure_ranges": exposure, "exposure_days": exposure_days}
    if scored.empty:
        return forecasts, {"verdict": "INCONCLUSIVE / no mature forecasts", "folds": folds, **counts}
    losses, metrics = scored[["time"]].copy(), {}
    calendar_years = exposure_days/365.25
    scoreable = events.loc[events.status.eq("CONFIRMED") & events.onset_time.apply(
        lambda t: bool(((scored.time < t) & (scored.time >= t-pd.Timedelta(days=7))).any()))]
    for name in (*H1A_MODELS, "placebo"):
        p = scored[[f"{name}_{k}" for k in ("none", "up", "down")]].to_numpy()
        bs, ll = forecast_losses(scored.y, p)
        losses[name] = bs
        event_p, actual = p[:, 1:].sum(axis=1), scored.y.ne(0).to_numpy()
        bins = []
        for lo, hi in ((0., .1), (.1, .25), (.25, .5), (.5, 1.000001)):
            mask = (event_p >= lo) & (event_p < hi)
            bins.append({"range": [lo, hi], "N": int(mask.sum()),
                         "predicted": float(event_p[mask].mean()) if mask.any() else None,
                         "observed": float(actual[mask].mean()) if mask.any() else None})
        metrics[name] = {"brier": float(bs.mean()), "log_loss": float(ll.mean()),
                         "event_pr_auc_AP": float(average_precision_score(actual, event_p)) if actual.any() else None,
                         "calibration_bins": bins}
        if name != "placebo":
            metrics[name]["alarms"] = alarm_metrics(scored, p, scored[f"{name}_threshold"].to_numpy(), events, calendar_years)
            metrics[name]["alarms"]["missed_scoreable_events"] = len(scoreable)-metrics[name]["alarms"]["detected_events"]
    gain = block_interval(losses, ["M1", "M2"], ratio=True)
    dates = scoreable.onset_time.sort_values()
    clusters, last = 0, None
    for t in dates:
        if last is None or (t-last).days > 28:
            clusters += 1
        last = t
    up, down = int(scoreable.side.eq(1).sum()), int(scoreable.side.eq(-1).sum())
    gate = h1a_gate(up, down, clusters, gain, metrics["M2"]["alarms"],core5=fast)
    proximity = []
    for label, mask in (("near_Z_boundary", scored.distance < .25), ("far_Z_boundary", scored.distance >= .25)):
        if mask.any():
            proximity.append({"group": label, "N": int(mask.sum()), "event_rate": float(scored.y[mask].ne(0).mean()),
                              "relative_gain": float(1-losses.M2[mask].mean()/losses.M1[mask].mean())})
    return forecasts, {**gate, "economic_simulation": "NOT RUN: full forecast/evidence/validation gates not passed",
                       "folds": folds, **counts, "event_row_prevalence": float(scored.y.ne(0).mean()),
                       "scoreable_onsets_up": up, "scoreable_onsets_down": down, "calendar_clusters_pm14d": clusters,
                       "metrics": metrics, "relative_brier_gain": gain,
                       "proximity_diagnostic": proximity, "independent_validation": False}


def mark_to_market_returns(x: pd.DataFrame, indices: list[int], side: int, hold_bars: int,
                           prepared: tuple[np.ndarray,np.ndarray,np.ndarray] | None = None) -> tuple[np.ndarray,np.ndarray]:
    """Vectorized open-to-close mark-to-market with the native engine's fees, slip and funding timing."""
    open_price,close,prefix=prepared if prepared is not None else (
        x.open.to_numpy(),x.close.to_numpy(),np.r_[0.0,np.cumsum((x.open*x.funding_rate).to_numpy())])
    ids=np.asarray(indices,dtype=int); entry=ids+1; exit_=entry+hold_bars-1
    raw_entry=open_price[entry]; raw_exit=close[exit_]
    fill_entry=raw_entry*(1+side*SLIPPAGE_BPS/10_000)
    fill_exit=raw_exit*(1-side*SLIPPAGE_BPS/10_000)
    fee=FEE_BPS/10_000*(fill_entry+fill_exit)
    funding=side*(prefix[exit_+1]-prefix[entry+1])
    return side*(raw_exit/raw_entry-1),(side*(fill_exit-fill_entry)-fee-funding)/raw_entry


def market_observations(x: pd.DataFrame, d: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Independent descriptive study. Daily decisions, native 4h execution, no H1-A trades."""
    _started = time.perf_counter()
    valid = d[H1A_MODELS["M2"] + ["atr", "funding3"]].notna().all(axis=1)
    # vectorbt's 42 bars represent seven days only on an uninterrupted 4h path.
    bad_step = x.time.diff().ne(pd.Timedelta(hours=4)).to_numpy(copy=True)
    bad_step[0] = False
    gap_prefix = np.r_[0, np.cumsum(bad_step)]
    continuous = np.zeros(len(d), dtype=bool)
    for row, value in enumerate(d.signal_i):
        if pd.notna(value):
            i = int(value)
            continuous[row] = i+HOLD_BARS < len(x) and gap_prefix[i+HOLD_BARS+1]-gap_prefix[i+1] == 0
    ds = d.loc[valid.to_numpy() & continuous].copy()
    # Daily ATR risk is frozen at the decision close, but fills/funding stay 4h.
    execution=x.copy(deep=False)
    execution["atr"]=x.atr.copy()
    execution.loc[ds.signal_i.to_numpy(int),"atr"]=ds.atr.to_numpy()
    open_price=x.open.to_numpy(); close=x.close.to_numpy()
    funding_prefix=np.r_[0.0,np.cumsum((x.open*x.funding_rate).to_numpy())]
    prepared=(open_price,close,funding_prefix)
    observations = []
    _perf_time("market_observations.preparation_seconds", time.perf_counter() - _started)
    for side in (1, -1):
        a = ds.copy()
        a["side"] = side
        a["momentum_aligned"] = np.sign(a.r5).eq(side)
        a["phase"] = (a.time.array.as_unit("ns").asi8 // pd.Timedelta(days=1).value) % 8
        q_values=d.q.to_numpy(); signal_idx=a.signal_i.to_numpy(dtype=int); row_ids=a.index.to_numpy()
        for h in (1,3,7):
            gross,net=mark_to_market_returns(x,signal_idx.tolist(),side,6*h,prepared)
            a.loc[row_ids,f"regime_survives_{h}d"]=np.array([
                np.all(q_values[i+1:i+h+1]==side) for i in row_ids],dtype=float)
            a.loc[row_ids,f"return_{h}d"]=gross
            a.loc[row_ids,f"net_return_{h}d"]=net
        _vbt_stage_started = time.perf_counter()
        matrix=evaluate_event_matrix(execution,signal_idx.tolist(),(3.,5.),side=side,hold_bars=HOLD_BARS)
        _perf_time("market_observations.matrix_seconds", time.perf_counter() - _vbt_stage_started)
        by_event={(row["signal_i"],int(row["target_r"])):row for row in matrix}
        for target in (3,5):
            results=[by_event[(i,target)] for i in signal_idx]
            a.loc[row_ids,f"net_R_{target}"]=[row["net_R"] for row in results]
            a.loc[row_ids,f"ex_funding_R_{target}"]=[row["net_R_ex_funding"] for row in results]
            a.loc[row_ids,f"gross_barrier_hit_{target}"]=[row["target_before_stop"] for row in results]
            a.loc[row_ids,f"exit_{target}"]=[row["exit"] for row in results]
            extra=np.array([12/10_000*(row["entry_price"]+row["exit_price"])/2/row["risk_usdt"] for row in results])
            a.loc[row_ids,f"stress_R_{target}"]=[row["net_R"] for row in results]-extra
        observations.append(a)
    obs = pd.concat(observations, ignore_index=True)
    aligned = obs.loc[obs.momentum_aligned].copy()
    conditions = {
        "all_momentum": np.ones(len(aligned), bool),
        "established_trend": aligned.q.eq(aligned.side),
        "neutral": aligned.q.eq(0),
        "new_threshold_crossing": aligned.crossing & aligned.q.eq(aligned.side),
        "compression": aligned.compression < .8,
        "expansion_today": aligned.r1.abs() > 2*aligned.rv20_prior,
        "high_volume": aligned.volume_surprise > np.log(1.5),
        "reversal_today": np.sign(aligned.r1).ne(aligned.side) & (aligned.r1.abs() > aligned.rv20),
        "funding_aligned_gt10bp3d": aligned.side*aligned.funding3 > .001,
        "funding_opposes_momentum": aligned.side*aligned.funding3 < 0,
    }
    summaries = []
    _summary_started = time.perf_counter()
    for label, mask in conditions.items():
        group = aligned.loc[mask].sort_values("time")
        result = {"condition": label, "N_overlapping_days": len(group), "horizons": {}, "barriers": {}}
        result["status"] = "INSUFFICIENT_DATA" if group.empty else "AVAILABLE"
        for h in (1, 3, 7):
            col = f"return_{h}d"
            values = group[col]
            result["horizons"][str(h)] = {"continuation": float((values > 0).mean()) if len(group) else None,
                "q_state_survival": float(group[f"regime_survives_{h}d"].mean()) if len(group) else None,
                "return_quantiles": values.quantile([.1, .5, .9]).tolist() if len(group) else [None, None, None],
                "mean_gross": float(values.mean()) if len(group) else None,
                "mean_net": float(group[f"net_return_{h}d"].mean()) if len(group) else None,
                "continuation_uncertainty": block_interval(group[["time", col]].assign(**{col: (values > 0).astype(float)}), [col]),
                "net_uncertainty": block_interval(group, [f"net_return_{h}d"])}
        for target in (3, 5):
            col = f"net_R_{target}"
            phases = group.groupby("phase")[col].mean()
            result["barriers"][str(target)] = {"exit_policy": f"gross_{target}R_target_or_1R_stop_or_7d_timeout",
                "gross_barrier_hit": float(group[f"gross_barrier_hit_{target}"].mean()) if len(group) else None,
                "net_target_attainment": float(group[col].ge(target).mean()) if len(group) else None,
                "stress_net_target_attainment": float(group[f"stress_R_{target}"].ge(target).mean()) if len(group) else None,
                "net_positive_rate": float(group[col].gt(0).mean()) if len(group) else None,
                "net_R_quantiles": group[col].quantile([.1, .5, .9]).tolist() if len(group) else [None, None, None],
                "net_R": float(group[col].mean()) if len(group) else None,
                "ex_funding_R": float(group[f"ex_funding_R_{target}"].mean()) if len(group) else None,
                "stress_net_R": float(group[f"stress_R_{target}"].mean()) if len(group) else None, "mean_ci": block_interval(group, [col]),
                "phase0_N": int(group.phase.eq(0).sum()),
                "phase0_net_R": float(group.loc[group.phase.eq(0), col].mean()) if group.phase.eq(0).any() else None,
                "phase_mean_range": [float(phases.min()), float(phases.max())] if len(group) else [None, None],
                "by_year": {str(y): {"N": len(g), "net_R": float(g[col].mean())} for y, g in group.groupby(group.time.dt.year)}}
        summaries.append(result)
    _perf_time("market_observations.summaries_seconds", time.perf_counter() - _summary_started)
    _perf_time("market_observations.total_seconds", time.perf_counter() - _started)
    return obs, summaries


def research_run(x: pd.DataFrame, readiness: dict, args) -> None:
    """Freeze the protocol before any labels/scores; never extend the input window."""
    import time

    import sklearn
    started = time.monotonic()
    specification = {
        "study_id": "ONEFILE-H1A-BTC-FEASIBILITY-01", "window": [DEV_START, DEV_END],
        "sealed": ["ONEFILE >=2025-01-01", "monograph G3 >=2026-04-01"],
        "deviations": ["BTC only, not core-5", "4 outer folds, not preferred >=5", "existing 4h source aggregated to complete 1D; no 1h Parquet"],
        "target": "risk: >=10 consecutive q=0; Z20 log-return score +/-1.25, log-return ER20>=0.35; onset confirmed 8 of NEXT 10 days; first confirmed onset next 7d",
        "models": H1A_MODELS, "train": "first 3y; expanding; 6mo outer; strict origin+17d<fold start",
        "inner": "6mo forward folds from 2021-07; C=.1/1/10 by Brier; temperature=.75/1/1.5/2 by inner logloss",
        "alerts": "signed max p; thresholds .1/.2/.3/.4/.5/.7/.9; maximize inner hits with <=3 false/year, median lead>=2; else disabled 1.01; global 7d cooldown",
        "primary": "1-BS(M2)/BS(M1); >=5% and 95% calendar-block lower>0",
        "uncertainty": "1000 paired circular calendar blocks each 28/56/84d; widest CI; empty days retained; seed42",
        "heldout_utility": "M2 outer alarms: detected_events>0, false alarms/asset-year<=3, median lead>=2d; separate from score gate; low event power remains inconclusive",
        "evidence_floor": {"up": 20, "down": 20, "pm14d_clusters": 12},
        "placebo": "60d historical lag of M2 extras; same M2 C/temperature, diagnostic only",
        "descriptive": "ten overlapping conditions; 1/3/7d; sign of past5d; 3R/5R, 7d max; 1R=1.5 daily ATR14; native4h vectorbt; all eight 8d phases; NO ranking/significance/edge selection",
        "conditions": ["all momentum5d", "q aligned", "q neutral", "new q crossing aligned", "RV5/RV20<.8", "|r1|>2*previous RV20", "volume/prior20dMedian>1.5", "r1 opposite r5 and |r1|>RV20", "side*funding3d>.001", "side*funding3d<0"],
        "descriptive_attainment": "gross barrier hits separate from realized net_R>=3/5 and stressed net_R>=3/5 under each gross target/1R stop/7d timeout exit policy; empty groups retain N=0 with null statistics",
        "costs": {"fee_bps_side": FEE_BPS, "slip_bps_side": SLIPPAGE_BPS, "stress_extra_rt_bps": 12, "funding": "historical8h, open-price proxy"},
        "trial_history": "unknown; this is historical development, NOT independent OOS; no positive edge certification",
        "h1a_economics": "forbidden unless full forecast/evidence gates pass; no live trading",
    }
    identity = {"specification": specification, "input_sha256": readiness["file_sha256"],
                "code_sha256": readiness["code_sha256"]}
    protocol = Path(args.protocol)
    if protocol.exists():
        frozen = json.loads(protocol.read_text())
        if frozen["identity"] != identity:
            raise ValueError("Frozen protocol/code/data mismatch; retain old attempt and use a new --protocol path")
    else:
        frozen = {"frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "identity": identity,
                  "note": "Local pre-score freeze, not an external/independent preregistration or git commit."}
        atomic_write(protocol, lambda f: f.write(json.dumps(frozen, indent=2)+"\n"))
    daily = daily_market(x)
    checks = 0
    for cutoff in pd.date_range("2023-01-01", DEV_END, freq="6MS", inclusive="left", tz="UTC"):
        earlier = daily.time <= cutoff
        for col in ("open", "high", "low", "close", "volume", "funding_rate"):
            mutant = x.copy()
            mutant.loc[mutant.time >= cutoff, col] = mutant.loc[mutant.time >= cutoff, col]*3+1
            changed = daily_market(mutant)
            pd.testing.assert_frame_equal(daily.loc[earlier, H1A_MODELS["M2"]+["q", "risk"]],
                                          changed.loc[earlier, H1A_MODELS["M2"]+["q", "risk"]])
            checks += 1
    d, events = onset_labels(daily)
    forecasts, h1a = h1a_experiment(d, events)
    obs, summaries = market_observations(x, d)
    evidence = {"protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(), "readiness": readiness,
                "h1a": h1a, "market_behavior": summaries, "daily_days": len(d), "risk_days": int(d.risk.sum()),
                "causality_audit": {"future_corruption_checks": checks, "status": "PASS"},
                "onset_status_counts": events.status.value_counts().to_dict(),
                "daily_q_counts": d.q.value_counts().to_dict(),
                "volatility_persistence_spearman": d.rv20.corr(np.sqrt(d.r1.pow(2).shift(-1).rolling(7).mean().shift(-6)), method="spearman"),
                "runtime_seconds": time.monotonic()-started, "engine": f"vectorbt {vbt.__version__}",
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
                "outputs": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [args.protocol]},
                "limits": {"causal_claim": False, "edge_proven": False, "sealed_data_used": False},
                "caveats": ["Known historical DEV; unknown prior trial debt", "Overlapping days/conditions are NOT independent N", "Funding settlement mark proxied by 4h open; entry-instant funding omitted", "1BTC unlimited cash, no account/margin/liquidation", "No causal participant identification from OHLCV/funding", "No H1-A economic simulation or sealed data access"]}
    for path, frame in ((args.out, obs), (args.forecasts, forecasts), (args.onsets, events)):
        atomic_write(Path(path), lambda f, frame=frame: frame.to_csv(f, index=False))
    if not np.isfinite(evidence["volatility_persistence_spearman"]):
        evidence["volatility_persistence_spearman"] = None
    evidence["outputs"] = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                           for p in [args.protocol, args.out, args.forecasts, args.onsets]}
    atomic_write(Path(args.evidence), lambda f: f.write(json.dumps(evidence, indent=2, allow_nan=False)+"\n"))
    print(json.dumps({"H1A": h1a["verdict"], "gain": h1a.get("relative_brier_gain"),
                      "onsets": evidence["onset_status_counts"], "seconds": evidence["runtime_seconds"]}, indent=2))
    print(f"Saved {args.out}, {args.forecasts}, {args.onsets}, {args.evidence}; protocol {args.protocol}")



def continuation_volatility_experiment(d: pd.DataFrame) -> dict:
    """Independent H1-B continuation and H1-C realized-volatility forecasting."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    d = d.copy().reset_index(drop=True)
    daily_return = d.r1
    d["future_rv5"] = sum(daily_return.shift(-h).pow(2) for h in range(1, 6))
    d["future_rv10"] = sum(daily_return.shift(-h).pow(2) for h in range(1, 11))
    d["future_signed_5"] = d.q * np.log(d.close.shift(-5)/d.close)
    features = ["r1", "r5", "r20", "z20", "er20", "rv5", "rv20", "compression", "volume_surprise", "funding3"]
    folds, all_events = [], []
    dates = d.time
    for start in pd.date_range("2023-01-01", DEV_END, freq="6MS", inclusive="left", tz="UTC"):
        end = start + pd.DateOffset(months=6)
        test_mask = (dates >= start) & (dates < end)
        train_mask = dates < start-pd.Timedelta(days=10)
        valid = d[features].notna().all(axis=1).to_numpy()
        train_idx = np.flatnonzero(train_mask.to_numpy() & valid & d.future_rv10.notna().to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy() & valid & d.future_rv10.notna().to_numpy())
        if len(train_idx) < 300 or not len(test_idx):
            continue
        scaler = StandardScaler().fit(d.loc[train_idx, features])
        xt, xv = scaler.transform(d.loc[train_idx, features]), scaler.transform(d.loc[test_idx, features])
        models = {}
        for horizon in (5, 10):
            y_train = np.log(d[f"future_rv{horizon}"].iloc[train_idx].to_numpy()+1e-12)
            actual_log = np.log(d[f"future_rv{horizon}"].iloc[test_idx].to_numpy()+1e-12)
            baseline_log = np.log(d.rv20.iloc[test_idx].to_numpy().clip(1e-12)**2*horizon+1e-12)
            predicted_log = Ridge(alpha=10.).fit(xt, y_train).predict(xv)
            actual, base, pred = np.exp(actual_log), np.exp(baseline_log), np.exp(predicted_log)
            qlike = lambda y, p: np.mean(np.log(p)+y/p)
            base_mse, pred_mse = np.mean((baseline_log-actual_log)**2), np.mean((predicted_log-actual_log)**2)
            models[str(horizon)] = {"N": len(test_idx), "log_mse_har": float(base_mse),
                "log_mse_ridge": float(pred_mse), "relative_log_mse_gain": float(1-pred_mse/base_mse),
                "qlike_har": float(qlike(actual,base)), "qlike_ridge": float(qlike(actual,pred)),
                "relative_qlike_gain": float(1-qlike(actual,pred)/qlike(actual,base))}
        events = test_mask & d.crossing & d.q.ne(0) & d.future_signed_5.notna()
        event_rows = [{"time": str(d.time.iloc[i]), "side": int(d.q.iloc[i]),
                       "signed_return_5d": float(d.future_signed_5.iloc[i]),
                       "continued": bool(d.future_signed_5.iloc[i] > 0),
                       "confirmed_5d": bool(d.q.iloc[i+1:i+6].eq(d.q.iloc[i]).mean() >= .6)}
                      for i in np.flatnonzero(events.to_numpy())]
        all_events.extend(event_rows)
        folds.append({"start": str(start), "end": str(end), "volatility": models,
                      "post_onset": {"N": len(event_rows),
                          "continuation_rate": float(np.mean([r["continued"] for r in event_rows])) if event_rows else None,
                          "mean_signed_log_return_5d": float(np.mean([r["signed_return_5d"] for r in event_rows])) if event_rows else None}})
    return {"H1B": {"definition": "post-onset q crossing; signed next-5d return; distinct from pre-onset H1-A",
                    "N_onsets": len(all_events), "continuation_rate": float(np.mean([r["continued"] for r in all_events])) if all_events else None,
                    "mean_signed_log_return_5d": float(np.mean([r["signed_return_5d"] for r in all_events])) if all_events else None,
                    "onsets": all_events},
            "H1C": {"definition": "forward realized variance=sum(next h daily log-return squared); HAR-style rv20 baseline vs expanding-fold Ridge using past-only features; log MSE and QLIKE",
                    "folds": folds},
            "economic_utility": "NOT TESTED; forecast skill is not net PnL"}

# Secondary studies: fixed, historical DEV-only extensions. They do not alter the
# frozen BTC-only H1-A result above and must never be interpreted as independent OOS.
def _fetch_core5_month(symbol: str, ym: str) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    root = f"{DATA_SOURCE}/klines/{symbol}/{INTERVAL}/{symbol}-{INTERVAL}-{ym}.zip"
    payload=cached_archive(root,"onefile-core5-research/1.0")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive, archive.open(archive.namelist()[0]) as stream:
        bars = pd.read_csv(stream, header=None)
    if str(bars.iloc[0, 0]).isalpha() or str(bars.iloc[0, 0]) == "open_time":
        bars = bars.iloc[1:].reset_index(drop=True)
    candles = pd.DataFrame({"time_ms": bars[0].astype("int64"), "open": bars[1].astype(float),
                            "high": bars[2].astype(float), "low": bars[3].astype(float),
                            "close": bars[4].astype(float), "volume": bars[5].astype(float)})
    funding = fetch_funding_month(ym, symbol=symbol)
    return symbol, candles, funding


def download_core5_dev(file: Path) -> dict:
    """Download canonical Binance monthly archives only through 2024-12 (never holdouts)."""
    jobs = [(symbol, ym) for symbol in CORE5_SYMBOLS for ym in
            pd.date_range(CORE5_FIRST_MONTH[symbol], DEV_END, freq="MS", inclusive="left").strftime("%Y-%m")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        chunks = list(pool.map(lambda job: _fetch_core5_month(*job), jobs))
    frames = []
    per_asset = {}
    for symbol in CORE5_SYMBOLS:
        candles = pd.concat([c for s, c, _ in chunks if s == symbol], ignore_index=True).sort_values("time_ms")
        funding = pd.concat([f for s, _, f in chunks if s == symbol], ignore_index=True).sort_values("time_ms")
        if candles.time_ms.duplicated().any() or funding.time_ms.duplicated().any():
            raise ValueError(f"Duplicate canonical archive records for {symbol}")
        end_ms = int(pd.Timestamp(DEV_END, tz="UTC").value // 1_000_000)
        first_ms = int(candles.time_ms.iloc[0])
        expected_bars = pd.date_range(pd.to_datetime(first_ms, unit="ms", utc=True),
                                      pd.Timestamp(DEV_END, tz="UTC"), freq="4h", inclusive="left").as_unit("ms").asi8
        actual_bars = candles.time_ms.to_numpy()
        interval_ms = 4*3_600_000
        if (not len(actual_bars) or np.any(np.diff(actual_bars) <= 0)
                or np.any(np.diff(actual_bars) % interval_ms != 0)
                or not np.isin(actual_bars, expected_bars).all()):
            raise ValueError(f"Off-grid/duplicate 4h history for {symbol}")
        missing_bar_count = len(expected_bars)-len(actual_bars)
        if funding.empty or funding.time_ms.iloc[0] >= end_ms:
            raise ValueError(f"No canonical funding records for {symbol}")
        first_funding = int(funding.time_ms.iloc[0])
        source_funding_count = len(funding)
        if (funding.time_ms.duplicated().any() or np.any(funding.time_ms.to_numpy() % 3_600_000 != 0)
                or not np.isfinite(funding.funding_rate.to_numpy()).all()):
            raise ValueError(f"Duplicate/off-hour/non-finite canonical funding for {symbol}")
        # Settlements inside a 4h candle are charged at the next candle open;
        # no intrabar execution ordering is invented. Rates mapping to one open sum.
        funding["time_ms"] = ((funding.time_ms+interval_ms-1)//interval_ms)*interval_ms
        mapped = funding.time_ms.isin(expected_bars)
        unmapped_funding_count = int((~mapped).sum())
        funding = funding.loc[mapped].groupby("time_ms", as_index=False).funding_rate.sum()
        merged = candles.merge(funding, on="time_ms", how="left", validate="one_to_one", indicator=True)
        merged["funding_observed"] = merged.pop("_merge").eq("both").astype("int8")
        merged.loc[merged.funding_observed.eq(0), "funding_rate"] = 0.0
        merged["symbol"] = symbol
        merged["time"] = pd.to_datetime(merged.time_ms, unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        frame = merged[DATA_COLUMNS]
        frames.append(frame)
        per_asset[symbol] = {"first_candle": str(pd.to_datetime(first_ms, unit="ms", utc=True)),
                             "first_funding": str(pd.to_datetime(first_funding, unit="ms", utc=True)),
                             "candles": len(candles), "funding_settlements": source_funding_count, "mapped_funding_bars": len(funding), "unmapped_funding_records": unmapped_funding_count,
                             "missing_candles": int(missing_bar_count), "missing_funding": 0}
    combined = pd.concat(frames, ignore_index=True).sort_values(["symbol", "time"]).reset_index(drop=True)
    atomic_write(file, lambda stream: combined.to_csv(stream, index=False))
    return {"source": DATA_SOURCE, "window_end_exclusive": DEV_END, "assets": per_asset,
            "file_sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "status": "PASS"}


def validate_core5(raw: pd.DataFrame, file: Path, file_hash: str | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    missing = set(DATA_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Core-5 data missing fields: {sorted(missing)}")
    x = raw[DATA_COLUMNS].copy()
    x["time"] = pd.to_datetime(x.time, utc=True, errors="raise")
    x["symbol"] = x.symbol.astype(str)
    if not x.symbol.isin(CORE5_SYMBOLS).all():
        raise ValueError("Core-5 file contains a symbol outside the frozen core-5 universe")
    if (x.time < pd.Timestamp(DEV_START, tz="UTC")).any() or (x.time >= pd.Timestamp(DEV_END, tz="UTC")).any():
        raise ValueError("Core-5 data exceeds the unlocked historical DEV window")
    assets, report = {}, {}
    for symbol in CORE5_SYMBOLS:
        a = x.loc[x.symbol.eq(symbol)].sort_values("time").reset_index(drop=True).copy()
        if len(a) < 365*6 or a.time.duplicated().any():
            raise ValueError(f"Insufficient/duplicate history for {symbol}")
        deltas = a.time.diff().iloc[1:].dt.total_seconds().to_numpy()
        bar_seconds = 4*60*60
        if (np.any(deltas <= 0) or np.any(deltas % bar_seconds != 0)):
            raise ValueError(f"Duplicate/off-grid 4h bars for {symbol}")
        numbers = ["open", "high", "low", "close", "volume", "funding_rate"]
        a[numbers] = a[numbers].apply(pd.to_numeric, errors="raise").astype(float)
        if not np.isfinite(a[numbers].to_numpy()).all() or (a[numbers[:4]] <= 0).any().any() or (a.volume < 0).any():
            raise ValueError(f"Invalid/non-finite OHLCV or funding data for {symbol}")
        if not ((a.high >= a[["open", "close", "low"]].max(axis=1)) &
                (a.low <= a[["open", "close", "high"]].min(axis=1))).all():
            raise ValueError(f"Invalid OHLC geometry for {symbol}")
        if not a.time.eq(a.time.dt.floor("4h")).all():
            raise ValueError(f"Off-grid candle for {symbol}")
        a["funding_observed"] = pd.to_numeric(a.funding_observed, errors="raise").astype("int8")
        if not a.funding_observed.isin([0, 1]).all():
            raise ValueError(f"funding_observed must be binary for {symbol}")
        observed_times = a.loc[a.funding_observed.eq(1), "time"]
        if not observed_times.eq(observed_times.dt.floor("4h")).all():
            raise ValueError(f"Funding timestamps are off candle grid for {symbol}")
        if a.loc[a.funding_observed.eq(0), "funding_rate"].ne(0).any():
            raise ValueError(f"Unobserved funding rate must be zero for {symbol}")
        assets[symbol] = a
        report[symbol] = {"bars": len(a), "first": str(a.time.iloc[0]), "last": str(a.time.iloc[-1]),
                          "funding_settlements": int(a.funding_observed.sum()),
                          "funding_nonzero": int(a.loc[a.funding_observed.eq(1), "funding_rate"].ne(0).sum()),
                          "missing_bars": int(np.maximum(a.time.diff().iloc[1:].dt.total_seconds().to_numpy() // (4*60*60)-1, 0).sum()),
                          "sha256": hashlib.sha256(a[DATA_COLUMNS].to_csv(index=False).encode()).hexdigest()}
    if file_hash is None:
        file_hash = hashlib.sha256(file.read_bytes()).hexdigest()
    return assets, {"file": str(file), "file_sha256": file_hash,
                    "rows": len(x), "assets": report, "integrity": "PASS"}


def panel_block_interval(frame: pd.DataFrame, columns: list[str], *, ratio: bool = False) -> dict:
    """Paired, date-block uncertainty for panels; coins sharing a shock stay together."""
    if frame.empty:
        return {"estimate": None, "ci95": [None, None], "sensitivity": []}
    daily = frame.groupby("time", sort=True)[columns].mean()
    daily.index = pd.DatetimeIndex(daily.index).tz_localize(None)
    daily = daily.reindex(pd.date_range(daily.index.min().normalize(), daily.index.max().normalize(), freq="D"))
    return _calendar_block_interval(daily.to_numpy(float), ratio=ratio)


def bocpd_recent_probability(returns: pd.Series, *, hazard: float = 1/60, max_run: int = 365) -> np.ndarray:
    """Online BOCPD: Gaussian mean with Normal-Inverse-Gamma predictive, fixed hazard."""
    from scipy.special import gammaln
    _started = time.perf_counter()
    r = pd.to_numeric(returns, errors="coerce").astype(float)
    scale = np.sqrt(r.pow(2).shift(1).rolling(20, min_periods=10).mean())
    x = (r/scale).clip(-10, 10).to_numpy()
    output = np.full(len(x), np.nan)
    probs = np.array([1.0])
    mu, kappa, alpha, beta = (np.array([0.0]), np.array([0.1]), np.array([2.5]), np.array([1.5]))
    for i, obs in enumerate(x):
        if not np.isfinite(obs):
            probs = np.array([1.0])
            mu, kappa, alpha, beta = (np.array([0.0]), np.array([0.1]), np.array([2.5]), np.array([1.5]))
            continue
        dof = 2*alpha
        scale2 = beta*(kappa+1)/(alpha*kappa)
        z2 = (obs-mu)**2/(dof*scale2)
        # Vectorized special function avoids hundreds of Python calls per day.
        log_like = (gammaln((dof+1)/2)-gammaln(dof/2)
                    -.5*np.log(dof*np.pi*scale2)-((dof+1)/2)*np.log1p(z2))
        log_joint = np.log(np.maximum(probs, 1e-300))+log_like
        log_growth = log_joint+math.log1p(-hazard)
        max_log = float(max(np.max(log_growth), np.max(log_joint)+math.log(hazard)))
        reset = np.exp(log_joint+math.log(hazard)-max_log).sum()
        growth = np.exp(log_growth-max_log)
        next_prob = np.concatenate(([reset], growth[:max_run]))
        next_prob /= next_prob.sum()
        output[i] = float(next_prob[:3].sum())
        new_mu = (kappa*mu+obs)/(kappa+1)
        new_kappa = kappa+1
        new_alpha = alpha+.5
        new_beta = beta+.5*kappa*(obs-mu)**2/(kappa+1)
        mu = np.concatenate(([obs/1.1], new_mu[:max_run]))
        kappa = np.concatenate(([1.1], new_kappa[:max_run]))
        alpha = np.concatenate(([3.0], new_alpha[:max_run]))
        beta = np.concatenate(([1.5+.5*.1*obs**2/1.1], new_beta[:max_run]))
        probs = next_prob
    _perf_time("h1b.bocpd_seconds", time.perf_counter() - _started)
    return output


def h1b_experiment(d: pd.DataFrame, events: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, dict]:
    """Secondary onset detection + post-detection continuation; never pre-onset H1-A."""
    d = d.copy().reset_index(drop=True)
    d["bocpd_recent_cp"] = bocpd_recent_probability(d.r1)
    alarms = []
    last_alarm = -10_000
    for i, probability in enumerate(d.bocpd_recent_cp):
        if np.isfinite(probability) and probability >= .5 and i-last_alarm >= 7:
            alarms.append(i)
            last_alarm = i
    confirmed = events.loc[events.status.eq("CONFIRMED")].sort_values("onset_i")
    used, rows = set(), []
    for e in confirmed.itertuples():
        i = int(e.onset_i)
        eligible = [j for j in alarms if i <= j <= i+10 and j not in used]
        detected_i = min(eligible) if eligible else None
        if detected_i is not None:
            used.add(detected_i)
        record = {"symbol": symbol, "onset_time": e.onset_time, "side": int(e.side),
                  "detected": detected_i is not None,
                  "delay_days": int(detected_i-i) if detected_i is not None else None,
                  "cp_probability_at_onset": float(d.bocpd_recent_cp.iloc[i]) if np.isfinite(d.bocpd_recent_cp.iloc[i]) else None,
                  "detection_time": d.time.iloc[detected_i] if detected_i is not None else pd.NaT}
        for horizon in (1, 3, 7):
            for label, j in (("onset", i), ("bocpd", detected_i)):
                key = f"{label}_signed_return_{horizon}d"
                record[key] = (float(e.side*np.log(d.close.iloc[j+horizon]/d.close.iloc[j]))
                               if j is not None and j+horizon < len(d) else None)
        rows.append(record)
    matched = pd.DataFrame(rows, columns=["symbol", "onset_time", "side", "detected", "delay_days",
                                          "cp_probability_at_onset", "detection_time",
                                          *[f"{name}_signed_return_{h}d" for h in (1,3,7)
                                            for name in ("onset", "bocpd")]])
    false_alarm_count = len(alarms)-len(used)
    exposure_years = max((d.time.max()-d.time.min()).days/365.25, 1e-9)
    summary = {"method": "BOCPD NIG Student-t predictive; hazard=1/60; alert P(run_length<=2)>=0.5; 7d cooldown",
               "confirmed_events": len(matched), "detected": int(matched.detected.sum()) if len(matched) else 0,
               "recall": float(matched.detected.mean()) if len(matched) else None,
               "all_alarms": len(alarms), "false_alarms": false_alarm_count,
               "false_alarms_per_asset_year": false_alarm_count/exposure_years,
               "median_detection_delay_days": float(matched.loc[matched.detected, "delay_days"].median()) if len(matched) and matched.detected.any() else None,
               "delay_ci": panel_block_interval(matched.loc[matched.detected, ["onset_time", "delay_days"]].rename(columns={"onset_time":"time"}), ["delay_days"]),
               "continuation": {}}
    for horizon in (1, 3, 7):
        onset_col, cpd_col = f"onset_signed_return_{horizon}d", f"bocpd_signed_return_{horizon}d"
        summary["continuation"][str(horizon)] = {
            "N_events": int(matched[onset_col].notna().sum()) if len(matched) else 0,
            "q_crossing_positive_rate": float(matched[onset_col].dropna().gt(0).mean()) if len(matched) and matched[onset_col].notna().any() else None,
            "q_crossing_mean_signed_return": float(matched[onset_col].mean()) if len(matched) and matched[onset_col].notna().any() else None,
            "q_crossing_uncertainty": panel_block_interval(matched.loc[matched[onset_col].notna(), ["onset_time", onset_col]].rename(columns={"onset_time":"time"}), [onset_col]),
            "bocpd_detected_N": int(matched[cpd_col].notna().sum()) if len(matched) else 0,
            "bocpd_positive_rate": float(matched[cpd_col].dropna().gt(0).mean()) if len(matched) and matched[cpd_col].notna().any() else None,
            "bocpd_mean_signed_return": float(matched[cpd_col].mean()) if len(matched) and matched[cpd_col].notna().any() else None,
            "bocpd_uncertainty": panel_block_interval(matched.loc[matched[cpd_col].notna(), ["detection_time", cpd_col]].rename(columns={"detection_time":"time"}), [cpd_col])}
    return d, summary | {"event_rows": rows, "alarm_indices": alarms}


def har_rv_experiment(d: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, dict]:
    """Walk-forward HAR-RV vs EWMA and 20d persistence for future 5/10d realized variance."""
    from sklearn.linear_model import LinearRegression
    d = d.copy().reset_index(drop=True)
    rv = d.r1.pow(2)
    d["rv_var1"] = rv
    d["rv_var5"] = rv.rolling(5).mean()
    d["rv_var22"] = rv.rolling(22).mean()
    d["rv_ewma"] = rv.ewm(alpha=.06, adjust=False, min_periods=22).mean()
    for h in (5, 10):
        d[f"future_var_{h}"] = sum(rv.shift(-j) for j in range(1, h+1))/h
    output = []
    for start in pd.date_range("2022-01-01", DEV_END, freq="6MS", inclusive="left", tz="UTC"):
        end = start+pd.DateOffset(months=6)
        for h in (5, 10):
            target = f"future_var_{h}"
            cols = ["rv_var1", "rv_var5", "rv_var22"]
            matured = d.time+pd.Timedelta(days=h) < start
            train = d.loc[matured & d[target].gt(0) & d[cols].gt(0).all(axis=1)].copy()
            test = d.loc[(d.time >= start) & (d.time < end) & d[target].gt(0) & d[cols].gt(0).all(axis=1)].copy()
            if len(train) < 250 or test.empty:
                continue
            model = LinearRegression().fit(np.log(train[cols]), np.log(train[target]))
            log_hat = model.predict(np.log(test[cols]))
            bias = float(np.var(np.log(train[target])-model.predict(np.log(train[cols])), ddof=1))
            pred = np.exp(np.clip(log_hat+.5*bias, -40, 5))
            test_out = pd.DataFrame({"time": test.time, "symbol": symbol, "horizon": h,
                                     "actual_variance": test[target].to_numpy(),
                                     "HAR": pred,
                                     "EWMA": test.rv_ewma.to_numpy(),
                                     "persistence_RV20": test.rv_var22.to_numpy(),
                                     "fold": str(start.date()), "train_latest_origin": str(train.time.max())})
            output.append(test_out)
    forecasts = pd.concat(output, ignore_index=True) if output else pd.DataFrame()
    if forecasts.empty:
        return forecasts, {"status": "INSUFFICIENT_HISTORY", "forecasts": 0}
    for model in ("HAR", "EWMA", "persistence_RV20"):
        forecasts[f"QLIKE_{model}"] = (np.log(forecasts[model].clip(lower=1e-15))+
                                        forecasts.actual_variance/forecasts[model].clip(lower=1e-15))
        forecasts[f"log_sqerr_{model}"] = (np.log(forecasts.actual_variance)-np.log(forecasts[model].clip(lower=1e-15)))**2
    result = {"status": "SCORED", "N": len(forecasts), "metrics": {}}
    for model in ("HAR", "EWMA", "persistence_RV20"):
        result["metrics"][model] = {"mean_QLIKE": float(forecasts[f"QLIKE_{model}"].mean()),
                                    "RMSE_log_variance": float(np.sqrt(forecasts[f"log_sqerr_{model}"].mean()))}
    return forecasts, result


def funding_increment_experiment(d: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, dict]:
    """Secondary expanding return forecast: momentum controls vs same model plus known funding."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    d = d.copy().reset_index(drop=True)
    cols = H1A_MODELS["M1"]
    output = []
    for h in (1, 3, 7):
        future = sum(d.r1.shift(-j) for j in range(1, h+1))
        d[f"future_return_{h}"] = future
        for start in pd.date_range("2022-01-01", DEV_END, freq="6MS", inclusive="left", tz="UTC"):
            end = start+pd.DateOffset(months=6)
            matured = d.time+pd.Timedelta(days=h) < start
            usable = d[cols+ ["funding3", f"future_return_{h}"]].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
            train = d.loc[matured & usable]
            test = d.loc[(d.time >= start) & (d.time < end) & usable & d[f"future_return_{h}"].notna()]
            if len(train) < 250 or test.empty:
                continue
            pred = {}
            for name, features in (("momentum", cols), ("momentum_funding", cols+["funding3"])):
                model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                model.fit(train[features], train[f"future_return_{h}"])
                pred[name] = model.predict(test[features])
            rows = pd.DataFrame({"time": test.time, "symbol": symbol, "horizon": h,
                                 "actual_return": test[f"future_return_{h}"].to_numpy(),
                                 "momentum_prediction": pred["momentum"],
                                 "funding_prediction": pred["momentum_funding"],
                                 "momentum_direction": np.sign(test.r5.to_numpy()), "fold": str(start.date()),
                                 "train_latest_origin": str(train.time.max())})
            output.append(rows)
    forecasts = pd.concat(output, ignore_index=True) if output else pd.DataFrame()
    if forecasts.empty:
        return forecasts, {"status": "INSUFFICIENT_HISTORY", "forecasts": 0}
    for name, pred in (("momentum", "momentum_prediction"), ("momentum_funding", "funding_prediction")):
        forecasts[f"mse_{name}"] = (forecasts.actual_return-forecasts[pred])**2
        forecasts[f"direction_hit_{name}"] = np.sign(forecasts[pred]).eq(np.sign(forecasts.actual_return)).astype(float)
    forecasts["direction_hit_momentum_signal"] = np.sign(forecasts.momentum_direction).eq(np.sign(forecasts.actual_return)).astype(float)
    return forecasts, {"status": "SCORED", "N": len(forecasts)}


def verify_core5_funding(assets: dict[str, pd.DataFrame]) -> dict:
    """Compare every cached/canonical funding settlement, independently per contract."""
    jobs = [(symbol, ym) for symbol in CORE5_SYMBOLS for ym in
            pd.date_range(CORE5_FIRST_MONTH[symbol], DEV_END, freq="MS", inclusive="left").strftime("%Y-%m")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        chunks = list(pool.map(lambda item: (item[0], fetch_funding_month(item[1], symbol=item[0])), jobs))
    checked = {}
    for symbol in CORE5_SYMBOLS:
        source = pd.concat([frame for name, frame in chunks if name == symbol], ignore_index=True).sort_values("time_ms")
        observed = assets[symbol].loc[assets[symbol].funding_observed.eq(1)]
        candles_ms = assets[symbol].time.array.as_unit("ms").asi8
        raw_times = source.time_ms.to_numpy(dtype="int64")
        bar_ms = 4*3_600_000
        source["effective_ms"] = ((source.time_ms+bar_ms-1)//bar_ms)*bar_ms
        matched = source.groupby("effective_ms", as_index=False).funding_rate.sum()
        matched = matched.loc[matched.effective_ms.isin(candles_ms)]
        supplied_times = observed.time.array.as_unit("ms").asi8
        if (source.time_ms.duplicated().any() or np.any(raw_times % 3_600_000 != 0)
                or not np.isfinite(source.funding_rate.to_numpy()).all()
                or not np.array_equal(matched.effective_ms.to_numpy(dtype="int64"), supplied_times)
                or not np.allclose(matched.funding_rate.to_numpy(), observed.funding_rate.to_numpy(), rtol=0, atol=1e-15)):
            raise ValueError(f"Funding data for {symbol} differs from canonical Binance cache/archive")
        checked[symbol] = len(source)
    return {"status":"VERIFIED", "method":"Binance monthly archives; local ZIP cache SHA-256 checked",
            "settlements_by_symbol":checked,"verified_at_utc":pd.Timestamp.now(tz="UTC").isoformat()}


def core5_research_run(raw: pd.DataFrame, readiness: dict, args) -> None:
    """Run fixed DEV-only H1-A/B/C and five-question extensions across liquid core-5 perps."""
    import time
    import sklearn
    started = time.monotonic()
    specification = {
        "study_id": "ONEFILE-CORE5-H1ABC-DEV-EXPLORATORY-01", "window": [DEV_START, DEV_END],
        "sealed_never_read": ["ONEFILE >=2025-01-01", "monograph G3 >=2026-04-01"],
        "assets": list(CORE5_SYMBOLS), "source": DATA_SOURCE,
        "H1A": "per-asset monograph onset; M0/M1/M2 expanding 6mo forecasts; >=17d purge; fixed logistic C=1; M2 alert threshold selected on inner forward folds; raw probabilities/calibration scored OOS; pooled paired 28/56/84d blocks",
        "H1B": "online Gaussian BOCPD NIG/Student-t, hazard=1/60, cap run length 365; P(run length<=2)>=.5 alarm; 7d cooldown; event match onset..onset+10; signed close-to-close 1/3/7d continuation vs q-crossing baseline",
        "H1C": "5/10d future mean squared daily log return; monthly expanding walk-forward HAR-RV(log variance on RV1/RV5/RV22) vs EWMA(lambda=.94) and RV22 persistence; H-day target purge; QLIKE primary",
        "funding": "known 3d settlement funding at daily close adds one predictor to locked M1-style return ridge; 1/3/7d expanding OOS; skill and funding PnL remain distinct",
        "asymmetry": "same 10 fixed descriptive conditions; native vectorbt 3R/5R gross barriers, 1R stop, 7d timeout, daily 1.5*ATR14 risk; 4bp/side fee+2bp/side slippage+historical funding proxy; no condition selection",
        "uncertainty": "paired 28/56/84 calendar-day circular bootstrap; symbols on the same dates resampled together; overlap and shared macro shocks acknowledged",
        "economic_mechanisms": "OHLCV/funding cannot identify trader identity, order-book liquidity, liquidation causality or spot-perp basis; no causal claim",
        "trial_history": "unknown; historical research extension chosen after BTC H1-A/descriptive results; exploratory, no independent OOS or edge certification",
        "costs": {"fee_bps_side": FEE_BPS, "slippage_bps_side": SLIPPAGE_BPS},
    }
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity = {"specification": specification, "input_sha256": readiness["file_sha256"], "code_sha256": source_hash}
    protocol_path = Path(args.protocol)
    if protocol_path.exists():
        frozen = json.loads(protocol_path.read_text())
        if frozen.get("identity") != identity:
            raise ValueError("Core-5 frozen protocol/data/code mismatch; retain old attempt and choose a new --protocol")
    else:
        frozen = {"frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "identity": identity,
                  "note": "Local historical freeze before this run, not independent preregistration or OOS."}
        atomic_write(protocol_path, lambda stream: stream.write(json.dumps(frozen, indent=2)+"\n"))
    assets, validated = validate_core5(raw, Path(args.csv))
    validated["funding_source_verification"] = readiness.get("funding_source_verification", {"status":"UNKNOWN"})
    by_asset, h1a_forecasts, h1a_events, h1b_rows, h1c_forecasts, funding_forecasts, observations = {}, [], [], [], [], [], []
    h1b_assets, h1c_assets, funding_assets, behavior_by_asset = {}, {}, {}, {}
    runtime_by_symbol={}
    for symbol in CORE5_SYMBOLS:
        asset_started=time.monotonic(); stages={}
        x = features(assets[symbol])
        stage=time.monotonic(); daily = daily_market(x); labeled, events = onset_labels(daily)
        stages["aggregate_labels_seconds"]=time.monotonic()-stage
        stage=time.monotonic(); fc, h1a = h1a_experiment(labeled, events, fast=True)
        stages["H1A_seconds"]=time.monotonic()-stage
        if len(fc):
            fc.insert(1, "symbol", symbol)
            h1a_forecasts.append(fc)
        if len(events):
            events.insert(0, "symbol", symbol)
            h1a_events.append(events)
        by_asset[symbol] = h1a
        stage=time.monotonic(); _, h1b = h1b_experiment(labeled, events.drop(columns="symbol", errors="ignore"), symbol)
        stages["H1B_seconds"]=time.monotonic()-stage
        h1b_assets[symbol] = {k:v for k,v in h1b.items() if k not in ("event_rows", "alarm_indices")}
        if h1b["event_rows"]:
            h1b_rows.extend(h1b["event_rows"])
        stage=time.monotonic(); h1c, h1c_summary = har_rv_experiment(daily, symbol)
        stages["H1C_seconds"]=time.monotonic()-stage
        h1c_forecasts.append(h1c)
        h1c_assets[symbol] = h1c_summary
        stage=time.monotonic(); fund_fc, fund_summary = funding_increment_experiment(daily, symbol)
        stages["funding_seconds"]=time.monotonic()-stage
        funding_forecasts.append(fund_fc)
        funding_assets[symbol] = fund_summary
        stage=time.monotonic(); behavior, summaries = market_observations(x, daily)
        stages["market_behavior_seconds"]=time.monotonic()-stage
        behavior.insert(0, "symbol", symbol)
        observations.append(behavior)
        behavior_by_asset[symbol] = summaries
        stages["total_seconds"]=time.monotonic()-asset_started
        runtime_by_symbol[symbol]=stages
        print(f"Finished {symbol}: {stages['total_seconds']:.2f}s",flush=True)
    _PERF_CONTEXT["core5_runtime_by_symbol"] = runtime_by_symbol
    forecasts = pd.concat([f for f in h1a_forecasts if not f.empty], ignore_index=True) if any(not f.empty for f in h1a_forecasts) else pd.DataFrame()
    events_all = pd.concat(h1a_events, ignore_index=True) if h1a_events else pd.DataFrame()
    h1b_frame = pd.DataFrame(h1b_rows)
    h1c_frame = pd.concat([f for f in h1c_forecasts if not f.empty], ignore_index=True) if any(not f.empty for f in h1c_forecasts) else pd.DataFrame()
    funding_frame = pd.concat([f for f in funding_forecasts if not f.empty], ignore_index=True) if any(not f.empty for f in funding_forecasts) else pd.DataFrame()
    obs = pd.concat(observations, ignore_index=True)
    h1a_summary = {"assets": by_asset, "N_forecasts": len(forecasts),
                   "N_scored": int(forecasts.y.notna().sum()) if len(forecasts) else 0,
                   "N_confirmed_events": int(events_all.status.eq("CONFIRMED").sum()) if len(events_all) else 0}
    if len(forecasts) and forecasts.y.notna().any():
        scored = forecasts.loc[forecasts.y.notna()].copy()
        for name in ("M0", "M1", "M2", "placebo"):
            p = scored[[f"{name}_{k}" for k in ("none", "up", "down")]].to_numpy()
            bs, ll = forecast_losses(scored.y, p)
            scored[f"{name}_brier"] = bs
            h1a_summary.setdefault("pooled_metrics", {})[name] = {"brier":float(bs.mean()), "log_loss":float(ll.mean())}
        h1a_summary["M2_vs_M1_relative_brier_gain"] = panel_block_interval(scored, ["M1_brier", "M2_brier"], ratio=True)
        unique_events = scored.loc[scored.y.ne(0), ["symbol", "target_onset", "y"]].drop_duplicates()
        up = int(unique_events.y.eq(1).sum()); down = int(unique_events.y.eq(-1).sum())
        dates = sorted(pd.to_datetime(unique_events.target_onset, utc=True).dropna().unique())
        clusters, last = 0, None
        for date in dates:
            if last is None or (pd.Timestamp(date)-pd.Timestamp(last)).days > 28:
                clusters += 1
            last = date
        alarm_archives = [r.get("metrics",{}).get("M2",{}).get("alarms",{}) for r in by_asset.values()]
        total_years = sum(r.get("exposure_days",0) for r in by_asset.values())/365.25
        alarms_n = sum(a.get("alarms",0) for a in alarm_archives)
        detected = sum(a.get("detected_events",0) for a in alarm_archives)
        false = sum(a.get("false_alarms",0) for a in alarm_archives)
        leads = [v["lead_days"] for a in alarm_archives for v in a.get("alarm_archive",[])
                 if v.get("lead_days") is not None]
        pooled_alarm = {"alarms":alarms_n,"detected_events":detected,"false_alarms":false,
                        "false_per_asset_year":false/max(total_years,1e-9),
                        "median_lead_days":float(np.median(leads)) if leads else None}
        gain = h1a_summary["M2_vs_M1_relative_brier_gain"]
        h1a_summary.update({"scoreable_up_events":up,"scoreable_down_events":down,"calendar_clusters_pm14d":clusters,
                            "pooled_M2_alarms":pooled_alarm,
                            "evidence_gate":h1a_gate(up,down,clusters,gain,pooled_alarm,core5=True),
                            "interpretation":"All assets share historical calendar shocks; this is exploratory cross-asset replication, not independent validation."})
    h1b_summary = {"assets":h1b_assets,"pooled_confirmed_event_rows":len(h1b_frame)}
    for h in (1,3,7):
        for col in (f"onset_signed_return_{h}d", f"bocpd_signed_return_{h}d"):
            if len(h1b_frame) and h1b_frame[col].notna().any():
                h1b_summary.setdefault("pooled_continuation",{})[col] = panel_block_interval(
                    h1b_frame.loc[h1b_frame[col].notna(), ["onset_time" if col.startswith("onset") else "detection_time",col]].rename(
                        columns={"onset_time":"time","detection_time":"time"}),[col])
    h1c_summary = {"assets":h1c_assets,"N_forecasts":len(h1c_frame),"pooled":{}}
    if len(h1c_frame):
        for h in (5,10):
            part=h1c_frame.loc[h1c_frame.horizon.eq(h)].copy()
            h1c_summary["pooled"][str(h)]={}
            for model in ("HAR","EWMA","persistence_RV20"):
                qlike=np.log(part[model].clip(lower=1e-15))+part.actual_variance/part[model].clip(lower=1e-15)
                part[f"qlike_{model}"]=qlike
                h1c_summary["pooled"][str(h)][model]={"mean_QLIKE":float(qlike.mean()),
                    "RMSE_log_variance":float(np.sqrt(((np.log(part.actual_variance)-np.log(part[model].clip(lower=1e-15)))**2).mean()))}
            h1c_summary["pooled"][str(h)]["HAR_vs_EWMA_QLIKE_skill"]=panel_block_interval(part,["qlike_EWMA","qlike_HAR"],ratio=True)
            h1c_summary["pooled"][str(h)]["HAR_vs_RV20_QLIKE_skill"]=panel_block_interval(part,["qlike_persistence_RV20","qlike_HAR"],ratio=True)
    funding_summary={"assets":funding_assets,"N_forecasts":len(funding_frame),"horizons":{}}
    if len(funding_frame):
        for h in (1,3,7):
            part=funding_frame.loc[funding_frame.horizon.eq(h)].copy()
            funding_summary["horizons"][str(h)]={"N":len(part),
                "funding_incremental_MSE_skill":panel_block_interval(part,["mse_momentum","mse_momentum_funding"],ratio=True),
                "momentum_direction_hit_rate":float(part.direction_hit_momentum_signal.mean()),
                "ridge_direction_hit_rate":float(part.direction_hit_momentum.mean()),
                "ridge_plus_funding_direction_hit_rate":float(part.direction_hit_momentum_funding.mean())}
    obs_summary={"assets":behavior_by_asset,"pooled":[]}
    aligned=obs.loc[np.sign(obs.r5).eq(obs.side)].copy()
    conditions={"all_momentum":np.ones(len(aligned),bool),"established_trend":aligned.q.eq(aligned.side),
                "neutral":aligned.q.eq(0),"new_threshold_crossing":aligned.crossing & aligned.q.eq(aligned.side),
                "compression":aligned.compression<.8,"expansion_today":aligned.r1.abs()>2*aligned.rv20_prior,
                "high_volume":aligned.volume_surprise>np.log(1.5),
                "reversal_today":np.sign(aligned.r1).ne(aligned.side)&(aligned.r1.abs()>aligned.rv20),
                "funding_aligned_gt10bp3d":aligned.side*aligned.funding3>.001,
                "funding_opposes_momentum":aligned.side*aligned.funding3<0}
    for label, mask in conditions.items():
        group=aligned.loc[mask].copy()
        item={"condition":label,"N_asset_days":len(group),"horizons":{},"barriers":{}}
        for h in (1,3,7):
            col=f"return_{h}d"
            item["horizons"][str(h)]={"continuation":float(group[col].gt(0).mean()) if len(group) else None,
                "continuation_uncertainty":panel_block_interval(group.assign(**{col:group[col].gt(0).astype(float)}),[col]),
                "mean_gross":float(group[col].mean()) if len(group) else None,
                "net_return_mean":float(group[f"net_return_{h}d"].mean()) if len(group) else None}
        for target in (3,5):
            col=f"net_R_{target}"
            item["barriers"][str(target)]={"gross_barrier_hit_rate":float(group[f"gross_barrier_hit_{target}"].mean()) if len(group) else None,
                "net_target_attainment":float(group[col].ge(target).mean()) if len(group) else None,
                "stress_net_target_attainment":float(group[f"stress_R_{target}"].ge(target).mean()) if len(group) else None,
                "net_R_mean":float(group[col].mean()) if len(group) else None,
                "net_R_uncertainty":panel_block_interval(group,[col]),
                "net_R_ex_funding_mean":float(group[f"ex_funding_R_{target}"].mean()) if len(group) else None}
        obs_summary["pooled"].append(item)
    evidence={"protocol_sha256":hashlib.sha256(protocol_path.read_bytes()).hexdigest(),"readiness":validated,
              "H1A":h1a_summary,"H1B":h1b_summary,"H1C":h1c_summary,"funding_increment":funding_summary,
              "five_research_questions":obs_summary,"runtime_by_symbol":runtime_by_symbol,
              "runtime_seconds":time.monotonic()-started,"versions":{"python":platform.python_version(),"numpy":np.__version__,
              "pandas":pd.__version__,"sklearn":sklearn.__version__,"vectorbt":vbt.__version__},
              "limits":{"sealed_data_read":False,"independent_oos":False,"causal_participant_identification":False,
              "economic_edge_proven":False,"all_findings_exploratory":True},
              "caveats":["Core-5 sampled assets are correlated; same-date block resampling is not five independent replications.",
              "Funding is a settlement/open-price proxy and not a spot-perp arbitrage return.",
              "Daily OHLCV cannot identify order flow, liquidation causality, trader identity, mark-price liquidation or order-book execution.",
              "H1-B onset event is algorithmic q-threshold crossing; BOCPD detection is a secondary diagnostic and cannot be renamed H1-A.",
              "H1-C forecasts variance; forecast skill is not a directional signal or economic profit.",
              "5R/3R tests are fixed-window event simulations, not portfolio/account equity or margin models."]}
    outputs=((args.forecasts,forecasts),(args.onsets,events_all),(args.h1b_events,h1b_frame),
             (args.h1c_forecasts,h1c_frame),(args.funding_forecasts,funding_frame),(args.out,obs))
    for path, frame in outputs:
        atomic_write(Path(path),lambda stream,frame=frame: frame.to_csv(stream,index=False))
    evidence["outputs"]={str(path):hashlib.sha256(Path(path).read_bytes()).hexdigest() for path,_ in outputs}
    atomic_write(Path(args.evidence),lambda stream:stream.write(json.dumps(evidence,indent=2,allow_nan=False)+"\n"))
    print(json.dumps({"H1A":h1a_summary.get("evidence_gate",{}).get("verdict","INCONCLUSIVE"),
                      "H1A_events":[h1a_summary.get("scoreable_up_events"),h1a_summary.get("scoreable_down_events")],
                      "H1B_confirmed_events":h1b_summary["pooled_confirmed_event_rows"],
                      "H1C_forecasts":len(h1c_frame),"funding_forecasts":len(funding_frame),
                      "runtime_seconds":evidence["runtime_seconds"]},indent=2))
    print(f"Saved core-5 outputs and evidence to {args.evidence}; protocol {protocol_path}")



# Regime forecasting v2: historical DEV only, 2020 training warmup and four
# calendar years (2021-2024) of expanding, purged six-month outer evaluations.
REGIME_FEATURES = H1A_MODELS["M2"] + [
    "r1", "q", "z5", "er5", "body_atr", "range_atr", "close_location", "funding3"]
REGIME_HORIZON = 7
REGIME_CONFIRM = 5
REGIME_GAP = REGIME_HORIZON + REGIME_CONFIRM
REGIME_THRESHOLDS = (.1, .2, .35, .5, .65, .8, .9)


def regime_features(d: pd.DataFrame) -> pd.DataFrame:
    """Completed daily observations only; no fitted transform or future data."""
    d = d.copy()
    d["z5"] = d.r5 / np.sqrt(d.r1.pow(2).rolling(5).sum() + 1e-12)
    d["er5"] = d.r5.abs() / (d.r1.abs().rolling(5).sum() + 1e-12)
    d["body_atr"] = (d.close-d.open) / d.atr.clip(lower=1e-12)
    d["range_atr"] = (d.high-d.low) / d.atr.clip(lower=1e-12)
    d["close_location"] = (2*d.close-d.high-d.low)/(d.high-d.low).clip(lower=1e-12)
    return d


def regime_labels(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All q crossings, including reversals; 3/5 future days must retain q.

    The old q state definition is unchanged. Failed crossings remain in the
    event archive and are negatives for forecasting. Origins may have any
    current q, but cannot forecast an onset into their already-current state.
    Overlapping warnings are not independent events. No 10-day neutral filter.
    """
    d = d.copy().reset_index(drop=True)
    q = d.q.to_numpy(float)
    crossing = np.flatnonzero(np.isfinite(q[1:]) & np.isfinite(q[:-1]) &
                             (q[1:] != 0) & (q[1:] != q[:-1]))+1
    records = []
    for i in crossing:
        future = q[i+1:i+REGIME_CONFIRM+1]
        mature = len(future) == REGIME_CONFIRM and np.isfinite(future).all()
        confirmed = mature and np.count_nonzero(future == q[i]) >= 3
        records.append({"onset_i": int(i), "onset_time": d.time.iloc[i], "side": int(q[i]),
                        "confirmation_time": d.time.iloc[i+REGIME_CONFIRM] if mature else pd.NaT,
                        "status": "CONFIRMED" if confirmed else "UNCONFIRMED" if mature else "CENSORED"})
    events = pd.DataFrame(records, columns=["onset_i", "onset_time", "side", "confirmation_time", "status"])
    d["y"] = np.nan
    d["target_onset"] = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns, UTC]")
    d["target_confirmation"] = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns, UTC]")
    d["maturity_time"] = d.time + pd.Timedelta(days=REGIME_GAP)
    if len(q) <= REGIME_GAP:
        return d, events
    windows = np.lib.stride_tricks.sliding_window_view(q, REGIME_GAP+1)
    origins = np.flatnonzero(np.isfinite(windows).all(axis=1))
    d.loc[origins, "y"] = 0.
    chosen = np.full(len(origins), len(d), dtype=int)
    chosen_side = np.zeros(len(origins), dtype=int)
    for side in (1, -1):
        candidates = events.loc[events.status.eq("CONFIRMED") & events.side.eq(side), "onset_i"].to_numpy(int)
        if not len(candidates):
            continue
        next_pos = np.searchsorted(candidates, origins, side="right")
        upcoming = candidates[np.minimum(next_pos, len(candidates)-1)]
        valid = ((next_pos < len(candidates)) & (upcoming <= origins+REGIME_HORIZON) &
                 (q[origins] != side) & (upcoming < chosen))
        chosen[valid], chosen_side[valid] = upcoming[valid], side
    hit = chosen < len(d)
    d.loc[origins[hit], "y"] = chosen_side[hit]
    d.loc[origins[hit], "target_onset"] = d.time.iloc[chosen[hit]].to_numpy()
    d.loc[origins[hit], "target_confirmation"] = d.time.iloc[chosen[hit]+REGIME_CONFIRM].to_numpy()
    return d, events


def regime_splits(d: pd.DataFrame, start: pd.Timestamp) -> tuple[pd.DataFrame, ...]:
    """Six-month validation: first half stopping, second calibration/policy.

    Fixed 12-day label information window plus one-day embargo at EVERY boundary.
    No random split. Strict legacy labels instead retain their 17-day window.
    """
    stop_start, cal_start = start-pd.DateOffset(months=6), start-pd.DateOffset(months=3)
    feature_ok = np.isfinite(d[REGIME_FEATURES]).all(axis=1) & d.forecast_eligible
    eligible = d.y.notna() & feature_ok
    mature = d.maturity_time + pd.Timedelta(days=1)
    train = d.loc[eligible & (mature < stop_start)]
    stopping = d.loc[eligible & (d.time >= stop_start) & (mature < cal_start)]
    calibration = d.loc[eligible & (d.time >= cal_start) & (mature < start)]
    test = d.loc[feature_ok & (d.time >= start) & (d.time < start+pd.DateOffset(months=6))]
    return train, stopping, calibration, test


def _regime_weights(rows: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Train-only capped sqrt class weights and inverse positive-event multiplicity."""
    counts = np.bincount(y, minlength=2).clip(min=1)
    w = np.minimum(3., np.sqrt(len(y)/(2*counts[y])))
    event = rows.target_onset.notna()
    if event.any():
        multiplicity = rows.loc[event, "target_onset"].map(rows.loc[event, "target_onset"].value_counts())
        w[event.to_numpy()] /= multiplicity.to_numpy(float)
    return w / w.mean()


def regime_probabilities(train: pd.DataFrame, stopping: pd.DataFrame,
                         calibration: pd.DataFrame, test: pd.DataFrame,
                         kind: str, threads: int = 1) -> tuple[np.ndarray, np.ndarray, dict]:
    """Factorized event/direction with sklearn sigmoid calibration on later data.

    Early stopping uses an earlier, separate validation segment. Refit retains
    the chosen tree count; calibration never fits on training or outer labels.
    """
    from scipy.special import logit
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    from xgboost import XGBClassifier
    started = time.perf_counter()
    refit = pd.concat([train, stopping]).sort_values("time")
    info = {"training_seconds": 0., "inference_seconds": 0., "calibration_seconds": 0., "heads": {}}
    if kind == "no_event":
        return np.tile([1., 0., 0.], (len(calibration), 1)), np.tile([1., 0., 0.], (len(test), 1)), info
    if kind == "prior":
        prior = (np.array([refit.y.eq(k).sum() for k in H1A_CLASSES])+1)/(len(refit)+3)
        return np.tile(prior, (len(calibration), 1)), np.tile(prior, (len(test), 1)), info
    outputs = []
    for head in ("event", "direction"):
        tr = train if head == "event" else train.loc[train.y.ne(0)]
        va = stopping if head == "event" else stopping.loc[stopping.y.ne(0)]
        fit = refit if head == "event" else refit.loc[refit.y.ne(0)]
        ca = calibration if head == "event" else calibration.loc[calibration.y.ne(0)]
        target = (lambda r: r.y.ne(0).to_numpy(int)) if head == "event" else (lambda r: r.y.eq(1).to_numpy(int))
        def constant(rows, p):
            return np.full(len(rows), p)
        ts = time.perf_counter()
        detail = {"fit_rows": len(fit), "fit_class_counts": np.bincount(target(fit), minlength=2).tolist()}
        if len(fit) < 20 or len(np.unique(target(fit))) < 2:
            prior = (target(fit).sum()+1)/(len(fit)+2)
            predict = lambda rows, prior=prior: constant(rows, prior)
            detail["fallback"] = "smoothed training prior"
        elif kind == "logistic":
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=200, solver="lbfgs"))
            with threadpool_limits(limits=1):
                model.fit(fit[REGIME_FEATURES].to_numpy(), target(fit),
                          logisticregression__sample_weight=_regime_weights(fit, target(fit)))
            predict = lambda rows: model.predict_proba(rows[REGIME_FEATURES].to_numpy())[:, 1]
        else:
            params = dict(tree_method="hist", device="cpu", n_jobs=threads, max_depth=2,
                          min_child_weight=10., learning_rate=.05, reg_lambda=10., reg_alpha=.1,
                          gamma=.1, subsample=.8, colsample_bytree=.8, max_bin=64,
                          random_state=SEED, eval_metric="logloss")
            rounds = 64
            if len(tr) >= 20 and len(np.unique(target(tr))) == 2 and len(va) >= 5:
                probe = XGBClassifier(**params, n_estimators=256, early_stopping_rounds=20)
                probe.fit(tr[REGIME_FEATURES].to_numpy(np.float32), target(tr),
                          sample_weight=_regime_weights(tr, target(tr)),
                          eval_set=[(va[REGIME_FEATURES].to_numpy(np.float32), target(va))], verbose=False)
                rounds = probe.best_iteration+1
                detail["stopping_log_loss"] = float(probe.best_score)
            model = XGBClassifier(**params, n_estimators=rounds)
            model.fit(fit[REGIME_FEATURES].to_numpy(np.float32), target(fit),
                      sample_weight=_regime_weights(fit, target(fit)))
            predict = lambda rows: model.predict_proba(rows[REGIME_FEATURES].to_numpy(np.float32))[:, 1]
            detail["rounds"] = rounds
        info["training_seconds"] += time.perf_counter()-ts
        ts = time.perf_counter()
        pc = predict(calibration) if len(calibration) else np.empty(0)
        pt = predict(test) if len(test) else np.empty(0)
        pf = predict(fit)
        info["inference_seconds"] += time.perf_counter()-ts
        from sklearn.metrics import log_loss
        detail["in_sample_log_loss"] = float(log_loss(target(fit), np.clip(pf, 1e-7, 1-1e-7), labels=[0, 1])) if len(fit) else None
        ts = time.perf_counter()
        mask = np.ones(len(calibration), bool) if head == "event" else calibration.y.ne(0).to_numpy()
        cy = target(ca)
        if len(cy) and np.bincount(cy, minlength=2).min() >= 5:
            scaler = LogisticRegression(C=1., solver="lbfgs")
            scaler.fit(logit(np.clip(pc[mask], 1e-6, 1-1e-6))[:, None], cy)
            detail["calibration"] = "sigmoid; independent later validation segment"
            detail["calibration_raw_log_loss"] = float(log_loss(cy, pc[mask], labels=[0, 1]))
            pc = scaler.predict_proba(logit(np.clip(pc, 1e-6, 1-1e-6))[:, None])[:, 1]
            if len(pt):
                pt = scaler.predict_proba(logit(np.clip(pt, 1e-6, 1-1e-6))[:, None])[:, 1]
        else:
            detail["calibration"] = "identity: fewer than 5 observations in a calibration class"
        if len(cy):
            detail["calibration_log_loss"] = float(log_loss(cy, np.clip(pc[mask], 1e-7, 1-1e-7), labels=[0, 1]))
        info["calibration_seconds"] += time.perf_counter()-ts
        info["heads"][head] = detail
        outputs.append((pc, pt))
    def joint(event, direction):
        return np.column_stack([1-event, event*direction, event*(1-direction)])
    info["total_seconds"] = time.perf_counter()-started
    return joint(outputs[0][0], outputs[1][0]), joint(outputs[0][1], outputs[1][1]), info


def regime_alarm_archive(rows: pd.DataFrame, p: np.ndarray, thresholds,
                         events: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    """Strict pre-onset, correct direction, one alarm per 7 days, one-to-one hits.

    Already matched events cannot validate a second alarm. Thresholds come only
    from preceding calibration data; NaN means an explicitly disabled policy.
    """
    confirmed = events.loc[events.status.eq("CONFIRMED")].sort_values("onset_time")
    times = pd.DatetimeIndex(rows.time).as_unit("ns").asi8
    q = rows.q.to_numpy()
    scoreable = []
    for event in confirmed.itertuples():
        lo = np.searchsorted(times, event.onset_time.value-pd.Timedelta(days=7).value)
        hi = np.searchsorted(times, event.onset_time.value, side="left")
        if np.any(q[lo:hi] != event.side):
            scoreable.append(event.onset_i)
    opportunity = confirmed.loc[confirmed.onset_i.isin(scoreable)].copy()
    allowed = p[:, 1:].copy()
    allowed[q == 1, 0] = -1.
    allowed[q == -1, 1] = -1.
    direction = np.argmax(allowed, axis=1)
    scores = allowed[np.arange(len(rows)), direction]
    limits = np.broadcast_to(np.asarray(thresholds, float), len(rows))
    selected = np.flatnonzero(np.isfinite(limits) & (scores >= limits))
    used, archive, last = set(), [], None
    for i in selected:
        when, side = rows.time.iloc[i], (1 if direction[i] == 0 else -1)
        if last is not None and when < last+pd.Timedelta(days=7):
            continue
        last = when
        # Labels name the FIRST future onset into a different current state.
        # Do not skip a wrong-direction/already-used first event to claim a
        # later convenient event in the same warning window.
        candidates = opportunity.loc[opportunity.side.ne(q[i]) & opportunity.onset_time.gt(when) &
                                    opportunity.onset_time.le(when+pd.Timedelta(days=7))]
        event = None if candidates.empty else candidates.iloc[0]
        if event is not None and (event.side != side or int(event.onset_i) in used):
            event = None
        key = None if event is None else int(event.onset_i)
        if key is not None:
            used.add(key)
        archive.append({"time": str(when), "side": side, "score": float(scores[i]), "event_id": key,
                        "onset_time": None if event is None else str(event.onset_time),
                        "lead_days": None if event is None else int((event.onset_time-when).days)})
    return archive, opportunity


def regime_thresholds(rows: pd.DataFrame, p: np.ndarray, events: pd.DataFrame) -> dict:
    """Two declared operating points; no thresholds are selected on outer tests."""
    options = []
    for threshold in REGIME_THRESHOLDS:
        alarms, opportunities = regime_alarm_archive(rows, p, threshold, events)
        n, hit = len(alarms), sum(a["event_id"] is not None for a in alarms)
        precision, recall = hit/max(n, 1), hit/max(len(opportunities), 1)
        options.append((threshold, n, precision, recall, 2*precision*recall/max(precision+recall, 1e-12)))
    balanced = [r for r in options if r[1] >= 3]
    high = [r for r in options if r[1] >= 5 and r[2] >= .8]
    return {"balanced": max(balanced, key=lambda r: (r[4], r[2], r[0]))[0] if balanced else None,
            "precision80": max(high, key=lambda r: (r[3], r[2], r[0]))[0] if high else None}


def regime_event_intervals(rows: pd.DataFrame, alarms: list[dict], events: pd.DataFrame) -> dict:
    """Calendar block bootstrap envelopes (28/56/84d), never IID warning rows."""
    if rows.empty:
        return {}
    end = max(rows.time.max(), events.onset_time.max()) if len(events) else rows.time.max()
    calendar = pd.date_range(rows.time.min(), end, freq="D")
    values = np.zeros((len(calendar), 6))
    def day(t):
        return int((pd.Timestamp(t)-calendar[0]).days)
    matched = {a["event_id"] for a in alarms if a["event_id"] is not None}
    for a in alarms:
        j = day(a["time"])
        values[j, :2] += [a["event_id"] is not None, 1]
    for e in events.itertuples():
        values[day(e.onset_time), 2:4] += [e.onset_i in matched, 1]
    for t in rows.time:
        values[day(t), 4] += 1
    n = len(values)
    prefix = np.vstack([np.zeros(6), np.cumsum(np.concatenate([values, values]), axis=0)])
    output = {k: [] for k in ("precision", "recall", "coverage")}
    for starts, requested in zip(_calendar_block_starts(n), (28, 56, 84)):
        length = min(requested, n)
        lengths = np.full(starts.shape[1], length)
        lengths[-1] = n-(starts.shape[1]-1)*length
        sums = (prefix[starts+lengths]-prefix[starts]).sum(axis=1)
        for label, numerator, denominator in (("precision", 0, 1), ("recall", 2, 3), ("coverage", 1, 4)):
            good = sums[:, denominator] > 0
            stat = sums[good, numerator]/sums[good, denominator]
            ci = np.quantile(stat, [.025, .975]).tolist() if len(stat) >= 900 else [None, None]
            output[label].append({"block_days": length, "ci95": ci, "valid_draws": len(stat)})
    return {k: {"ci95": [min(r["ci95"][0] for r in v), max(r["ci95"][1] for r in v)]
                        if all(r["ci95"][0] is not None for r in v) else [None, None],
                "sensitivity": v} for k, v in output.items()}


def regime_metrics(rows: pd.DataFrame, p: np.ndarray, thresholds, events: pd.DataFrame,
                   *, intervals: bool = True, exposure_days: int | None = None) -> dict:
    from sklearn.metrics import f1_score, confusion_matrix, average_precision_score
    mature = rows.y.notna().to_numpy()
    censored = int((~mature).sum())
    thresholds = np.broadcast_to(np.asarray(thresholds, float), len(rows))[mature]
    rows, p = rows.loc[mature].reset_index(drop=True), p[mature]
    if rows.empty:
        return {"N": 0}
    archive, opportunity = regime_alarm_archive(rows, p, thresholds, events)
    hits = sum(a["event_id"] is not None for a in archive)
    cm = confusion_matrix(rows.y, H1A_CLASSES[p.argmax(axis=1)], labels=H1A_CLASSES)
    recalls = np.divide(cm.diagonal(), cm.sum(axis=1), out=np.zeros(3), where=cm.sum(axis=1)>0)
    brier, ll = forecast_losses(rows.y, p)
    days = exposure_days if exposure_days is not None else (rows.time.max()-rows.time.min()).days+1
    leads = [a["lead_days"] for a in archive if a["event_id"] is not None]
    out = {"N": len(rows), "censored_forecasts": censored, "target_counts": {str(k): int(rows.y.eq(k).sum()) for k in H1A_CLASSES},
           "accuracy": float(np.trace(cm)/len(rows)), "macro_F1": float(f1_score(rows.y, H1A_CLASSES[p.argmax(axis=1)], labels=H1A_CLASSES, average="macro", zero_division=0)),
           "balanced_accuracy": float(recalls.mean()), "class_recall": recalls.tolist(), "confusion_matrix": cm.tolist(),
           "brier": float(brier.mean()), "log_loss": float(ll.mean()),
           "event_AP": float(average_precision_score(rows.y.ne(0), p[:, 1:].sum(axis=1))) if rows.y.ne(0).any() else None,
           "scoreable_events": len(opportunity), "scoreable_up": int(opportunity.side.eq(1).sum()),
           "scoreable_down": int(opportunity.side.eq(-1).sum()), "alarms": len(archive), "detected_events": hits,
           "precision": hits/len(archive) if archive else None, "recall": hits/len(opportunity) if len(opportunity) else None,
           "coverage": len(archive)/len(rows), "exposure_days": days, "false_alarms": len(archive)-hits,
           "false_alarms_per_year": (len(archive)-hits)*365.25/days,
           "median_lead_days": float(np.median(leads)) if leads else None,
           "intervals": regime_event_intervals(rows, archive, opportunity) if intervals else {},
           "alarm_archive": archive}
    return out


def regime_walk_forward(d: pd.DataFrame, events: pd.DataFrame, threads: int) -> tuple[pd.DataFrame, dict]:
    """One fixed capacity; eight outer folds, independent stopping/calibration."""
    predictions, folds = [], []
    for start in pd.date_range("2021-01-01", DEV_END, freq="6MS", inclusive="left", tz="UTC"):
        tr, va, ca, te = regime_splits(d, start)
        if te.empty or len(tr) < 30:
            raise ValueError(f"Insufficient chronological history at {start}; no silently omitted fold")
        record = {"start": str(start), "train_N": len(tr), "stopping_N": len(va), "calibration_N": len(ca), "test_N": len(te), "test_censored_N": int(te.y.isna().sum()),
                  "train_latest_maturity": str(tr.maturity_time.max()),
                  "stopping_start": str(va.time.min()), "stopping_latest_maturity": str(va.maturity_time.max()),
                  "calibration_start": str(ca.time.min()), "calibration_latest_maturity": str(ca.maturity_time.max()), "models": {}}
        for name in ("no_event", "prior", "logistic", "xgb"):
            pc, pt, fit_info = regime_probabilities(tr, va, ca, te, name, threads)
            policy_started = time.perf_counter()
            thresholds = {"balanced": None, "precision80": None} if name == "no_event" else regime_thresholds(ca, pc, events)
            fit_info["policy_selection_seconds"] = time.perf_counter()-policy_started
            part = te[["time", "q", "y", "maturity_time", "target_onset"]].copy()
            part["fold"], part["model"] = str(start.date()), name
            part[["none", "up", "down"]] = pt
            for policy, threshold in thresholds.items():
                part[policy] = np.nan if threshold is None else threshold
            record["models"][name] = {**fit_info, "thresholds": thresholds}
            predictions.append(part)
        folds.append(record)
    forecasts = pd.concat(predictions, ignore_index=True)
    gap = (d.maturity_time-d.time).iloc[0]
    calendar_end = min(pd.Timestamp(DEV_END, tz="UTC"), d.time.max()-gap+pd.Timedelta(days=1))
    exposure = (calendar_end-pd.Timestamp("2021-01-01", tz="UTC")).days
    results = {}
    for name, group in forecasts.groupby("model", sort=False):
        group = group.sort_values("time").reset_index(drop=True)
        p = group[["none", "up", "down"]].to_numpy()
        results[name] = {policy: regime_metrics(group, p, group[policy].to_numpy(), events, exposure_days=exposure)
                         for policy in ("balanced", "precision80")}
        results[name]["by_year"] = {}
        for year, yearly in group.groupby(group.time.dt.year):
            yp = yearly[["none", "up", "down"]].to_numpy()
            year_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
            year_end = min(calendar_end, pd.Timestamp(f"{year+1}-01-01", tz="UTC"))
            results[name]["by_year"][str(year)] = regime_metrics(yearly, yp, yearly.balanced.to_numpy(), events, intervals=False, exposure_days=(year_end-year_start).days)
    return forecasts, {"folds": folds, "models": results}


def regime_research_run(x: pd.DataFrame, readiness: dict, args) -> None:
    """Frozen exploratory protocol; never accesses 2025+ or trades on future labels."""
    import sklearn
    import xgboost
    if (x.time >= pd.Timestamp(DEV_END, tz="UTC")).any():
        raise ValueError("Regime research cannot access sealed 2025+ observations")
    started = time.perf_counter()
    specification = {"study": "ONEFILE-REGIME-XGB-DEV-V1", "input_window": [DEV_START, DEV_END],
                     "evaluation": ["2021-01-01", DEV_END], "features": REGIME_FEATURES,
                     "event": "q crossing including reversal; >=3 of next5 q retain direction; failed crossings negative",
                     "target": "first confirmed onset within next1..7 days, different from current q",
                     "legacy": "unchanged neutral>=10d and >=8/10 confirmation; four-year extension starts2021",
                     "purge": "fixed target information window12d(new)/17d(legacy), plus1d embargo",
                     "folds": "expanding fit, separate preceding3mo stopping +3mo calibration/policy, six-month tests",
                     "capacity": "XGB hist CPU depth2 child10 eta.05 L2=10 L1=.1 gamma.1 bins64 subsample.8 columns.8; <=256 rounds,20 earlystop",
                     "imbalance": "capped sqrt train-only class weights; inverse event multiplicity; sigmoid calibration, >=5 per class else identity",
                     "thresholds": list(REGIME_THRESHOLDS), "policies": {"balanced": "max calibration eventF1 with>=3alarms", "precision80": "max calibration recall with precision>=.8 and>=5alarms; otherwise disabled"},
                     "matching": "strict signed lead1..7d, 7d alarm cooldown across folds; one-to-one",
                     "coverage": "emitted alarms / eligible mature forecast origins; recall covers every scoreable confirmed event",
                     "CI": "1000 circular calendar-block draws at28/56/84d; widest95% envelope",
                     "trial_status": "definition chosen after previous DEV findings; historical exploratory, never fresh holdout",
                     "success_gate": "precision CI lower>=.8, recall CI lower>=.3, >=50alarms,>=30events,>=12 28d event clusters, positive paired Brier gain CI in>=3years; otherwise NO_EDGE",
                     "sealed_data_used": False, "trading_PnL": "not measured; forecasting evaluation only"}
    identity = {"specification": specification, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "input_sha256": readiness["file_sha256"], "threads": args.xgb_threads,
                "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__}}
    path = Path(args.protocol)
    if path.exists():
        if json.loads(path.read_text())["identity"] != identity:
            raise ValueError("Regime protocol/code/data mismatch; retain prior attempt and choose a new protocol path")
    else:
        atomic_write(path, lambda f: f.write(json.dumps({"identity": identity, "frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat()}, indent=2)+"\n"))
    ts = time.perf_counter()
    daily = daily_market(x)
    timings = {"daily_features_seconds": time.perf_counter()-ts}
    ts = time.perf_counter()
    enriched = regime_features(daily)
    timings["extra_features_seconds"] = time.perf_counter()-ts
    timings["features_seconds"] = timings["daily_features_seconds"]+timings["extra_features_seconds"]
    ts = time.perf_counter()
    strict, strict_events = onset_labels(enriched)
    strict["maturity_time"] = strict.time+pd.Timedelta(days=17)
    strict["forecast_eligible"] = strict.risk
    expanded, expanded_events = regime_labels(enriched)
    expanded["forecast_eligible"] = expanded.q.notna()
    timings["labels_seconds"] = time.perf_counter()-ts
    ts = time.perf_counter()
    old_stage = {key: _PERF_TIMINGS.get("h1a."+key, {}).get("total_seconds", 0.) for key in ("training_seconds", "inference_seconds")}
    legacy_forecasts, legacy_result = h1a_experiment(strict, strict_events, evaluation_start="2021-01-01", inner_start="2020-07-01")
    timings["legacy_original_seconds"] = time.perf_counter()-ts
    for key, before in old_stage.items():
        timings["legacy_"+key] = _PERF_TIMINGS.get("h1a."+key, {}).get("total_seconds", 0.)-before
    scored = legacy_forecasts.loc[legacy_forecasts.y.notna()].merge(daily[["time", "q"]], on="time").sort_values("time")
    legacy_metric = regime_metrics(scored, scored[[f"M2_{k}" for k in ("none", "up", "down")]].to_numpy(), scored.M2_threshold.to_numpy(), strict_events, exposure_days=legacy_result["exposure_days"])
    runs, summaries = {}, {}
    for task, frame, events in (("strict", strict, strict_events), ("expanded", expanded, expanded_events)):
        ts = time.perf_counter()
        runs[task], summaries[task] = regime_walk_forward(frame, events, args.xgb_threads)
        timings[task+"_walk_forward_seconds"] = time.perf_counter()-ts
    paired = runs["expanded"].loc[runs["expanded"].y.notna()].pivot(index="time", columns="model", values=["none", "up", "down"])
    y = runs["expanded"].loc[runs["expanded"].model.eq("xgb"), ["time", "y"]].set_index("time").reindex(paired.index).y
    losses = pd.DataFrame({"time": paired.index})
    for model in ("prior", "logistic", "xgb"):
        pp = np.column_stack([paired[(k, model)].to_numpy() for k in ("none", "up", "down")])
        losses[model] = forecast_losses(y.to_numpy(), pp)[0]
    gains = {}
    for model in ("prior", "logistic"):
        losses["gain"] = losses[model]-losses.xgb
        gains[model] = block_interval(losses, ["gain"])
    chosen = summaries["expanded"]["models"]["xgb"]["precision80"]
    ci_p = chosen["intervals"]["precision"]["ci95"][0]
    ci_r = chosen["intervals"]["recall"]["ci95"][0]
    opportunities = expanded_events.loc[expanded_events.status.eq("CONFIRMED") & expanded_events.onset_time.ge(pd.Timestamp("2021-01-01", tz="UTC"))]
    clusters = opportunities.onset_time.dt.floor("28D").nunique()
    annual_positive = 0
    for _, group in losses.groupby(losses.time.dt.year):
        g = group.assign(gain=group.logistic-group.xgb)
        ci = block_interval(g, ["gain"])["ci95"]
        annual_positive += ci[0] is not None and ci[0] > 0
    passed = (chosen["alarms"] >= 50 and chosen["scoreable_events"] >= 30 and clusters >= 12 and
              ci_p is not None and ci_p >= .8 and ci_r is not None and ci_r >= .3 and
              gains["logistic"]["ci95"][0] > 0 and annual_positive >= 3)
    result = {"identity": identity, "readiness": readiness, "legacy_original": {"metrics": legacy_metric, "result": legacy_result},
              "comparisons": summaries, "paired_brier_gain": gains, "positive_years_with_CI": annual_positive,
              "event_clusters_28d": clusters, "timings": timings, "verdict": "PROMISING_EXPLORATORY_REQUIRES_INDEPENDENT_VALIDATION" if passed else "NO_EDGE",
              "independent_validation": False, "sealed_data_used": False, "economic_edge_tested": False}
    # Retain both failures and censored events, as well as every mature forecast.
    output = pd.concat([f.assign(task=k) for k, f in runs.items()], ignore_index=True)
    legacy_export = legacy_forecasts.merge(daily[["time", "q"]], on="time").rename(
        columns={"label_maturity_time": "maturity_time", "M2_none": "none", "M2_up": "up", "M2_down": "down", "M2_threshold": "balanced"})
    legacy_export["model"], legacy_export["task"], legacy_export["precision80"] = "legacy_original", "strict", np.nan
    output = pd.concat([output, legacy_export[output.columns]], ignore_index=True)
    event_output = pd.concat([strict_events.assign(task="strict"), expanded_events.assign(task="expanded")], ignore_index=True)
    atomic_write(Path(args.forecasts), lambda f: output.to_csv(f, index=False))
    atomic_write(Path(args.onsets), lambda f: event_output.to_csv(f, index=False))
    atomic_write(Path(args.out), lambda f: expanded.to_csv(f, index=False))
    result["timings"]["total_research_seconds"] = time.perf_counter()-started
    result["outputs_sha256"] = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (args.protocol, args.forecasts, args.onsets, args.out)}
    atomic_write(Path(args.evidence), lambda f: f.write(json.dumps(result, indent=2, allow_nan=False)+"\n"))
    _PERF_CONTEXT["regime_timings"] = result["timings"]
    print(json.dumps({"verdict": result["verdict"], "legacy": legacy_metric,
                      "expanded_xgb": summaries["expanded"]["models"]["xgb"], "timings": result["timings"]}, indent=2))


# Regime forecasting v3: pooled core-5 DEV, same expanded labels as v2. The
# rolling z20/er20 state has a KNOWN base effect: returns that leave the window
# in the next 1..7 days are already observed, so a random-walk crossing
# probability is causal. Shared harness: every model sees identical rows,
# labels, purged folds, history-only thresholds and alarm budget.
REGIME3_TUNE_END = "2023-01-01"   # model choices may look at 2021-2022 outer folds only
REGIME3_STRUCTURAL = ["rw_up", "rw_down", "need_up", "need_down", "drop_r7", "er_proj"]
REGIME3_BASE = REGIME_FEATURES + REGIME3_STRUCTURAL
REGIME3_BUDGET = 12.  # alarms per asset-year for the fixed-budget policy
REGIME3_MIN_HISTORY_ALARMS = 10
# Worker-owned model configs. Changing them must not touch the harness above.
REGIME3_LOGIT = {"features": REGIME3_BASE, "C": 1., "class_weight": "sqrt"}
REGIME3_XGB = {"features": REGIME3_STRUCTURAL + ["q", "z20", "er20"], "max_depth": 2, "min_child_weight": 2., "learning_rate": .05,
               "reg_lambda": 10., "reg_alpha": .1, "gamma": .1, "subsample": .8, "colsample_bytree": .8,
               "max_bin": 64, "rounds": 30, "class_weight": "sqrt", "base_margin": True}


def regime3_features(d: pd.DataFrame) -> pd.DataFrame:
    """Random-walk crossing odds from already-observed drop-off returns (prefix causal)."""
    from scipy.stats import norm
    d = regime_features(d)
    sigma = d.rv20.clip(lower=1e-12)
    up = np.zeros(len(d)); down = np.zeros(len(d))
    need_up = np.full(len(d), np.inf); need_down = np.full(len(d), np.inf)
    for k in range(1, REGIME_HORIZON+1):
        kept = d.r1.rolling(20-k).sum()
        scale = np.sqrt(d.r1.pow(2).rolling(20-k).sum()+k*sigma**2)
        nu = ((1.25*scale-kept)/(sigma*np.sqrt(k))).to_numpy()
        nd = ((1.25*scale+kept)/(sigma*np.sqrt(k))).to_numpy()
        need_up, need_down = np.fmin(need_up, nu), np.fmin(need_down, nd)
        up, down = np.fmax(up, norm.sf(nu)), np.fmax(down, norm.sf(nd))
    invalid = d.rv20.isna().to_numpy()
    d["rw_up"] = np.where(invalid, np.nan, np.where(d.q.eq(1), 0., up))
    d["rw_down"] = np.where(invalid, np.nan, np.where(d.q.eq(-1), 0., down))
    d["need_up"] = np.where(invalid, np.nan, np.clip(need_up, -10, 10))
    d["need_down"] = np.where(invalid, np.nan, np.clip(need_down, -10, 10))
    d["drop_r7"] = (d.r20-d.r1.rolling(20-REGIME_HORIZON).sum())/(sigma*np.sqrt(REGIME_HORIZON))
    kept = d.r1.rolling(20-REGIME_HORIZON).sum()
    d["er_proj"] = kept.abs()/(d.r1.abs().rolling(20-REGIME_HORIZON).sum()+REGIME_HORIZON*d.r1.abs().rolling(20).mean()+1e-12)
    return d


def regime3_frame(assets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-asset daily labels; event ids offset per asset so pooled matching is one-to-one."""
    frames, events = [], []
    for k, (symbol, a) in enumerate(sorted(assets.items())):
        d, e = regime_labels(regime3_features(daily_market(a)))
        d["symbol"], e["symbol"] = symbol, symbol
        e["onset_i"] = e.onset_i+100_000*k
        d["event_key"] = np.where(d.target_onset.notna(), symbol+"|"+d.target_onset.astype(str), "")
        frames.append(d); events.append(e)
    return pd.concat(frames, ignore_index=True), pd.concat(events, ignore_index=True)


def regime3_splits(d: pd.DataFrame, start: pd.Timestamp) -> tuple[pd.DataFrame, ...]:
    """Same 12d label window + 1d embargo at every boundary, applied to all assets by time."""
    stop_start, cal_start = start-pd.DateOffset(months=6), start-pd.DateOffset(months=3)
    feature_ok = np.isfinite(d[REGIME3_BASE]).all(axis=1) & d.q.notna()
    eligible = d.y.notna() & feature_ok
    mature = d.maturity_time+pd.Timedelta(days=1)
    return (d.loc[eligible & (mature < stop_start)], d.loc[eligible & (d.time >= stop_start) & (mature < cal_start)],
            d.loc[eligible & (d.time >= cal_start) & (mature < start)],
            d.loc[feature_ok & (d.time >= start) & (d.time < start+pd.DateOffset(months=6))])


def _regime3_weights(rows: pd.DataFrame, y: np.ndarray, scheme: str) -> np.ndarray:
    counts = np.bincount(y, minlength=2).clip(min=1)
    w = {"none": np.ones(len(y)), "sqrt": np.minimum(3., np.sqrt(len(y)/(2*counts[y]))),
         "balanced": len(y)/(2*counts[y])}[scheme]
    key = rows.event_key.to_numpy()
    event = key != ""
    if event.any():
        w = w.copy()
        w[event] /= pd.Series(key[event]).map(pd.Series(key[event]).value_counts()).to_numpy(float)
    return w/w.mean()


def regime3_fit(kind: str, train: pd.DataFrame, stopping: pd.DataFrame, calibration: pd.DataFrame,
                threads: int = 1) -> tuple[Callable[[pd.DataFrame], np.ndarray], dict]:
    """Return predict(rows)->[none, up, down]. Event/direction heads, sigmoid calibration later.

    Fit = train+stopping (stopping only picks XGB rounds); calibration fits one
    sigmoid per head on the later segment. Outer test rows never enter here.
    """
    from scipy.special import logit
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    refit = pd.concat([train, stopping])
    info = {"heads": {}}
    if kind == "no_event":
        return lambda rows: np.tile([1., 0., 0.], (len(rows), 1)), info
    if kind == "prior":
        prior = (np.array([refit.y.eq(k).sum() for k in H1A_CLASSES])+1)/(len(refit)+3)
        return lambda rows: np.tile(prior, (len(rows), 1)), info
    if kind == "structural":
        def structural(rows):
            up, down = rows.rw_up.to_numpy(), rows.rw_down.to_numpy()
            total = np.maximum(1., up+down)
            return np.column_stack([1-(up+down)/total, up/total, down/total])
        return structural, info
    if kind == "xgb":
        # Two direct side heads (up: q!=+1, down: q!=-1), each a residual on the structural
        # random-walk logit (base_margin). Sigmoid calibration per head on the later segment.
        import xgboost as xgb
        cfg = REGIME3_XGB; cols = cfg["features"]
        params = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "device": "cpu",
                  "nthread": threads, "seed": SEED, "max_depth": cfg["max_depth"], "min_child_weight": cfg["min_child_weight"],
                  "eta": cfg["learning_rate"], "lambda": cfg["reg_lambda"], "alpha": cfg["reg_alpha"], "gamma": cfg["gamma"],
                  "subsample": cfg["subsample"], "colsample_bytree": cfg["colsample_bytree"], "max_bin": cfg["max_bin"]}
        def margin(rows, col):
            if not cfg["base_margin"]:
                return np.zeros(len(rows))
            return logit(np.clip(rows[col].to_numpy(float), 1e-6, 1-1e-6))
        def dmatrix(rows, target, col, weights=True):
            w = _regime3_weights(rows, target, cfg["class_weight"]) if weights else None
            return xgb.DMatrix(rows[cols].to_numpy(np.float32), label=target, weight=w, base_margin=margin(rows, col))
        heads = []
        for side, col, name in ((1, "rw_up", "up"), (-1, "rw_down", "down")):
            pick = lambda r, s=side: r.loc[r.q.ne(s)]
            target = lambda r, s=side: r.y.eq(s).to_numpy(int)
            tr, va, fit, ca = pick(train), pick(stopping), pick(refit), pick(calibration)
            detail = {"fit_rows": len(fit), "fit_class_counts": np.bincount(target(fit), minlength=2).tolist()}
            if len(fit) < 20 or len(np.unique(target(fit))) < 2:
                prior = (target(fit).sum()+1)/(len(fit)+2)
                raw = lambda rows, prior=prior: np.full(len(rows), prior)
                detail["fallback"] = "smoothed training prior"
            else:
                rounds = cfg["rounds"]
                mono = "(" + ",".join("1" if c == col else "0" for c in cols) + ")"
                booster = xgb.train({**params, "monotone_constraints": mono}, dmatrix(fit, target(fit), col), num_boost_round=rounds)
                raw = lambda rows, b=booster, c=col: b.predict(xgb.DMatrix(rows[cols].to_numpy(np.float32), base_margin=margin(rows, c)))
                detail["rounds"] = rounds
            cy = target(ca)
            calibrate = None
            if len(cy) and np.bincount(cy, minlength=2).min() >= 5:
                calibrate = LogisticRegression(C=1., solver="lbfgs").fit(logit(np.clip(raw(ca), 1e-6, 1-1e-6))[:, None], cy)
            detail["calibration"] = "sigmoid on later segment" if calibrate is not None else "identity (<5 per class)"
            info["heads"][name] = detail
            heads.append(lambda rows, raw=raw, cal=calibrate: raw(rows) if cal is None else
                         cal.predict_proba(logit(np.clip(raw(rows), 1e-6, 1-1e-6))[:, None])[:, 1])
        def predict_xgb(rows):
            if rows.empty:
                return np.empty((0, 3))
            up = np.where(rows.q.eq(1), 0., heads[0](rows)); down = np.where(rows.q.eq(-1), 0., heads[1](rows))
            total = np.maximum(1., up+down)
            return np.column_stack([1-(up+down)/total, up/total, down/total])
        return predict_xgb, info
    heads = []
    for head in ("event", "direction"):
        pick = (lambda r: r) if head == "event" else (lambda r: r.loc[r.y.ne(0)])
        target = (lambda r: r.y.ne(0).to_numpy(int)) if head == "event" else (lambda r: r.y.eq(1).to_numpy(int))
        tr, va, fit, ca = pick(train), pick(stopping), pick(refit), pick(calibration)
        detail = {"fit_rows": len(fit), "fit_class_counts": np.bincount(target(fit), minlength=2).tolist()}
        if len(fit) < 20 or len(np.unique(target(fit))) < 2:
            prior = (target(fit).sum()+1)/(len(fit)+2)
            raw = lambda rows, prior=prior: np.full(len(rows), prior)
            detail["fallback"] = "smoothed training prior"
        elif kind == "logistic":
            cfg = REGIME3_LOGIT; cols = cfg["features"]
            scaler = StandardScaler().fit(fit[cols].to_numpy())
            model = LogisticRegression(C=cfg["C"], max_iter=500, solver="lbfgs")
            with threadpool_limits(limits=1):
                model.fit(scaler.transform(fit[cols].to_numpy()), target(fit),
                          sample_weight=_regime3_weights(fit, target(fit), cfg["class_weight"]))
            raw = lambda rows, m=model, s=scaler, c=cols: m.predict_proba(s.transform(rows[c].to_numpy()))[:, 1]
            detail["coef"] = dict(zip(cols, np.round(model.coef_[0], 4).tolist()))
        else:
            raise ValueError(f"Unknown regime3 model {kind}")
        cy = target(ca)
        calibrate = None
        if len(cy) and np.bincount(cy, minlength=2).min() >= 5:
            calibrate = LogisticRegression(C=1., solver="lbfgs").fit(logit(np.clip(raw(ca), 1e-6, 1-1e-6))[:, None], cy)
        detail["calibration"] = "sigmoid on later segment" if calibrate is not None else "identity (<5 per class)"
        heads.append(lambda rows, raw=raw, cal=calibrate: raw(rows) if cal is None else
                     cal.predict_proba(logit(np.clip(raw(rows), 1e-6, 1-1e-6))[:, None])[:, 1])
        info["heads"][head] = detail
    def predict(rows):
        if rows.empty:
            return np.empty((0, 3))
        e, u = heads[0](rows), heads[1](rows)
        return np.column_stack([1-e, e*u, e*(1-u)])
    return predict, info


def _regime3_candidates(rows: pd.DataFrame, p: np.ndarray, events: pd.DataFrame) -> dict:
    """Per-row inputs of regime_alarm_archive, computed once (threshold-independent)."""
    out = {"symbol": [], "time": [], "score": [], "side": [], "event": [], "event_side": [], "onset": [], "opp": []}
    for symbol, idx in rows.groupby("symbol", sort=True).indices.items():
        sub = rows.iloc[idx]
        times = pd.DatetimeIndex(sub.time).as_unit("ns").asi8
        q = sub.q.to_numpy()
        conf = events.loc[events.status.eq("CONFIRMED") & events.symbol.eq(symbol)].sort_values("onset_time")
        week = pd.Timedelta(days=7).value
        onset = pd.DatetimeIndex(conf.onset_time).as_unit("ns").asi8
        lo, hi = np.searchsorted(times, onset-week), np.searchsorted(times, onset, side="left")
        keep = np.array([np.any(q[a:b] != s) for a, b, s in zip(lo, hi, conf.side)], bool)
        opp = conf.loc[keep] if len(conf) else conf
        o_t, o_s, o_i = pd.DatetimeIndex(opp.onset_time).as_unit("ns").asi8, opp.side.to_numpy(), opp.onset_i.to_numpy()
        allowed = p[idx, 1:].copy()
        allowed[q == 1, 0] = -1.
        allowed[q == -1, 1] = -1.
        direction = np.argmax(allowed, axis=1)
        event = np.full(len(idx), -1); event_side = np.zeros(len(idx), int); event_t = np.zeros(len(idx), np.int64)
        for j in range(len(idx)):  # first opportunity in (t, t+7d] into a different state
            a = np.searchsorted(o_t, times[j], side="right"); b = np.searchsorted(o_t, times[j]+week, side="right")
            hit = np.flatnonzero(o_s[a:b] != q[j])
            if len(hit):
                k = a+hit[0]; event[j], event_side[j], event_t[j] = o_i[k], o_s[k], o_t[k]
        out["symbol"].append(np.full(len(idx), symbol, object)); out["time"].append(times)
        out["score"].append(allowed[np.arange(len(idx)), direction]); out["side"].append(np.where(direction == 0, 1, -1))
        out["event"].append(event); out["event_side"].append(event_side); out["onset"].append(event_t); out["opp"].append(opp)
    return {k: (pd.concat(v, ignore_index=True) if k == "opp" else np.concatenate(v)) for k, v in out.items()} if out["time"] else None


def regime3_alarms(rows: pd.DataFrame, p: np.ndarray, thresholds, events: pd.DataFrame,
                   cand: dict | None = None) -> tuple[list[dict], pd.DataFrame]:
    """v2 strict matching (regime_alarm_archive semantics) per asset, cooldown per asset; pooled."""
    cand = cand if cand is not None else _regime3_candidates(rows, p, events)
    if cand is None:
        return [], events.iloc[:0]
    limits = np.broadcast_to(np.asarray(thresholds, float), len(cand["time"]))
    selected = np.flatnonzero(np.isfinite(limits) & (cand["score"] >= limits))
    week = pd.Timedelta(days=7).value
    archive, last, used = [], {}, set()
    for i in selected:
        symbol, when = cand["symbol"][i], cand["time"][i]
        if symbol in last and when < last[symbol]+week:
            continue
        last[symbol] = when
        key = int(cand["event"][i])
        ok = key >= 0 and cand["event_side"][i] == cand["side"][i] and key not in used
        if ok:
            used.add(key)
        archive.append({"time": str(pd.Timestamp(when, tz="UTC")), "symbol": symbol, "side": int(cand["side"][i]),
                        "score": float(cand["score"][i]), "event_id": key if ok else None,
                        "lead_days": int((cand["onset"][i]-when)//pd.Timedelta(days=1).value) if ok else None})
    return archive, cand["opp"]


def regime3_policy(rows: pd.DataFrame, p: np.ndarray, events: pd.DataFrame) -> dict:
    """Thresholds from PAST out-of-sample scores only (prior folds + calibration segment)."""
    asset_years = max(rows.groupby("symbol").time.agg(lambda t: (t.max()-t.min()).days+1).sum()/365.25, 1e-9)
    scores = np.max(p[:, 1:], axis=1)
    grid = np.unique(np.quantile(scores, np.linspace(.5, .999, 120))) if len(scores) else []
    options = []
    cand = _regime3_candidates(rows, p, events)
    for t in grid:
        alarms, opp = regime3_alarms(rows, p, t, events, cand)
        n, hit = len(alarms), sum(a["event_id"] is not None for a in alarms)
        options.append({"threshold": float(t), "alarms": n, "hits": hit, "precision": hit/n if n else 0.,
                        "recall": hit/max(len(opp), 1), "rate": n/asset_years})
    high = [o for o in options if o["alarms"] >= REGIME3_MIN_HISTORY_ALARMS and o["precision"] >= .8]
    budget = [o for o in options if o["rate"] <= REGIME3_BUDGET and o["alarms"] >= 1]
    return {"precision80": min(high, key=lambda o: o["threshold"])["threshold"] if high else None,
            "budget": min(budget, key=lambda o: o["threshold"])["threshold"] if budget else None,
            "history_rows": len(rows)}


def regime3_walk_forward(d: pd.DataFrame, events: pd.DataFrame, kinds: tuple[str, ...], scope: str,
                         threads: int = 1) -> tuple[pd.DataFrame, list[dict]]:
    """Expanding six-month outer folds; 'tune' stops before 2023 (2023-2024 untouched)."""
    end = REGIME3_TUNE_END if scope == "tune" else DEV_END
    predictions, folds = [], []
    history = {k: [] for k in kinds}
    for start in pd.date_range("2021-01-01", end, freq="6MS", inclusive="left", tz="UTC"):
        tr, va, ca, te = regime3_splits(d, start)
        record = {"start": str(start.date()), "train_N": len(tr), "stopping_N": len(va), "calibration_N": len(ca), "test_N": len(te), "models": {}}
        for kind in kinds:
            ts = time.perf_counter()
            predict, info = regime3_fit(kind, tr, va, ca, threads)
            info["fit_seconds"] = time.perf_counter()-ts
            pc, pt = predict(ca), predict(te)
            # History pool: earlier outer predictions already mature before this fold, + calibration.
            past = [h.loc[h.maturity_time+pd.Timedelta(days=1) < start] for h in history[kind]]
            pool = pd.concat(past+[ca.assign(none=pc[:, 0], up=pc[:, 1], down=pc[:, 2])], ignore_index=True)
            pool = pool.loc[pool.y.notna()].sort_values(["symbol", "time"]).reset_index(drop=True)
            ts = time.perf_counter()
            info["policy"] = regime3_policy(pool, pool[["none", "up", "down"]].to_numpy(), events) if kind != "no_event" else {"precision80": None, "budget": None}
            info["policy_seconds"] = time.perf_counter()-ts
            part = te[["symbol", "time", "q", "y", "maturity_time", "target_onset", "event_key"]].copy()
            part[["none", "up", "down"]] = pt
            part["fold"], part["model"] = str(start.date()), kind
            for policy in ("precision80", "budget"):
                part[policy] = np.nan if info["policy"][policy] is None else info["policy"][policy]
            history[kind].append(part)
            predictions.append(part)
            record["models"][kind] = info
        folds.append(record)
    return pd.concat(predictions, ignore_index=True), folds


def _wilson(hits: int, n: int, z: float = 1.96) -> list:
    if not n:
        return [None, None]
    p = hits/n; c = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    centre = (p+z*z/(2*n))/(1+z*z/n)
    return [float(centre-c), float(centre+c)]


def regime3_metrics(rows: pd.DataFrame, events: pd.DataFrame, policy: str) -> dict:
    from sklearn.metrics import average_precision_score
    rows = rows.loc[rows.y.notna()].sort_values(["symbol", "time"]).reset_index(drop=True)
    if rows.empty:
        return {"N": 0}
    p = rows[["none", "up", "down"]].to_numpy()
    archive, opp = regime3_alarms(rows, p, rows[policy].to_numpy(), events)
    hits = [a for a in archive if a["event_id"] is not None]
    brier, _ = forecast_losses(rows.y, p)
    event = rows.y.ne(0)
    weeks = {pd.Timestamp(a["time"]).floor("7D") for a in hits}
    return {"N": len(rows), "positives": int(event.sum()), "scoreable_events": len(opp),
            "alarms": len(archive), "hits": len(hits), "false_alarms": len(archive)-len(hits),
            "precision": len(hits)/len(archive) if archive else None, "precision_wilson95": _wilson(len(hits), len(archive)),
            "recall": len(hits)/len(opp) if len(opp) else None, "distinct_hit_weeks": len(weeks),
            "event_AP": float(average_precision_score(event, p[:, 1:].sum(axis=1))) if event.any() else None,
            "base_rate": float(event.mean()), "brier": float(brier.mean()),
            "median_lead_days": float(np.median([a["lead_days"] for a in hits])) if hits else None,
            "alarm_archive": archive}


def regime3_ap_gain(a: pd.DataFrame, b: pd.DataFrame, draws: int = 500) -> dict:
    """Paired 28-day calendar-block bootstrap of event-AP(a)-AP(b); assets share dates."""
    from sklearn.metrics import average_precision_score
    key = ["symbol", "time"]
    m = a.loc[a.y.notna(), key+["y", "up", "down"]].merge(b.loc[b.y.notna(), key+["up", "down"]], on=key, suffixes=("_a", "_b"))
    y = m.y.ne(0).to_numpy(); sa = (m.up_a+m.down_a).to_numpy(); sb = (m.up_b+m.down_b).to_numpy()
    block = ((m.time-m.time.min()).dt.days//28).to_numpy()
    groups = [np.flatnonzero(block == g) for g in range(block.max()+1)]
    rng = np.random.default_rng(SEED); stats = []
    for _ in range(draws):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        if y[idx].any():
            stats.append(average_precision_score(y[idx], sa[idx])-average_precision_score(y[idx], sb[idx]))
    return {"estimate": float(average_precision_score(y, sa)-average_precision_score(y, sb)),
            "ci95": np.quantile(stats, [.025, .975]).tolist()}


def regime3_report(forecasts: pd.DataFrame, events: pd.DataFrame) -> dict:
    out = {}
    for kind, g in forecasts.groupby("model", sort=False):
        out[kind] = {policy: {k: v for k, v in regime3_metrics(g, events, policy).items() if k != "alarm_archive"}
                     for policy in ("precision80", "budget")}
        out[kind]["by_year"] = {str(y): {k: v for k, v in regime3_metrics(yg, events, "precision80").items() if k != "alarm_archive"}
                                for y, yg in g.groupby(g.time.dt.year)}
        out[kind]["by_asset"] = {s: {k: v for k, v in regime3_metrics(sg, events, "precision80").items() if k != "alarm_archive"}
                                 for s, sg in g.groupby("symbol")}
    base = forecasts.loc[forecasts.model.eq("structural")]
    for kind in ("logistic", "xgb"):
        if kind in out and not base.empty:
            out[kind]["AP_gain_vs_structural"] = regime3_ap_gain(forecasts.loc[forecasts.model.eq(kind)], base)
    return out


def regime3_run(assets: dict[str, pd.DataFrame], args) -> dict:
    """DEV-only pooled core-5 regime v3; --regime3-scope tune never scores 2023-2024."""
    for a in assets.values():
        if (a.time >= pd.Timestamp(DEV_END, tz="UTC")).any():
            raise ValueError("Regime v3 cannot access sealed 2025+ observations")
    started = time.perf_counter()
    d, events = regime3_frame(assets)
    kinds = tuple(args.regime3_models.split(","))
    forecasts, folds = regime3_walk_forward(d, events, kinds, args.regime3_scope, args.xgb_threads)
    report = regime3_report(forecasts, events)
    result = {"scope": args.regime3_scope, "evaluation": ["2021-01-01", REGIME3_TUNE_END if args.regime3_scope == "tune" else DEV_END],
              "assets": sorted(assets), "configs": {"logistic": REGIME3_LOGIT, "xgb": REGIME3_XGB},
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "folds": folds, "models": report,
              "runtime_seconds": time.perf_counter()-started, "sealed_data_used": False}
    if args.forecasts:
        atomic_write(Path(args.forecasts), lambda f: forecasts.to_csv(f, index=False))
    atomic_write(Path(args.evidence), lambda f: f.write(json.dumps(result, indent=2, default=str)+"\n"))
    brief = {k: {p: {m: v[p].get(m) for m in ("alarms", "hits", "precision", "precision_wilson95", "recall", "scoreable_events", "event_AP")}
                 for p in ("precision80", "budget")} for k, v in report.items()}
    print(json.dumps({"scope": args.regime3_scope, "summary": brief, "runtime_seconds": result["runtime_seconds"]}, indent=2, default=str))
    return result


# Per-invocation performance evidence (kept separate from scientific evidence).
_PERF_TIMINGS: dict[str, dict] = {}
_PERF_CONTEXT: dict = {}


def _perf_time(label: str, seconds: float) -> None:
    rec = _PERF_TIMINGS.setdefault(label, {"count": 0, "total_seconds": 0.0,
                                            "min_seconds": None, "max_seconds": 0.0})
    rec["count"] += 1
    rec["total_seconds"] += seconds
    rec["min_seconds"] = seconds if rec["min_seconds"] is None else min(rec["min_seconds"], seconds)
    rec["max_seconds"] = max(rec["max_seconds"], seconds)


def _loop_catalog() -> list[dict]:
    """Inventory source loops, including comprehensions; absence of hits is not zero iterations."""
    class Inventory(ast.NodeVisitor):
        def __init__(self):
            self.function = []
            self.loops = []

        def visit_FunctionDef(self, node):
            self.function.append(node.name)
            self.generic_visit(node)
            self.function.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_For(self, node):
            self._add(node, "for")
            self.generic_visit(node)

        def visit_While(self, node):
            self._add(node, "while")
            self.generic_visit(node)

        def visit_AsyncFor(self, node):
            self._add(node, "async-for")
            self.generic_visit(node)

        def visit_ListComp(self, node):
            self._add(node, "list-comprehension")
            self.generic_visit(node)

        def visit_SetComp(self, node):
            self._add(node, "set-comprehension")
            self.generic_visit(node)

        def visit_DictComp(self, node):
            self._add(node, "dict-comprehension")
            self.generic_visit(node)

        def visit_GeneratorExp(self, node):
            self._add(node, "generator")
            self.generic_visit(node)

        def _add(self, node, kind):
            self.loops.append({"line": node.lineno, "end_line": node.end_lineno,
                               "function": self.function[-1] if self.function else "<module>",
                               "kind": kind})
    visitor = Inventory()
    visitor.visit(ast.parse(Path(__file__).read_text(encoding="utf-8")))
    return visitor.loops


def _run_with_performance_audit(argv: list[str] | None = None) -> None:
    """Always persist a run summary, even for failed CLI invocations.

    Pyinstrument samples live stacks when installed. The opt-in line_profiler
    traces every Python line for exact hit counts; this is intentionally not
    on by default because line tracing distorts fast runtime measurements.
    """
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument("--perf-dir", default="perf")
    options.add_argument("--no-perf", action="store_true")
    options.add_argument("--profile-loops", action="store_true")
    options.add_argument("--perf-baseline", default=None)
    opts, _ = options.parse_known_args(sys.argv[1:] if argv is None else argv)
    if opts.no_perf:
        main(argv)
        return

    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_id = f"{stamp}-{os.getpid()}-{time.time_ns() % 1_000_000_000:09d}"
    base = Path(opts.perf_dir) / run_id
    pyinstrument_profiler = None
    line_profiler = None
    sampling = "unavailable: install pyinstrument"
    loop_mode = "inventory_only"
    if opts.profile_loops:
        from line_profiler import LineProfiler
        line_profiler = LineProfiler()
        for value in list(globals().values()):
            if inspect.isfunction(value) and value.__module__ == __name__:
                line_profiler.add_function(value)
        loop_mode = "line_profiler"
    else:
        try:
            from pyinstrument import Profiler
        except ImportError:
            pass
        else:
            pyinstrument_profiler = Profiler(interval=0.002, async_mode="disabled")
            sampling = "pyinstrument_2ms"
    clock_start = time.perf_counter()
    cpu_start = time.process_time()
    failure = None
    started = False
    try:
        if line_profiler is not None:
            line_profiler.enable_by_count()
            started = True
        if pyinstrument_profiler is not None:
            pyinstrument_profiler.start()
            started = True
        main(argv)
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        elapsed = time.perf_counter() - clock_start
        cpu = time.process_time() - cpu_start
        if line_profiler is not None and started:
            line_profiler.disable_by_count()
        if pyinstrument_profiler is not None and started:
            pyinstrument_profiler.stop()
        try:
            base.parent.mkdir(parents=True, exist_ok=True)
            loops = _loop_catalog()
            paths = {}
            if pyinstrument_profiler is not None:
                try:
                    html_path = base.with_suffix(".html")
                    pyinstrument_profiler.write_html(html_path)
                    paths["flamegraph_html"] = str(html_path)
                    sample_text = base.with_suffix(".txt")
                    sample_text.write_text(pyinstrument_profiler.output_text(color=False, show_all=True), encoding="utf-8")
                    paths["call_tree_text"] = str(sample_text)
                except Exception as exc:
                    sampling = f"sample_output_error: {exc}"
            if line_profiler is not None:
                raw_path = base.with_suffix(".lprof")
                line_profiler.dump_stats(str(raw_path))
                paths["line_profile_lprof"] = str(raw_path)
                stats = line_profiler.get_stats()
                by_line = {}
                for key, values in stats.timings.items():
                    filename, first_line, fn = key
                    if Path(filename).resolve() != Path(__file__).resolve():
                        continue
                    for lineno, hits, ticks in values:
                        by_line.setdefault((fn, lineno), []).append((hits, ticks * stats.unit))
                for loop in loops:
                    header = by_line.get((loop["function"], loop["line"]), [])
                    loop["header_hits"] = sum(hits for hits, _ in header) if header else 0
                    loop["header_seconds"] = sum(seconds for _, seconds in header)
                    loop["body_line_seconds_approx"] = sum(
                        seconds for (name, lineno), records in by_line.items()
                        if name == loop["function"] and loop["line"] <= lineno <= loop["end_line"]
                        for _, seconds in records)
                loop_mode = "line_profiler_active"
            hotspots = sorted(
                ({"operation": name, "seconds": rec["total_seconds"],
                  "percent_of_e2e": round(100 * rec["total_seconds"] / max(elapsed, 1e-12), 2),
                  "calls": rec["count"]}
                 for name, rec in _PERF_TIMINGS.items()),
                key=lambda item: item["seconds"], reverse=True)
            stages = {}
            for stage_values in _PERF_CONTEXT.get("core5_runtime_by_symbol", {}).values():
                for name, seconds in stage_values.items():
                    if name != "total_seconds":
                        stages[name] = stages.get(name, 0.) + seconds
            core5_hotspots = sorted(
                ({"stage": name, "seconds": seconds,
                  "percent_of_e2e": round(100 * seconds / max(elapsed, 1e-12), 2)}
                 for name, seconds in stages.items()),
                key=lambda item: item["seconds"], reverse=True)
            report = {
                "schema": "onefile-performance-v1",
                "run_id": run_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "argv": sys.argv[1:] if argv is None else argv,
                "workload": _PERF_CONTEXT,
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "status": "FAIL" if failure else "SUCCESS",
                "error": failure,
                "wall_seconds": elapsed,
                "cpu_seconds": cpu,
                "cpu_to_wall_ratio": cpu / max(elapsed, 1e-12),
                "sampled_hotspots": hotspots,
                "core5_stage_hotspots": core5_hotspots,
                "sampling": sampling if line_profiler is None else "off_during_exact_line_tracing",
                "loop_profile_mode": loop_mode,
                "timings": _PERF_TIMINGS,
                "loop_inventory": loops,
                "artifacts": paths,
                "limitations": [
                    "Wall time includes profiler overhead, I/O and cold JIT on this process.",
                    "Loop inventory is static; without --profile-loops, iteration counts are unknown.",
                    "Line tracing is expensive and changes wall time; compare runs with matching profiling modes.",
                    "Loop body line-time is approximate and nested loop totals overlap; do not sum them.",
                    "Hotspots are inclusive/overlapping timings; do not add their percentages.",
                    "Only module-level functions are directly instrumented by line_profiler; nested closures may not have hit counts.",
                    "Numba-compiled inner loops are not visible as Python line hits.",
                ],
            }
            if opts.perf_baseline:
                try:
                    old = json.loads(Path(opts.perf_baseline).read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    old = {}
                    report["comparison"] = {"status": "UNAVAILABLE_BASELINE", "reason": str(exc)}
                if old and any((old.get("workload") or {}).get(key) != _PERF_CONTEXT.get(key) for key in ("input_sha256", "study_mode", "unlock_oos", "offline", "download", "behavior", "target_r", "trials")):
                    report["comparison"] = {"status": "NOT_COMPARABLE", "reason": "Input SHA or study options mismatch"}
                elif old and (old.get("loop_profile_mode") != loop_mode or old.get("sampling") != report["sampling"]):
                    report["comparison"] = {"status": "NOT_COMPARABLE", "reason": "Different profiling modes"}
                elif old and old.get("wall_seconds", 0) <= 0:
                    report["comparison"] = {"status": "NOT_COMPARABLE", "reason": "No valid baseline timing"}
                elif old:
                    baseline = old["wall_seconds"]
                    old_timings = old.get("timings") or {}
                    stage_changes = {}
                    for name, rec in _PERF_TIMINGS.items():
                        older = old_timings.get(name, {})
                        previous = older.get("total_seconds", 0)
                        if previous > 0:
                            stage_changes[name] = {
                                "baseline_seconds": previous, "current_seconds": rec["total_seconds"],
                                "time_reduction_percent": 100 * (previous - rec["total_seconds"]) / previous,
                                "baseline_calls": older.get("count"), "current_calls": rec["count"]}
                    report["comparison"] = {
                        "status": "COMPARABLE_PROFILE_MODE_AND_INPUT",
                        "baseline_wall_seconds": baseline,
                        "wall_reduction_percent": 100 * (baseline - elapsed) / baseline,
                        "speedup_x": baseline / elapsed,
                        "operation_time_reductions": stage_changes}
            path = base.with_suffix(".json")
            atomic_write(path, lambda stream: stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n"))
            print(f"Performance audit: {path} ({elapsed:.3f}s, {loop_mode})", file=sys.stderr)
        except Exception as exc:
            print(f"Performance audit could not be saved: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ONEFILE Brick 1: exploratory BTC 4h event study")
    parser.add_argument("--csv", default="btc_4h.csv", help="Validated CSV with funding_observed and symbol columns; legacy files require --download")
    parser.add_argument("--perf-dir", default="perf", help="Directory for automatic E2E performance reports")
    parser.add_argument("--no-perf", action="store_true", help="Disable automatic performance audit (baseline timing only)")
    parser.add_argument("--profile-loops", action="store_true", help="Trace Python loop lines with optional line_profiler (slow)")
    parser.add_argument("--perf-baseline", default=None, help="Previous performance JSON to compare on matching input")
    parser.add_argument("--symbol", default=SYMBOL, choices=[SYMBOL], help="Locked research universe: BTCUSDT only")
    parser.add_argument("--target-r", type=float, default=TARGET_R, help="Target R-multiple (default: 3.0)")
    parser.add_argument("--download", action="store_true", help="Download canonical USD-M 4h candles + funding")
    parser.add_argument("--offline", action="store_true", help="Skip source funding comparison; unverified data cannot get statistical PASS")
    parser.add_argument("--trials", type=int, default=None,
                        help="Total hypotheses/parameter choices ever inspected, not just this run; unknown by default")
    parser.add_argument("--behavior", action="store_true", help="Run Stage 02 conditional market behavior matrix")
    parser.add_argument("--research", action="store_true", help="Run locked BTC DEV market study + nested H1-A feasibility, not H1-A trades")
    parser.add_argument("--multi-research", action="store_true", help="Run historical core-5 H1-A/B/C, funding and 3R/5R studies; never opens sealed data")
    parser.add_argument("--regime-research", action="store_true", help="Four-year DEV-only XGBoost regime forecasts and original H1-A comparison")
    parser.add_argument("--regime3", action="store_true", help="Pooled core-5 DEV regime v3 (shared harness for logistic/XGB)")
    parser.add_argument("--regime3-scope", choices=["tune", "final"], default="tune", help="tune scores 2021-2022 only; final scores 2021-2024")
    parser.add_argument("--regime3-models", default="no_event,prior,structural,logistic,xgb")
    parser.add_argument("--xgb-threads", type=int, default=1, help="XGBoost CPU threads (default 1; small daily dataset)")
    parser.add_argument("--protocol", default=None, help="Immutable pre-score protocol/code/data identity")
    parser.add_argument("--forecasts", default=None)
    parser.add_argument("--onsets", default=None)
    parser.add_argument("--h1b-events", default=None)
    parser.add_argument("--h1c-forecasts", default=None)
    parser.add_argument("--funding-forecasts", default=None)
    parser.add_argument("--unlock-oos", action="store_true",
                        help="Evaluate locked OOS holdout [2025-01-01, 2026-04-01) (Stage 06 only)")
    parser.add_argument("--self-test", action="store_true", help="Run vectorbt execution and data integrity regression tests")
    parser.add_argument("--out", default=None,
                        help="Output individual observations (default depends on study)")
    parser.add_argument("--evidence", default=None, help="Run metadata and native execution audit")
    args = parser.parse_args(argv)
    if args.xgb_threads < 1:
        parser.error("--xgb-threads must be positive")
    if args.regime_research and (args.research or args.multi_research or args.unlock_oos or args.behavior or args.target_r != TARGET_R):
        parser.error("--regime-research is a separate DEV-only study; sealed OOS cannot be opened")
    args.multi_research = args.multi_research
    if args.research and args.multi_research:
        parser.error("--research and --multi-research are separate frozen studies")
    if args.multi_research and (args.unlock_oos or args.behavior or args.target_r != TARGET_R):
        parser.error("--multi-research is restricted to the historical DEV window and locked 3R/5R protocol")
    args.protocol = args.protocol or ("core5_research_protocol.json" if args.multi_research else "research_protocol.json")
    args.forecasts = args.forecasts or ("core5_h1a_forecasts.csv" if args.multi_research else "h1a_forecasts.csv")
    args.onsets = args.onsets or ("core5_h1a_onsets.csv" if args.multi_research else "h1a_onsets.csv")
    args.h1b_events = args.h1b_events or "core5_h1b_events.csv"
    args.h1c_forecasts = args.h1c_forecasts or "core5_h1c_forecasts.csv"
    args.funding_forecasts = args.funding_forecasts or "core5_funding_forecasts.csv"
    args.out = args.out or ("core5_market_observations.csv" if args.multi_research else ("market_observations.csv" if args.research else "brick1_trades.csv"))
    args.evidence = args.evidence or ("core5_research_evidence.json" if args.multi_research else ("research_evidence.json" if args.research else "brick1_evidence.json"))
    if args.regime_research:
        # Use distinct defaults; never overwrite the frozen original evidence.
        supplied = sys.argv[1:] if argv is None else argv
        for option, attr, default in (("--protocol", "protocol", "regime_protocol.json"),
                                      ("--forecasts", "forecasts", "regime_forecasts.csv"),
                                      ("--onsets", "onsets", "regime_onsets.csv"),
                                      ("--out", "out", "regime_daily.csv"),
                                      ("--evidence", "evidence", "regime_evidence.json")):
            if not any(a == option or a.startswith(option+"=") for a in supplied):
                setattr(args, attr, default)
    if args.research and (args.unlock_oos or args.behavior or args.target_r != TARGET_R):
        parser.error("--research is frozen DEV only; no --unlock-oos, --behavior or alternate --target-r")
    if not np.isfinite(args.target_r) or args.target_r <= 0:
        parser.error("--target-r must be finite and positive")
    if args.trials is not None and args.trials < 1:
        parser.error("--trials must be >= 1")
    if args.self_test:
        self_test()
        return
    if args.regime3:
        if args.unlock_oos or args.research or args.multi_research or args.regime_research:
            parser.error("--regime3 is a separate DEV-only study; sealed OOS cannot be opened")
        file = Path(args.csv if args.csv != "btc_4h.csv" else "core5_4h.csv")
        if not file.exists():
            download_core5_dev(file)
        raw = pd.read_csv(file, float_precision="round_trip")
        assets, _ = validate_core5(raw, file)
        supplied = sys.argv[1:] if argv is None else argv
        if not any(a == "--evidence" or a.startswith("--evidence=") for a in supplied):
            args.evidence = f"regime3_{args.regime3_scope}.json"
        if not any(a == "--forecasts" or a.startswith("--forecasts=") for a in supplied):
            args.forecasts = f"regime3_{args.regime3_scope}_forecasts.csv"
        regime3_run(assets, args)
        return

    start_date = DEV_END if args.unlock_oos else DEV_START
    end_date = OOS_END if args.unlock_oos else DEV_END
    split_label = f"OOS HOLDOUT [{start_date}, {end_date})" if args.unlock_oos else f"DEV [{start_date}, {end_date})"

    file = Path(args.csv)
    try:
        extra = {}
        if args.research or args.multi_research or args.regime_research:
            extra.update({"--protocol":Path(args.protocol),"--forecasts":Path(args.forecasts),"--onsets":Path(args.onsets)})
        if args.multi_research:
            extra.update({"--h1b-events":Path(args.h1b_events),"--h1c-forecasts":Path(args.h1c_forecasts),
                          "--funding-forecasts":Path(args.funding_forecasts)})
        validate_file_paths(file, Path(args.out), Path(args.evidence), **extra)
        if args.download:
            if args.multi_research:
                download_core5_dev(file)
            else:
                download(file, symbol=args.symbol, end_date=end_date)
        if not file.exists():
            raise ValueError(f"No data at {file}. Run: python one.py --download")
        file_bytes = file.read_bytes()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        _PERF_CONTEXT.update({"input_sha256": file_hash, "input_path": str(file), "study_mode": "core5" if args.multi_research else "research" if args.research else "brick1", "unlock_oos": args.unlock_oos, "offline": args.offline, "download": args.download, "behavior": args.behavior, "target_r": args.target_r, "trials": args.trials})
        raw = pd.read_csv(io.BytesIO(file_bytes), float_precision="round_trip")
        if args.regime_research and (pd.to_datetime(raw.time, utc=True) >= pd.Timestamp(DEV_END, tz="UTC")).any():
            raise ValueError("Regime input contains sealed 2025+ data; supply historical DEV only")
        if args.multi_research:
            assets, readiness = validate_core5(raw, file, file_hash=file_hash)
            source = {"status":"VERIFIED" if args.download else "UNVERIFIED",
                      "method":"fresh canonical multi-asset download" if args.download else "explicit offline mode"}
            if not args.download and not args.offline:
                source = verify_core5_funding(assets)
            readiness["funding_source_verification"] = source
            core5_research_run(raw, readiness, args)
            return
        history = validate(raw, end_date=end_date)
        require_data_readiness(check_data_readiness(history, file, end_date=end_date, file_hash=file_hash))
        source = {"status": "UNVERIFIED", "method": "explicit offline mode"}
        if args.download:
            source = {"status": "VERIFIED", "method": "fresh canonical HTTPS download",
                      "verified_at_utc": pd.Timestamp.now(tz="UTC").isoformat()}
        elif not args.offline:
            source = verify_funding_source(history, end_date)
        df, readiness = prepare_study(raw, file, start_date=start_date, end_date=end_date,
                                      file_hash=file_hash)
        readiness["funding_source_verification"] = source
        if args.regime_research:
            _PERF_CONTEXT["study_mode"] = "regime"
            regime_research_run(df, readiness, args)
            return
        if args.research:
            research_run(df, readiness, args)
            return
        all_events = select_setup_events(df)
        events, control = select_events(df)
        all_signal_rows = evaluate_events(df, all_events, target_r=args.target_r)
        by_time = {r["signal_time"]: r for r in all_signal_rows}
        signal_rows = [by_time[str(df.time.iloc[i])] for i in events]
        control_rows = evaluate_events(df, control, target_r=args.target_r)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))

    print("=" * 78)
    print("ONEFILE / BRICK 1: PRICE + VOLUME MOMENTUM-CONTINUATION (4H)")
    print("=" * 78)
    print("1. UNIVERSE & BOUNDS SPECIFICATION (B1.1, B1.5):")
    print(f"  Asset / Pair:       {args.symbol} (Binance USD-M Perpetual)")
    print(f"  Candle Timeframe:   {INTERVAL}")
    print(f"  Canonical Source:   {DATA_SOURCE}")
    print(f"  DEV Range:          [{DEV_START}, {DEV_END}) (Default In-Sample)")
    print(f"  OOS Range:          [{DEV_END}, {OOS_END}) (Locked Holdout)")
    print(f"  Evaluation Window:  {split_label}")
    print(f"  Data Span:          {df.time.iloc[0]} to {df.time.iloc[-1]}")
    print("-" * 78)
    print("2. DATA IDENTITY & REPRODUCIBILITY (B1.6, B1.7):")
    print(f"  Data File:          {file.name} ({len(df):,} QC-passed bars, {file.stat().st_size:,} bytes)")
    print(f"  Input SHA-256:      {readiness['file_sha256']}")
    print(f"  Code SHA-256:       {readiness['code_sha256']}")
    print(f"  Engine / Libs:      vectorbt {vbt.__version__}, python {platform.python_version()}")
    print("-" * 78)
    print("3. B1 DATA ACCEPTANCE GATE (B1.2, B1.3, B1.4, B1.8):")
    print(f"  [{readiness['candle_integrity']}] Candle Continuity:  {readiness['actual_candles']:,} / {readiness['expected_candles']:,} bars ({readiness['missing_candles']} missing)")
    print(f"  [{readiness['funding_coverage']}] 8h Funding Coverage: {readiness['actual_8h_settlements']:,} / {readiness['expected_8h_settlements']:,} observed settlements ({readiness['non_zero_funding_settlements']:,} non-zero)")
    print(f"  [{readiness['qc_integrity']}] QC Integrity:       Strict fail-closed OHLCV, funding observations and symbol")
    print(f"  [{readiness['chronological_bounds']}] Chronological Gate: Complete active window {split_label}")
    print(f"  Funding source:    {source['status']} ({source['method']})")
    print(f"  B1 DATA VERDICT:    [{readiness['verdict']}] ACTIVE DATA ACCEPTED & READY")
    print("=" * 78)
    print(f"Execution model: next-open entry; ATR stop {ATR_MULT}x; target {args.target_r}R; time stop {HOLD_BARS} bars")
    print(f"Assumed fee {FEE_BPS:.1f} bps/side; slip {SLIPPAGE_BPS:.1f} bps/side + historical 8h funding")
    if args.behavior:
        analyze_behavior(df, target_r=args.target_r)
    print("Research units: 1 BTC / unlimited research capital; NOT a $500 account, margin or liquidation simulation.")
    print(f"Common support: {len(events)}/{len(all_events)} setups matched; {len(all_events)-len(events)} unmatched retained below.")
    print("Matched-subset comparisons do not generalize to unmatched setups; matching never uses returns.")
    summarize("ALL SETUPS (descriptive)", all_signal_rows)
    summarize("MATCHED SETUPS (comparison subset)", signal_rows)
    summarize("MONTH-MATCHED TREND-ONLY CONTROL", control_rows)
    summarize_by_year(signal_rows, control_rows)

    matched_times = {r["signal_time"] for r in signal_rows}
    output = pd.DataFrame([{**r, "group": "setup", "matched": r["signal_time"] in matched_times}
                           for r in all_signal_rows] +
                          [{**r, "group": "control", "matched": True} for r in control_rows],
                          columns=list(all_signal_rows[0]) + ["group", "matched"] if all_signal_rows else
                          ["signal_time", "group", "matched"])
    gate = evaluate_gate(signal_rows, control_rows, trials=args.trials, source_verified=source["status"] == "VERIFIED")
    atomic_write(Path(args.out), lambda stream: output.to_csv(stream, index=False))
    print(f"\nIndividual entries exported to {args.out}")
    if signal_rows and control_rows:
        delta_ex = (np.mean([r["net_R_ex_funding"] for r in signal_rows]) -
                    np.mean([r["net_R_ex_funding"] for r in control_rows]))
        delta_fnd = (np.mean([r["net_R"] for r in signal_rows]) -
                     np.mean([r["net_R"] for r in control_rows]))
        print(f"\nDescriptive mean net-R difference (ex-funding):   {delta_ex:+.3f} R (setup minus control)")
        print(f"Descriptive mean net-R difference (incl. funding): {delta_fnd:+.3f} R (setup minus control)")
    evidence = {
        "b1_universe": {
            "symbol": args.symbol,
            "interval": INTERVAL,
            "data_source": DATA_SOURCE,
            "dev_start": DEV_START,
            "dev_end": DEV_END,
            "oos_end": OOS_END,
            "active_split": split_label,
        },
        "b1_data_readiness": readiness,
        "engine": "vectorbt.Portfolio.from_signals",
        "vectorbt": vbt.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "input_sha256": readiness["file_sha256"],
        "code_sha256": readiness["code_sha256"],
        "split": split_label,
        "parameters": vars(args),
        "hold_bars": HOLD_BARS,
        "atr_mult": ATR_MULT,
        "fee_bps": FEE_BPS,
        "slippage_bps": SLIPPAGE_BPS,
        "seed": SEED,
        "bootstrap_n": BOOTSTRAP_N,
        "setup_events": len(all_signal_rows),
        "matched_setup_events": len(signal_rows),
        "unmatched_setup_events": len(all_signal_rows) - len(signal_rows),
        "matching_policy": "same-month; controls disjoint from ALL setup horizons; outcome-blind common-support subset",
        "decision_gate": gate,
        "behavior_sampling": "fixed UTC 43-bar grid; all 43 phases sensitivity, descriptive only" if args.behavior else None,
        "control_events": len(control_rows),
        "bars": len(df),
        "funding": "external per-unit estimate; 4h open proxy, not settlement mark",
        "simulation": "fixed 1 BTC event study with unlimited research cash; NOT account performance; no margin/liquidation engine",
    }
    atomic_write(Path(args.evidence), lambda stream: stream.write(json.dumps(evidence, indent=2) + "\n"))
    print(f"Run evidence & data identity exported to {args.evidence}")


if __name__ == "__main__":
    _run_with_performance_audit()
