"""Canonical Binance data: download, cache verification, validation and readiness."""
from __future__ import annotations


import concurrent.futures
import hashlib
import io


import os

import tempfile
import urllib.request
import zipfile


from pathlib import Path


import numpy as np
import pandas as pd

from core import ATR_LEN, CORE5_FIRST_MONTH, CORE5_SYMBOLS, DATA_COLUMNS, DATA_SOURCE, DEV_END, DEV_START, INTERVAL, SYMBOL, atomic_write


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


def download_core5_dev(file: Path, first_month: dict[str, str] = CORE5_FIRST_MONTH) -> dict:
    """Download canonical Binance monthly archives only through 2024-12 (never holdouts)."""
    symbols = tuple(first_month)
    jobs = [(symbol, ym) for symbol in symbols for ym in
            pd.date_range(first_month[symbol], DEV_END, freq="MS", inclusive="left").strftime("%Y-%m")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        chunks = list(pool.map(lambda job: _fetch_core5_month(*job), jobs))
    frames = []
    per_asset = {}
    for symbol in symbols:
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


def validate_core5(raw: pd.DataFrame, file: Path, file_hash: str | None = None,
                   symbols: tuple[str, ...] = CORE5_SYMBOLS) -> tuple[dict[str, pd.DataFrame], dict]:
    missing = set(DATA_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Core-5 data missing fields: {sorted(missing)}")
    x = raw[DATA_COLUMNS].copy()
    x["time"] = pd.to_datetime(x.time, utc=True, errors="raise")
    x["symbol"] = x.symbol.astype(str)
    if not x.symbol.isin(symbols).all():
        raise ValueError("Core-5 file contains a symbol outside the frozen core-5 universe")
    if (x.time < pd.Timestamp(DEV_START, tz="UTC")).any() or (x.time >= pd.Timestamp(DEV_END, tz="UTC")).any():
        raise ValueError("Core-5 data exceeds the unlocked historical DEV window")
    assets, report = {}, {}
    for symbol in symbols:
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


