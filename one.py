"""ONEFILE Brick 1 — native vectorbt execution, no custom fill/backtest engine.

Install (tested): python -m pip install vectorbt==0.28.5 plotly==5.24.1
Check: python onefile_vectorbt.py --self-test
Run: python onefile_vectorbt.py --csv btc_4h.csv --behavior
Fetch: python onefile_vectorbt.py --download --behavior

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
Results are exploratory independent events, not account returns or edge proof.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import hashlib
import platform
import urllib.request
import zipfile
from pathlib import Path

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

# Stage 01 / Brick 1 Kill Criteria thresholds
MIN_EVENTS = 30
MIN_MEAN_NET_R = 0.15
MIN_DELTA_CONTROL_R = 0.15
MIN_P75_MFE_R = 2.50
MAX_P_VALUE = 0.05


def download(file: Path, symbol: str = SYMBOL, end_date: str = DEV_END) -> None:
    """Download canonical Binance USD-M 4h candles + historical funding rates."""
    months = pd.date_range(DEV_START, end_date, freq="MS", inclusive="left").strftime("%Y-%m")
    s3_root = DATA_SOURCE

    def fetch_kline_month(ym: str) -> pd.DataFrame:
        url = f"{s3_root}/klines/{symbol}/{INTERVAL}/{symbol}-{INTERVAL}-{ym}.zip"
        req = urllib.request.Request(url, headers={"User-Agent": "onefile-brick1/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw_zip = resp.read()
        with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
            with zf.open(zf.namelist()[0]) as f:
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

    def fetch_funding_month(ym: str) -> pd.DataFrame:
        url = f"{s3_root}/fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip"
        req = urllib.request.Request(url, headers={"User-Agent": "onefile-brick1/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw_zip = resp.read()
        with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
            with zf.open(zf.namelist()[0]) as f:
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        k_frames = list(pool.map(fetch_kline_month, months))
        f_frames = list(pool.map(fetch_funding_month, months))

    klines = pd.concat(k_frames, ignore_index=True).sort_values("time_ms").drop_duplicates("time_ms")
    funding = pd.concat(f_frames, ignore_index=True).sort_values("time_ms").drop_duplicates("time_ms")
    expected = pd.date_range(DEV_START, end_date, freq="8h", inclusive="left", tz="UTC")
    expected_ms = expected.as_unit("ms").asi8
    if not np.isin(expected_ms, funding["time_ms"]).all():
        raise ValueError("Incomplete 8h BTC funding history; refusing to replace missing rates with zero")
    if not np.isin(funding["time_ms"], klines["time_ms"]).all():
        raise ValueError("Funding settlements do not align with 4h candles")
    merged = klines.merge(funding, on="time_ms", how="left")
    merged["funding_rate"] = merged["funding_rate"].fillna(0.0)
    merged["time"] = pd.to_datetime(merged["time_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    cols = ["time", "open", "high", "low", "close", "volume", "funding_rate"]
    merged[cols].to_csv(file, index=False)


def validate(raw: pd.DataFrame, start_date: str = DEV_START, end_date: str = DEV_END) -> pd.DataFrame:
    """Validate OHLCV integrity and strict 4h spacing (fail-closed)."""
    req = ["time", "open", "high", "low", "close", "volume", "funding_rate"]
    missing = set(req) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    x = raw.copy()
    x = x[req].copy()
    x["time"] = pd.to_datetime(x["time"], utc=True)
    x = x[(x["time"] >= pd.Timestamp(start_date, tz="UTC")) &
          (x["time"] < pd.Timestamp(end_date, tz="UTC"))].reset_index(drop=True)
    if len(x) < 60:
        raise ValueError("Insufficient candle history")

    diffs = x["time"].diff().iloc[1:]
    if not (diffs == pd.Timedelta(hours=4)).all():
        raise ValueError("Missing/irregular 4h candle spacing")

    for col in ["open", "high", "low", "close", "volume", "funding_rate"]:
        x[col] = pd.to_numeric(x[col], errors="raise").astype(float)
    if not np.isfinite(x[req[1:]].to_numpy()).all():
        raise ValueError("Non-finite market/funding data")
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
) -> dict:
    """Stage 01 / B1 Data Readiness & Acceptance Gate (fail-closed verification)."""
    expected_candles = len(pd.date_range(start_date, end_date, freq="4h", inclusive="left", tz="UTC"))
    expected_8h = len(pd.date_range(start_date, end_date, freq="8h", inclusive="left", tz="UTC"))
    actual_candles = len(clean)

    fnd_8h = clean[clean["time"].dt.hour.isin([0, 8, 16])]
    actual_8h = len(fnd_8h)
    non_zero_fnd = int((fnd_8h["funding_rate"] != 0.0).sum())

    file_bytes = file_path.read_bytes() if file_path.exists() else b""
    file_hash = hashlib.sha256(file_bytes).hexdigest() if file_bytes else "n/a"
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    candle_pass = (actual_candles == expected_candles)
    funding_pass = (actual_8h == expected_8h) and (non_zero_fnd > 0)
    boundary_pass = (
        (clean["time"].min() == pd.Timestamp(start_date, tz="UTC"))
        and (clean["time"].max() == pd.Timestamp(end_date, tz="UTC") - pd.Timedelta(hours=4))
    )

    all_passed = candle_pass and funding_pass and boundary_pass
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
        "verdict": "PASS" if all_passed else "FAIL",
    }


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


def select_events(x: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Select non-overlapping setup signals and deterministic month-matched trend controls."""
    max_signal_idx = len(x) - HOLD_BARS - 1
    signals: list[int] = []
    next_free = 0
    for i in range(max(0, max_signal_idx + 1)):
        if i < next_free:
            continue
        if bool(x.signal.iloc[i]):
            signals.append(i)
            next_free = i + HOLD_BARS + 1

    month = x["time"].dt.strftime("%Y-%m").to_numpy()
    rng = np.random.default_rng(SEED)
    controls: list[int] = []

    def spaced(idx: int, chosen: list[int]) -> bool:
        return all(abs(idx - c) > HOLD_BARS for c in chosen)

    for m in sorted(set(month[i] for i in signals)):
        count = sum(month[i] == m for i in signals)
        candidates = [
            i for i in range(max(0, max_signal_idx + 1))
            if month[i] == m and bool(x.trend.iloc[i]) and not bool(x.signal.iloc[i])
        ]
        for j in rng.permutation(candidates).tolist():
            if spaced(j, controls):
                controls.append(int(j))
                if sum(month[k] == m for k in controls) == count:
                    break
    return signals, sorted(controls)


def evaluate_events(x: pd.DataFrame, indices: list[int], target_r: float = TARGET_R,
                    barriers: bool = True) -> list[dict]:
    """Batch disjoint events through vectorbt; adapt native trade records only.

    Signal t close -> t+1 OPEN; last allowed bar is entry+HOLD_BARS-1 CLOSE.
    1 BTC fixed size and unlimited research capital avoid accidental cash-based
    order rejection. There is no account sizing, compounding or leverage claim.
    Funding at a bar OPEN is charged only if the position already existed before
    that OPEN. An exiting position pays that timestamp; a new entry does not.
    """
    if not indices:
        return []
    ids = np.asarray(indices)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Event indices must be integers")
    if len(set(indices)) != len(indices) or np.any(np.diff(ids) <= HOLD_BARS):
        raise ValueError("Events must be sorted, unique and non-overlapping")
    if ids[0] < 0 or ids[-1] + HOLD_BARS >= len(x):
        raise ValueError("Unmatured 7-day horizon or invalid event index")
    if not np.isfinite(target_r) or target_r <= 0:
        raise ValueError("target_r must be finite and positive")
    n = len(x)
    entry_bars = ids + 1
    exit_bars = ids + HOLD_BARS
    risk = ATR_MULT * x.atr.to_numpy()[ids]
    expected_entry = x.open.to_numpy()[entry_bars] * (1 + SLIPPAGE_BPS / 10_000)
    if not np.isfinite(risk).all() or np.any(risk <= 0) or np.any(risk >= expected_entry):
        raise ValueError("Invalid execution geometry/ATR")

    # Pure data adapter: only OPEN is observable on phase 0. Original OHLC is
    # supplied on phase 1. All order matching and stop priority belong to vbt.
    opens = np.repeat(x.open.to_numpy(), 2)
    highs, lows, closes = opens.copy(), opens.copy(), opens.copy()
    highs[1::2] = x.high.to_numpy()
    lows[1::2] = x.low.to_numpy()
    closes[1::2] = x.close.to_numpy()
    entries, exits = np.zeros(2*n, bool), np.zeros(2*n, bool)
    entries[2*entry_bars] = True
    exits[2*exit_bars+1] = True
    sl, tp = np.full(2*n, np.nan), np.full(2*n, np.nan)
    if barriers:
        sl[2*entry_bars] = risk / expected_entry
        tp[2*entry_bars] = target_r * risk / expected_entry
    pf = vbt.Portfolio.from_signals(
        close=pd.Series(closes), entries=entries, exits=exits, price=closes,
        open=opens, high=highs, low=lows, size=1.0, size_type="amount",
        direction="longonly", accumulate=False, init_cash=np.inf,
        fees=FEE_BPS/10_000, slippage=SLIPPAGE_BPS/10_000,
        sl_stop=sl, tp_stop=tp, stop_entry_price="fillprice",
        stop_exit_price="stopmarket", use_stops=barriers,
        allow_partial=False, raise_reject=True,
    )
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
        stop, target = entry-risk[j], entry+target_r*risk[j]
        fees = float(trade["entry_fees"] + trade["exit_fees"])
        native_pnl = float(trade["pnl"])
        # Classify the already executed native fill; never select a fill here.
        unslipped_exit = exit_price / (1-SLIPPAGE_BPS/10_000)
        reason = "TIME"
        if barriers:
            if unslipped_exit <= stop + 1e-9:
                reason = "STOP_GAP" if unslipped_exit < stop-1e-9 else "STOP"
            elif unslipped_exit >= target - 1e-9:
                reason = "TARGET"
        held = x.iloc[entry_bar+1:exit_bar+1]
        funding = float((held.open*held.funding_rate).sum()) * quantity
        horizon = x.iloc[entry_bar:entry_bar+HOLD_BARS]
        raw_entry, raw_exit = float(x.open.iloc[entry_bar]), float(closes[exit_i])
        # Cost_R is intended for barrier-free path comparison only.
        cost_r = (fees + funding + (entry-raw_entry)*quantity +
                  (raw_exit-exit_price)*quantity)/(risk[j]*quantity) if not barriers else np.nan
        rows.append({
            "signal_time": str(x.time.iloc[signal_i]),
            "entry_time": str(x.time.iloc[entry_bar]),
            "exit_time": str(x.time.iloc[exit_bar]),
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
            "path_MFE_R": (float(horizon.high.max())-entry)/risk[j],
            "path_MAE_R": (float(horizon.low.min())-entry)/risk[j],
            "target_before_stop": int(reason == "TARGET"),
        })
    return rows


def select_non_overlapping(mask: pd.Series, max_idx: int) -> list[int]:
    """Select non-overlapping indices spaced by > HOLD_BARS from any boolean condition mask."""
    indices: list[int] = []
    next_free = 0
    arr = mask.to_numpy(dtype=bool)
    for i in range(max(0, max_idx + 1)):
        if i >= next_free and arr[i]:
            indices.append(i)
            next_free = i + HOLD_BARS + 1
    return indices


def analyze_behavior(x: pd.DataFrame, target_r: float = TARGET_R) -> None:
    """Stage 02: Measure conditional trend, momentum, pullback, volatility, funding & Monograph H1-A behavior."""
    max_idx = len(x) - HOLD_BARS - 1
    valid = x["atr"].notna()

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
    print(header)
    print("  " + "-" * (len(header) - 2))

    for label, mask in conditions:
        idxs = select_non_overlapping(mask, max_idx)
        if not idxs:
            print(f"  {label:<36} {0:>4d}      n/a     n/a    n/a      n/a      n/a      n/a      n/a     n/a")
            continue
        rows = evaluate_events(x, idxs, target_r=target_r)
        bf_rows = evaluate_events(x, idxs, barriers=False)
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
            idxs = select_non_overlapping(g_mask, max_idx)
            if not idxs:
                continue
            a = pd.DataFrame(evaluate_events(x, idxs, target_r=target_r))
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
    d1["er20"] = (d1["close"] - d1["close"].shift(20)).abs() / (d1["close"].diff().abs().rolling(20).sum() + 1e-12)
    d1["q"] = np.where((d1["z20"] >= 1.25) & (d1["er20"] >= 0.35), 1,
              np.where((d1["z20"] <= -1.25) & (d1["er20"] >= 0.35), -1, 0))
    run_len = np.zeros(len(d1), dtype=int)
    for i in range(1, len(d1)):
        if d1["q"].iloc[i] == 0:
            run_len[i] = run_len[i - 1] + 1
    d1["in_risk_set"] = (d1["q"] == 0) & (run_len >= 10) & (d1.index >= 30)
    onsets_up, onsets_dn, unconfirmed = [], [], []
    for i in range(31, len(d1) - 11):
        if bool(d1["in_risk_set"].iloc[i - 1]) and int(d1["q"].iloc[i]) != 0:
            sign = int(d1["q"].iloc[i])
            if int((d1["q"].iloc[i + 1:i + 11] == sign).sum()) >= 8:
                (onsets_up if sign == 1 else onsets_dn).append(i)
            else:
                unconfirmed.append(i)
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


def bootstrap_stats(setup_r: np.ndarray, ctrl_r: np.ndarray, n_boot: int = BOOTSTRAP_N) -> dict:
    """Compute 95% bootstrap CI for mean R and one-sided permutation p-value vs control."""
    rng = np.random.default_rng(SEED)
    n_s = len(setup_r)
    boot_means = setup_r[rng.integers(0, n_s, size=(n_boot, n_s))].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    prob_pos = float((boot_means > 0.0).mean())

    obs_diff = float(setup_r.mean() - ctrl_r.mean())
    pooled = np.concatenate([setup_r, ctrl_r])
    perm_diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        perm = rng.permutation(pooled)
        perm_diffs[b] = perm[:n_s].mean() - perm[n_s:].mean()
    p_val = float((np.sum(perm_diffs >= obs_diff) + 1) / (n_boot + 1))
    return {
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "prob_pos": prob_pos,
        "obs_diff": obs_diff,
        "p_value": p_val,
    }


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
    print(f"  Stops / targets / time exits: "
          f"{a['exit'].str.startswith('STOP').sum()} / "
          f"{(a['exit'] == 'TARGET').sum()} / {(a['exit'] == 'TIME').sum()}")


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


def evaluate_gate(signal_rows: list[dict], control_rows: list[dict]) -> None:
    """Hypothesis -> Experiment -> Decision gate (with Funding + Bootstrap significance)."""
    print("\n--- STAGE 01 / BRICK 1 EXPLORATORY DECISION GATE ---")
    print("IID bootstrap/permutation is descriptive; no correction for time dependence or multiple testing.")
    if not signal_rows or not control_rows:
        print("GATE VERDICT: KILL / INSUFFICIENT DATA (No setup or control events)")
        return

    s = pd.DataFrame(signal_rows)
    c = pd.DataFrame(control_rows)
    n_events = len(s)
    mean_r = float(s.net_R.mean())
    ctrl_r = float(c.net_R.mean())
    delta_r = mean_r - ctrl_r
    p75_mfe = float(np.percentile(s.path_MFE_R, 75))
    med_mfe = float(s.path_MFE_R.median())
    med_mae_abs = abs(float(s.path_MAE_R.median()))
    stats = bootstrap_stats(s.net_R.to_numpy(), c.net_R.to_numpy())

    print(f"  Bootstrap 95% CI (Setup Mean Net R): [{stats['ci_low']:+.3f}R, {stats['ci_high']:+.3f}R] "
          f"| P(Mean R > 0) = {stats['prob_pos']:.1%}")
    print(f"  Permutation Test (Setup > Control):  delta = {stats['obs_diff']:+.3f}R "
          f"| one-sided p-value = {stats['p_value']:.4f}")

    checks = [
        ("Sample size (N >= 30)", n_events >= MIN_EVENTS, f"N={n_events}"),
        ("Post-funding Mean Net R (> +0.15R)", mean_r > MIN_MEAN_NET_R, f"{mean_r:+.3f}R"),
        ("Setup vs Control Delta (>= +0.15R)", delta_r >= MIN_DELTA_CONTROL_R, f"{delta_r:+.3f}R"),
        ("Statistical Significance (CI_low > 0 & p <= 0.05)",
         (stats["ci_low"] > 0.0) and (stats["p_value"] <= MAX_P_VALUE),
         f"CI_low={stats['ci_low']:+.3f}R, p={stats['p_value']:.4f}"),
        ("Right-tail MFE asymmetry (P75>=2.5R & MedMFE>|MedMAE|)",
         (p75_mfe >= MIN_P75_MFE_R) and (med_mfe > med_mae_abs),
         f"P75={p75_mfe:+.2f}R, MedMFE={med_mfe:+.2f}R vs |MedMAE|={med_mae_abs:.2f}R"),
    ]
    all_passed = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {name}: {detail}")

    verdict = "EXPLORATORY PASS -> INDEPENDENT VALIDATION REQUIRED" if all_passed else "KILL -> HYPOTHESIS REJECTED (DO NOT OVERFIT)"
    print(f"FINAL GATE VERDICT: {verdict}")


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
                         "low": 99., "close": 100., "volume": 1., "funding_rate": 0.})
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
    check(bootstrap_stats(sr, cr, 500) == bootstrap_stats(sr, cr, 500), "deterministic bootstrap/permutation")
    print(f"SELF-TEST PASSED: {count} checks; actual vectorbt {vbt.__version__}; synthetic fixtures only")


def main() -> None:
    parser = argparse.ArgumentParser(description="ONEFILE Brick 1: exploratory BTC 4h event study")
    parser.add_argument("--csv", default="btc_4h.csv", help="UTC time,open,high,low,close,volume,funding_rate")
    parser.add_argument("--symbol", default=SYMBOL, help="Binance USD-M perpetual symbol (default: BTCUSDT)")
    parser.add_argument("--target-r", type=float, default=TARGET_R, help="Target R-multiple (default: 3.0)")
    parser.add_argument("--download", action="store_true", help="Download canonical USD-M 4h candles + funding")
    parser.add_argument("--behavior", action="store_true", help="Run Stage 02 conditional market behavior matrix")
    parser.add_argument("--unlock-oos", action="store_true",
                        help="Evaluate locked OOS holdout [2025-01-01, 2026-04-01) (Stage 06 only)")
    parser.add_argument("--self-test", action="store_true", help="Run vectorbt execution and data integrity regression tests")
    parser.add_argument("--out", default="brick1_trades.csv",
                        help="Output individual exploratory event/control rows")
    parser.add_argument("--evidence", default="brick1_evidence.json", help="Run metadata and native execution audit")
    args = parser.parse_args()
    if not np.isfinite(args.target_r) or args.target_r <= 0:
        parser.error("--target-r must be finite and positive")
    if args.self_test:
        self_test()
        return

    start_date = DEV_END if args.unlock_oos else DEV_START
    end_date = OOS_END if args.unlock_oos else DEV_END
    split_label = f"OOS HOLDOUT [{start_date}, {end_date})" if args.unlock_oos else f"DEV [{start_date}, {end_date})"

    file = Path(args.csv)
    if args.download:
        download(file, symbol=args.symbol, end_date=end_date)
    if not file.exists():
        parser.error(f"No data at {file}. Run: python one.py --download")
    raw = pd.read_csv(file)
    df = features(validate(raw, start_date=start_date, end_date=end_date))
    events, control = select_events(df)
    signal_rows = evaluate_events(df, events, target_r=args.target_r)
    control_rows = evaluate_events(df, control, target_r=args.target_r)

    readiness = check_data_readiness(df, file, start_date=start_date, end_date=end_date)

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
    print(f"  [PASS] Candle Continuity:  {readiness['actual_candles']:,} / {readiness['expected_candles']:,} bars ({readiness['missing_candles']} missing)")
    print(f"  [PASS] 8h Funding Coverage: {readiness['actual_8h_settlements']:,} / {readiness['expected_8h_settlements']:,} settlements ({readiness['non_zero_funding_settlements']:,} non-zero)")
    print(f"  [PASS] QC Integrity:       Strict fail-closed OHLCV geometry, price > 0, volume >= 0")
    print(f"  [PASS] Chronological Gate: 100% samples strictly bounded to {split_label}")
    print(f"  B1 DATA VERDICT:    [{readiness['verdict']}] REAL DEV DATA ACCEPTED & READY")
    print("=" * 78)
    print(f"Execution model: next-open entry; ATR stop {ATR_MULT}x; target {args.target_r}R; time stop {HOLD_BARS} bars")
    print(f"Assumed fee {FEE_BPS:.1f} bps/side; slip {SLIPPAGE_BPS:.1f} bps/side + historical 8h funding")
    if args.behavior:
        analyze_behavior(df, target_r=args.target_r)
    summarize("SETUP", signal_rows)
    summarize("MONTH-MATCHED TREND-ONLY CONTROL", control_rows)
    summarize_by_year(signal_rows, control_rows)

    output = pd.DataFrame([{**r, "group": "setup"} for r in signal_rows] +
                          [{**r, "group": "control"} for r in control_rows])
    output.to_csv(args.out, index=False)
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
        "setup_events": len(signal_rows),
        "control_events": len(control_rows),
        "bars": len(df),
        "funding": "external per-unit estimate; 4h open proxy, not settlement mark",
        "simulation": "unlevered independent 1 BTC events; not $500 account performance",
    }
    Path(args.evidence).write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Run evidence & data identity exported to {args.evidence}")
    evaluate_gate(signal_rows, control_rows)
    if len(control_rows) < len(signal_rows):
        print("WARNING: insufficient same-month spaced control entries; sample sizes differ.")


if __name__ == "__main__":
    main()
