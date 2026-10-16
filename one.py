from __future__ import annotations

import argparse
import concurrent.futures
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

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
    s3_root = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision/data/futures/um/monthly"

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
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw_zip = resp.read()
        except Exception:
            return pd.DataFrame(columns=["time_ms", "funding_rate"])
        with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
            with zf.open(zf.namelist()[0]) as f:
                df = pd.read_csv(f, header=None)
        if str(df.iloc[0, 0]).isalpha() or str(df.iloc[0, 0]) == "calc_time":
            df = df.iloc[1:].reset_index(drop=True)
        # Snap calc_time to nearest hour (in ms) in case of millisecond drift
        snapped_ms = ((df[0].astype("int64") + 1800_000) // 3600_000) * 3600_000
        return pd.DataFrame({
            "time_ms": snapped_ms,
            "funding_rate": df[2].astype(float),
        })

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        k_frames = list(pool.map(fetch_kline_month, months))
        f_frames = list(pool.map(fetch_funding_month, months))

    klines = pd.concat(k_frames, ignore_index=True).sort_values("time_ms").drop_duplicates("time_ms")
    funding = pd.concat(f_frames, ignore_index=True).sort_values("time_ms").drop_duplicates("time_ms")
    merged = klines.merge(funding, on="time_ms", how="left")
    merged["funding_rate"] = merged["funding_rate"].fillna(0.0)
    merged["time"] = pd.to_datetime(merged["time_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    cols = ["time", "open", "high", "low", "close", "volume", "funding_rate"]
    merged[cols].to_csv(file, index=False)


def validate(raw: pd.DataFrame, start_date: str = DEV_START, end_date: str = DEV_END) -> pd.DataFrame:
    """Validate OHLCV integrity and strict 4h spacing (fail-closed)."""
    req = ["time", "open", "high", "low", "close", "volume"]
    missing = set(req) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    x = raw.copy()
    if "funding_rate" not in x.columns:
        x["funding_rate"] = 0.0
    x = x[req + ["funding_rate"]].copy()
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
    if (x[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Non-positive price detected")
    if (x["volume"] < 0).any():
        raise ValueError("Negative volume detected")
    if ((x["high"] < x[["open", "close", "low"]].max(axis=1)) |
            (x["low"] > x[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Invalid OHLC bar geometry")
    return x


def features(x: pd.DataFrame) -> pd.DataFrame:
    """Compute strictly causal ATR and Price+Volume Momentum-Continuation signals."""
    df = x.copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_LEN, min_periods=ATR_LEN).mean()

    ema_fast = df["close"].ewm(span=20, adjust=False).mean()
    ema_slow = df["close"].ewm(span=50, adjust=False).mean()
    vol_sma = df["volume"].rolling(20, min_periods=20).mean()

    # 1. Trend regime: fast EMA > slow EMA and close > slow EMA
    df["trend"] = (ema_fast > ema_slow) & (df["close"] > ema_slow) & df["atr"].notna()

    # 2. Volume-backed impulse in the recent 6 bars (24h window prior to trigger)
    bull_expansion = ((df["close"] - df["open"]) >= 1.0 * df["atr"]) & (df["volume"] >= 1.2 * vol_sma)
    recent_impulse = bull_expansion.shift(1).rolling(6, min_periods=1).max() > 0

    # 3. Controlled pullback: prior 1-3 bars dipped near EMA20 without breaking EMA50
    pullback_low = df["low"].shift(1).rolling(3, min_periods=1).min()
    touched_pullback = pullback_low <= (ema_fast.shift(1) * 1.008)
    held_structure = (
        (pullback_low > ema_slow.shift(1))
        & (df["close"].shift(1) < df["high"].shift(1).rolling(6, min_periods=1).max())
    )

    # 4. Continuation trigger: bullish bar reclaiming prior bar high and fast EMA
    continuation = (
        (df["close"] > df["open"])
        & (df["close"] > df["high"].shift(1))
        & (df["close"] > ema_fast)
    )

    df["signal"] = (
        df["trend"] & recent_impulse & touched_pullback & held_structure & continuation
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


def evaluate(x: pd.DataFrame, signal_idx: int, target_r: float = TARGET_R) -> dict:
    """Entry next bar open. Intrabar stop/target tie -> STOP FIRST."""
    entry_i = signal_idx + 1
    raw_entry = float(x.open.iloc[entry_i])
    atr = float(x.atr.iloc[signal_idx])
    entry = raw_entry * (1 + SLIPPAGE_BPS / 10_000)
    risk = ATR_MULT * atr
    stop = entry - risk
    target = entry + target_r * risk
    if not 0 < stop < entry < target:
        raise ValueError("Invalid execution geometry")

    exit_price = float("nan")
    exit_i = -1
    reason = "TIME"
    horizon = x.iloc[entry_i:entry_i + HOLD_BARS]
    if len(horizon) != HOLD_BARS:
        raise ValueError("Unmatured 7-day horizon")
    # Seven-day forward extrema are exploratory *path* excursions, not
    # necessarily excursions experienced before a trade's exit.
    mfe_r = (float(horizon.high.max()) - entry) / risk
    mae_r = (float(horizon.low.min()) - entry) / risk
    for k in range(entry_i, entry_i + HOLD_BARS):
        bar = x.iloc[k]
        # A price gap can cause a worse-than-stop fill.
        if bar.open <= stop:
            exit_price, reason = float(bar.open), "STOP_GAP"
        elif bar.open >= target:
            exit_price, reason = target, "TARGET"
        elif bar.low <= stop:
            exit_price, reason = stop, "STOP"  # Also resolves ambiguous bars.
        elif bar.high >= target:
            exit_price, reason = target, "TARGET"
        elif k == entry_i + HOLD_BARS - 1:
            exit_price, reason = float(bar.close), "TIME"
        else:
            continue
        exit_i = k
        break

    execution_exit = exit_price * (1 - SLIPPAGE_BPS / 10_000)
    fees = (entry + execution_exit) * FEE_BPS / 10_000
    held = x.iloc[entry_i:exit_i + 1]
    funding_usdt = float((held["open"] * held["funding_rate"]).sum()) if "funding_rate" in held else 0.0
    net_r_ex_funding = (execution_exit - entry - fees) / risk
    net_r = (execution_exit - entry - fees - funding_usdt) / risk
    return {"signal_time": str(x.time.iloc[signal_idx]),
            "entry_time": str(x.time.iloc[entry_i]),
            "exit_time": str(x.time.iloc[exit_i]),
            "exit": reason, "entry_price": round(entry, 5),
            "stop_price": round(stop, 5), "target_price": round(target, 5),
            "exit_price": round(execution_exit, 5), "risk_usdt": round(risk, 5),
            "fee_usdt_per_btc": round(fees, 5),
            "funding_usdt_per_btc": round(funding_usdt, 5),
            "net_R_ex_funding": net_r_ex_funding,
            "net_R": net_r,
            "path_MFE_R": mfe_r, "path_MAE_R": mae_r,
            "target_before_stop": int(reason == "TARGET")}


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
    print(f"  {'Year':<6} {'N_setup':>7} {'Win%':>7} {'Setup_R':>9} {'Ctrl_R':>9} {'Delta_R':>9}")
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
    print("\n--- STAGE 01 / BRICK 1 SCIENTIFIC DECISION GATE ---")
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

    verdict = "PASS -> PROCEED TO STAGE 02/03" if all_passed else "KILL -> HYPOTHESIS REJECTED (DO NOT OVERFIT)"
    print(f"FINAL GATE VERDICT: {verdict}")


def self_test() -> None:
    # A candle that touches BOTH stop and target must conservatively stop out.
    now = pd.Timestamp("2023-01-01", tz="UTC")
    t = pd.date_range(now, periods=180, freq="4h")
    close = np.linspace(100.0, 140.0, len(t))
    fixture = pd.DataFrame({"time": t, "open": close, "high": close + 1,
                            "low": close - 1, "close": close,
                            "volume": np.ones(len(t))})
    fixture = features(validate(fixture))
    fixture.loc[60, "atr"] = 2.0
    fixture.loc[61, "open"] = 120.0
    fixture.loc[61, "high"] = 135.0
    fixture.loc[61, "low"] = 110.0
    fixture.loc[61, "close"] = 121.0
    result = evaluate(fixture, 60)
    assert result["exit"] == "STOP", result
    assert result["net_R_ex_funding"] < -1.0

    # Positive funding rate reduces long net_R relative to net_R_ex_funding.
    funded_fixture = fixture.copy()
    funded_fixture.loc[61, "funding_rate"] = 0.001
    funded_res = evaluate(funded_fixture, 60)
    assert funded_res["net_R"] < funded_res["net_R_ex_funding"]

    # An unambiguous target bar must score as a win.
    target_fixture = fixture.copy()
    target_fixture.loc[61, "low"] = 119.0
    assert evaluate(target_fixture, 60)["exit"] == "TARGET"

    # A later price change cannot change features at an earlier signal close.
    a = features(validate(pd.DataFrame({
        "time": t, "open": close, "high": close + 1,
        "low": close - 1, "close": close, "volume": np.ones(len(t))})))
    b = a[["time", "open", "high", "low", "close", "volume"]].copy()
    b.loc[151:, ["open", "high", "low", "close"]] *= 2
    b = features(validate(b))
    pd.testing.assert_series_equal(a.signal.iloc[:150], b.signal.iloc[:150])
    pd.testing.assert_series_equal(a.atr.iloc[:150], b.atr.iloc[:150])

    # Same candles and seed -> deterministic signal/control selection & bootstrap.
    assert select_events(a) == select_events(a)
    s_arr = np.array([2.0, -1.0, 2.0, 2.0, -1.0])
    c_arr = np.array([-1.0, -1.0, 2.0, -1.0, -1.0])
    assert bootstrap_stats(s_arr, c_arr, n_boot=500) == bootstrap_stats(s_arr, c_arr, n_boot=500)

    # Missing bars fail closed; never silently forward-fill a market candle.
    try:
        validate(a.drop(index=80).reset_index(drop=True))
    except ValueError as exc:
        assert "Missing/irregular" in str(exc)
    else:
        raise AssertionError("Missing candle passed validation")
    print("SELF-TEST PASSED: stop-first, funding cost, causal features, deterministic control/bootstrap, missing-bar rejection")


def main() -> None:
    parser = argparse.ArgumentParser(description="ONEFILE Brick 1: exploratory BTC 4h event study")
    parser.add_argument("--csv", default="btc_4h.csv", help="UTC time,open,high,low,close,volume,funding_rate")
    parser.add_argument("--symbol", default=SYMBOL, help="Binance USD-M perpetual symbol (default: BTCUSDT)")
    parser.add_argument("--target-r", type=float, default=TARGET_R, help="Target R-multiple (default: 2.0)")
    parser.add_argument("--download", action="store_true", help="Download canonical USD-M 4h candles + funding")
    parser.add_argument("--unlock-oos", action="store_true",
                        help="Evaluate locked OOS holdout [2025-01-01, 2026-04-01) (Stage 06 only)")
    parser.add_argument("--self-test", action="store_true", help="Run deterministic safety tests")
    parser.add_argument("--out", default="brick1_trades.csv",
                        help="Output individual exploratory event/control rows")
    args = parser.parse_args()
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
    signal_rows = [evaluate(df, i, target_r=args.target_r) for i in events]
    control_rows = [evaluate(df, i, target_r=args.target_r) for i in control]

    print("ONEFILE / BRICK 1 — TREND + PULLBACK + CONTINUATION")
    print(f"Dataset: {args.symbol} USD-M perpetual {INTERVAL} | "
          f"{df.time.iloc[0]} to {df.time.iloc[-1]}")
    print(f"{len(df):,} QC-passed complete candles | {split_label}")
    print(f"Long-only; next-open entry; ATR stop {ATR_MULT}x; target {args.target_r}R; "
          f"time stop {HOLD_BARS} bars")
    print(f"Assumed fee {FEE_BPS:.1f} bps/side; slip {SLIPPAGE_BPS:.1f} bps/side + historical 8h funding")
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
    evaluate_gate(signal_rows, control_rows)
    if len(control_rows) < len(signal_rows):
        print("WARNING: insufficient same-month spaced control entries; sample sizes differ.")


if __name__ == "__main__":
    main()
