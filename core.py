"""Shared constants, output helpers, block statistics and performance timers."""
from __future__ import annotations


import time


import os

import tempfile


from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd


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
# Regime v4 cross-sectional holdout: fixed before any of these assets was examined.
# DEV window only (<2025); never used for features, labels, tuning or policy choice.
HOLDOUT_FIRST_MONTH = {"ADAUSDT": "2020-01", "LINKUSDT": "2020-01", "LTCUSDT": "2020-01", "DOGEUSDT": "2020-07",
                       "DOTUSDT": "2020-08", "TRXUSDT": "2020-01", "BCHUSDT": "2020-01", "ETCUSDT": "2020-01",
                       "ATOMUSDT": "2020-02", "AVAXUSDT": "2020-09", "XLMUSDT": "2020-01", "FILUSDT": "2020-10"}
HOLDOUT_SYMBOLS = tuple(HOLDOUT_FIRST_MONTH)


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


