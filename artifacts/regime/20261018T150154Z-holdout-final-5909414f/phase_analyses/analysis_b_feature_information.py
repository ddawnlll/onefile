"""Worker B: incremental information of regime features over mc, ablation and block-permutation nulls.

Usage (cwd = worktree):
  python analysis_b.py build          # build DEV frame + causal extra features (cached in scratchpad)
  python analysis_b.py w1             # task 1 + task 4 in ONE walk-forward call (all specs share folds)
  python analysis_b.py null GROUP SEED  # task 2: one walk-forward call on a block-permuted GROUP

Only one.py's labels, splits, policies and metrics are used. This file monkeypatches
one.regime3_fit to add free-form logistic specs (same head as one._regime3_logit_head).
Every walk-forward call is appended to the scratchpad log WF_LOG.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

import one

CSV = Path("/home/user/onefile/core5_4h.csv")  # DEV 2020-2024 only
SCRATCH = Path("/tmp/claude-0/-home-user-onefile/2b305038-135c-5d7b-8763-ad999fc0d488/scratchpad")
FRAME, EVENTS, WF_LOG = SCRATCH/"frame_b.pkl", SCRATCH/"events_b.pkl", SCRATCH/"wf_log_b.jsonl"

GROUPS = {
    "G_struct": ["rw_up", "rw_down", "need_up", "need_down", "er_proj", "drop_r7"],
    "G_vol": ["rv5", "rv20", "compression", "range_atr"],
    "G_mom": ["r5", "r20", "r63", "z5", "er5", "dz5", "body_atr", "close_location"],
    "G_funding": ["funding3", "funding7", "funding30", "funding3_z"],
    "G_volume": ["volume_surprise"],
    "G_cross": ["btc_z20", "btc_q", "breadth_up", "breadth_dn", "mean_z20"],
}
BASE = ["mc_up_l", "mc_down_l"]
CUR = ["lrw_up", "lrw_down", "need_up", "need_down", "er_proj"]  # == one.REGIME3_LOGIT["features"]
NEW_FILLED = ["funding7", "funding30", "funding3_z", "btc_z20", "btc_q", "breadth_up", "breadth_dn", "mean_z20"]

SPECS: dict = {}
_orig_fit = one.regime3_fit


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """Causal extras: every value uses the symbol's own past plus same-day (close-known) cross-asset state."""
    d = d.sort_values(["symbol", "time"], kind="stable").reset_index(drop=True)
    g = d.groupby("symbol", sort=False)
    d["funding7"] = g.funding.transform(lambda s: s.rolling(7, min_periods=1).sum())
    d["funding30"] = g.funding.transform(lambda s: s.rolling(30, min_periods=1).sum())
    f3 = d.funding3
    m = f3.groupby(d.symbol).transform(lambda s: s.rolling(90, min_periods=30).mean())
    sd = f3.groupby(d.symbol).transform(lambda s: s.rolling(90, min_periods=30).std())
    d["funding3_z"] = (f3-m)/(sd+1e-12)
    wq = d.pivot(index="time", columns="symbol", values="q")
    wz = d.pivot(index="time", columns="symbol", values="z20")
    btc = [c for c in wq.columns if c.startswith("BTC")][0]
    n = wq.notna().sum(axis=1).clip(lower=1)
    cross = pd.DataFrame({"btc_z20": wz[btc], "btc_q": wq[btc], "breadth_up": wq.eq(1).sum(axis=1)/n,
                          "breadth_dn": wq.eq(-1).sum(axis=1)/n, "mean_z20": wz.mean(axis=1)})
    d = d.join(cross, on="time")
    for side in ("up", "down"):
        p = d[f"mc_{side}"].clip(1e-3, 1-1e-3)
        d[f"mc_{side}_l"] = logit(p)
    for c in NEW_FILLED:
        d[c] = d[c].fillna(0.)
    return d


def build() -> None:
    raw = pd.read_csv(CSV)
    assets, _ = one.validate_core5(raw, CSV)
    d, events = one.regime3_frame(assets)
    d = add_features(d)
    d.to_pickle(FRAME)
    events.to_pickle(EVENTS)
    print("rows", len(d), "events", len(events), "symbols", sorted(d.symbol.unique()))


def logistic_fit(spec: dict, train, stopping, calibration):
    """Same head construction as one.regime3_fit('logistic') with a per-spec feature list."""
    refit = pd.concat([train, stopping])
    info = {"heads": {}}
    cfg = {"features": spec["features"], "C": 1., "class_weight": spec["cw"]}
    sides = {}
    for side, name in ((1, "up"), (-1, "down")):
        pick = lambda r, s=side: r.loc[r.q.ne(s)]
        target = lambda r, s=side: r.y.eq(s).to_numpy(int)
        sides[side], info["heads"][name] = one._regime3_logit_head(pick(refit), pick(calibration), target, cfg)

    def predict(rows):
        if rows.empty:
            return np.empty((0, 3))
        if rows[spec["features"]].isna().to_numpy().any():
            raise ValueError("non-finite feature reached the model")
        up = np.where(rows.q.eq(1), 0., sides[1](rows))
        down = np.where(rows.q.eq(-1), 0., sides[-1](rows))
        total = np.maximum(1., up+down)
        return np.column_stack([1-(up+down)/total, up/total, down/total])
    return predict, info


def _fit(kind, train, stopping, calibration, threads=1):
    if kind in SPECS:
        return logistic_fit(SPECS[kind], train, stopping, calibration)
    return _orig_fit(kind, train, stopping, calibration, threads)


one.regime3_fit = _fit


def permute_blocks(d: pd.DataFrame, cols: list, seed: int) -> tuple[pd.DataFrame, dict]:
    """Shuffle `cols` across 28-day blocks; same block permutation for all assets; rows stay aligned.

    Only blocks with all 28 calendar days are permuted; each row takes the group values of the
    same asset at the same offset inside the source block. Non-permuted rows keep their values.
    """
    rng = np.random.default_rng(seed)
    day = ((d.time-d.time.min())/pd.Timedelta(days=1)).round().astype(int).to_numpy()
    block, off = day//28, day % 28
    cnt = pd.Series(day).groupby(block).nunique()
    full = cnt.index[cnt.to_numpy() == 28].to_numpy()
    perm = rng.permutation(full)
    mapping = pd.Series(perm, index=full)
    m = np.isin(block, full)
    src_block = block.copy()
    src_block[m] = mapping.loc[block[m]].to_numpy()
    src_day = src_block*28+off
    pos = pd.Series(np.arange(len(d)), index=pd.MultiIndex.from_arrays([d.symbol.to_numpy(), day]))
    found = pos.reindex(pd.MultiIndex.from_arrays([d.symbol.to_numpy(), src_day])).to_numpy()
    # A source row must exist and carry finite group values (warm-up rows are NaN); otherwise keep the target row.
    finite = np.isfinite(d[cols].to_numpy(float)).all(axis=1)
    found_i = np.where(np.isnan(found), -1, found).astype(int)
    ok = m & (found_i >= 0)
    ok[ok] &= finite[found_i[ok]]
    src = found_i[ok]
    out = d.copy()
    for c in cols:
        v = d[c].to_numpy().copy()
        v[ok] = d[c].to_numpy()[src]
        out[c] = v
    return out, {"full_blocks": int(len(full)), "permuted_rows": int(ok.sum()), "unpermuted_rows": int((~ok).sum())}


def run(tag: str, frame: pd.DataFrame, specs: dict, kinds=("mc",), extra: dict | None = None) -> dict:
    """One walk-forward call (scope final, 2021-2024). Logs every call and every model in it."""
    global SPECS
    SPECS = specs
    events = pd.read_pickle(EVENTS)
    t0 = time.perf_counter()
    f, folds = one.regime3_walk_forward(frame, events, tuple(kinds)+tuple(specs), "final")
    seconds = time.perf_counter()-t0
    mc = f.loc[f.model.eq("mc")]
    rec = {"run": 0, "tag": tag, "seconds": round(seconds, 1), "folds": len(folds), "models": {}, "extra": extra or {}}
    for name in list(kinds)+list(specs):
        g = f.loc[f.model.eq(name)]
        m = {p: one.regime3_metrics(g, events, p) for p in ("fixed70", "budget")}
        entry = {"features": specs[name]["features"] if name in specs else None,
                 "N": m["fixed70"]["N"], "event_AP": m["fixed70"]["event_AP"],
                 "fixed70": {k: m["fixed70"][k] for k in ("alarms", "hits", "precision", "recall")},
                 "budget": {k: m["budget"][k] for k in ("alarms", "hits", "precision", "recall")}}
        if name != "mc":
            entry["AP_gain_vs_mc"] = one.regime3_ap_gain(g, mc)
        rec["models"][name] = entry
    n = sum(1 for _ in open(WF_LOG)) if WF_LOG.exists() else 0
    rec["run"] = n+1
    f.to_pickle(SCRATCH/f"forecasts_run{rec['run']}.pkl")
    with open(WF_LOG, "a") as fh:
        fh.write(json.dumps(rec, default=float)+"\n")
    print(json.dumps(rec, indent=1, default=float))
    return rec


def w1(frame: pd.DataFrame) -> None:
    specs = {"base_mc": {"features": BASE, "cw": "sqrt"}, "cur": {"features": CUR, "cw": "sqrt"}}
    for i in range(len(CUR)):
        drop = CUR[:i]+CUR[i+1:]
        specs[f"cur-{CUR[i]}"] = {"features": drop, "cw": "sqrt"}
    for g, cols in GROUPS.items():
        specs[f"base+{g}"] = {"features": BASE+cols, "cw": "sqrt"}
    run("W1 base/groups/ablation", frame, specs, kinds=("mc", "logistic"))


def null(frame: pd.DataFrame, group: str, seed: int) -> None:
    """Null call: base, the REAL base+group (deterministic replicate of W1, checks reproducibility), and one permuted draw."""
    cols = GROUPS[group]
    permuted, stats = permute_blocks(frame, cols, seed)
    combined = frame.copy()
    for c in cols:
        combined[c+"__null"] = permuted[c].to_numpy()
    null_cols = [c+"__null" for c in cols]
    specs = {"base": {"features": BASE, "cw": "sqrt"}, f"real_base+{group}": {"features": BASE+cols, "cw": "sqrt"},
             f"null_base+{group}": {"features": BASE+null_cols, "cw": "sqrt"}}
    run(f"NULL {group} seed{seed}", combined, specs, kinds=("mc",), extra={"group": group, "seed": seed, **stats})


FLOW_FRAME = SCRATCH/"frame_flow_b.pkl"
BINANCE_CACHE = Path.home()/".cache"/"onefile"/"binance"
KLINE_ROOT = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision/data/futures/um/monthly/klines"


def taker_ratio_days(symbol: str) -> pd.DataFrame:
    """Daily taker-buy base volume / volume from CACHED 4h kline ZIPs only (no download).

    Column 5 = volume, column 9 = taker buy base volume. Bars are matched to the CSV rows
    by open_time (CSV time convention is checked by the caller via the match rate).
    """
    import hashlib
    import io
    import zipfile
    parts = []
    for ym in pd.date_range("2020-01-01", "2024-12-01", freq="MS").strftime("%Y-%m"):
        url = f"{KLINE_ROOT}/{symbol}/4h/{symbol}-4h-{ym}.zip"
        p = BINANCE_CACHE/(hashlib.sha256(url.encode()).hexdigest()+".zip")
        if not p.exists():
            continue
        with zipfile.ZipFile(p) as zf, zf.open(zf.namelist()[0]) as fh:
            k = pd.read_csv(fh, header=None, usecols=[0, 5, 9])
        if not str(k.iloc[0, 0]).lstrip("-").isdigit():  # header row, if any
            k = k.iloc[1:]
        parts.append(pd.DataFrame({"open_ms": k[0].astype("int64"), "vol": k[5].astype(float), "taker": k[9].astype(float)}))
    return pd.concat(parts, ignore_index=True)


def add_flow(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """G_flow: causal daily taker-buy ratio. Day t is known at its close (row time = t+1).

    Kline bars are matched to the raw CSV bars (not to daily rows); the CSV bars are then grouped
    into UTC days exactly as one.daily_market does, and only complete 6-bar days are kept.
    """
    d = d.copy()
    diag = {}
    raw = pd.read_csv(CSV)
    raw["bar"] = pd.to_datetime(raw.time, utc=True)
    d["tbr1"] = d["tbr5"] = d["tbr_dev20"] = np.nan
    for symbol in sorted(d.symbol.unique()):
        bars = raw.loc[raw.symbol.eq(symbol), ["bar", "volume"]].copy()
        bars["ms"] = (bars.bar-pd.Timestamp(0, tz="UTC"))//pd.Timedelta(milliseconds=1)
        k = taker_ratio_days(symbol)
        best = None
        for shift_ms in (0, 4*3600*1000, -4*3600*1000):
            m = bars.merge(k.assign(ms=k.open_ms+shift_ms)[["ms", "vol", "taker"]], on="ms", how="left")
            ok = m.vol.notna()
            rel = (np.abs(m.vol[ok]-m.volume[ok])/m.volume[ok].clip(lower=1e-12)).median() if ok.any() else np.nan
            score = (ok.mean(), -rel)
            if best is None or score > best[0]:
                best = (score, shift_ms, m)
        (match_rate, neg_rel), shift_ms, m = best
        diag[symbol] = {"csv_bar_match_rate": round(float(match_rate), 4), "shift_hours": shift_ms/3600e3,
                        "median_abs_rel_volume_diff": round(-float(neg_rel), 8)}
        m = m.loc[m.vol.notna()]
        day_all = bars.bar.dt.floor("D")
        counts = day_all.value_counts()
        complete = counts.index[counts.eq(6)]
        m = m.assign(day=m.bar.dt.floor("D"))
        day = m.groupby("day").agg(vol=("vol", "sum"), taker=("taker", "sum"), n=("vol", "size"))
        day = day.loc[day.index.isin(complete) & day.n.eq(6)]
        day["tbr"] = day.taker/day.vol.clip(lower=1e-12)
        day["tbr5"] = day.tbr.rolling(5, min_periods=3).mean()
        day["dev20"] = day.tbr-day.tbr.rolling(20, min_periods=10).mean()
        day["time"] = day.index+pd.Timedelta(days=1)  # causal: known at that day's close
        day = day.reset_index(drop=True)
        mask = d.symbol.eq(symbol)
        joined = d.loc[mask, ["time"]].merge(day[["time", "tbr", "tbr5", "dev20"]], on="time", how="left")
        d.loc[mask, "tbr1"] = joined.tbr.to_numpy()
        d.loc[mask, "tbr5"] = joined.tbr5.to_numpy()
        d.loc[mask, "tbr_dev20"] = joined.dev20.to_numpy()
        diag[symbol]["days_with_taker"] = int(len(day))
        diag[symbol]["rows_missing_tbr"] = int(joined.tbr.isna().sum())
    for c in ("tbr1", "tbr5", "tbr_dev20"):
        d[c] = d[c].fillna(0.)
    return d, diag


def flow_check() -> None:
    """Verify the cached kline volume reproduces the CSV volume (same bars, same day grouping)."""
    d = pd.read_pickle(FRAME)
    d2, diag = add_flow(d)
    print(json.dumps(diag, indent=1))
    d2.to_pickle(FLOW_FRAME)
    print("flow columns", d2[["tbr1", "tbr5", "tbr_dev20"]].describe().round(4).to_string())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "w1":
        w1(pd.read_pickle(FRAME))
    elif cmd == "null":
        null(pd.read_pickle(FRAME), sys.argv[2], int(sys.argv[3]))
    elif cmd == "flow_check":
        flow_check()
    elif cmd == "flow":
        flow = pd.read_pickle(FLOW_FRAME)
        specs = {"base_mc": {"features": BASE, "cw": "sqrt"},
                 "base+G_flow": {"features": BASE+["tbr1", "tbr5", "tbr_dev20"], "cw": "sqrt"}}
        run("W-FLOW base+G_flow", flow, specs, kinds=("mc",))
    else:
        raise SystemExit(f"unknown command {cmd}")
