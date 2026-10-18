"""Market regime and direction research: H1-A, regime v2-v4, return gate, direction v5."""
from __future__ import annotations


import time

import hashlib
import io
import json


import zipfile
from collections.abc import Callable

from pathlib import Path


import numpy as np
import pandas as pd

from core import CORE5_FIRST_MONTH, DATA_SOURCE, DEV_END, DEV_START, FEE_BPS, INTERVAL, SEED, SLIPPAGE_BPS, _PERF_CONTEXT, _PERF_TIMINGS, _calendar_block_starts, _perf_time, atomic_write, block_interval
from data import cached_archive


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
REGIME3_FIXED = .70   # declared after 2021-2022 tuning; probability rule, not refit per fold
REGIME3_POLICIES = ("precision80", "budget", "fixed70")
# Designated after 2021-2022 tuning, committed before any 2023-2024 scoring.
REGIME3_PRIMARY = ("mc", "fixed70")
REGIME3_SECONDARY = (("logistic", "precision80"), ("xgb", "budget"))
# Holdout pre-registration (committed before any holdout asset is scored). Phase 1-4
# found no feature, drift or volatility model that adds information beyond mc, so
# the unchanged v3 system is tested once on 12 unseen assets with the v3 gate;
# the whole 2021-2024 holdout window counts as untouched.
REGIME_HOLDOUT_PREREGISTRATION = {
    "primary": REGIME3_PRIMARY, "secondary": [("logistic", "precision80"), ("xgb", "precision80"), ("mc", "budget")],
    "code": "regime v3 models/configs unchanged from PR #6", "gate": "regime3_gate on the holdout window",
    "expectation": "mechanical drop-off skill should transfer; non-mechanical (need>0) skill is not expected",
    "dependence": "holdout assets share market days with core-5; evidence is cross-sectional, not temporal OOS"}
# Worker-owned model configs. Changing them must not touch the harness above.
REGIME3_LOGIT = {"features": ["lrw_up", "lrw_down", "need_up", "need_down", "er_proj"], "C": 1., "class_weight": "sqrt"}
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
    from scipy.special import logit
    d["lrw_up"] = logit(d.rw_up.clip(1e-4, 1-1e-4))
    d["lrw_down"] = logit(d.rw_down.clip(1e-4, 1-1e-4))
    return d


def regime3_mc(d: pd.DataFrame, paths: int = 400, df: float = 4.) -> tuple[np.ndarray, np.ndarray]:
    """P(label=up/down) by simulating the EXACT label on known window + Student-t returns.

    Future daily returns ~ rv20(t) * t(df) scaled to unit variance; the observed
    last 20 returns are fixed. q, crossing, >=3/5 confirmation and 'different
    from current state' follow regime_labels. Fixed seed; no fitted parameters.
    """
    rng = np.random.default_rng(SEED)
    h, c = REGIME_HORIZON, REGIME_CONFIRM
    eps = rng.standard_t(df, (paths, h+c))*np.sqrt((df-2)/df)
    r, sigma, q = d.r1.to_numpy(), d.rv20.to_numpy(), d.q.to_numpy()
    up, down = np.full(len(d), np.nan), np.full(len(d), np.nan)
    ends = np.arange(20, 21+h+c)
    for t in range(19, len(d)):
        past = r[t-19:t+1]
        if not (np.isfinite(past).all() and np.isfinite(sigma[t]) and np.isfinite(q[t])):
            continue
        path = np.concatenate([np.tile(past, (paths, 1)), sigma[t]*eps], axis=1)
        cum = [np.cumsum(np.pad(v, ((0, 0), (1, 0))), axis=1) for v in (path, path**2, np.abs(path))]
        total, square, absolute = [x[:, ends]-x[:, ends-20] for x in cum]
        z, er = total/np.sqrt(square+1e-12), np.abs(total)/(absolute+1e-12)
        state = np.where((z >= 1.25) & (er >= .35), 1, np.where((z <= -1.25) & (er >= .35), -1, 0))
        state[:, 0] = q[t]
        result = np.zeros(paths)
        for i in range(1, h+1):
            onset = (state[:, i] != 0) & (state[:, i] != state[:, i-1]) & (state[:, i] != q[t]) & (result == 0)
            confirmed = (state[:, i+1:i+c+1] == state[:, [i]]).sum(axis=1) >= 3
            result[onset & confirmed] = state[onset & confirmed, i]
        up[t], down[t] = (result == 1).mean(), (result == -1).mean()
    return up, down


def regime3_frame(assets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-asset daily labels; event ids offset per asset so pooled matching is one-to-one."""
    frames, events = [], []
    for k, (symbol, a) in enumerate(sorted(assets.items())):
        d, e = regime_labels(regime3_features(daily_market(a)))
        d["mc_up"], d["mc_down"] = regime3_mc(d)
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


def _regime3_logit_head(fit: pd.DataFrame, cal: pd.DataFrame, target: Callable, cfg: dict) -> tuple[Callable, dict]:
    """One regularized logistic head for a binary target, sigmoid-calibrated on the later segment."""
    from scipy.special import logit
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    cols = cfg["features"]
    y = target(fit)
    detail = {"fit_rows": len(fit), "fit_class_counts": np.bincount(y, minlength=2).tolist()}
    if len(fit) < 20 or len(np.unique(y)) < 2:
        prior = (y.sum()+1)/(len(fit)+2)
        raw = lambda rows: np.full(len(rows), prior)
        detail["fallback"] = "smoothed training prior"
    else:
        scaler = StandardScaler().fit(fit[cols].to_numpy())
        model = LogisticRegression(C=cfg["C"], max_iter=500, solver="lbfgs")
        with threadpool_limits(limits=1):
            model.fit(scaler.transform(fit[cols].to_numpy()), y, sample_weight=_regime3_weights(fit, y, cfg["class_weight"]))
        raw = lambda rows: model.predict_proba(scaler.transform(rows[cols].to_numpy()))[:, 1]
        detail["coef"] = dict(zip(cols, np.round(model.coef_[0], 4).tolist()))
    cy = target(cal)
    calibrate = None
    if len(cy) and np.bincount(cy, minlength=2).min() >= 5:
        calibrate = LogisticRegression(C=1., solver="lbfgs").fit(logit(np.clip(raw(cal), 1e-6, 1-1e-6))[:, None], cy)
    detail["calibration"] = "sigmoid on later segment" if calibrate is not None else "identity (<5 per class)"
    def predict(rows):
        p = raw(rows)
        return p if calibrate is None else calibrate.predict_proba(logit(np.clip(p, 1e-6, 1-1e-6))[:, None])[:, 1]
    return predict, detail


def regime3_fit(kind: str, train: pd.DataFrame, stopping: pd.DataFrame, calibration: pd.DataFrame,
                threads: int = 1) -> tuple[Callable[[pd.DataFrame], np.ndarray], dict]:
    """Return predict(rows)->[none, up, down]; up/down are direct side-specific heads.

    Fit = train+stopping; calibration fits one sigmoid per head on the later
    segment. Outer test rows never enter here.
    """
    from scipy.special import logit
    from sklearn.linear_model import LogisticRegression
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
    if kind == "mc":
        def simulated(rows):
            up, down = rows.mc_up.fillna(0).to_numpy(), rows.mc_down.fillna(0).to_numpy()
            return np.column_stack([1-up-down, up, down])
        return simulated, info
    if kind == "logistic":
        # Two direct side heads: P(up onset) on rows not already up, P(down onset) on rows not already down.
        sides = {}
        for side, name in ((1, "up"), (-1, "down")):
            pick = lambda r, s=side: r.loc[r.q.ne(s)]
            target = lambda r, s=side: r.y.eq(s).to_numpy(int)
            sides[side], info["heads"][name] = _regime3_logit_head(pick(refit), pick(calibration), target, REGIME3_LOGIT)
        def logistic(rows):
            if rows.empty:
                return np.empty((0, 3))
            up = np.where(rows.q.eq(1), 0., sides[1](rows))
            down = np.where(rows.q.eq(-1), 0., sides[-1](rows))
            total = np.maximum(1., up+down)
            return np.column_stack([1-(up+down)/total, up/total, down/total])
        return logistic, info
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
            fit, ca = pick(refit), pick(calibration)
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
    raise ValueError(f"Unknown regime3 model {kind}")


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
            part["fixed70"] = np.nan if kind in ("no_event", "prior") else REGIME3_FIXED
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
                     for policy in REGIME3_POLICIES}
        for split, key in (("by_year", g.time.dt.year), ("by_asset", g.symbol)):
            out[kind][split] = {policy: {str(v): {k: x for k, x in regime3_metrics(sg, events, policy).items() if k != "alarm_archive"}
                                         for v, sg in g.groupby(key)} for policy in REGIME3_POLICIES}
    for ref in ("structural", "mc"):
        base = forecasts.loc[forecasts.model.eq(ref)]
        for kind in ("mc", "logistic", "xgb"):
            if kind in out and kind != ref and not base.empty:
                out[kind][f"AP_gain_vs_{ref}"] = regime3_ap_gain(forecasts.loc[forecasts.model.eq(kind)], base)
    return out


def regime3_gate(model: dict, policy: str) -> dict:
    """Pre-registered before any 2023-2024 scoring; all conditions must hold."""
    pooled, untouched = model[policy], model.get("untouched", {}).get(policy, {})
    years = [v for v in model["by_year"][policy].values() if v.get("alarms", 0) >= 5]
    checks = {"precision>=.80": (pooled.get("precision") or 0) >= .8,
              "wilson_lower>=.70": (pooled["precision_wilson95"][0] or 0) >= .7,
              "alarms>=40": pooled["alarms"] >= 40, "hit_weeks>=20": pooled["distinct_hit_weeks"] >= 20,
              "recall>=.10": (pooled.get("recall") or 0) >= .1,
              "untouched_precision>=.75_with>=15alarms": untouched.get("alarms", 0) >= 15 and (untouched.get("precision") or 0) >= .75,
              "every_year_precision>=.60": all((v.get("precision") or 0) >= .6 for v in years)}
    return {"checks": checks, "verdict": "EDGE_80_CANDIDATE" if all(checks.values()) else "NO_EDGE_80"}


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
    if args.regime3_scope == "final":
        # Holdout assets were never examined: their whole window is untouched.
        untouched_start = "2021-01-01" if getattr(args, "regime3_universe", "core5") == "holdout" else REGIME3_TUNE_END
        late = forecasts.time >= pd.Timestamp(untouched_start, tz="UTC")
        for kind, model in report.items():
            g = forecasts.loc[late & forecasts.model.eq(kind)]
            model["untouched_start"] = untouched_start
            model["untouched"] = {p: {k: v for k, v in regime3_metrics(g, events, p).items() if k != "alarm_archive"}
                                            for p in REGIME3_POLICIES}
            model["gate"] = {p: regime3_gate(model, p) for p in REGIME3_POLICIES}
        kind, policy = REGIME3_PRIMARY
        report["verdict"] = {"primary": list(REGIME3_PRIMARY), **report[kind]["gate"][policy]}
    result = {"scope": args.regime3_scope, "evaluation": ["2021-01-01", REGIME3_TUNE_END if args.regime3_scope == "tune" else DEV_END],
              "assets": sorted(assets), "configs": {"logistic": REGIME3_LOGIT, "xgb": REGIME3_XGB},
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "folds": folds, "models": report,
              "runtime_seconds": time.perf_counter()-started, "sealed_data_used": False}
    if args.forecasts:
        atomic_write(Path(args.forecasts), lambda f: forecasts.to_csv(f, index=False))
    atomic_write(Path(args.evidence), lambda f: f.write(json.dumps(result, indent=2, default=str)+"\n"))
    if getattr(args, "artifacts_dir", None):
        meta = {"universe": getattr(args, "regime3_universe", "core5"), "scope": args.regime3_scope,
                "data": getattr(args, "data_provenance", {}), "assets": sorted(assets), "models": list(kinds),
                "policies": list(REGIME3_POLICIES), "primary": list(REGIME3_PRIMARY),
                "holdout_preregistration": REGIME_HOLDOUT_PREREGISTRATION if getattr(args, "regime3_universe", "core5") == "holdout" else None,
                "config": {"logistic": REGIME3_LOGIT, "xgb": REGIME3_XGB, "fixed": REGIME3_FIXED, "budget": REGIME3_BUDGET,
                           "min_history_alarms": REGIME3_MIN_HISTORY_ALARMS, "horizon": REGIME_HORIZON, "confirm": REGIME_CONFIRM,
                           "mc": {"paths": 400, "df": 4., "seed": SEED}, "tune_end": REGIME3_TUNE_END},
                "features": REGIME3_BASE, "timings": {"research_seconds": result["runtime_seconds"]}}
        ts = time.perf_counter()
        ref = getattr(args, "regime3_reference", None)
        path = regime_audit(d, events, forecasts, folds, report, meta, Path(args.artifacts_dir), Path(ref) if ref else None)
        result["artifacts"] = str(path)
        print(f"Audit artifacts: {path} ({time.perf_counter()-ts:.1f}s)")
    brief = {k: {p: {m: v[p].get(m) for m in ("alarms", "hits", "precision", "precision_wilson95", "recall", "scoreable_events", "event_AP")}
                 for p in REGIME3_POLICIES} for k, v in report.items() if k != "verdict"}
    brief["verdict"] = report.get("verdict")
    print(json.dumps({"scope": args.regime3_scope, "summary": brief, "runtime_seconds": result["runtime_seconds"]}, indent=2, default=str))
    return result


# Regime evidence audit: post-hoc, read-only over forecasts/folds/report. It
# never feeds back into fitting or thresholds; parity re-derives every metric
# from the written predictions and must match the in-memory report exactly.
REGIME_AUDIT_SNAPSHOT = ["q", "z20", "er20", "rv20", "funding3", "rw_up", "rw_down", "need_up", "need_down", "mc_up", "mc_down"]


def _git_state() -> dict:
    import subprocess
    try:
        run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True,
                                        cwd=Path(__file__).parent).stdout.strip()
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def regime_event_table(d: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Every onset with its NATURAL regime duration: consecutive days q keeps the new side."""
    rows = []
    for symbol, g in d.groupby("symbol", sort=True):
        q, times = g.q.to_numpy(), g.time.to_numpy()
        index = {t: i for i, t in enumerate(pd.DatetimeIndex(g.time).as_unit("ns").asi8)}
        for e in events.loc[events.symbol.eq(symbol)].itertuples():
            i = index.get(pd.Timestamp(e.onset_time).as_unit("ns").value)
            run = 0
            while i is not None and i+run < len(q) and q[i+run] == e.side:
                run += 1
            censored = i is None or i+run >= len(q)
            rows.append({"symbol": symbol, "event_id": int(e.onset_i), "onset_time": e.onset_time, "side": int(e.side),
                         "status": e.status, "duration_days": run, "duration_censored": censored,
                         "end_time": None if censored else times[i+run]})
    return pd.DataFrame(rows)


def _alarm_reason(alarm: dict, ev: pd.DataFrame) -> str:
    """Why an alarm was false, from realized onsets only (diagnostic, after the fact)."""
    t, side = pd.Timestamp(alarm["time"]), alarm["side"]
    e = ev.loc[ev.symbol.eq(alarm["symbol"])]
    window = e.loc[(e.onset_time > t) & (e.onset_time <= t+pd.Timedelta(days=7))]
    same = window.loc[window.side.eq(side)]
    if same.empty:
        late = e.loc[e.side.eq(side) & (e.onset_time > t+pd.Timedelta(days=7)) & (e.onset_time <= t+pd.Timedelta(days=14))]
        if not late.empty:
            return "late_onset_8_14d"
        return "opposite_onset" if not window.empty else "no_onset"
    status = same.iloc[0].status
    if status == "UNCONFIRMED":
        return "onset_failed_confirmation"
    if status == "CENSORED":
        return "censored_onset"
    return "matched_elsewhere_or_preceded"


def regime_audit_tables(d: pd.DataFrame, events: pd.DataFrame, forecasts: pd.DataFrame,
                        policies: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    from sklearn.metrics import average_precision_score
    etable = regime_event_table(d, events)
    snap = d[["symbol", "time"]+[c for c in REGIME_AUDIT_SNAPSHOT if c in d]]
    alarms, event_rows, fold_rows, calib = [], [], [], []
    for kind, g in forecasts.groupby("model", sort=False):
        g = g.loc[g.y.notna()].sort_values(["symbol", "time"]).reset_index(drop=True)
        if g.empty:
            continue
        p = g[["none", "up", "down"]].to_numpy()
        cand = _regime3_candidates(g, p, events)
        score = pd.Series(cand["score"], index=pd.MultiIndex.from_arrays([cand["symbol"], cand["time"]]))
        for policy in policies:
            archive, opp = regime3_alarms(g, p, g[policy].to_numpy(), events, cand)
            caught = {a["event_id"]: a for a in archive if a["event_id"] is not None}
            limit = g.set_index(["symbol", "time"])[policy]
            for a in archive:
                alarms.append({"model": kind, "policy": policy, **a, "data_cutoff": a["time"],
                               "threshold": float(limit.loc[(a["symbol"], pd.Timestamp(a["time"]))]),
                               "outcome": "TP" if a["event_id"] is not None else "FP",
                               "fp_reason": None if a["event_id"] is not None else _alarm_reason(a, events)})
            for e in opp.itertuples():
                hit = caught.get(int(e.onset_i))
                window = g.loc[g.symbol.eq(e.symbol) & (g.time >= e.onset_time-pd.Timedelta(days=7)) &
                               (g.time < e.onset_time) & g.q.ne(e.side)]
                side_score = window.up.to_numpy() if e.side == 1 else window.down.to_numpy()
                limits = window[policy].to_numpy()
                if hit is not None:
                    reason = None
                elif window.empty:
                    reason = "no_eligible_origin"
                elif not np.isfinite(limits).any():
                    reason = "policy_disabled"
                elif not np.any(side_score >= limits):
                    reason = "score_below_threshold"
                else:
                    reason = "cooldown_or_other_side_alarm"
                event_rows.append({"model": kind, "policy": policy, "symbol": e.symbol, "event_id": int(e.onset_i),
                                   "onset_time": e.onset_time, "side": int(e.side), "caught": hit is not None,
                                   "lead_days": None if hit is None else hit["lead_days"], "miss_reason": reason,
                                   "max_side_score_7d": float(side_score.max()) if len(side_score) else None})
            for fold, fg in g.groupby("fold"):
                lo, hi = fg.time.min(), fg.time.max()
                fa = [a for a in archive if lo <= pd.Timestamp(a["time"]) <= hi]
                fo = opp.loc[opp.onset_time.between(lo, hi+pd.Timedelta(days=7))]
                hits = sum(a["event_id"] is not None for a in fa)
                ev = fg.y.ne(0)
                fold_rows.append({"model": kind, "policy": policy, "fold": fold, "rows": len(fg), "positives": int(ev.sum()),
                                  "threshold": float(fg[policy].iloc[0]) if np.isfinite(fg[policy].iloc[0]) else None,
                                  "alarms": len(fa), "hits": hits, "precision": hits/len(fa) if fa else None,
                                  "events": len(fo), "event_AP": float(average_precision_score(ev, fg.up+fg.down)) if ev.any() else None})
        edges = np.linspace(0, 1, 11)
        prob = p[:, 1:].sum(axis=1)
        bins = np.clip(np.digitize(prob, edges[1:-1]), 0, 9)
        for b in range(10):
            m = bins == b
            if m.any():
                calib.append({"model": kind, "bin_low": edges[b], "bin_high": edges[b+1], "n": int(m.sum()),
                              "mean_predicted": float(prob[m].mean()), "realized_rate": float(g.y.ne(0)[m].mean())})
    alarm_frame = pd.DataFrame(alarms)
    if not alarm_frame.empty:
        alarm_frame["time"] = pd.to_datetime(alarm_frame.time, utc=True)
        alarm_frame = alarm_frame.merge(snap, on=["symbol", "time"], how="left")
    events_frame = pd.DataFrame(event_rows)
    if not events_frame.empty:
        events_frame = events_frame.merge(etable[["symbol", "event_id", "duration_days", "duration_censored", "end_time"]],
                                          on=["symbol", "event_id"], how="left")
    return {"alarms": alarm_frame, "events": events_frame, "fold_metrics": pd.DataFrame(fold_rows),
            "calibration": pd.DataFrame(calib), "event_table": etable}


def _distribution(values) -> dict:
    v = pd.Series(values, dtype=float).dropna()
    if v.empty:
        return {"n": 0}
    return {"n": int(len(v)), "mean": float(v.mean()), "p10": float(v.quantile(.1)), "median": float(v.median()),
            "p90": float(v.quantile(.9)), "counts": {str(int(k)): int(c) for k, c in v.value_counts().sort_index().items()}}


def _mechanics(a: pd.DataFrame) -> dict:
    """Mechanical alarm: the crossing happens even if every future return is zero (need<=0)."""
    if a.empty or "need_up" not in a:
        return {}
    need = np.where(a.side.eq(1), a.need_up, a.need_down)
    out = {}
    for name, m in (("mechanical_need_le0", need <= 0), ("requires_move_need_gt0", need > 0)):
        n, hits = int(m.sum()), int((a.outcome.eq("TP") & m).sum())
        out[name] = {"alarms": n, "hits": hits, "precision": hits/n if n else None}
    return out


def regime_audit(d: pd.DataFrame, events: pd.DataFrame, forecasts: pd.DataFrame, folds: list[dict], report: dict,
                 meta: dict, root: Path, reference: Path | None = None) -> Path:
    """Write artifacts/regime/<run_id>/ and prove parity with the in-memory report."""
    config = {k: v for k, v in meta.items() if k not in ("timings",)}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:8]
    run_id = f"{pd.Timestamp.now(tz='UTC'):%Y%m%dT%H%M%SZ}-{meta['universe']}-{meta['scope']}-{digest}"
    out = root/run_id
    out.mkdir(parents=True, exist_ok=False)
    policies = tuple(p for p in REGIME3_POLICIES if p in forecasts)
    tables = regime_audit_tables(d, events, forecasts, policies)
    files = {}
    def save(name, frame, **kw):
        path = out/name
        frame.to_csv(path, index=False, **kw)
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    save("predictions.csv.gz", forecasts, compression={"method": "gzip", "mtime": 0})
    feats = sorted({c for c in REGIME3_BASE+REGIME_AUDIT_SNAPSHOT+list(REGIME3_LOGIT["features"])+list(REGIME3_XGB["features"]) if c in d})
    save("features.csv.gz", d[["symbol", "time", "y", "target_onset", "maturity_time"]+feats], compression={"method": "gzip", "mtime": 0})
    for name in ("alarms", "events", "fold_metrics", "calibration"):
        save(f"{name}.csv", tables[name])
    # Parity: re-read predictions from disk; every policy metric must match the report.
    back = pd.read_csv(out/"predictions.csv.gz", parse_dates=["time", "maturity_time", "target_onset"])
    back["time"] = pd.to_datetime(back.time, utc=True)
    keys = ("alarms", "hits", "precision", "recall", "event_AP", "scoreable_events")
    mismatches = []
    for kind, g in back.groupby("model", sort=False):
        for policy in policies:
            again = regime3_metrics(g, events, policy)
            for k in keys:
                a, b = again.get(k), report[kind][policy].get(k)
                if not (a == b or (a is not None and b is not None and abs(a-b) < 1e-9)):
                    mismatches.append([kind, policy, k, a, b])
    parity = {"predictions_roundtrip": "PASS" if not mismatches else "FAIL", "mismatches": mismatches}
    if reference is not None and reference.exists():
        ref = json.loads(reference.read_text())["models"]
        diff = [[kind, policy, k, report[kind][policy].get(k), ref[kind][policy].get(k)]
                for kind in report if kind in ref and kind != "verdict" for policy in policies if policy in ref[kind]
                for k in keys if report[kind][policy].get(k) != ref[kind][policy].get(k)]
        parity["reference"] = {"path": str(reference), "status": "PASS" if not diff else "DIFF", "differences": diff}
    al, ev = tables["alarms"], tables["events"]
    comparison = {}
    for kind in [k for k in report if k != "verdict"]:
        for policy in policies:
            r = report[kind][policy]
            a = al.loc[al.model.eq(kind) & al.policy.eq(policy)] if not al.empty else al
            e = ev.loc[ev.model.eq(kind) & ev.policy.eq(policy)] if not ev.empty else ev
            comparison[f"{kind}/{policy}"] = {
                **{k: r.get(k) for k in ("alarms", "hits", "precision", "precision_wilson95", "recall", "scoreable_events", "event_AP", "brier")},
                "coverage": r.get("alarms", 0)/max(r.get("N", 1), 1),
                "lead_days": _distribution(a.loc[a.outcome.eq("TP"), "lead_days"]) if len(a) else {"n": 0},
                "hits_lead_ge2": int((a.outcome.eq("TP") & a.lead_days.ge(2)).sum()) if len(a) else 0,
                "early_recall_lead_ge2": float((a.outcome.eq("TP") & a.lead_days.ge(2)).sum()/max(r.get("scoreable_events", 0), 1)) if len(a) else 0.,
                "by_mechanics": _mechanics(a),
                "fp_reasons": a.loc[a.outcome.eq("FP"), "fp_reason"].value_counts().to_dict() if len(a) else {},
                "miss_reasons": e.loc[~e.caught, "miss_reason"].value_counts().to_dict() if len(e) else {},
                "duration_caught": _distribution(e.loc[e.caught, "duration_days"]) if len(e) else {"n": 0},
                "duration_missed": _distribution(e.loc[~e.caught, "duration_days"]) if len(e) else {"n": 0}}
    (out/"comparison.json").write_text(json.dumps(comparison, indent=2, default=str)+"\n")
    lines = ["| model/policy | alarms | hits | precision | recall | AP | hits lead≥2 | median lead |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, c in comparison.items():
        f = lambda x: "—" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)
        lines.append(f"| {k} | {c['alarms']} | {c['hits']} | {f(c['precision'])} | {f(c['recall'])} | {f(c['event_AP'])} | "
                     f"{c['hits_lead_ge2']} | {f(c['lead_days'].get('median'))} |")
    (out/"comparison.md").write_text(f"# {run_id}\n\n"+"\n".join(lines)+"\n")
    for name in ("comparison.json", "comparison.md"):
        files[name] = hashlib.sha256((out/name).read_bytes()).hexdigest()
    manifest = {"run_id": run_id, "created_utc": pd.Timestamp.now(tz="UTC").isoformat(), "git": _git_state(),
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **meta,
                "splits": [{k: v for k, v in f.items() if k != "models"} | {"models": {m: {"policy": i.get("policy"), "heads": i.get("heads"),
                           "fit_seconds": i.get("fit_seconds")} for m, i in f["models"].items()}} for f in folds],
                "verdict": report.get("verdict"), "parity": parity, "files": files, "sealed_data_used": False}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2, default=str)+"\n")
    if parity["predictions_roundtrip"] != "PASS":
        raise ValueError(f"Audit parity failed: {mismatches[:5]}")
    return out


# Return gate (pre-registered before scoring; tune years only). Model-free: it asks
# whether the DIRECTION of confirmed onsets earns money after costs. Entry is the
# close of the confirmation day (causal: confirmation_time), exit five days later.
RETURN_GATE = {"study": "ONEFILE-RETURN-GATE-TUNE-V1", "window_onset": ["2021-01-01", "2023-01-01"],
               "entry": "close of confirmation day (onset_i+5)", "exit": "close of onset_i+10",
               "direction": "side of the confirmed onset", "hold_days": 5,
               "costs": "FEE_BPS and SLIPPAGE_BPS on entry and exit; daily funding paid by the side over the 5 held days",
               "gate": "net mean block-CI lower > 0 AND directional gross mean block-CI lower > 0 AND net mean > 0 in both 2021 and 2022",
               "block_days": 28, "draws": 2000, "trials": 1, "sealed_data_used": False}


def regime_return_rows(assets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per confirmed onset in the tune window with a fully observed 5-day hold."""
    start, end = pd.Timestamp(RETURN_GATE["window_onset"][0], tz="UTC"), pd.Timestamp(RETURN_GATE["window_onset"][1], tz="UTC")
    cost = 2*(FEE_BPS+SLIPPAGE_BPS)/1e4
    rows = []
    for symbol, a in sorted(assets.items()):
        d, events = regime_labels(regime3_features(daily_market(a)))
        d = d.reset_index(drop=True)
        for e in events.loc[events.status.eq("CONFIRMED")].itertuples():
            i = int(e.onset_i)
            if not (start <= e.onset_time < end) or i+10 >= len(d):
                continue
            entry, exit_ = d.close.iloc[i+5], d.close.iloc[i+10]
            funding = d.funding.iloc[i+6:i+11]
            if not (np.isfinite([entry, exit_]).all() and len(funding) == 5 and funding.notna().all()):
                continue
            gross = e.side*np.log(exit_/entry)
            funding_cost = e.side*funding.sum()
            rows.append({"symbol": symbol, "onset_time": e.onset_time, "side": int(e.side), "gross_signed": gross,
                         "funding_cost": funding_cost, "cost": cost, "net": gross-funding_cost-cost})
    return pd.DataFrame(rows)


def _calendar_mean_ci(frame: pd.DataFrame, column: str) -> dict:
    """Paired 28-day calendar-block bootstrap of the mean; assets sharing a date stay together."""
    t0 = frame.onset_time.min()
    block = ((frame.onset_time-t0).dt.days//RETURN_GATE["block_days"]).to_numpy()
    groups = [np.flatnonzero(block == g) for g in range(block.max()+1)]
    groups = [g for g in groups if len(g)]
    values = frame[column].to_numpy(float)
    rng = np.random.default_rng(SEED)
    draws = [values[np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])].mean()
             for _ in range(RETURN_GATE["draws"])]
    return {"mean": float(values.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist()}


def regime_return_gate(assets: dict[str, pd.DataFrame], out: Path) -> dict:
    rows = regime_return_rows(assets)
    if rows.empty:
        raise ValueError("No confirmed tune-window onsets with a complete hold")
    net, gross = _calendar_mean_ci(rows, "net"), _calendar_mean_ci(rows, "gross_signed")
    years = {str(y): {"n": int(len(g)), "net_mean": float(g.net.mean()), "gross_mean": float(g.gross_signed.mean())}
             for y, g in rows.groupby(rows.onset_time.dt.year)}
    passed = (net["ci95"][0] > 0 and gross["ci95"][0] > 0 and
              all(years.get(y, {}).get("net_mean", -1) > 0 for y in ("2021", "2022")))
    result = {"spec": RETURN_GATE, "events": int(len(rows)), "assets": int(rows.symbol.nunique()),
              "net_mean": net, "gross_directional_mean": gross,
              "hit_rate_net_gt0": float((rows.net > 0).mean()), "by_year": years,
              "funding_cost_mean": float(rows.funding_cost.mean()), "cost_per_trade": float(rows.cost.iloc[0]),
              "verdict": "PASS_RETURN_GATE_TUNE" if passed else "NO_RETURN_EDGE_TUNE"}
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(out, lambda f: f.write(json.dumps(result, indent=2, default=str)+"\n"))
    rows.to_csv(out.with_suffix(".rows.csv"), index=False)
    return result


# Direction v5 (ONEFILE market-mechanisms plan). Pre-registered before any scoring.
# Primary: 5-day normalized direction (3-class) from causal price/funding/flow features;
# 3 and 7 days secondary. DEV only (2020-2024), BTCUSDT and ETHUSDT.
DIR_PROTOCOL = {
    "study": "ONEFILE-DIRECTION-V5-DEV-PRE-REG",
    "symbols": ["BTCUSDT", "ETHUSDT"],
    "decision": "completed daily bar t; features use only bars up to t; available at t+1 00:00 UTC",
    "target": "Z_h = (log close_{t+h} - log close_t) / (s_t sqrt(h)); UP if Z>0.75, DOWN if Z<-0.75, else NEUTRAL",
    "scale": "s_t = EWMA of squared daily log returns, half-life 21 days",
    "primary": {"model": "logit_full", "horizon": 5}, "secondary_horizons": [3, 7],
    "baselines": ["prior", "persist_m3", "momentum_m21", "logit_price"],
    "candidates": ["logit_price", "logit_full", "xgb_price", "xgb_full"],
    "features": {"price": "m1 m3 m7 m21 accel3 eff7 breakout21 compression3_21 log_risk21 relvol3 close_loc causal_age",
                 "funding": "funding3 z90 and 7-day change (spot not available: no cached spot klines)",
                 "flow": "perp taker-buy ratio 3d and 21d (cached futures klines; OI not cached for 2020-2021)"},
    "test": "90-day blocks from 2021-01-01 to 2024-12-31; labels needing 2025 data are dropped (sealed)",
    "purge": "training pool requires t+7 < block_start-8 (7-day label horizon + 8-day embargo)",
    "calibration": "last 90 days of pool; stopping: previous 90 days (XGB rounds only); temperature scaling",
    "coverage": 0.20, "cooldown_days": 5, "costs": "fee+slippage 4+2 bp per side on entry and exit; funding paid over held days",
    "gate": "precision>=0.80, Wilson lower>=0.70, realized coverage>=0.10, UP+DOWN recall>=0.20, alarms>=100, "
            ">=4 blocks with alarms, precision above persist_m3 at the same coverage, net mean block-CI lower>0",
    "deviation": "history starts 2020-01-01: the 24-month warm-up is impossible; first blocks have thin fit sets",
    "sealed_data_used": False}
DIR_THRESHOLD_Z, DIR_HALFLIFE, DIR_BLOCK_DAYS, DIR_EMBARGO_DAYS = .75, 21, 90, 8
DIR_COST_BP = FEE_BPS+SLIPPAGE_BPS  # per side
DIR_COVERAGE, DIR_COOLDOWN, DIR_HORIZONS = .20, 5, (3, 5, 7)
DIR_PRICE = ["m1", "m3", "m7", "m21", "accel3", "eff7", "breakout21", "compression3_21", "log_risk21", "relvol3", "close_loc", "causal_age"]
DIR_FUND, DIR_FLOW = ["fund_z90", "fund_chg7"], ["taker3", "taker21"]
DIR_MODELS = {"prior": None, "persist_m3": ["m3"], "momentum_m21": ["m21"], "logit_price": DIR_PRICE,
              "logit_full": DIR_PRICE+DIR_FUND+DIR_FLOW, "xgb_price": DIR_PRICE, "xgb_full": DIR_PRICE+DIR_FUND+DIR_FLOW}


def direction_taker_daily(symbol: str) -> pd.DataFrame:
    """Perp taker-buy base volume per UTC day from CACHED futures 4h klines (no download)."""
    parts = []
    for ym in pd.date_range(CORE5_FIRST_MONTH[symbol], DEV_END, freq="MS", inclusive="left").strftime("%Y-%m"):
        payload = cached_archive(f"{DATA_SOURCE}/klines/{symbol}/{INTERVAL}/{symbol}-{INTERVAL}-{ym}.zip", "onefile-core5-research/1.0")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive, archive.open(archive.namelist()[0]) as stream:
            bars = pd.read_csv(stream, header=None, usecols=[0, 5, 9])
        bars = bars[pd.to_numeric(bars[0], errors="coerce").notna()]
        parts.append(pd.DataFrame({"day": pd.to_datetime(bars[0].astype("int64"), unit="ms", utc=True).dt.floor("D"),
                                   "vol": bars[5].astype(float), "taker": bars[9].astype(float)}))
    return pd.concat(parts).groupby("day").sum()


def direction_daily(a: pd.DataFrame, taker: pd.DataFrame) -> pd.DataFrame:
    """Causal daily features and normalized targets for one asset (features use bars up to t)."""
    d = daily_market(a).set_index("day")
    d = d.join(taker, how="left")
    c, r = np.log(d.close), np.log(d.close).diff()
    alpha = 1-.5**(1/DIR_HALFLIFE)
    s = np.sqrt((r**2).ewm(alpha=alpha, adjust=False).mean())
    m = {k: (c-c.shift(k))/(s*np.sqrt(k)) for k in (1, 3, 7, 21)}
    out = pd.DataFrame(index=d.index)
    out["close"], out["funding"], out["s"] = d.close, d.funding, s
    out["m1"], out["m3"], out["m7"], out["m21"] = m[1], m[3], m[7], m[21]
    out["accel3"] = m[3]-m[3].shift(1)
    out["eff7"] = (c-c.shift(7)).abs()/r.abs().rolling(7).sum().clip(lower=1e-12)
    hi, lo = d.high.shift(1).rolling(21).max(), d.low.shift(1).rolling(21).min()
    out["breakout21"] = (d.close-lo)/(hi-lo).clip(lower=1e-12)
    rv3, rv21 = np.sqrt(r.pow(2).rolling(3).mean()), np.sqrt(r.pow(2).rolling(21).mean())
    out["compression3_21"] = rv3/rv21.clip(lower=1e-12)
    out["log_risk21"] = np.log(rv21.clip(lower=1e-12))
    qv = d.volume*d.close
    out["relvol3"] = np.log(qv.rolling(3).mean()/qv.shift(3).rolling(21).mean().clip(lower=1e-12))
    out["close_loc"] = (d.close-d.low)/(d.high-d.low).clip(lower=1e-12)
    state = np.where(m[3] > .25, 1, np.where(m[3] < -.25, -1, 0))
    age, run = np.zeros(len(state)), 0
    for i, st in enumerate(state):
        run = run+1 if i and st == state[i-1] and st != 0 else (1 if st != 0 else 0)
        age[i] = min(run, DIR_HALFLIFE)
    out["causal_age"] = age
    f3 = d.funding.rolling(3).sum()
    out["fund_z90"] = (f3-f3.shift(1).rolling(90).mean())/f3.shift(1).rolling(90).std().clip(lower=1e-12)
    out["fund_chg7"] = (f3-f3.shift(7))/f3.shift(1).rolling(90).std().clip(lower=1e-12)
    tk = 2*d.taker/d.vol.clip(lower=1e-12)-1
    out["taker3"], out["taker21"] = tk.rolling(3).mean(), tk.rolling(21).mean()
    for h in DIR_HORIZONS:
        z = (c.shift(-h)-c)/(s*np.sqrt(h))
        out[f"Z{h}"] = z
        out[f"y{h}"] = np.where(z > DIR_THRESHOLD_Z, 1, np.where(z < -DIR_THRESHOLD_Z, -1, 0)).astype(float)
        out.loc[z.isna(), f"y{h}"] = np.nan
    out["r_fwd5"] = c.shift(-5)-c
    out["fund_fwd5"] = d.funding.rolling(5).sum().shift(-5)
    return out.reset_index()


def direction_panel(assets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for symbol in DIR_PROTOCOL["symbols"]:
        f = direction_daily(assets[symbol], direction_taker_daily(symbol))
        f["symbol"] = symbol
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def _direction_fit(kind: str, cols: list[str], fit: pd.DataFrame, stop: pd.DataFrame, cal: pd.DataFrame,
                   test: pd.DataFrame, h: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Calibrated [down, neutral, up] probabilities for calibration and test rows (no test fitting)."""
    from scipy.optimize import minimize_scalar
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    y_name = f"y{h}"
    train = pd.concat([fit, stop]).dropna(subset=cols+[y_name])
    yi = train[y_name].to_numpy(int)+1
    info = {"fit_rows": int(len(fit)), "train_rows": int(len(train)), "class_counts": np.bincount(yi, minlength=3).tolist()}
    if len(train) < 30 or len(np.unique(yi)) < 3:
        prior = (np.bincount(yi, minlength=3)+1)/(len(yi)+3)
        pc, pt = np.tile(prior, (len(cal), 1)), np.tile(prior, (len(test), 1))
        info["fallback"] = "smoothed prior"
    elif kind == "xgb":
        from xgboost import XGBClassifier
        params = dict(objective="multi:softprob", num_class=3, tree_method="hist", max_depth=2, min_child_weight=10.,
                      learning_rate=.05, reg_lambda=10., reg_alpha=.1, gamma=.1, subsample=.8, colsample_bytree=.8,
                      max_bin=64, random_state=SEED, n_jobs=1, eval_metric="mlogloss")
        xf = fit.dropna(subset=cols+[y_name])
        sf = stop.dropna(subset=cols+[y_name])
        if len(np.unique(xf[y_name])) == 3 and len(np.unique(sf[y_name])) == 3:
            probe = XGBClassifier(**params, n_estimators=200, early_stopping_rounds=20)
            probe.fit(xf[cols].to_numpy(np.float32), xf[y_name].to_numpy(int)+1,
                      eval_set=[(sf[cols].to_numpy(np.float32), sf[y_name].to_numpy(int)+1)], verbose=False)
            rounds = probe.best_iteration+1
        else:
            rounds, info["stopping_fallback"] = 64, "class missing in fit or stopping segment; fixed 64 rounds"
        model = XGBClassifier(**params, n_estimators=rounds).fit(train[cols].to_numpy(np.float32), yi)
        info["rounds"] = rounds
        pc = model.predict_proba(cal[cols].fillna(0).to_numpy(np.float32))
        pt = model.predict_proba(test[cols].fillna(0).to_numpy(np.float32))
    else:
        scaler = StandardScaler().fit(train[cols].to_numpy())
        model = LogisticRegression(C=1., max_iter=1000).fit(scaler.transform(train[cols].to_numpy()), yi)
        pc = model.predict_proba(scaler.transform(cal[cols].fillna(0).to_numpy()))
        pt = model.predict_proba(scaler.transform(test[cols].fillna(0).to_numpy()))
        info["coef"] = np.round(model.coef_, 3).tolist()
    cy = cal[y_name].to_numpy(int)+1 if len(cal) else np.empty(0, int)
    keep = np.isfinite(pc).all(axis=1) & np.isin(cy, [0, 1, 2]) if len(cal) else np.empty(0, bool)
    T = 1.
    if keep.sum() >= 30:
        logp = np.log(np.clip(pc[keep], 1e-12, 1))
        def nll(t):
            z = logp/t
            z = z-z.max(axis=1, keepdims=True)
            q = np.exp(z)/np.exp(z).sum(axis=1, keepdims=True)
            return -np.mean(np.log(np.clip(q[np.arange(len(cy[keep])), cy[keep]], 1e-12, 1)))
        T = float(minimize_scalar(nll, bounds=(.05, 20), method="bounded").x)
    def apply(p):
        z = np.log(np.clip(p, 1e-12, 1))/T
        z = z-z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e/e.sum(axis=1, keepdims=True)
    info["temperature"] = T
    return apply(pc), apply(pt), info


def direction_walk(panel: pd.DataFrame, models: list[str], horizons: tuple[int, ...]) -> pd.DataFrame:
    """Purged expanding walk-forward; each test row is predicted once per model and horizon."""
    rows, folds = [], []
    last_label_day = panel.day.max()
    # Same complete-feature rows for every model, so differences are not sample artifacts.
    panel = panel.dropna(subset=sorted(set(DIR_PRICE+DIR_FUND+DIR_FLOW)))
    for start in pd.date_range("2021-01-01", "2024-12-31", freq=f"{DIR_BLOCK_DAYS}D", tz="UTC"):
        end = min(start+pd.Timedelta(days=DIR_BLOCK_DAYS), pd.Timestamp("2025-01-01", tz="UTC"))
        pool_end = start-pd.Timedelta(days=7+DIR_EMBARGO_DAYS)
        pool = panel.loc[panel.day < pool_end]
        cal = pool.loc[pool.day >= pool_end-pd.Timedelta(days=DIR_BLOCK_DAYS)]
        stop = pool.loc[(pool.day >= pool_end-pd.Timedelta(days=2*DIR_BLOCK_DAYS)) & (pool.day < pool_end-pd.Timedelta(days=DIR_BLOCK_DAYS))]
        fit = pool.loc[pool.day < pool_end-pd.Timedelta(days=2*DIR_BLOCK_DAYS)]
        test = panel.loc[(panel.day >= start) & (panel.day < end) & (panel.day <= last_label_day)]
        record = {"block": str(start.date()), "fit": len(fit), "stop": len(stop), "cal": len(cal), "test": len(test), "models": {}}
        for name in models:
            cols = DIR_MODELS[name]
            for h in horizons:
                if cols is None:
                    prior_fit = pd.concat([fit, stop]).dropna(subset=[f"y{h}"])
                    cls = np.bincount(prior_fit[f"y{h}"].to_numpy(int)+1, minlength=3)+1
                    p = cls/cls.sum()
                    pc, pt, info = np.tile(p, (len(cal), 1)), np.tile(p, (len(test), 1)), {"fallback": "training frequencies"}
                elif len(cols) == 1:
                    pc, pt, info = _direction_fit("logit", cols, fit, stop, cal, test, h)
                else:
                    kind = "xgb" if name.startswith("xgb") else "logit"
                    pc, pt, info = _direction_fit(kind, cols, fit, stop, cal, test, h)
                record["models"][f"{name}/h{h}"] = info
                for split, frame, p in (("cal", cal, pc), ("test", test, pt)):
                    part = frame[["symbol", "day", "close", "funding", f"y{h}", f"Z{h}", "r_fwd5", "fund_fwd5"]].copy()
                    part[["p_down", "p_neutral", "p_up"]] = p
                    part["model"], part["h"], part["split"], part["block"] = name, h, split, str(start.date())
                    rows.append(part)
        folds.append(record)
    out = pd.concat(rows, ignore_index=True)
    out.attrs["folds"] = folds
    return out


def direction_evaluate(pred: pd.DataFrame, name: str, h: int) -> dict:
    """Coverage-20% alarms (threshold from the block's calibration rows), 5-day cooldown per asset,
    precision/recall per direction, proper scores and cost-adjusted net economics on the same rows."""
    from sklearn.metrics import average_precision_score
    sub = pred.loc[(pred.model == name) & (pred.h == h)]
    cal, test = sub.loc[sub.split == "cal"].copy(), sub.loc[sub.split == "test"].dropna(subset=[f"y{h}"]).copy()
    if test.empty:
        return {"N": 0}
    cal["score"] = np.maximum(cal.p_up, cal.p_down)
    thr = cal.groupby("block").score.quantile(1-DIR_COVERAGE)
    test["score"] = np.maximum(test.p_up, test.p_down)
    test["dir"] = np.where(test.p_up >= test.p_down, 1, -1)
    test["raw"] = test.score >= test.block.map(thr).astype(float)
    test = test.sort_values(["symbol", "day"]).reset_index(drop=True)
    alarm, last = np.zeros(len(test), bool), {}
    for i, row in enumerate(test.itertuples()):
        if row.raw and (row.symbol not in last or (row.day-last[row.symbol]).days >= DIR_COOLDOWN):
            alarm[i], last[row.symbol] = True, row.day
    test["alarm"] = alarm
    y = test[f"y{h}"].to_numpy(int)
    al = test.loc[test.alarm]
    correct = al.dir.to_numpy() == al[f"y{h}"].to_numpy(int)
    precision = float(correct.mean()) if len(al) else None
    wilson = _wilson(int(correct.sum()), len(al)) if len(al) else [None, None]
    up_true, down_true = (y == 1), (y == -1)
    net = al.dir*al.r_fwd5-al.dir*al.fund_fwd5-2*DIR_COST_BP/1e4
    ev = al.assign(hit=correct, net=net.to_numpy())
    ci = _calendar_mean_ci(ev.assign(onset_time=ev.day).reset_index(drop=True), "net") if len(ev) else None
    return {"N": int(len(test)), "alarms": int(len(al)), "coverage_realized": float(len(al)/len(test)),
            "precision": precision, "precision_wilson95": wilson,
            "recall_up": float((ev.dir.eq(1) & ev[f"y{h}"].eq(1)).sum()/max(up_true.sum(), 1)),
            "recall_down": float((ev.dir.eq(-1) & ev[f"y{h}"].eq(-1)).sum()/max(down_true.sum(), 1)),
            "recall_combined": float(correct.sum()/max((y != 0).sum(), 1)),
            "alarms_up": int(ev.dir.eq(1).sum()), "alarms_down": int(ev.dir.eq(-1).sum()),
            "alarm_blocks": int(ev.block.nunique()),
            "AP_up": float(average_precision_score(up_true, test.p_up)) if up_true.any() else None,
            "AP_down": float(average_precision_score(down_true, test.p_down)) if down_true.any() else None,
            "base_up": float(up_true.mean()), "base_down": float(down_true.mean()),
            "logloss": float(-np.mean(np.log(np.clip(test[["p_down", "p_neutral", "p_up"]].to_numpy()[np.arange(len(y)), y+1], 1e-12, 1)))),
            "brier": float(np.mean(np.sum((test[["p_down", "p_neutral", "p_up"]].to_numpy()-np.eye(3)[y+1])**2, axis=1))),
            "net_mean_per_trade": float(net.mean()) if len(ev) else None, "net_ci95": ci["ci95"] if ci else None,
            "gross_mean_per_trade": float((ev.dir*ev.r_fwd5).mean()) if len(ev) else None,
            "hit_rate": float(correct.mean()) if len(ev) else None}


def direction_run(assets: dict[str, pd.DataFrame], args) -> dict:
    started = time.perf_counter()
    panel = direction_panel(assets)
    panel = panel.loc[panel.symbol.isin(DIR_PROTOCOL["symbols"])]
    pred = direction_walk(panel, list(DIR_MODELS), DIR_HORIZONS)
    result = {"protocol": DIR_PROTOCOL, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "panel_rows": int(len(panel)), "folds": pred.attrs.get("folds", []), "metrics": {}}
    for name in DIR_MODELS:
        for h in DIR_HORIZONS:
            result["metrics"][f"{name}/h{h}"] = direction_evaluate(pred, name, h)
    primary = result["metrics"][f"{DIR_PROTOCOL['primary']['model']}/h{DIR_PROTOCOL['primary']['horizon']}"]
    base = result["metrics"][f"persist_m3/h{DIR_PROTOCOL['primary']['horizon']}"]
    passed = (primary.get("precision") is not None and primary["precision"] >= .80 and primary["precision_wilson95"][0] >= .70
              and primary["coverage_realized"] >= .10 and primary["recall_combined"] >= .20 and primary["alarms"] >= 100
              and primary["alarm_blocks"] >= 4 and primary["precision"] > base["precision"]
              and primary["net_ci95"] is not None and primary["net_ci95"][0] > 0)
    result["verdict"] = "PASS_DIRECTION_DEV" if passed else "NO_EDGE_DIRECTION_DEV"
    result["runtime_seconds"] = time.perf_counter()-started
    out = Path(args.evidence or "artifacts/direction/direction_v5_dev.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(out, lambda f: f.write(json.dumps(result, indent=2, default=str)+"\n"))
    pred.to_csv(out.with_name(out.stem+"_predictions.csv.gz"), index=False, compression="gzip")
    return result


