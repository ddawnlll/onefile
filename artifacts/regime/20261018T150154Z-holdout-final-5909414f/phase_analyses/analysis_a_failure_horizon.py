"""Worker A: failure and horizon analysis of ONEFILE regime onset forecasts (DEV data only).

Reads the frozen audit artifacts and recomputes labels/features with one.py (no label or model changes).
Run: .venv/bin/python analysis_a.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import one

ART = Path('/home/user/onefile/artifacts/regime/20261018T144655Z-core5-final-53da8b51')
CSV = Path('/home/user/onefile/core5_4h.csv')
DAY = pd.Timedelta(days=1)
T0, T1 = pd.Timestamp('2021-01-01', tz='UTC'), pd.Timestamp('2025-01-01', tz='UTC')
MODELS = ['mc', 'structural', 'logistic', 'xgb']


def ap(y, s) -> float:
    y = np.asarray(y, bool)
    if y.sum() == 0:
        return float('nan')
    return float(average_precision_score(y, np.asarray(s, float)))


def load():
    raw = pd.read_csv(CSV)
    assets, _ = one.validate_core5(raw, CSV)
    d, ev = one.regime3_frame(assets)
    rt = one.regime_event_table(d, ev)
    rt['onset_time'] = pd.to_datetime(rt.onset_time, utc=True)
    p = pd.read_csv(ART / 'predictions.csv.gz')
    p['time'] = pd.to_datetime(p.time, utc=True)
    p['target_onset'] = pd.to_datetime(p.target_onset, utc=True)
    p = p[p.y.notna()].copy()
    p['pos'] = p.y.ne(0)
    p['days'] = (p.target_onset - p.time) / DAY
    p['score'] = p.up + p.down
    return d, rt, p


def phase1(p):
    print('\n=== 1a. Alarms by need_side bucket (alarms.csv) ===')
    al = pd.read_csv(ART / 'alarms.csv')
    al['need_side'] = np.where(al.side.eq(1), al.need_up, al.need_down)
    al = al.dropna(subset=['need_side']).copy()
    al['bucket'] = pd.cut(al.need_side, bins=[-np.inf, 0, .5, 1, np.inf],
                          labels=['<0', '0-0.5', '0.5-1', '>1'], right=False)
    rows = []
    for (m, pol) in [('mc', 'fixed70'), ('logistic', 'precision80'), ('xgb', 'precision80'), ('mc', 'budget')]:
        g = al[al.model.eq(m) & al.policy.eq(pol)]
        for b, h in list(g.groupby('bucket', observed=False)) + [('all', g)]:
            tp = h.outcome.eq('TP')
            l1 = tp & h.lead_days.eq(1)
            l2 = tp & h.lead_days.ge(2)
            n = len(h)
            rows.append({'model/policy': f'{m}/{pol}', 'bucket': str(b), 'alarms': n, 'TP': int(tp.sum()),
                         'precision': round(tp.sum() / n, 3) if n else np.nan,
                         'TP_lead1': int(l1.sum()), 'TP_lead2-7': int(l2.sum()),
                         'prec_lead1': round(l1.sum() / n, 3) if n else np.nan,
                         'prec_lead2-7': round(l2.sum() / n, 3) if n else np.nan})
    print(pd.DataFrame(rows).to_string(index=False))

    print('\n=== 1b. Event-AP of score=up+down (predictions.csv.gz, y notna) ===')
    rows = []
    for m in MODELS:
        g = p[p.model.eq(m)]
        subsets = {
            'i all': g,
            'ii neg vs pos@1d': g[~g.pos | g.days.eq(1)],
            'iii neg vs pos@2-7d': g[~g.pos | g.days.between(2, 7)],
            'q!=0 (reversal)': g[g.q.ne(0)],
            'q==0 (from neutral)': g[g.q.eq(0)],
        }
        for name, s in subsets.items():
            base = s.pos.mean()
            a = ap(s.pos, s.score)
            rows.append({'model': m, 'subset': name, 'n': len(s), 'n_pos': int(s.pos.sum()),
                         'base_rate': round(base, 4), 'AP': round(a, 4), 'AP/base': round(a / base, 2)})
    print(pd.DataFrame(rows).to_string(index=False))

    print('\n=== 1c. FP reasons per model/policy (alarms.csv) ===')
    fp = al[al.outcome.eq('FP')]
    fr = fp.groupby(['model', 'policy', 'fp_reason']).size().unstack(fill_value=0)
    fr['total_FP'] = fr.sum(axis=1)
    fr['share_onset_failed_conf'] = (fr['onset_failed_confirmation'] / fr['total_FP']).round(3)
    fr = fr.join(al.groupby(['model', 'policy']).size().rename('alarms'))
    print(fr.to_string())

    print('\n=== 1d. Misses for mc/fixed70 (events.csv): max_side_score_7d ===')
    ev = pd.read_csv(ART / 'events.csv')
    mm = ev[ev.model.eq('mc') & ev.policy.eq('fixed70')]
    miss = mm[~mm.caught.astype(bool)]
    print(f'events={len(mm)} caught={int(mm.caught.astype(bool).sum())} missed={len(miss)}')
    print(miss.max_side_score_7d.describe().round(4).to_string())
    h = pd.cut(miss.max_side_score_7d, bins=[0, .1, .3, .5, .7, 1.01], right=False).value_counts().sort_index()
    print(h.to_string())
    print('misses with score in [0.5,0.7):', int(((miss.max_side_score_7d >= .5) & (miss.max_side_score_7d < .7)).sum()))
    print('miss_reason counts:', miss.miss_reason.value_counts().to_dict())


def phase2(d, rt, p):
    print('\n=== 2e. Natural duration buckets (CONFIRMED onsets 2021-2024) and recall ===')
    c = rt[rt.status.eq('CONFIRMED') & rt.onset_time.ge(T0) & rt.onset_time.lt(T1)].copy()
    c['bucket'] = pd.cut(c.duration_days, bins=[-np.inf, 4, 14, np.inf],
                         labels=['short<=4', 'medium5-14', 'long>=15'])
    print('events:', len(c), '| censored durations:', int(c.duration_censored.sum()))
    print(c.duration_days.describe().round(2).to_string())
    print(c.bucket.value_counts().sort_index().to_string())
    ev = pd.read_csv(ART / 'events.csv')
    evc = ev.merge(c[['symbol', 'event_id', 'bucket', 'duration_days']], on=['symbol', 'event_id'],
                   how='inner', suffixes=('', '_rt'))
    print('events.csv rows matched to CONFIRMED 2021-2024 onsets, by model/policy:',
          evc.groupby(['model', 'policy']).size().to_dict())
    print('duration_days matches rt:', bool((evc.duration_days == evc.duration_days_rt).all()))
    rows = []
    for (m, pol), g in evc.groupby(['model', 'policy']):
        if (m, pol) not in {('mc', 'fixed70'), ('mc', 'budget')}:
            continue
        for b, h in g.groupby('bucket', observed=False):
            rows.append({'model/policy': f'{m}/{pol}', 'bucket': str(b), 'n': len(h),
                         'caught': int(h.caught.astype(bool).sum()),
                         'recall': round(h.caught.astype(bool).mean(), 3)})
    print(pd.DataFrame(rows).to_string(index=False))

    print('\n=== 2f. Horizon targets (mc and structural), 2021-2024 origins with y notna ===')
    keys = p[p.model.eq('mc')][['symbol', 'time', 'q', 'pos', 'days']].set_index(['symbol', 'time'])
    wide = p.pivot(index=['symbol', 'time'], columns='model', values='score')
    keys = keys.join(wide[['mc', 'structural']], how='left').reset_index()
    keys = keys[keys.time.ge(T0) & keys.time.lt(T1)].copy()
    longc = rt[rt.status.eq('CONFIRMED') & rt.duration_days.ge(10)]
    by_sym = {s: g for s, g in longc.groupby('symbol')}

    def long_flag(r):
        g = by_sym.get(r.symbol)
        if g is None:
            return False
        return bool(((g.onset_time > r.time) & (g.onset_time <= r.time + 7 * DAY) & g.side.ne(r.q)).any())

    keys['LONG'] = [long_flag(r) for r in keys.itertuples()]
    keys['L1'] = keys.pos & keys.days.eq(1)
    keys['L2_7'] = keys.pos & keys.days.ge(2)
    sets = {
        'L1 (all rows)': (keys, keys.L1),
        'L2_7 (rows w/o L1 onset)': (keys[~keys.L1], keys.loc[~keys.L1, 'L2_7']),
        'LONG (onset<=7d, dur>=10)': (keys, keys.LONG),
    }
    rows = []
    for name, (s, lab) in sets.items():
        lab = lab.astype(bool).to_numpy()
        n = len(s)
        row = {'target': name, 'n': n, 'pos': int(lab.sum()), 'base': round(lab.mean(), 4)}
        for col in ['mc', 'structural']:
            sc = s[col].to_numpy()
            row[f'AP_{col}'] = round(ap(lab, sc), 4)
            order = np.argsort(-sc, kind='stable')
            for frac in (0.01, 0.05):
                k = int(np.ceil(frac * n))
                row[f'p@{int(frac * 100)}%_{col}'] = round(lab[order[:k]].mean(), 3)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    print('\n=== 2g. mc side-probability before confirmed onsets vs matched negatives (2021-2024) ===')
    dd = d[['symbol', 'time', 'q', 'y', 'mc_up', 'mc_down']]
    idx = {(s, t): (q, u, dn) for s, t, q, u, dn in zip(dd.symbol, dd.time, dd.q, dd.mc_up, dd.mc_down)}
    neg = dd[dd.y.eq(0)]
    pool = neg.groupby(['symbol', 'q'])[['mc_up', 'mc_down']].mean()
    ev2 = rt[rt.status.eq('CONFIRMED') & rt.onset_time.ge(T0) & rt.onset_time.lt(T1)]
    acc = {k: {'ev': [], 'neg': []} for k in range(1, 8)}
    for e in ev2.itertuples():
        col = 0 if e.side == 1 else 1
        for k in range(1, 8):
            v = idx.get((e.symbol, e.onset_time - k * DAY))
            if v is None or np.isnan(v[1]):
                continue
            acc[k]['ev'].append(v[1 + col])
            key = (e.symbol, v[0])
            if key in pool.index:
                acc[k]['neg'].append(pool.loc[key].iloc[col])
    rows = []
    for k in range(7, 0, -1):
        ev_m = float(np.mean(acc[k]['ev'])) if acc[k]['ev'] else float('nan')
        ng_m = float(np.mean(acc[k]['neg'])) if acc[k]['neg'] else float('nan')
        rows.append({'day': f'-{k}', 'n_events': len(acc[k]['ev']), 'mean_mc_side_onset': round(ev_m, 4),
                     'mean_mc_side_matched_neg': round(ng_m, 4), 'diff': round(ev_m - ng_m, 4)})
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    d, rt, p = load()
    chk = p[p.model.eq('mc')].merge(d[['symbol', 'time', 'mc_up', 'mc_down']], on=['symbol', 'time'])
    print('mc consistency rows:', len(chk), 'max|up diff|:', float((chk.up - chk.mc_up).abs().max()),
          'max|down diff|:', float((chk.down - chk.mc_down).abs().max()))
    print('pred rows per model:', p.model.value_counts().to_dict())
    phase1(p)
    phase2(d, rt, p)


if __name__ == '__main__':
    main()
