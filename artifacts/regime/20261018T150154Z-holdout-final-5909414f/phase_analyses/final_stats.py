import sys,json; sys.path.insert(0,'.'); import numpy as np, pandas as pd, one
from pathlib import Path
runs={'core5':'artifacts/regime/20261018T150333Z-core5-final-cd00be92','holdout':'artifacts/regime/20261018T150154Z-holdout-final-5909414f'}
out={}
for name,R in runs.items():
    raw=pd.read_csv('core5_4h.csv' if name=='core5' else 'holdout_4h.csv')
    assets,_=one.validate_core5(raw,Path('x.csv') if False else Path('core5_4h.csv' if name=='core5' else 'holdout_4h.csv'),symbols=one.CORE5_SYMBOLS if name=='core5' else one.HOLDOUT_SYMBOLS)
    d,ev=one.regime3_frame(assets)
    p=pd.read_csv(R+'/predictions.csv.gz',parse_dates=['time','maturity_time','target_onset']); p['time']=pd.to_datetime(p.time,utc=True)
    a=pd.read_csv(R+'/alarms.csv'); a['time']=pd.to_datetime(a.time,utc=True)
    res={}
    # calendar 28d block bootstrap of precision & recall (assets on same dates stay together)
    for k,pol in (('mc','fixed70'),('logistic','precision80'),('xgb','fixed70'),('structural','fixed70'),('mc','budget')):
        x=a[(a.model==k)&(a.policy==pol)]
        e=pd.read_csv(R+'/events.csv',parse_dates=['onset_time']); e=e[(e.model==k)&(e.policy==pol)]; e['onset_time']=pd.to_datetime(e.onset_time,utc=True)
        t0=pd.Timestamp('2021-01-01',tz='UTC'); nb=int(np.ceil((pd.Timestamp('2025-01-01',tz='UTC')-t0).days/28))
        ab=((x.time-t0).dt.days//28).to_numpy(); eb=((e.onset_time-t0).dt.days//28).clip(0,nb-1).to_numpy()
        H=np.bincount(ab,weights=x.outcome.eq('TP').to_numpy(float),minlength=nb); A=np.bincount(ab,minlength=nb)
        C=np.bincount(eb,weights=e.caught.to_numpy(float),minlength=nb); E=np.bincount(eb,minlength=nb)
        rng=np.random.default_rng(42); pr=[];rc=[]
        for _ in range(2000):
            i=rng.integers(0,nb,nb)
            if A[i].sum(): pr.append(H[i].sum()/A[i].sum())
            rc.append(C[i].sum()/max(E[i].sum(),1))
        res[f'{k}/{pol}']={'precision':float(H.sum()/max(A.sum(),1)),'precision_block95':np.quantile(pr,[.025,.975]).round(3).tolist(),'recall':float(C.sum()/max(E.sum(),1)),'recall_block95':np.quantile(rc,[.025,.975]).round(3).tolist(),'P(precision>=.8)':float(np.mean(np.array(pr)>=.8))}
    # descriptive PR frontier for mc at fixed thresholds (NOT used for selection)
    g=p[(p.model=='mc')&p.y.notna()].sort_values(['symbol','time']).reset_index(drop=True); pp=g[['none','up','down']].to_numpy(); c=one._regime3_candidates(g,pp,ev)
    fr=[]
    for t in (.5,.6,.65,.7,.75,.8,.85):
        al,op=one.regime3_alarms(g,pp,t,ev,c); h=sum(z['event_id'] is not None for z in al); fr.append((t,len(al),h,round(h/max(len(al),1),3),round(h/len(op),3)))
    res['mc_frontier']=fr
    # durations
    et=one.regime_event_table(d,ev); et=et[et.status.eq('CONFIRMED')&et.onset_time.ge(pd.Timestamp('2021-01-01',tz='UTC'))]
    res['duration']={'n':len(et),'median':float(et.duration_days.median()),'short<=4':float((et.duration_days<=4).mean()),'medium5_14':float(et.duration_days.between(5,14).mean()),'long>=15':float((et.duration_days>=15).mean())}
    e=pd.read_csv(R+'/events.csv'); e=e[(e.model=='mc')&(e.policy=='fixed70')]
    res['mc_fixed70_recall_by_duration']={b:(int(m.sum()),round(float(e.caught[m].mean()),3)) for b,m in (('short',e.duration_days<=4),('medium',e.duration_days.between(5,14)),('long',e.duration_days>=15))}
    out[name]=res
print(json.dumps(out,indent=1))
json.dump(out,open(sys.argv[1],'w'),indent=1)
