import sys; sys.path.insert(0,'.'); import numpy as np, pandas as pd, one
d=pd.read_pickle(sys.argv[1]+'/d.pkl'); ev=pd.read_pickle(sys.argv[1]+'/ev.pkl')
def mcx(g, sig, drift):
    h=g.copy(); h['rv20']=sig  # regime3_mc reads rv20 as sigma
    if drift is None: return one.regime3_mc(h)
    # add drift by shifting simulated returns: emulate via modified function
    rng=np.random.default_rng(one.SEED); H,C=one.REGIME_HORIZON,one.REGIME_CONFIRM; df=4.
    eps=rng.standard_t(df,(400,H+C))*np.sqrt((df-2)/df)
    r,s,q=h.r1.to_numpy(),h.rv20.to_numpy(),h.q.to_numpy(); up=np.full(len(h),np.nan); dn=up.copy(); ends=np.arange(20,21+H+C)
    for t in range(19,len(h)):
        past=r[t-19:t+1]
        if not (np.isfinite(past).all() and np.isfinite(s[t]) and np.isfinite(q[t])): continue
        path=np.concatenate([np.tile(past,(400,1)), drift[t]+s[t]*eps],axis=1)
        cum=[np.cumsum(np.pad(v,((0,0),(1,0))),axis=1) for v in (path,path**2,np.abs(path))]
        tot,sq,ab=[x[:,ends]-x[:,ends-20] for x in cum]; z=tot/np.sqrt(sq+1e-12); er=np.abs(tot)/(ab+1e-12)
        st=np.where((z>=1.25)&(er>=.35),1,np.where((z<=-1.25)&(er>=.35),-1,0)); st[:,0]=q[t]; res=np.zeros(400)
        for i in range(1,H+1):
            on=(st[:,i]!=0)&(st[:,i]!=st[:,i-1])&(st[:,i]!=q[t])&(res==0); cf=(st[:,i+1:i+C+1]==st[:,[i]]).sum(axis=1)>=3; res[on&cf]=st[on&cf,i]
        up[t],dn[t]=(res==1).mean(),(res==-1).mean()
    return up,dn
rows=[]
for kappa,vol in ((0.,'ewm'),(.1,'rv20'),(.2,'rv20')):
    U=[];D=[]
    for s_,g in d.groupby('symbol',sort=False):
        g=g.reset_index(drop=True); ewv=np.sqrt((g.r1**2).ewm(alpha=.06).mean()).to_numpy()
        sig=ewv if vol=='ewm' else g.rv20.to_numpy()
        u,dn=mcx(g,sig,None if kappa==0 else kappa*(g.r20/20).to_numpy()); U.append(u); D.append(dn)
    e=d.copy(); e['mc_up']=np.concatenate(U); e['mc_down']=np.concatenate(D)
    f,_=one.regime3_walk_forward(e,ev,("mc",),"final"); f['model']=f'mc_k{kappa}_{vol}'
    rows.append(f)
base,_=one.regime3_walk_forward(d,ev,("mc",),"final")
for f in rows:
    k=f.model.iloc[0]; m={p:one.regime3_metrics(f,ev,p) for p in ('fixed70','budget')}
    print(k,{p:(m[p]['alarms'],m[p]['hits'],round(m[p]['precision'] or 0,3),round(m[p]['recall'],3)) for p in m},'AP',round(m['fixed70']['event_AP'],3),'gain_vs_mc',one.regime3_ap_gain(f,base))
