"""QC + hourly aggregation + coverage filtering. Output: data/proc/hourly_H.parquet (T x sites), meta."""
import numpy as np, pandas as pd, os, json
os.makedirs("data/proc",exist_ok=True)
th=pd.read_csv("data/raw/thresholds.csv",dtype={"site_no":str})
idx=pd.date_range("2016-10-01 00:00","2026-09-30 23:00",freq="1h",tz="UTC")
Hs={};Qs={};stats=[]
for r in th.itertuples():
    f=f"data/raw/iv_{r.site_no}.csv.gz"
    if not os.path.exists(f): continue
    d=pd.read_csv(f,parse_dates=["t"]); 
    if "H" not in d.columns: stats.append(dict(site_no=r.site_no,reason="no stage")); continue
    d=d.set_index("t").sort_index()
    n_raw=len(d)
    H=d["H"].where((d["H"]>-20)&(d["H"]<300))
    # spike QC: 15-min jump larger than 12*MAD-scale of 15-min changes (robust), isolated
    dh=H.diff(); s=1.4826*np.nanmedian(np.abs(dh-np.nanmedian(dh)))+1e-3
    big=dh.abs()>max(12*s,1.0); 
    spike=big & (big.shift(-1,fill_value=False)) & (np.sign(dh)!=np.sign(dh.shift(-1)))
    H=H.mask(spike|spike.shift(1,fill_value=False))
    h=H.resample("1h").mean().reindex(idx)
    q=d["Q"].where(d["Q"]>=0).resample("1h").mean().reindex(idx) if "Q" in d.columns else pd.Series(np.nan,index=idx)
    # dead-sensor screen: any 180-day window with stage range <= 0.05 ft is a stuck/dead sensor -> treated as outage
    rg=(h.rolling(4320,min_periods=4000).max()-h.rolling(4320,min_periods=4000).min())
    fl=(rg<=0.05).astype(float).fillna(0)
    dead=(fl[::-1].rolling(4320,min_periods=1).max()[::-1]>0).values & h.notna().values
    n_dead=int(dead.sum()); h=h.mask(dead)
    cov=h.notna().mean()
    stats.append(dict(site_no=r.site_no,region=r.region,lid=r.lid,name=r.name,lat=r.lat,lon=r.lon,n_raw=n_raw,coverage=cov,
                      action=r.action,minor=r.minor,moderate=r.moderate,major=r.major,
                      dead_hours=n_dead,hmax=h.max(),hmed=h.median(),hstd=h.std(),reason="ok"))
    Hs[r.site_no]=h;Qs[r.site_no]=q
st=pd.DataFrame(stats); keep=st[(st.coverage>=0.85)&(st.reason=="ok")].site_no.tolist()
H=pd.DataFrame({k:Hs[k] for k in keep}); Q=pd.DataFrame({k:Qs[k] for k in keep})
H.to_pickle("data/proc/H.pkl"); Q.to_pickle("data/proc/Q.pkl")
meta=st[st.site_no.isin(keep)].reset_index(drop=True); meta.to_csv("data/proc/meta.csv",index=False)
st.to_csv("data/proc/site_screening.csv",index=False)
print(len(st),"screened;",len(keep),"kept"); print(meta.groupby("region").agg(n=("site_no","size"),cov=("coverage","mean"),thr=("minor",lambda x:x.notna().sum())))
print("raw 15-min readings:",int(st.n_raw.sum()))
