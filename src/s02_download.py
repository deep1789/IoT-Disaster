"""Download 10 water years of 15-min stage (00065) and discharge (00060) from USGS NWIS for sampled gauges."""
import requests, io, os, time, sys
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
inv=pd.read_csv("data/raw/inventory_all.csv",dtype={"site_no":str})
inv=inv[inv.lid.notna()].reset_index(drop=True)
def hav(lat1,lon1,lat2,lon2):
    p=np.pi/180; a=np.sin((lat2-lat1)*p/2)**2+np.cos(lat1*p)*np.cos(lat2*p)*np.sin((lon2-lon1)*p/2)**2
    return 2*6371*np.arcsin(np.sqrt(a))
sel=[]
for reg,g in inv.groupby("region"):
    g=g.reset_index(drop=True)
    D=np.array([hav(r.lat,r.lon,g.lat.values,g.lon.values) for r in g.itertuples()])
    c=int(np.argmax((D<70).sum(1)))           # densest point
    sel.append(g.iloc[np.argsort(D[c])[:32]])
sel=pd.concat(sel); sel.to_csv("data/raw/candidates.csv",index=False)
print(sel.groupby("region").size(),flush=True)
CH=[("2016-10-01","2019-09-30"),("2019-10-01","2022-09-30"),("2022-10-01","2026-09-30")]
def fetch(site):
    try: return _fetch(site)
    except Exception as e: return site,f'error {type(e).__name__}'
def _fetch(site):
    out=f"data/raw/iv_{site}.csv.gz"
    if os.path.exists(out): return site,"cached"
    parts=[]
    for a,b in CH:
        u=(f"https://nwis.waterservices.usgs.gov/nwis/iv/?format=rdb&sites={site}&parameterCd=00060,00065"
           f"&startDT={a}T00:00Z&endDT={b}T23:59Z")
        for k in range(4):
            try:
                r=requests.get(u,timeout=240)
                if r.status_code==200: break
            except Exception: pass
            time.sleep(3*(k+1))
        else: return site,"fail"
        t=[l for l in r.text.splitlines() if not l.startswith('#')]
        if len(t)<3: continue
        df=pd.read_csv(io.StringIO("\n".join(t)),sep="\t",dtype=str).iloc[1:]
        if "tz_cd" not in df.columns or "datetime" not in df.columns: continue
        cols=[c for c in df.columns if c.endswith("_00060") or c.endswith("_00065")]
        d=pd.DataFrame({"t":pd.to_datetime(df["datetime"]+" "+df["tz_cd"].map({"EST":"-0500","EDT":"-0400","CST":"-0600","CDT":"-0500"}).fillna("+0000"),utc=True,errors="coerce")})
        for c in cols: d["Q" if c.endswith("00060") else "H"]=pd.to_numeric(df[c],errors="coerce")
        parts.append(d)
    if not parts: return site,"empty"
    d=pd.concat(parts).dropna(subset=["t"]).drop_duplicates("t").sort_values("t")
    d.to_csv(out,index=False,compression="gzip"); return site,f"ok {len(d)}"
with ThreadPoolExecutor(4) as ex:
    for s,m in ex.map(fetch,sel.site_no.tolist()): print(s,m,flush=True)
print("DONE")
