"""Fetch official NWS flood categories for candidate gauges (NOAA NWPS)."""
import requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor
c=pd.read_csv("data/raw/candidates.csv",dtype={"site_no":str})
def get(lid):
    for _ in range(3):
        try:
            d=requests.get(f"https://api.water.noaa.gov/nwps/v1/gauges/{lid}",timeout=60).json()
            cat=d["flood"]["categories"]; 
            g=lambda k: cat[k]["stage"] if cat[k]["stage"] not in (None,-9999,-999) else None
            return dict(lid=lid,action=g("action"),minor=g("minor"),moderate=g("moderate"),major=g("major"),
                        units=d["flood"].get("stageUnits"),river=d.get("name"))
        except Exception: pass
    return dict(lid=lid)
with ThreadPoolExecutor(8) as ex: rows=list(ex.map(get,c.lid.tolist()))
t=pd.DataFrame(rows); out=c.merge(t,on="lid",how="left"); out.to_csv("data/raw/thresholds.csv",index=False)
print(out[["action","minor","moderate","major"]].notna().sum()); print(out.groupby("region")[["action","minor"]].count())
