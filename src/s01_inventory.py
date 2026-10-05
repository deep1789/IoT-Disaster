"""Build gauge inventory: USGS NWIS stage sites with 15-min telemetry + NOAA NWPS flood-stage metadata."""
import requests, io, json, math, sys
import pandas as pd
REGIONS = {
 "MidAtlantic": (-79.8, 37.6, -76.6, 40.2),
 "TexasHill":   (-100.2, 29.0, -97.0, 31.2),
 "OhioValley":  (-84.6, 37.0, -79.8, 40.0),
 "UpperMidwest":(-94.8, 41.0, -90.0, 43.8),
}
def nwis_sites(b):
    u=("https://nwis.waterservices.usgs.gov/nwis/site/?format=rdb&bBox=%.4f,%.4f,%.4f,%.4f"
       "&siteType=ST&siteStatus=active&hasDataTypeCd=iv&parameterCd=00065,00060&siteOutput=expanded")%b
    r=requests.get(u,timeout=120); r.raise_for_status()
    t=[l for l in r.text.splitlines() if not l.startswith('#')]
    df=pd.read_csv(io.StringIO("\n".join(t)),sep="\t",dtype=str).iloc[1:]
    return df
def nwps(b):
    u="https://api.water.noaa.gov/nwps/v1/gauges?bbox.xmin=%.4f&bbox.ymin=%.4f&bbox.xmax=%.4f&bbox.ymax=%.4f&srid=EPSG_4326"%b
    return requests.get(u,timeout=120).json()["gauges"]
def hav(a,b,c,d):
    R=6371;p=math.pi/180
    x=math.sin((c-a)*p/2)**2+math.cos(a*p)*math.cos(c*p)*math.sin((d-b)*p/2)**2
    return 2*R*math.asin(math.sqrt(x))
rows=[]
for reg,b in REGIONS.items():
    s=nwis_sites(b); g=nwps(b)
    print(reg,len(s),"nwis sites",len(g),"nwps gauges",flush=True)
    s["lat"]=s.dec_lat_va.astype(float); s["lon"]=s.dec_long_va.astype(float)
    s["da"]=pd.to_numeric(s.contrib_drain_area_va,errors="coerce")
    for _,r in s.iterrows():
        best=None
        for q in g:
            d=hav(r.lat,r.lon,q["latitude"],q["longitude"])
            if d<0.6 and (best is None or d<best[0]): best=(d,q)
        rows.append(dict(region=reg,site_no=r.site_no,name=r.station_nm,lat=r.lat,lon=r.lon,da_sqmi=r.da,
                         lid=best[1]["lid"] if best else None))
inv=pd.DataFrame(rows); inv.to_csv("data/raw/inventory_all.csv",index=False)
print(inv.groupby("region").agg(n=("site_no","size"),with_lid=("lid",lambda x:x.notna().sum())))
