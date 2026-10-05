import pickle, numpy as np, pandas as pd
from lib import *
H, meta = load(); net = Net(H, meta); S = H.shape[1]
sel = pickle.load(open("data/proc/prior.pkl", "rb"))
raw = pd.read_csv("data/proc/site_screening.csv", dtype={"site_no": str}).set_index("site_no")
out = {"n_sites": S, "n_regions": 4, "hours": len(H), "raw_readings": int(raw.loc[meta.site_no, "n_raw"].sum()),
       "hourly_obs": int(np.isfinite(H.values).sum()), "coverage_mean": float(meta.coverage.mean()), "coverage_min": float(meta.coverage.min())}
rows = []
for reg, g in meta.groupby("region"):
    rows.append(dict(region=reg, sites=len(g), coverage=g.coverage.mean(), minor=int(g.minor.notna().sum()), action=int(g.action.notna().sum()),
                     lat_span=g.lat.max() - g.lat.min(), lon_span=g.lon.max() - g.lon.min(),
                     hmed=g.hmed.median(), hstd=g.hstd.median(), sig_ft=float(np.median(net.sig[g.index]))))
pd.DataFrame(rows).to_csv("results/table_regions.csv", index=False)
# natural gaps
m = ~np.isfinite(H.values); gaps = []
for s in range(S):
    x = m[:, s]; d = np.diff(np.r_[0, x.astype(int), 0]); st = np.where(d == 1)[0]; en = np.where(d == -1)[0]; gaps += list(en - st)
gaps = np.array(gaps); out.update(missing_frac=float(m.mean()), n_gaps=int(len(gaps)), gap_median=float(np.median(gaps)), gap_p90=float(np.percentile(gaps, 90)), gap_gt6=float((gaps > 6).mean()))
# events per split
def events(s, thr, mask):
    h = net.Ht[:, s]; idx = np.where(mask)[0]; ab = h[idx] >= thr
    up = idx[np.where(ab[1:] & ~ab[:-1])[0] + 1]; n = 0; last = -10**9
    for c in up:
        if c - last >= 48: n += 1
        last = c
    return n
for lev, col in (("minor", meta.minor), ("action", meta.action)):
    for nm, mk in (("train", net.tr), ("val", net.va), ("test", net.te)):
        out[f"events_{lev}_{nm}"] = int(sum(events(s, col[s], mk) for s in range(S) if np.isfinite(col[s])))
    out[f"sites_with_test_{lev}_event"] = int(sum(events(s, col[s], net.te) > 0 for s in range(S) if np.isfinite(col[s])))
# prior
def hav(a, b, c, d):
    p = np.pi / 180; x = np.sin((c - a) * p / 2) ** 2 + np.cos(a * p) * np.cos(c * p) * np.sin((d - b) * p / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(x))
pr = []
for j, lst in sel.items():
    for (i, lag, rho, dr, z) in lst:
        dk = hav(meta.lat[i], meta.lon[i], meta.lat[j], meta.lon[j]); pr.append(dict(j=j, i=i, region=meta.region[j], lag=lag, rho=rho, direct=dr, rho0=z, dist_km=dk, celerity_kmh=dk / lag))
pr = pd.DataFrame(pr); pr.to_csv("results/prior_edges.csv", index=False)
out.update(n_edges=len(pr), nbr_mean=float(len(pr) / S), sites_with_4=int(sum(len(v) == 4 for v in sel.values())), sites_with_0=int(sum(len(v) == 0 for v in sel.values())),
           lag_median=float(pr.lag.median()), lag_p25=float(pr.lag.quantile(.25)), lag_p75=float(pr.lag.quantile(.75)), rho_median=float(pr.rho.median()),
           dist_median=float(pr.dist_km.median()), celerity_median=float(pr.celerity_kmh.median()), celerity_p10=float(pr.celerity_kmh.quantile(.1)), celerity_p90=float(pr.celerity_kmh.quantile(.9)),
           frac_lag_ge1=float((pr.lag >= 1).mean()), corr_dist_lag=float(np.corrcoef(pr.dist_km, pr.lag)[0, 1]))
print(pd.DataFrame(rows).round(3).to_string()); print(pr.groupby("region")[["lag", "rho", "dist_km", "celerity_kmh"]].median().round(2))
save_json(out, "results/dataset_stats.json"); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()})
