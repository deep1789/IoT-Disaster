import numpy as np, json, pandas as pd
from sklearn.metrics import average_precision_score
from evalcore import *
H, meta = load(); net = Net(H, meta); ev = Ev(net, meta)
M = ["persist", "ar", "tf", "lgbm", "gru", "full"]; rng = np.random.default_rng(7); B = 200
res = {}
for lev in ("minor", "action"):
    data = {}
    for m in M:
        seeds = ["s0"] if m in ("persist", "ar", "tf", "lgbm") else ["s0", "s1", "s2"]
        data[m] = []
        for sd in seeds:
            a = ev.alert_data(ev.load(f"results/preds/temporal_{m}_{sd}.npz"), lev); ok = a["ok"]
            data[m].append((a["ss"][ok], a["y"][ok], a["sc"][ok]))
    gauges = np.unique(data["full"][0][0][data["full"][0][1]]) if False else np.unique(data["full"][0][0])
    pos = {g: data["full"][0][1][data["full"][0][0] == g].sum() for g in gauges}
    idx = {m: [{g: np.where(d[0] == g)[0] for g in gauges} for d in data[m]] for m in M}
    aps = {m: [] for m in M}
    for b in range(B):
        gs = rng.choice(gauges, len(gauges), replace=True)
        for m in M:
            vals = []
            for k, d in enumerate(data[m]):
                ii = np.concatenate([idx[m][k][g] for g in gs])
                y = d[1][ii]
                vals.append(average_precision_score(y, d[2][ii]) if y.sum() > 0 else np.nan)
            aps[m].append(np.nanmean(vals))
    r = {m: dict(mean=float(np.nanmean(aps[m])), lo=float(np.nanpercentile(aps[m], 2.5)), hi=float(np.nanpercentile(aps[m], 97.5))) for m in M}
    for o in ("tf", "gru", "lgbm", "persist"):
        d = np.array(aps["full"]) - np.array(aps[o]); r[f"diff_full_minus_{o}"] = dict(mean=float(np.nanmean(d)), lo=float(np.nanpercentile(d, 2.5)), hi=float(np.nanpercentile(d, 97.5)), p_pos=float(np.nanmean(d > 0)))
    res[lev] = r; print(lev, {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in r.items()})
save_json(res, "results/bootstrap_ap.json")
