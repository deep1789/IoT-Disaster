import numpy as np, pandas as pd, json
from evalcore import *
H, meta = load(); net = Net(H, meta); ev = Ev(net, meta); sig = net.sig
regs = ["MidAtlantic", "OhioValley", "TexasHill", "UpperMidwest"]; reg = meta.region.values
rows = []
def met(d, sites):
    m = np.isin(d["ss"], sites); dd = dict(tt=d["tt"][m], ss=d["ss"][m], med=d["med"][m])
    err, yh, ph, h0 = ev.accuracy(dd); ss = dd["ss"]; nerr = err / sig[ss][:, None]
    dz = (yh - h0[:, None]) / sig[ss][:, None]; act = np.nanmax(np.abs(dz[:, [5, 11, 23]]), axis=1) > 1
    g3p = (ph[:, 2] - h0) / 3; g3t = (yh[:, 2] - h0) / 3
    a = ev.alert_data(dd, "action"); am = ev.alert_data(dd, "minor")
    return dict(nMAE=nerr.mean(), nMAE_active=nerr[act].mean(), active_share=act.mean(), grad_corr=np.corrcoef(g3p, g3t)[0, 1], AP_action=ev.ap(a), pos_action=int(a["y"][a["ok"]].sum()))
for r in regs:
    sites = np.where(reg == r)[0]
    for n in ["persist", "tf", "gru", "full"]:
        for proto in (["temporal", "loro-" + r] if n != "persist" else ["temporal"]):
            seeds = ["s0", "s1", "s2"] if (n in ("gru", "full") and proto == "temporal") else ["s0"]
            ms = [met(ev.load(f"results/preds/{proto}_{n}_{s}.npz"), sites) for s in seeds]
            rows.append(dict(region=r, method=n, protocol="LORO" if proto.startswith("loro") else "temporal", **{k: float(np.mean([m[k] for m in ms])) for k in ms[0]}))
df = pd.DataFrame(rows); df.to_csv("results/region_table.csv", index=False)
print(df.round(3).to_string())
