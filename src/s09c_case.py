import json, numpy as np, pandas as pd, matplotlib.pyplot as plt, pickle
from figstyle import *; setup()
from evalcore import *
H, meta = load(); net = Net(H, meta); ev = Ev(net, meta); sig = net.sig
sel = pickle.load(open("data/proc/prior.pkl", "rb"))
F = ev.load("results/preds/temporal_full_s0.npz"); P = ev.load("results/preds/temporal_persist_s0.npz"); G = ev.load("results/preds/temporal_gru_s0.npz")
a = ev.alert_data(F, "minor"); d_f = json.load(open("results/tables/m_temporal_full_s0.json"))["delta_minor"]
e = ev.events(a, d_f, level="minor"); e = e[e.lead >= 0].copy()
# candidate events: large lead for full, big rise above threshold
rows = []
for _, r in e.iterrows():
    s, c = int(r.s), int(r.t); thr = ev.thr_minor[s]
    pk = np.nanmax(net.Ht[c:c + 24, s]); rows.append((s, c, r.lead, pk - thr, len(sel[s])))
cand = pd.DataFrame(rows, columns=["s", "c", "lead", "excess", "nn"]); cand = cand[(cand.lead >= 8) & (cand.nn >= 2)].sort_values("excess", ascending=False)
print(cand.head(5))
s, c = int(cand.iloc[0].s), int(cand.iloc[0].c); thr = ev.thr_minor[s]
nb = [x[0] for x in sel[s][:2]]
t0, t1 = c - 40, c + 16; tx = np.arange(t0, t1)
fig, ax = plt.subplots(2, 1, figsize=(6.4, 5.0), sharex=True, gridspec_kw=dict(height_ratios=[1, 2.1]))
for j, col in zip(nb, (C["aqua"], C["yellow"])):
    ax[0].plot(tx - c, net.Ht[tx, j], color=col, lw=1.4, label=f"upstream {meta.site_no[j]}")
ax[0].set_ylabel("Upstream stage (ft)"); ax[0].legend(fontsize=7, loc="upper left")
ax[1].plot(tx - c, net.Ht[tx, s], color=C["ink"], lw=2, label=f"observed {meta.site_no[s]}")
ax[1].axhline(thr, color=C["orange"], ls="--", lw=1.2); ax[1].text(t0 - c + 0.5, thr, " NWS minor flood stage", color=C["orange"], fontsize=7, va="bottom")
for lag, col in ((-14, C["violet"]), (-8, C["blue"]), (-4, C["magenta"])):
    to = c + lag; to -= to % 2
    for D, nm, ls in ((F, "twin", "-"),):
        m = (D["tt"] == to) & (D["ss"] == s)
        if m.sum() == 0: continue
        i = np.where(m)[0][0]; h = np.arange(1, 25)
        med = ev.phys(D["med"][i:i + 1], np.array([s]))[0]; lo = ev.phys(D["lo"][i:i + 1], np.array([s]))[0]; hi = ev.phys(D["hi"][i:i + 1], np.array([s]))[0]
        ax[1].plot(to + h - c, med, color=col, lw=1.6, ls=ls, label=f"twin forecast issued {abs(to - c)} h before crossing")
        ax[1].fill_between(to + h - c, lo, hi, color=col, alpha=0.15, lw=0)
        ax[1].scatter([to - c], [net.Ht[to, s]], color=col, s=14, zorder=5)
        mp = (P["tt"] == to) & (P["ss"] == s)
ax[1].set_xlabel("Hours relative to threshold crossing"); ax[1].set_ylabel("Stage (ft)"); ax[1].legend(fontsize=6.5, loc="upper left")
ax[0].set_title(f"{meta.name[s].title()} — {pd.Timestamp(net.index[c]).strftime('%d %b %Y')}", fontsize=8, loc="left")
fig.tight_layout(); fig.savefig("figures/fig4_case.png"); plt.close()
json.dump(dict(site=meta.site_no[s], name=meta.name[s], t=str(net.index[c]), thr=float(thr), lead=float(cand.iloc[0].lead), excess=float(cand.iloc[0].excess)), open("results/case_study.json", "w"))
