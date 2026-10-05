import json, glob, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from figstyle import *; setup()
from evalcore import *
M = json.load(open("results/all_metrics.json"))
def tags(name, proto="temporal"): return [k for k in M if k.startswith(f"{proto}_{name}_s")]
def avg(name, key, proto="temporal"):
    t = tags(name, proto); return np.mean([M[k][key] for k in t], axis=0) if t else None
def sty(n): lab, col, mk, ls = METH[n]; return dict(color=col, marker=mk, ls=ls, lw=1.6, ms=3.5, label=lab)
order = ["persist", "ar", "tf", "lgbm", "gru", "full"]
# Fig 5: accuracy vs horizon
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
for n in order:
    c = avg(n, "curve_nMAE"); ca = avg(n, "curve_nMAE_active")
    if c is None: continue
    ax[0].plot(range(1, 25), c, **sty(n)); ax[1].plot(range(1, 25), ca, **sty(n))
ax[0].set_title("(a) All test origins", loc="left"); ax[1].set_title("(b) Active origins (|Δ|>1σ)", loc="left")
for a in ax: a.set_xlabel("Lead time (h)"); a.set_ylabel("Normalised MAE (σ units)")
ax[1].legend(ncol=2, fontsize=7); fig.tight_layout(); fig.savefig("figures/fig5_accuracy.png"); plt.close()
# alert curves from predictions
H, meta = load(); net = Net(H, meta); ev = Ev(net, meta)
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.8))
leads = {}
for n in order:
    f = f"results/preds/temporal_{n}_s0.npz"
    if not os.path.exists(f): continue
    a = ev.alert_data(ev.load(f), "minor"); ok = a["ok"]; p, r, _ = precision_recall_curve(a["y"][ok], a["sc"][ok])
    st = sty(n); ax[0].plot(r, p, color=st["color"], ls=st["ls"], lw=1.6, label=st["label"])
    sc = ev.alert_scores(a); ax[1].plot(1 - sc.FAR, sc.POD, color=st["color"], ls=st["ls"], lw=1.6)
    b = sc.loc[sc.CSI.idxmax()]; ax[1].scatter([1 - b.FAR], [b.POD], color=st["color"], marker=st["marker"], s=22, zorder=4)
    leads[n] = [x for x in json.load(open(f"results/tables/m_temporal_{n}_s0.json"))["_leads_minor"]] if os.path.exists(f"results/tables/m_temporal_{n}_s0.json") else []
ax[0].set_xlabel("Recall (POD)"); ax[0].set_ylabel("Precision"); ax[0].set_title("(a) Precision–recall", loc="left"); ax[0].set_ylim(0, 1); ax[0].legend(fontsize=7)
xs = np.linspace(0.01, 1, 100)
for c in (0.2, 0.4, 0.6): ax[1].plot(xs, c * xs / (xs + xs - c * xs + 1e-9) if False else 1 / (1 / c - 1 / xs + 1) if False else xs * 0 + np.nan, lw=0)
ax[1].set_xlabel("Success ratio (1 − FAR)"); ax[1].set_ylabel("POD"); ax[1].set_title("(b) Performance diagram", loc="left"); ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1)
bp = [leads[n] for n in order if n in leads and len(leads[n])]; nm = [METH[n][0].replace(" (routing)", "").replace("PG-FaultTwin", "PG-FT").replace("Persistence", "Persist.").replace("Vanilla GRU", "GRU") for n in order if n in leads and len(leads[n])]
ax[2].boxplot(bp, tick_labels=nm, showmeans=True, widths=0.6, boxprops=dict(color=C["sec"]), medianprops=dict(color=C["violet"]), meanprops=dict(markerfacecolor=C["orange"], markeredgecolor="none"))
ax[2].set_ylabel("Lead time (h)"); ax[2].set_title("(c) Event lead time", loc="left"); ax[2].tick_params(axis="x", rotation=40, labelsize=7)
fig.tight_layout(); fig.savefig("figures/fig6_alerts.png"); plt.close()
# Fig 7: economic value
RS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
fig, ax = plt.subplots(figsize=(3.6, 2.8))
for n in order:
    v = avg(n, "V_curve")
    if v is not None: ax.plot(RS, v, **sty(n))
ax.set_xscale("log"); ax.set_xlabel("Cost–loss ratio C/L"); ax.set_ylabel("Relative economic value V"); ax.legend(fontsize=6.5); fig.tight_layout(); fig.savefig("figures/fig7_value.png"); plt.close()
print("figs ok")
