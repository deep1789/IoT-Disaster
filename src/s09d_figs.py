import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from figstyle import *; setup()
f = json.load(open("results/faults.json"))["results"]; g = json.load(open("results/faults_noaug.json"))["results"]
# Fig 9: robustness
def curve(scen, key, var, src=f): return [src[s][var][key] for s in scen]
fig, ax = plt.subplots(1, 3, figsize=(7.8, 2.9))
dr = ["clean_0", "dropout_0.1", "dropout_0.3", "dropout_0.5", "dropout_0.7"]; x = [0, 10, 30, 50, 70]
sty = {"persist": ("Persistence", C["grey"], "o", ":"), "tf": ("TF", C["blue"], "^", "--"), "gru": ("Vanilla GRU", C["magenta"], "D", "-"), "full": ("PG-FaultTwin", C["violet"], "o", "-")}
for v, (lab, col, mk, ls) in sty.items(): ax[0].plot(x, curve(dr, "nMAE_active", v), color=col, marker=mk, ls=ls, lw=1.6, ms=3.5, label=lab)
ax[0].plot(x, curve(dr, "nMAE_active", "noaug", g), color=C["yellow"], marker="s", ls="-.", lw=1.4, ms=3.5, label="Twin w/o curriculum")
ax[0].set_xlabel("Readings missing (%)"); ax[0].set_ylabel("Active-state nMAE"); ax[0].set_title("(a) Block dropouts", loc="left"); ax[0].legend(fontsize=6.5)
lt = ["clean_0", "latency_3", "latency_6", "latency_12"]; xl = [0, 3, 6, 12]
for v, (lab, col, mk, ls) in sty.items(): ax[1].plot(xl, curve(lt, "nMAE_active", v), color=col, marker=mk, ls=ls, lw=1.6, ms=3.5)
ax[1].set_xlabel("Telemetry delay (h)"); ax[1].set_title("(b) Latency", loc="left")
sp = ["clean_0", "spike_0.01", "spike_0.03"]; xs = np.arange(3); w = 0.16
vars_ = [("persist", f, C["grey"], "Persist."), ("gru", f, C["magenta"], "GRU"), ("full", f, C["violet"], "Twin"), ("full+QC", f, C["aqua"], "Twin+QC")]
for i, (v, src, col, lab) in enumerate(vars_): ax[2].bar(xs + (i - 1.5) * w, np.minimum(curve(sp, "nMAE", v, src), 2.5), w, color=col, label=lab, edgecolor="#fcfcfb", linewidth=0.8)
ax[2].set_xticks(xs); ax[2].set_xticklabels(["clean", "1% spikes", "3% spikes"]); ax[2].set_ylabel("nMAE (all origins, capped at 2.5)"); ax[2].set_title("(c) Spikes and QC gate", loc="left"); ax[2].legend(fontsize=6.5)
fig.tight_layout(); fig.savefig("figures/fig9_faults.png"); plt.close()
# Fig 10: conformal
c = json.load(open("results/conformal.json")); d = c["0.1"]; h = np.arange(1, 25)
fig, ax = plt.subplots(1, 3, figsize=(7.8, 2.8))
ax[0].plot(h, d["cov_raw"], color=C["grey"], ls=":", lw=1.6, label="raw quantile heads"); ax[0].plot(h, d["cov_cqr"], color=C["violet"], lw=1.8, label="CQR (marginal)"); ax[0].plot(h, d["cov_split"], color=C["blue"], ls="--", lw=1.4, label="split-conformal, symmetric")
ax[0].axhline(0.9, color=C["orange"], lw=1); ax[0].set_ylim(0.7, 1); ax[0].set_xlabel("Lead time (h)"); ax[0].set_ylabel("Empirical coverage"); ax[0].set_title("(a) Marginal coverage", loc="left"); ax[0].legend(fontsize=6.5, loc="lower right")
ax[1].plot(h, d["cov_cqr_rising"], color=C["violet"], lw=1.8, label="CQR, rising"); ax[1].plot(h, d["cov_mond_rising"], color=C["aqua"], lw=1.8, label="Mondrian, rising"); ax[1].plot(h, d["cov_split_rising"], color=C["blue"], ls="--", lw=1.4, label="split, rising")
ax[1].plot(h, d["cov_cqr_active"], color=C["magenta"], ls="-.", lw=1.4, label="CQR, realised active"); ax[1].axhline(0.9, color=C["orange"], lw=1); ax[1].set_ylim(0, 1); ax[1].set_xlabel("Lead time (h)"); ax[1].set_title("(b) Coverage by regime", loc="left"); ax[1].legend(fontsize=6.3, loc="lower left")
ax[2].plot(h, d["width_raw_ft"], color=C["grey"], ls=":", lw=1.6, label="raw"); ax[2].plot(h, d["width_cqr_ft"], color=C["violet"], lw=1.8, label="CQR"); ax[2].plot(h, d["width_split_ft"], color=C["blue"], ls="--", lw=1.4, label="split, symmetric")
ax[2].set_xlabel("Lead time (h)"); ax[2].set_ylabel("Mean interval width (ft)"); ax[2].set_title("(c) Sharpness", loc="left"); ax[2].legend(fontsize=6.5)
fig.tight_layout(); fig.savefig("figures/fig10_conformal.png"); plt.close()
# Fig 11: physics consistency
n = pd.read_csv("results/neighbour_importance.csv")
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.6))
ax[0].scatter(n.rho, n.d_nMAE, s=8, color=C["violet"], alpha=0.6); ax[0].set_xlabel("Inferred peak correlation ρ*"); ax[0].set_ylabel("Δ active nMAE when neighbour removed"); ax[0].axhline(0, color=C["grey"], lw=0.8); ax[0].set_title("(a)", loc="left")
ax[1].boxplot([n[n["rank"] == r].d_nMAE for r in (1, 2, 3, 4)], tick_labels=["1", "2", "3", "4"], showfliers=False, boxprops=dict(color=C["sec"]), medianprops=dict(color=C["violet"])); ax[1].axhline(0, color=C["grey"], lw=0.8); ax[1].set_xlabel("Neighbour rank"); ax[1].set_title("(b)", loc="left")
b = pd.cut(n.lag, [0, 1, 2, 4, 100], labels=["1 h", "2 h", "3–4 h", ">4 h"]); ax[2].boxplot([n[b == k].d_nMAE for k in b.cat.categories], tick_labels=list(b.cat.categories), showfliers=False, boxprops=dict(color=C["sec"]), medianprops=dict(color=C["violet"])); ax[2].axhline(0, color=C["grey"], lw=0.8); ax[2].set_xlabel("Inferred travel time τ*"); ax[2].set_title("(c)", loc="left")
fig.tight_layout(); fig.savefig("figures/fig11_physics.png"); plt.close(); print("ok")
