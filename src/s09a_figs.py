import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figstyle import *; setup()
m = pd.read_csv("data/proc/meta.csv", dtype={"site_no": str}); e = pd.read_csv("results/prior_edges.csv")
# Fig 2: map + edges
fig, ax = plt.subplots(2, 2, figsize=(7.2, 6.4))
for a, (r, (nm, col)) in zip(ax.ravel(), REG.items()):
    g = m[m.region == r]
    for _, x in e[e.region == r].iterrows():
        i, j = int(x.i), int(x.j)
        a.annotate("", xy=(m.lon[j], m.lat[j]), xytext=(m.lon[i], m.lat[i]), arrowprops=dict(arrowstyle="->", color=C["grey"], lw=0.5 + 1.5 * max(x.rho, 0), alpha=0.6))
    a.scatter(g.lon, g.lat, s=np.where(g.minor.notna(), 38, 18), c=col, edgecolor="#fcfcfb", linewidth=0.8, zorder=3)
    a.set_title(f"{nm} (n={len(g)})", fontsize=9); a.set_xlabel("Longitude (°)"); a.set_ylabel("Latitude (°)"); a.grid(False)
fig.suptitle("Gauge networks and inferred upstream→downstream edges (arrow width ∝ ρ*; large dots: official flood stage)", fontsize=9)
fig.tight_layout(); fig.savefig("figures/fig2_map.png"); plt.close()
# Fig 3: lag & celerity diagnostics
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6))
for r, (nm, col) in REG.items():
    q = e[e.region == r]; ax[0].hist(q.lag, bins=np.arange(0.5, 14.5, 1), alpha=0.55, color=col, label=nm, histtype="stepfilled")
ax[0].set_xlabel("Inferred travel time τ* (h)"); ax[0].set_ylabel("Edges"); ax[0].legend()
ax[1].scatter(e.dist_km, e.lag, s=8, c=[REG[r][1] for r in e.region], alpha=0.7); ax[1].set_xlabel("Straight-line distance (km)"); ax[1].set_ylabel("τ* (h)"); ax[1].set_xlim(0, 150); ax[1].set_ylim(0, 25)
xs = np.linspace(1, 150, 50)
for c, ls in ((3, ":"), (11, "--")): ax[1].plot(xs, xs / c, color=C["sec"], ls=ls, lw=1); ax[1].text(148, min(148 / c, 24), f"{c} km/h", ha="right", fontsize=7, color=C["sec"])
ax[2].hist(np.clip(e.celerity_kmh, 0, 60), bins=30, color=C["violet"], alpha=0.8); ax[2].axvspan(3.6, 11, color=C["aqua"], alpha=0.2); ax[2].set_xlabel("Apparent celerity (km/h)"); ax[2].set_ylabel("Edges")
for a, t in zip(ax, "abc"): a.set_title(f"({t})", loc="left", fontsize=9)
fig.tight_layout(); fig.savefig("figures/fig3_prior.png"); plt.close()
# Fig 1: architecture
fig, ax = plt.subplots(figsize=(9, 3.4)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 44)
def box(x, y, w, h, t, col, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=col, ec="none")); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=5.6, color=tc)
def arr(x1, y1, x2, y2): ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, color=C["sec"], lw=1))
box(1, 28, 20, 12, "SENSING LAYER\n99 telemetered gauges\n15-min stage → hourly\n(dropouts, spikes, stuck,\nbias, latency)", C["grey"])
box(26, 28, 22, 12, "EDGE LAYER (gateway)\ncausal QC gate\nmask + gap bookkeeping\nrouting prior (ridge)\nGRU residual (51k params)", C["blue"])
box(53, 28, 22, 12, "TWIN LAYER\nstage + sensor trust\n24-h trajectory\nCQR/ACI intervals\nwhat-if queries", C["violet"])
box(80, 28, 19, 12, "DECISION LAYER\nrise rate, time to\nthreshold, alert margin δ\n/ cost–loss ratio r", C["orange"])
for x in (21.5, 48.5, 75.5): arr(x, 34, x + 4, 34)
box(26, 4, 22, 15, "TOPOLOGY-FREE PRIOR\nlagged cross-correlation\n→ upstream set, τ*\n(training years only)", C["aqua"], "white")
box(53, 4, 22, 15, "FAULT CURRICULUM\nrandom/block/stale/lost\nchannel dropout in training;\nmask is a model input", C["yellow"], C["ink"])
box(80, 4, 19, 15, "VERIFICATION\nPOD / FAR / CSI,\nlead time, false alarms,\neconomic value", C["magenta"], "white")
arr(37, 19.5, 37, 27); arr(64, 19.5, 64, 27); arr(89.5, 27, 89.5, 19.5)
fig.savefig("figures/fig1_architecture.png", bbox_inches="tight"); plt.close(); print("ok")
