"""Generate result tables (markdown) and significance tests from results/tables/*.json"""
import json, glob, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
T = {os.path.basename(f)[2:-5]: json.load(open(f)) for f in glob.glob("results/tables/m_*.json")}
def grp(name, proto="temporal"): return [v for k, v in T.items() if k.startswith(f"{proto}_{name}_s")]
def ms(name, key, d=3, proto="temporal", sd=True):
    g = grp(name, proto)
    if not g: return "n/a"
    x = np.array([v[key] for v in g], float)
    return f"{x.mean():.{d}f}" + (f" ± {x.std(ddof=1):.{d}f}" if len(x) > 1 and sd else "")
NAMES = {"persist": "Persistence", "ar": "AR (own gauge)", "tf": "TF (routing prior)", "lgbm": "LightGBM", "gru": "Vanilla GRU", "full": "**PG-FaultTwin**"}
out = {}
# T4 accuracy
rows = ["| Method | nMAE (all) | nMAE quiet | nMAE active | MAE h=6 (ft) | MAE h=24 (ft) | RMSE h=24 (ft) | nRMSE active |", "|---|---|---|---|---|---|---|---|"]
for n in NAMES:
    rows.append(f"| {NAMES[n]} | " + " | ".join(ms(n, k, 3 if "nM" in k or "nR" in k else 3) for k in ["nMAE", "nMAE_quiet", "nMAE_active", "MAE_h6", "MAE_h24", "RMSE_h24", "nRMSE_active"]) + " |")
out["acc"] = "\n".join(rows)
# T5 gradient
rows = ["| Method | 3-h rise-rate MAE (ft/h) | 3-h rise-rate MAE active | corr. 3-h | corr. 6-h | rapid-rise F1 |", "|---|---|---|---|---|---|"]
for n in NAMES:
    rows.append(f"| {NAMES[n]} | " + " | ".join(ms(n, k, 4 if "MAE" in k else 3) for k in ["grad3_MAE", "grad3_MAE_active", "grad3_corr", "grad6_corr", "rapid_rise_F1"]) + " |")
out["grad"] = "\n".join(rows)
# T6 alerts
for lev in ("minor", "action"):
    rows = ["| Method | AP | POD | FAR | CSI | events detected / total | mean lead (h) | mean lead of detected (h) | false-alarm episodes per gauge-year |", "|---|---|---|---|---|---|---|---|---|"]
    for n in NAMES:
        g = grp(n)
        if not g: continue
        det = np.mean([v[f"detected_{lev}"] for v in g]); tot = g[0][f"events_{lev}"]
        rows.append(f"| {NAMES[n]} | " + " | ".join(ms(n, f"{k}_{lev}", 3) for k in ["AP", "POD", "FAR", "CSI"]) + f" | {det:.0f} / {tot} | " + " | ".join(ms(n, f"{k}_{lev}", 2) for k in ["lead_mean", "lead_detected_mean", "FA_per_gauge_year"]) + " |")
    out[f"alert_{lev}"] = "\n".join(rows)
# T7 ablation
AB = [("full", "Full PG-FaultTwin"), ("noroute", "– routing prior"), ("noaug", "– fault curriculum"), ("geo", "geographic instead of physics-selected neighbours"), ("own", "own gauge only (no neighbours)"), ("evsamp", "+ event-focused sampling (weight 7)"), ("gru", "Vanilla GRU (all elements removed)")]
rows = ["| Variant | nMAE | nMAE active | grad. corr. 3-h | AP (minor) | AP (action) | CSI (action) |", "|---|---|---|---|---|---|---|"]
for n, lab in AB:
    g = grp(n)
    if not g: continue
    s0 = [v for k, v in T.items() if k == f"temporal_{n}_s0"][0]
    rows.append(f"| {lab} | " + " | ".join(f"{s0[k]:.3f}" for k in ["nMAE", "nMAE_active", "grad3_corr", "AP_minor", "AP_action", "CSI_action"]) + " |")
out["abl"] = "\n".join(rows)
# T8 LORO
regs = ["MidAtlantic", "OhioValley", "TexasHill", "UpperMidwest"]
rows = ["| Held-out region | Method | nMAE | nMAE active | grad. corr. 3-h | AP (action) | CSI (action) |", "|---|---|---|---|---|---|---|"]
for r in regs:
    for n in ["tf", "lgbm", "gru", "full"]:
        k = f"loro-{r}_{n}_s0"
        if k in T: rows.append(f"| {r} | {NAMES[n]} | " + " | ".join(f"{T[k][x]:.3f}" for x in ["nMAE", "nMAE_active", "grad3_corr", "AP_action", "CSI_action"]) + " |")
out["loro"] = "\n".join(rows)
# significance (per gauge)
def pg(name, key="_pergauge"):
    g = grp(name)
    if not g: return None
    ks = set.intersection(*[set(v[key].keys()) for v in g])
    return pd.Series({int(k): np.mean([v[key][k] for v in g]) for k in ks})
rows = ["| Comparison | metric | gauges | median Δ (σ) | % gauges improved | p (Wilcoxon) | p (BH-adj.) |", "|---|---|---|---|---|---|---|"]; ps = []; tmp = []
for other in ["persist", "ar", "tf", "lgbm", "gru"]:
    for key, lab in (("_pergauge", "nMAE all"), ("_pergauge_active", "nMAE active")):
        a, b = pg("full", key), pg(other, key)
        if a is None or b is None: continue
        d = (a - b).dropna()
        d = d[np.isfinite(d)]
        p = wilcoxon(d).pvalue if len(d) > 5 else np.nan
        tmp.append((other, lab, len(d), d.median(), (d < 0).mean(), p))
ps = np.array([t[5] for t in tmp]); o = np.argsort(ps); adj = np.empty(len(ps)); m = len(ps); run = 1
for rank, i in enumerate(o[::-1]): 
    r = m - rank; run = min(run, ps[i] * m / r); adj[i] = run
for (other, lab, n, med, fr, p), a in zip(tmp, adj): rows.append(f"| PG-FaultTwin vs {NAMES[other].replace('**','')} | {lab} | {n} | {med:+.4f} | {100*fr:.0f}% | {p:.2g} | {a:.2g} |")
out["sig"] = "\n".join(rows)
for k, v in out.items(): open(f"manuscript/tbl_{k}.md", "w").write(v)
print("\n\n".join(f"## {k}\n{v}" for k, v in out.items()))
