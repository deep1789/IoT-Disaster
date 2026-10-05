"""Main evaluation: accuracy, gradient, alerts, events, economic value, ablations, LORO, stats."""
import glob, os, json, re, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from evalcore import *
H, meta = load(); net = Net(H, meta); ev = Ev(net, meta); sig = net.sig
RS = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7])
os.makedirs("results/tables", exist_ok=True)
cache = {}


def metrics(f, with_events=True):
    d = ev.load(f); tt, ss = d["tt"], d["ss"]
    err, yh, ph, h0 = ev.accuracy(d)
    nerr = err / sig[ss][:, None]
    dz = (yh - h0[:, None]) / sig[ss][:, None]
    active = np.nanmax(np.abs(dz[:, [5, 11, 23]]), axis=1) > 1.0
    rec = dict(n=len(tt), active_share=float(active.mean()))
    rec["nMAE"] = float(nerr.mean()); rec["nMAE_active"] = float(nerr[active].mean()); rec["nMAE_quiet"] = float(nerr[~active].mean())
    for h in (1, 3, 6, 12, 24):
        rec[f"MAE_h{h}"] = float(err[:, h - 1].mean()); rec[f"RMSE_h{h}"] = float(np.sqrt((err[:, h - 1] ** 2).mean()))
        rec[f"nMAE_h{h}"] = float(nerr[:, h - 1].mean()); rec[f"nMAEa_h{h}"] = float(nerr[active, h - 1].mean())
    rec["nRMSE_active"] = float(np.sqrt((nerr[active] ** 2).mean()))
    rec["nRMSE"] = float(np.sqrt((nerr ** 2).mean()))
    rec["curve_nMAE"] = nerr.mean(0).tolist(); rec["curve_nMAE_active"] = nerr[active].mean(0).tolist()
    # gradient: 3-h rise rate (ft/h)
    for k in (3, 6):
        g_pred = (ph[:, k - 1] - h0) / k; g_true = (yh[:, k - 1] - h0) / k
        rec[f"grad{k}_MAE"] = float(np.abs(g_pred - g_true).mean()); rec[f"grad{k}_MAE_active"] = float(np.abs(g_pred - g_true)[active].mean())
        rec[f"grad{k}_corr"] = float(np.corrcoef(g_pred, g_true)[0, 1])
    rr_true = (yh[:, 2] - h0) / sig[ss] > 0.5; rr_pred = (ph[:, 2] - h0) / sig[ss] > 0.5
    tp = (rr_true & rr_pred).sum(); fp = (~rr_true & rr_pred).sum(); fn = (rr_true & ~rr_pred).sum()
    rec["rapid_rise_F1"] = float(2 * tp / max(2 * tp + fp + fn, 1)); rec["rapid_rise_n"] = int(rr_true.sum())
    # per-gauge nMAE (for tests)
    pg = pd.DataFrame({"s": ss, "e": nerr.mean(1), "ea": np.where(active, nerr.mean(1), np.nan)}).groupby("s").mean()
    rec["_pergauge"] = pg.e.to_dict(); rec["_pergauge_active"] = pg.ea.to_dict()
    # alerts
    for lev in ("minor", "action"):
        a = ev.alert_data(d, lev)
        rec[f"AP_{lev}"] = float(ev.ap(a)); rec[f"pos_{lev}"] = int(a["y"][a["ok"]].sum())
        sc = ev.alert_scores(a); b = sc.loc[sc.CSI.idxmax()]
        rec[f"CSI_{lev}"] = float(b.CSI); rec[f"POD_{lev}"] = float(b.POD); rec[f"FAR_{lev}"] = float(b.FAR); rec[f"delta_{lev}"] = float(b.delta)
        if with_events:
            e = ev.events(a, b.delta, level=lev)
            evs = e[e.lead >= 0]; fa = e[e.lead == -1]
            ng = int(np.isfinite((ev.thr_minor if lev == "minor" else ev.thr_action)[np.unique(ss)]).sum())
            rec[f"events_{lev}"] = int(len(evs)); rec[f"detected_{lev}"] = int((evs.lead > 0).sum())
            rec[f"lead_mean_{lev}"] = float(evs.lead.mean()); rec[f"lead_median_{lev}"] = float(evs.lead.median())
            rec[f"lead_detected_mean_{lev}"] = float(evs[evs.lead > 0].lead.mean()) if (evs.lead > 0).any() else 0.0
            rec[f"FA_per_gauge_year_{lev}"] = float(len(fa) / (ng * 3.0))
            rec[f"_leads_{lev}"] = evs.lead.tolist()
        if lev == "minor":
            ok = a["ok"]; rec["V_curve"] = cost_loss_value(a["sc"][ok], a["y"][ok], RS).tolist()
            rec["PR_sc"] = None
    return rec


files = sorted(glob.glob("results/preds/*.npz")); files = [f for f in files if not f.endswith("_val.npz")]
allrec = {}
for f in files:
    tag = os.path.basename(f)[:-4]
    out_f = f"results/tables/m_{tag}.json"
    if os.path.exists(out_f) and os.path.getmtime(out_f) > os.path.getmtime(f):
        allrec[tag] = json.load(open(out_f)); continue
    print("eval", tag, flush=True)
    r = metrics(f); allrec[tag] = r; save_json(r, out_f)
save_json({k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in allrec.items()}, "results/all_metrics.json")
print("done", len(allrec))
