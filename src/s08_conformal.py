"""Conformal calibration (split CQR, Mondrian, ACI) for the full twin; ensemble/raw-quantile comparators."""
import json, numpy as np, pandas as pd
from evalcore import *
H, meta = load(); net = Net(H, meta); ev = Ev(net, meta); sig = net.sig
tag = "temporal_full_s0"
V = ev.load(f"results/preds/{tag}_val.npz"); T = ev.load(f"results/preds/{tag}.npz")
def prep(d):
    tt, ss = d["tt"], d["ss"]; yh, h0 = ev.truth(tt, ss)
    y = (yh - net.med[ss][:, None]) / sig[ss][:, None]
    ok = np.isfinite(y).all(1)
    now = net.Zin[tt, ss]
    rising = (net.Zin[tt, ss] - net.Zin[tt - 3, ss]) > 0.25
    return {k: (d[k][ok].astype(np.float32) if k in ("med", "lo", "hi") else d[k][ok]) for k in ("tt", "ss", "med", "lo", "hi")}, y[ok], rising[ok]
Vd, yv, rv = prep(V); Td, yt, rt = prep(T)
out = {}
for alpha in (0.2, 0.1, 0.05):
    q = cqr_calibrate(Vd["lo"], Vd["hi"], yv, alpha)
    lo, hi = Td["lo"] - q, Td["hi"] + q
    cov = ((yt >= lo) & (yt <= hi))
    raw = ((yt >= Td["lo"]) & (yt <= Td["hi"]))
    w = (hi - lo) * sig[Td["ss"]][:, None]
    # Gaussian-residual baseline: symmetric interval from validation residual quantile of |y-med| per horizon (split conformal on point forecast)
    rv_abs = np.abs(yv - Vd["med"]); qa = np.array([np.quantile(rv_abs[:, h], min(1, np.ceil((len(rv_abs) + 1) * (1 - alpha)) / len(rv_abs))) for h in range(24)])
    cov_s = (np.abs(yt - Td["med"]) <= qa)
    w_s = 2 * qa[None, :] * sig[Td["ss"]][:, None]
    # Mondrian by rising state
    qm = {}
    for st in (False, True):
        qm[st] = cqr_calibrate(Vd["lo"][rv == st], Vd["hi"][rv == st], yv[rv == st], alpha)
    covm = np.zeros_like(cov); wm = np.zeros(cov.shape)
    for st in (False, True):
        m = rt == st; l2 = Td["lo"][m] - qm[st]; h2 = Td["hi"][m] + qm[st]
        covm[m] = (yt[m] >= l2) & (yt[m] <= h2); wm[m] = (h2 - l2) * sig[Td["ss"][m]][:, None]
    out[str(alpha)] = dict(
        nominal=1 - alpha, cov_raw=raw.mean(0).tolist(), cov_cqr=cov.mean(0).tolist(), cov_split=cov_s.mean(0).tolist(), cov_mond=covm.mean(0).tolist(),
        width_raw_ft=((Td["hi"] - Td["lo"]) * sig[Td["ss"]][:, None]).mean(0).tolist(), width_cqr_ft=w.mean(0).tolist(), width_split_ft=w_s.mean(0).tolist(), width_mond_ft=wm.mean(0).tolist(),
        cov_cqr_rising=cov[rt].mean(0).tolist(), cov_cqr_notrising=cov[~rt].mean(0).tolist(), cov_mond_rising=covm[rt].mean(0).tolist(),
        cov_split_rising=cov_s[rt].mean(0).tolist(), cov_raw_rising=raw[rt].mean(0).tolist(), rising_share=float(rt.mean()), q=q.tolist())
    # by active regime
    act = np.abs(yt - net.Zin[Td["tt"], Td["ss"]][:, None]).max(1) > 1.0
    out[str(alpha)]["cov_cqr_active"] = cov[act].mean(0).tolist(); out[str(alpha)]["cov_mond_active"] = covm[act].mean(0).tolist()
    out[str(alpha)]["cov_cqr_quiet"] = cov[~act].mean(0).tolist(); out[str(alpha)]["active_share"] = float(act.mean())
    # per-gauge coverage spread at 6 h and 24 h
    df = pd.DataFrame({"s": Td["ss"], "c6": cov[:, 5], "c24": cov[:, 23]}).groupby("s").mean()
    out[str(alpha)]["gauge_cov6_p10_p90"] = [float(df.c6.quantile(.1)), float(df.c6.quantile(.9))]; out[str(alpha)]["gauge_cov24_p10_p90"] = [float(df.c24.quantile(.1)), float(df.c24.quantile(.9))]
# ACI online on a subset of gauges (stream per gauge, horizons 6 and 24)
alpha = 0.1; res = {}
rng = np.random.default_rng(0); gauges = np.unique(Td["ss"]); gauges = rng.choice(gauges, min(30, len(gauges)), replace=False)
for h in (5, 23):
    cs, cf = [], []
    for g in gauges:
        m = Td["ss"] == g; o = np.argsort(Td["tt"][m])
        lo, hi, y = Td["lo"][m][o, h], Td["hi"][m][o, h], yt[m][o, h]
        covA, qs = aci_run(lo, hi, y, None, alpha=alpha, gamma=0.005, delay=(h + 1) // 2)
        mv = Vd["ss"] == g
        qf = np.quantile(np.maximum(Vd["lo"][mv, h] - yv[mv, h], yv[mv, h] - Vd["hi"][mv, h]), 1 - alpha) if mv.sum() > 20 else 0
        covF = (y >= lo - qf) & (y <= hi + qf)
        cs.append(covA.mean()); cf.append(covF.mean())
    res[f"h{h+1}"] = dict(aci_mean=float(np.mean(cs)), aci_sd=float(np.std(cs)), static_mean=float(np.mean(cf)), static_sd=float(np.std(cf)), aci_dev=float(np.mean(np.abs(np.array(cs) - 0.9))), static_dev=float(np.mean(np.abs(np.array(cf) - 0.9))))
out["aci"] = res
# alert rule from conformal upper bound: margin sweep using upper bound at alpha
from sklearn.metrics import average_precision_score
tt, ss = Td["tt"], Td["ss"]; thr = ev.thr_minor[ss]; h0 = net.Ht[tt, ss]; ok = np.isfinite(thr) & (h0 < thr)
yh, _ = ev.truth(tt, ss); ytrue = np.nanmax(yh, axis=1) >= thr
for alpha in (0.4, 0.2, 0.1, 0.05):
    q = cqr_calibrate(Vd["lo"], Vd["hi"], yv, alpha)
    hiC = (Td["hi"] + q) * sig[ss][:, None] + net.med[ss][:, None]
    al = (np.nanmax(hiC, axis=1) >= thr) & ok; yy = ytrue & ok
    tp = (al & yy).sum(); fp = (al & ~yy).sum(); fn = (~al & yy).sum()
    out[f"alert_upper_{alpha}"] = dict(POD=float(tp / max(tp + fn, 1)), FAR=float(fp / max(tp + fp, 1)), CSI=float(tp / max(tp + fp + fn, 1)))
save_json(out, "results/conformal.json"); print({k: (v if not isinstance(v, dict) else list(v)[:3]) for k, v in out.items() if k in ("aci",)}); print(out["0.1"]["cov_cqr"][5], out["0.1"]["cov_raw"][5], out["0.1"]["cov_mond_rising"][5], out["0.1"]["cov_cqr_rising"][5])
