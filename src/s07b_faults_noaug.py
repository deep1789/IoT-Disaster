"""Sensor-fault injection + edge QC gate + robustness of forecasting models. Output: results/faults.json"""
import sys, os, pickle, json, time, zlib
import numpy as np, pandas as pd, torch
from lib import *
from models import *
from evalcore import *
torch.set_num_threads(3)
H, meta = load(); S = H.shape[1]; net = Net(H, meta); ev = Ev(net, meta)
Hraw = H.values.astype(np.float32).copy()
sel_phys = pickle.load(open("data/proc/prior.pkl", "rb")); sel_geo = geo_neighbors(meta)
nidx_p, nv_p = nbr_index(sel_phys, S); nidx_g, nv_g = nbr_index(sel_geo, S)
T = len(Hraw); sig = net.sig; trn = net.tr
rate = np.nanmax(np.abs(np.diff(net.Ht[trn], axis=0)), axis=0)              # largest 1-h change in training
rng = np.random.default_rng(123)


def markov(frac, mean_len, rng):
    pe = 1.0 / mean_len; ps = frac * pe / (1 - frac)
    st = np.zeros(S, bool); out = np.zeros((T, S), bool)
    for t in range(T):
        r = rng.random(S)
        st = np.where(st, r > pe, r < ps); out[t] = st
    return out


def inject(kind, level, rng):
    Hc = Hraw.copy(); lab = np.zeros((T, S), bool)
    if kind == "dropout":
        m = markov(level, 8, rng); Hc[m] = np.nan; lab = m
    elif kind == "stuck":
        m = markov(level, 24, rng)
        for t in range(1, T):
            s = m[t] & m[t - 1]
            Hc[t, s] = Hc[t - 1, s]
        lab = m
    elif kind == "spike":
        m = rng.random((T, S)) < level
        mag = rng.uniform(5, 20, (T, S)) * rng.choice([-1, 1], (T, S)) * sig[None, :]
        Hc = np.where(m, Hc + mag, Hc); lab = m
    elif kind == "bias":
        who = rng.random(S) < 0.3; off = rng.normal(0, level, S) * sig * who
        Hc = Hc + off[None, :]; lab[:, who] = True
    elif kind == "latency":
        d = int(level); Hc = np.vstack([np.full((d, S), np.nan, np.float32), Hc[:-d]])
    elif kind == "combined":
        Hc, lab = inject("dropout", 0.2, rng)
        H2, l2 = inject("stuck", 0.1, rng); H3, l3 = inject("spike", 0.01, rng)
        Hc = np.where(l2, H2, Hc); Hc = np.where(l3, H3, Hc); Hc = np.where(np.isnan(Hc) & ~lab, Hc, Hc)
        lab = lab | l2 | l3
    return Hc, lab


def gate(Hc):
    """Causal edge QC: rate-of-change/spike gate and cross-sensor flatline gate. Returns cleaned array, flag mask."""
    Hg = Hc.copy(); flag = np.zeros_like(Hc, bool)
    last = np.where(np.isfinite(Hc[0]), Hc[0], np.nanmedian(Hraw[trn], axis=0)); age = np.zeros(S)
    run = np.zeros(S); prev = Hc[0].copy()
    nb = nidx_p[:, 1:]; nbv = nv_p[:, 1:]
    for t in range(1, T):
        x = Hc[t]
        fin = np.isfinite(x)
        allowed = 1.5 * rate * (1 + 0.5 * age)
        bad = fin & (np.abs(x - last) > allowed)
        # flatline: exact repeat of previous reading
        same = fin & np.isfinite(prev) & (x == prev)
        run = np.where(same, run + 1, 0)
        act = (np.abs(Hc[t][nb] - Hc[t - 1][nb]) > 0.02 * sig[nb])
        act = np.nan_to_num(act.astype(float)) * nbv
        nb_act = act.sum(1) / np.maximum(nbv.sum(1), 1)
        stuck = (run >= 12) & (nb_act >= 0.5)
        f = bad | stuck
        flag[t] = f; Hg[t, f] = np.nan
        good = fin & ~f; last = np.where(good, x, last); age = np.where(good, 0, np.minimum(age + 1, 48))
        prev = x
    return Hg, flag


def load_twin(tag, geo=False, route=True):
    m = Twin(use_route=route); m.load_state_dict(torch.load(f"results/models/{tag}.pt")); m.eval()
    W = np.load(f"results/models/{tag}_W.npy") if route else None
    return m, W


def predict(model, W, nidx, nvalid, tt, ss):
    return run_model(model, net, tt, ss, nidx, nvalid, W)[0]


models = {}
models["noaug"] = (*load_twin("temporal_noaug_s0"), nidx_p, nv_p)
Wtf = np.load("results/models/temporal_tf_s0_W.npy")
tt, ss = net.valid_origins(net.te & (np.arange(T) % 12 == 0), np.arange(S))
yh, h0 = ev.truth(tt, ss)
active = np.nanmax(np.abs(yh[:, [5, 11, 23]] - h0[:, None]), axis=1) > net.sig[ss]


def score(pred_z, name):
    ph = ev.phys(pred_z, ss); err = np.abs(ph - yh)
    res = dict(MAE6=float(err[:, 5].mean()), MAE24=float(err[:, 23].mean()),
               RMSE24=float(np.sqrt(((ph[:, 23] - yh[:, 23]) ** 2).mean())),
               nMAE=float((err / net.sig[ss][:, None]).mean()),
               nMAE_active=float((err[active] / net.sig[ss][active][:, None]).mean()))
    d = dict(tt=tt, ss=ss, med=pred_z)
    a = ev.alert_data(d); res["AP_minor"] = float(ev.ap(a))
    return res


def eval_inputs(Hc, label, Hg=None):
    out = {}
    for variant, gated in [("noaug", False), ("noaug", True)]:
        Hin = Hg if gated else Hc
        net.set_inputs(Hin)
        if variant == "persist":
            pz = np.repeat(net.Zin[tt, ss][:, None], 24, axis=1)
        elif variant == "tf":
            X, M, Y, nv = net.gather(tt, ss, nidx_p, nv_p); pz = X[:, 0, -1:] + route_predict(Wtf, X, nv, ss)
        else:
            m, W, ni, nvv = models[variant]; pz = predict(m, W, ni, nvv, tt, ss)
        out[variant + ("+QC" if gated else "")] = score(pz, variant)
    return out


results = {}
scen = [("clean", 0)] + [("dropout", p) for p in (0.1, 0.3, 0.5, 0.7)] + [("stuck", p) for p in (0.1, 0.3)] + \
       [("spike", 0.01), ("spike", 0.03)] + [("bias", 0.5), ("bias", 1.0)] + [("latency", d) for d in (3, 6, 12)] + [("combined", 0)]
det = {}
for kind, lvl in scen:
    t0 = time.time()
    if kind == "clean": Hc, lab = Hraw.copy(), np.zeros((T, S), bool)
    else: Hc, lab = inject(kind, lvl, np.random.default_rng(zlib.crc32(f"{kind}{lvl}".encode())))
    key = f"{kind}_{lvl}"
    Hg, fl = gate(Hc)
    results[key] = eval_inputs(Hc, key, Hg)
    if kind in ("stuck", "spike", "combined"):                        # detection quality of the QC gate on test period
        te = net.te
        l = lab[te]; f = fl[te]
        if kind == "combined":     # only count non-missing injected faults (stuck/spike)
            l = l & np.isfinite(Hc[te])
        tp = (l & f).sum(); fp = (~l & f).sum(); fn = (l & ~f).sum()
        det[key] = dict(precision=float(tp / max(tp + fp, 1)), recall=float(tp / max(tp + fn, 1)), fp_rate_clean=float(fp / max((~l).sum(), 1)))
    print(key, {k: round(v["nMAE"], 4) for k, v in results[key].items()}, f"[{time.time() - t0:.0f}s]", flush=True)
net.set_inputs(Hraw)
save_json(dict(results=results, detection=det), "results/faults_noaug.json")
