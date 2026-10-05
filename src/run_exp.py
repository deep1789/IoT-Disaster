"""Run one experiment: python run_exp.py <protocol> <variant> <seed>
protocol: temporal | loro:<Region>
variants: persist ar tf lgbm | full noroute noaug geo own noevent gru
"""
import sys, os, pickle, time, joblib
import numpy as np, pandas as pd, torch
from lib import *
from models import *
torch.set_num_threads(int(os.environ.get("NT", "4")))
proto, variant, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs("results/preds", exist_ok=True); os.makedirs("results/models", exist_ok=True); os.makedirs("data/proc", exist_ok=True)
tag = f"{proto.replace(':', '-')}_{variant}_s{seed}"
log = lambda *a: print(f"[{tag}]", *a, flush=True)
if os.path.exists(f"results/preds/{tag}.npz") and (variant in ("persist","ar","tf","lgbm") or os.path.exists(f"results/preds/{tag}_val.npz")):
    print("skip", tag); sys.exit(0)

H, meta = load(); S = H.shape[1]
net = Net(H, meta)
region = meta.region.values
# ---- topology-free prior from TRAIN period only (cached)
cf = "data/proc/prior.pkl"
if os.path.exists(cf): sel_phys = pickle.load(open(cf, "rb"))
else:
    sel_phys = topology_free_prior(net.Ht[net.tr], meta, region); pickle.dump(sel_phys, open(cf, "wb"))
sel_geo = geo_neighbors(meta)
if proto == "temporal":
    train_sites = np.arange(S); test_sites = np.arange(S)
else:
    hold = proto.split(":")[1]
    train_sites = np.where(region != hold)[0]; test_sites = np.where(region == hold)[0]

use_geo = variant in ("geo", "gru", "lgbm_geo")
sel = sel_geo if use_geo else sel_phys
nidx, nvalid = nbr_index(sel, S)
if variant in ("own", "ar", "persist"):
    nvalid = nvalid.copy(); nvalid[:, 1:] = 0

# ---- test origins (stride 2) and val origins (stride 3)
tt_te, ss_te = net.valid_origins(net.te & (np.arange(len(net.index)) % 2 == 0), test_sites)
tt_va, ss_va = net.valid_origins(net.va & (np.arange(len(net.index)) % 3 == 0), test_sites)

# ---- linear routing prior
def get_W():
    if proto == "temporal":
        return fit_route(net, nidx, nvalid, train_sites, net.tr)
    # transfer: pooled ridge over training-region sites, shared by all sites
    As, Ts = [], []
    for s in train_sites:
        tt, ss = net.valid_origins(net.tr & (np.arange(len(net.index)) % 4 == 0), [s])
        X, M, Y, nv = net.gather(tt, ss, nidx, nvalid)
        f = route_features(X) * nv_expand(nv)
        As.append(np.concatenate([f, np.ones((len(f), 1), dtype=np.float32)], axis=1)); Ts.append(Y - X[:, 0, -1:])
    Wp = irls_ridge(np.concatenate(As), np.concatenate(Ts), 3.0)
    return np.repeat(Wp[None], S, axis=0)

def save(name, med, lo=None, hi=None):
    d = dict(tt=tt_te, ss=ss_te, med=med.astype(np.float16))
    if lo is not None: d.update(lo=lo.astype(np.float16), hi=hi.astype(np.float16))
    np.savez(f"results/preds/{tag}.npz", **d)

def predict_linear(W, tt, ss):
    outs = []
    for i in range(0, len(tt), 200000):
        X, M, Y, nv = net.gather(tt[i:i + 200000], ss[i:i + 200000], nidx, nvalid)
        outs.append(X[:, 0, -1:] + route_predict(W, X, nv, ss[i:i + 200000]))
    return np.concatenate(outs)

t0 = time.time()
if variant == "persist":
    X, _, _, _ = net.gather(tt_te, ss_te, nidx, nvalid); save(tag, np.repeat(X[:, 0, -1:], 24, axis=1))
elif variant in ("ar", "tf"):
    W = get_W(); np.save(f"results/models/{tag}_W.npy", W); save(tag, predict_linear(W, tt_te, ss_te))
elif variant in ("lgbm", "lgbm_geo"):
    import lightgbm as lgb
    tt, ss = net.valid_origins(net.tr, train_sites)
    fut = net.Zt[tt[:, None] + np.array([6, 12, 24])[None, :], ss[:, None]] - net.Zt[tt, ss][:, None]
    act = np.nanmax(np.abs(fut), axis=1); p = 1 + 6 * (act > 1.0); p = p / p.sum()
    pick = np.random.default_rng(seed).choice(len(tt), 350000, p=p); tt, ss = tt[pick], ss[pick]
    def feats(tt, ss):
        out = []
        for i in range(0, len(tt), 200000):
            X, M, Y, nv = net.gather(tt[i:i + 200000], ss[i:i + 200000], nidx, nvalid)
            f = route_features(X) * nv_expand(nv)
            own = X[:, 0, :]
            ex = np.stack([own.max(1) - own[:, -1], own.min(1) - own[:, -1], own.std(1), net.age[tt[i:i + 200000], ss[i:i + 200000]]], 1)
            out.append(np.concatenate([f, ex], 1))
        return np.concatenate(out).astype(np.float32)
    F_tr = feats(tt, ss); X0 = net.Zin[tt, ss]
    Y_tr = net.Zt[tt[:, None] + HOR[None, :], ss[:, None]] - X0[:, None]
    F_te = feats(tt_te, ss_te); X0e = net.Zin[tt_te, ss_te]
    vt, vs = net.valid_origins(net.va & (np.arange(len(net.index)) % 5 == 0), train_sites)
    F_va = feats(vt, vs); Y_va = net.Zt[vt[:, None] + HOR[None, :], vs[:, None]] - net.Zin[vt, vs][:, None]
    anchors = [1, 3, 6, 12, 18, 24]; D = np.zeros((len(tt_te), 7), dtype=np.float32)
    for k, h in enumerate(anchors):
        ok = np.isfinite(Y_tr[:, h - 1]); okv = np.isfinite(Y_va[:, h - 1])
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, objective="huber", alpha=1.0, n_jobs=2, verbose=-1)
        m.fit(F_tr[ok], Y_tr[ok, h - 1], eval_set=[(F_va[okv], Y_va[okv, h - 1])], callbacks=[lgb.early_stopping(40, verbose=False)])
        D[:, k + 1] = m.predict(F_te); log(f"h={h} iters={m.best_iteration_} [{time.time() - t0:.0f}s]")
        joblib.dump(m, f"results/models/{tag}_h{h}.pkl")
    ax = np.array([0] + anchors)
    med = np.stack([np.interp(HOR, ax, D[i]) for i in range(len(D))]) if len(D) < 0 else None
    # vectorised linear interpolation across anchors
    med = np.zeros((len(D), 24), dtype=np.float32)
    for h in HOR:
        j = np.searchsorted(ax, h);
        if ax[j] == h: med[:, h - 1] = D[:, j]
        else:
            w = (h - ax[j - 1]) / (ax[j] - ax[j - 1]); med[:, h - 1] = (1 - w) * D[:, j - 1] + w * D[:, j]
    save(tag, X0e[:, None] + med)
else:
    cfg = dict(route=True, aug=True, event_w=0.0, epochs=int(os.environ.get("EP", "8")), n_per_epoch=int(os.environ.get("NPE", "160000")))
    if variant == "noroute": cfg["route"] = False
    if variant == "noaug": cfg["aug"] = False
    if variant == "evsamp": cfg["event_w"] = 6.0
    if variant == "ev1": cfg["event_w"] = 1.0
    if variant == "gru": cfg.update(route=False, aug=False, event_w=0.0)
    W = get_W() if cfg["route"] else None
    model = train_model(net, nidx, nvalid, W, train_sites, cfg, seed, log)
    torch.save(model.state_dict(), f"results/models/{tag}.pt")
    if W is not None: np.save(f"results/models/{tag}_W.npy", W)
    out = run_model(model, net, tt_te, ss_te, nidx, nvalid, W)
    save(tag, out[0], out[1], out[2])
    # validation predictions (for conformal calibration)
    outv = run_model(model, net, tt_va, ss_va, nidx, nvalid, W)
    np.savez(f"results/preds/{tag}_val.npz", tt=tt_va, ss=ss_va, med=outv[0].astype(np.float16), lo=outv[1].astype(np.float16), hi=outv[2].astype(np.float16))
log(f"done in {time.time() - t0:.0f}s")
