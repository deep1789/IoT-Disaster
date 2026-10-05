"""Physics consistency: neighbour-ablation importance vs inferred correlation; what-if upstream scaling."""
import pickle, numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from lib import *; from models import *
torch.set_num_threads(4)
H, meta = load(); net = Net(H, meta); S = H.shape[1]
sel = pickle.load(open("data/proc/prior.pkl", "rb")); nidx, nv = nbr_index(sel, S)
m = Twin(); m.load_state_dict(torch.load("results/models/temporal_full_s0.pt")); m.eval(); W = np.load("results/models/temporal_full_s0_W.npy")
rows = []
for j in range(S):
    if len(sel[j]) == 0: continue
    tt, ss = net.valid_origins(net.te & (np.arange(len(net.index)) % 3 == 0), [j])
    if len(tt) < 300: continue
    Y = net.Zt[tt[:, None] + HOR[None, :], ss[:, None]]
    act = np.abs(Y[:, [5, 11, 23]] - net.Zin[tt, ss][:, None]).max(1) > 1.0
    if act.sum() < 30: continue
    base = np.abs(run_model(m, net, tt, ss, nidx, nv, W)[0] - Y)[act]
    for c, (i, lag, rho, dr, z0) in enumerate(sel[j]):
        nv2 = nv.copy(); nv2[j, c + 1] = 0
        e = np.abs(run_model(m, net, tt, ss, nidx, nv2, W)[0] - Y)[act]
        rows.append(dict(j=j, i=i, rank=c + 1, lag=lag, rho=rho, direct=dr, d_nMAE=float((e - base).mean()), d_nMAE_h6=float((e - base)[:, 5].mean()), d_nMAE_h12=float((e - base)[:, 11].mean())))
df = pd.DataFrame(rows); df.to_csv("results/neighbour_importance.csv", index=False)
res = dict(n_edges=len(df), spearman_rho_imp=list(map(float, spearmanr(df.rho, df.d_nMAE))), spearman_direct_imp=list(map(float, spearmanr(df.direct, df.d_nMAE))),
           spearman_rank_imp=list(map(float, spearmanr(df["rank"], df.d_nMAE))), frac_positive=float((df.d_nMAE > 0).mean()), mean_d=float(df.d_nMAE.mean()),
           mean_d_by_rank=df.groupby("rank").d_nMAE.mean().to_dict(), mean_d_lag_le2=float(df[df.lag <= 2].d_nMAE.mean()), mean_d_lag_gt2=float(df[df.lag > 2].d_nMAE.mean()),
           spearman_lag_imp=list(map(float, spearmanr(df.lag, df.d_nMAE))))
# what-if: scale the upstream rise
out = []
for j in range(S):
    if len(sel[j]) == 0: continue
    tt, ss = net.valid_origins(net.te & (np.arange(len(net.index)) % 3 == 0), [j])
    if len(tt) < 300: continue
    X, M, Y, n = net.gather(tt, ss, nidx, nv)
    up = np.abs(X[:, 1, -1] - X[:, 1, -7]) / 1.0
    own_rise = X[:, 0, -1] - X[:, 0, -7]
    k = np.argsort(-up)[:max(5, len(tt) // 50)]
    for f in (0.5, 1.0, 1.5, 2.0):
        X2 = X[k].copy(); ref = X2[:, 1, :1]; X2[:, 1, :] = ref + f * (X2[:, 1, :] - ref)
        X2[:, 2:, :] = X[k][:, 2:, :]
        Xt = torch.from_numpy(X2.astype(np.float32)); Mt = torch.from_numpy(M[k]); nvt = torch.from_numpy(n[k])
        rd = t_route(torch.from_numpy(W), Xt, nvt, torch.from_numpy(ss[k]))
        with torch.no_grad():
            med, lo, hi = m(Xt, Mt, nvt, rd, torch.from_numpy(net.age[tt[k], ss[k]]) / 72)
        out.append(dict(j=j, f=f, d6=float(med[:, 5].mean()), d12=float(med[:, 11].mean()), d24=float(med[:, 23].mean()), route12=float(rd[:, 11].mean())))
w = pd.DataFrame(out).groupby("f")[["d6", "d12", "d24", "route12"]].mean(); print(w)
byj = pd.DataFrame(out).pivot(index="j", columns="f", values="d12"); res["whatif_monotone_frac"] = float(((byj[1.0] >= byj[0.5]) & (byj[1.5] >= byj[1.0]) & (byj[2.0] >= byj[1.5])).mean())
res["whatif_mean"] = w.to_dict()
save_json(res, "results/physics_consistency.json"); print(res)
