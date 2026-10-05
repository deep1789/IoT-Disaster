"""PG-FaultTwin: physics-guided (linear routing prior) + GRU residual twin with sensor-dropout training and quantile heads."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time
from lib import *


def t_route_features(X):
    now = X[:, :, -1]
    d = [X[:, :, -1 - l] - now for l in RLAGS]
    return torch.cat(d, dim=1)


def t_route(W, X, nv, ss):
    f = t_route_features(X)
    nvm = torch.cat([nv] * len(RLAGS), dim=1)
    A = torch.cat([f * nvm, torch.ones(len(f), 1)], dim=1)
    return torch.einsum("bf,bfh->bh", A, W[ss])


def locf_window(X, M):
    """Fill unobserved steps in window by last observed value (leading gaps by first observed; none -> 0)."""
    B, C, Lw = X.shape
    ar = torch.arange(Lw).view(1, 1, Lw).expand(B, C, Lw)
    idx = torch.where(M > 0, ar, torch.zeros_like(ar)).cummax(dim=2).values
    Xf = torch.gather(X, 2, idx)
    any_obs = (M > 0).any(dim=2, keepdim=True)
    first = torch.where(M > 0, ar, torch.full_like(ar, Lw)).min(dim=2, keepdim=True).values.clamp(max=Lw - 1)
    lead = torch.gather(X, 2, first)
    pre = (ar < first) & (M.cumsum(2) == 0)
    Xf = torch.where(pre, lead.expand_as(Xf), Xf)
    return torch.where(any_obs, Xf, torch.zeros_like(Xf))


def augment(X, M, g):
    """Random sensor faults on windows: i.i.d. dropout, block outages, tail (stale) outages, full-channel loss."""
    B, C, Lw = X.shape
    keep = torch.ones_like(M)
    act = (torch.rand(B, 1, 1, generator=g) < 0.35).float()          # only a fraction of windows are corrupted
    p = torch.rand(B, 1, 1, generator=g) * 0.3 * act
    keep = keep * (torch.rand(B, C, Lw, generator=g) > p).float()
    # block outage
    use = (torch.rand(B, C, 1, generator=g) < 0.4).float() * act
    st = (torch.rand(B, C, 1, generator=g) * Lw).long(); ln = (torch.rand(B, C, 1, generator=g) * 24 + 2).long()
    ar = torch.arange(Lw).view(1, 1, Lw)
    blk = ((ar >= st) & (ar < st + ln)).float() * use
    keep = keep * (1 - blk)
    # stale tail
    use = (torch.rand(B, C, 1, generator=g) < 0.25).float() * act; n = (torch.rand(B, C, 1, generator=g) * 24).long()
    keep = keep * (1 - ((ar >= Lw - n).float() * use))
    # lost channel (never for guaranteed-own in half of cases)
    use = (torch.rand(B, C, 1, generator=g) < 0.06).float() * act
    keep = keep * (1 - use)
    return keep * M


class Twin(nn.Module):
    def __init__(self, C=K + 1, hid=64, use_route=True, nq=True):
        super().__init__()
        self.C = C; self.use_route = use_route
        self.gru = nn.GRU(2 * C + C, hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid + 2 * C + 2, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.o_med = nn.Linear(128, len(HOR)); self.o_lo = nn.Linear(128, len(HOR)); self.o_hi = nn.Linear(128, len(HOR))
        nn.init.zeros_(self.o_med.weight); nn.init.zeros_(self.o_med.bias)

    def forward(self, X, M, nv, route_delta, age_own):
        now = X[:, :, -1:]
        rel = (X - now)
        dX = torch.cat([torch.zeros_like(X[:, :, :1]), X[:, :, 1:] - X[:, :, :-1]], dim=2)
        seq = torch.cat([rel, M, dX * 3], dim=1).transpose(1, 2)           # (B,L,3C)
        _, h = self.gru(seq)
        z = torch.cat([h[-1], now.squeeze(2), nv, age_own[:, None], M[:, 0, -1:]], dim=1)
        z = self.head(z)
        res = self.o_med(z)
        med = (route_delta if self.use_route else 0) + res
        lo = -F.softplus(self.o_lo(z) - 1.0); hi = F.softplus(self.o_hi(z) - 1.0)
        return med, lo, hi


def pinball(y, q, tau):
    d = y - q
    return torch.maximum(tau * d, (tau - 1) * d)


def make_batch(net, tt, ss, nidx, nvalid, W, aug, g):
    X, M, Y, nv = net.gather(tt, ss, nidx, nvalid)
    X = torch.from_numpy(X); M = torch.from_numpy(M); Y = torch.from_numpy(Y); nv = torch.from_numpy(nv)
    M = M * nv[:, :, None]
    if aug: M = augment(X, M, g)
    Xf = locf_window(X, M)
    Xf = Xf * nv[:, :, None]
    return Xf, M, Y, nv


def run_model(model, net, tt, ss, nidx, nvalid, W, bs=8192):
    model.eval(); outs = []
    Wt = torch.from_numpy(W) if W is not None else None
    with torch.no_grad():
        for i in range(0, len(tt), bs):
            t_, s_ = tt[i:i + bs], ss[i:i + bs]
            X, M, Y, nv = make_batch(net, t_, s_, nidx, nvalid, W, False, None)
            rd = t_route(Wt, X, nv, torch.from_numpy(s_)) if Wt is not None else torch.zeros(len(t_), len(HOR))
            age = torch.from_numpy(net.age[t_, s_]) / 72
            med, lo, hi = model(X, M, nv, rd, age)
            now = X[:, 0, -1:]
            outs.append(torch.stack([now + med, now + med + lo, now + med + hi], 0).numpy().astype(np.float32))
    return np.concatenate(outs, axis=1)                                # (3, N, 24) normalised absolute


def train_model(net, nidx, nvalid, W, train_sites, cfg, seed=0, log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = Twin(use_route=cfg.get("route", True))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 2e-3), weight_decay=1e-4)
    tt, ss = net.valid_origins(net.tr, train_sites)
    # importance sampling: emphasise rising / high-activity samples
    fut = net.Zt[tt[:, None] + np.array([6, 12, 24])[None, :], ss[:, None]] - net.Zt[tt, ss][:, None]
    act = np.nanmax(np.abs(fut), axis=1)
    w = 1.0 + (cfg.get("event_w", 6.0)) * (act > 1.0) + 0.0 * act
    p = (w / w.sum())
    vt, vs = net.valid_origins(net.va, train_sites)
    vsel = np.random.default_rng(1).choice(len(vt), min(len(vt), 60000), replace=False); vt, vs = vt[vsel], vs[vsel]
    N = cfg.get("n_per_epoch", 200000); E = cfg.get("epochs", 10); bs = cfg.get("bs", 1024)
    steps = E * (N // bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.get("lr", 2e-3), total_steps=steps)
    Wt = torch.from_numpy(W) if W is not None else None
    best = (1e9, None); step = 0; t0 = time.time()
    rng = np.random.default_rng(seed)
    for ep in range(E):
        model.train(); idx = rng.choice(len(tt), N, p=p); tl = 0
        for b in range(N // bs):
            bi = idx[b * bs:(b + 1) * bs]
            X, M, Y, nv = make_batch(net, tt[bi], ss[bi], nidx, nvalid, W, cfg.get("aug", True), g)
            rd = t_route(Wt, X, nv, torch.from_numpy(ss[bi])) if Wt is not None else torch.zeros(len(bi), len(HOR))
            age = torch.from_numpy(net.age[tt[bi], ss[bi]]) / 72
            med, lo, hi = model(X, M, nv, rd, age)
            now = X[:, 0, -1:]
            yd = Y - now
            loss = F.huber_loss(med, yd, delta=1.0)
            loss = loss + 0.5 * (pinball(yd, med + lo, 0.1).mean() + pinball(yd, med + hi, 0.9).mean())
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); sched.step()
            tl += loss.item()
        out = run_model(model, net, vt, vs, nidx, nvalid, W)
        Yv = net.Zt[vt[:, None] + HOR[None, :], vs[:, None]]
        ok = np.isfinite(Yv).all(1)
        vmae = np.abs(out[0][ok] - Yv[ok]).mean()
        log(f"  ep{ep} loss {tl / (N // bs):.4f} val_nMAE {vmae:.4f} [{time.time() - t0:.0f}s]")
        if vmae < best[0]: best = (vmae, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    return model
