"""Shared utilities: data loading, splits, topology-free physics prior, windowing, metrics."""
import numpy as np, pandas as pd, torch, os, json

HOR = np.arange(1, 25)                     # forecast horizons (h)
L = 36                                     # input window (h)
K = 4                                      # upstream neighbours
RLAGS = np.array([1, 2, 3, 4, 6, 9, 12, 18, 24, 30])   # lags used by linear routing prior
T_TRAIN_END = pd.Timestamp("2022-10-01", tz="UTC")
T_VAL_END = pd.Timestamp("2023-10-01", tz="UTC")


def load():
    H = pd.read_pickle("data/proc/H.pkl")
    meta = pd.read_csv("data/proc/meta.csv", dtype={"site_no": str})
    meta = meta.set_index("site_no").loc[H.columns].rename_axis("site_no").reset_index()
    return H, meta


def splits(index):
    tr = np.asarray(index < T_TRAIN_END)
    va = np.asarray((index >= T_TRAIN_END) & (index < T_VAL_END))
    te = np.asarray(index >= T_VAL_END)
    return tr, va, te


def interp_short(H, limit=6):
    """Linear interpolation of short gaps only (<= limit h) -> 'truth' series."""
    return H.interpolate(limit=limit, limit_area="inside")


def locf(A, limit=72):
    return pd.DataFrame(A).ffill(limit=limit).values


# ---------------------------------------------------------------- physics prior
def lagged_xcorr(dx, i, j, maxlag=48):
    """corr(dx_i(t), dx_j(t+lag)) for lag=0..maxlag  -> i leads j when peak lag>0."""
    a = dx[:, i]; b = dx[:, j]
    ok = np.isfinite(a) & np.isfinite(b)
    out = np.zeros(maxlag + 1)
    for lag in range(maxlag + 1):
        if lag == 0:
            m = ok
            x, y = a[m], b[m]
        else:
            m = ok[:-lag] & ok[lag:]
            x, y = a[:-lag][m], b[lag:][m]
        if len(x) > 500 and x.std() > 0 and y.std() > 0:
            out[lag] = np.corrcoef(x, y)[0, 1]
    return out


def topology_free_prior(Htr, meta, region_of, k=K, maxlag=48):
    """Estimate upstream neighbours, lead times and celerity from telemetry only.
    Returns dict site_idx -> list of (nbr_idx, lag_h, rho, directionality)."""
    S = Htr.shape[1]
    # event-scale signal: 3-h change smoothed, high-pass to remove diurnal/seasonal baseflow trends
    X = np.log1p(np.clip(Htr - np.nanmin(Htr, axis=0) , 0, None))
    dx = pd.DataFrame(X).diff(3).rolling(3, min_periods=1).mean().values
    sel = {}; cc = {}
    for reg in np.unique(region_of):
        ids = np.where(region_of == reg)[0]
        P = {}
        for i in ids:
            for j in ids:
                if i == j: continue
                P[(i, j)] = lagged_xcorr(dx, i, j, maxlag)
        for j in ids:
            cand = []
            for i in ids:
                if i == j: continue
                f = P[(i, j)]; r = P[(j, i)]
                lag = int(np.argmax(f[1:]) + 1); rho = f[lag]
                rev = r[1:].max()
                # i leads j if forward peak exceeds reverse peak and zero-lag corr not dominating
                direct = rho - rev
                cand.append((i, lag, float(rho), float(direct), float(f[0])))
            cand = [c for c in cand if c[3] > 0.0 and c[2] > 0.15]
            cand.sort(key=lambda c: -(c[2] * (1 + c[3])))
            sel[j] = cand[:k]
    return sel


def geo_neighbors(meta, k=K):
    lat = meta.lat.values; lon = meta.lon.values
    S = len(meta); out = {}
    for j in range(S):
        d = np.sqrt(((lat - lat[j]) * 111) ** 2 + ((lon - lon[j]) * 111 * np.cos(np.deg2rad(lat[j]))) ** 2)
        d[meta.region.values != meta.region.values[j]] = 1e9; d[j] = 1e9
        o = np.argsort(d)[:k]
        out[j] = [(int(i), 0, 0.0, 0.0, 0.0) for i in o]
    return out


def nbr_index(sel, S, k=K):
    """(S, k+1) channel index array; missing neighbours padded by self (masked out later)."""
    idx = np.zeros((S, k + 1), dtype=np.int64); valid = np.zeros((S, k + 1), dtype=np.float32)
    for j in range(S):
        idx[j, 0] = j; valid[j, 0] = 1
        for c, (i, lag, rho, d, z) in enumerate(sel.get(j, [])[:k]):
            idx[j, c + 1] = i; valid[j, c + 1] = 1
        for c in range(len(sel.get(j, [])[:k]) + 1, k + 1):
            idx[j, c] = j
    return idx, valid


# ---------------------------------------------------------------- data container
class Net:
    """Holds the normalised timeline tensors and builds windows by gathering."""
    def __init__(self, H, meta, Hobs=None):
        self.meta = meta; self.index = H.index
        self.S = H.shape[1]
        tr, va, te = splits(H.index)
        self.tr, self.va, self.te = tr, va, te
        Ht = interp_short(H).values.astype(np.float32)              # truth (short gaps closed)
        self.med = np.nanmedian(Ht[tr], axis=0).astype(np.float32)
        d24 = Ht[24:] - Ht[:-24]
        self.sig = np.nanstd(d24[tr[24:]], axis=0).astype(np.float32) + 1e-3
        self.Ht = Ht
        self.Zt = (Ht - self.med) / self.sig                        # normalised truth
        self.set_inputs(H.values.astype(np.float32) if Hobs is None else Hobs)

    def set_inputs(self, Hobs):
        """Hobs: possibly corrupted/missing raw readings (T,S), NaN = no reading."""
        Z = (Hobs - self.med) / self.sig
        self.obs = np.isfinite(Z).astype(np.float32)
        Zl = locf(Z, 72)
        Zl = np.where(np.isfinite(Zl), Zl, 0.0).astype(np.float32)
        self.Zin = Zl
        # age since last observation (h), capped
        age = np.zeros_like(self.obs); a = np.zeros(self.S, dtype=np.float32)
        for t in range(len(Z)):
            a = np.where(self.obs[t] > 0, 0, np.minimum(a + 1, 72)); age[t] = a
        self.age = age

    def valid_origins(self, mask_time, sites=None, need_future=True):
        T = len(self.index)
        ts = np.where(mask_time)[0]
        ts = ts[(ts >= L + 30) & (ts + 24 < T)]
        sites = np.arange(self.S) if sites is None else np.asarray(sites)
        tt, ss = np.meshgrid(ts, sites, indexing="ij")
        tt = tt.ravel(); ss = ss.ravel()
        if need_future:
            ok = np.isfinite(self.Zt[tt, ss])
            for h in range(1, 25): ok &= np.isfinite(self.Zt[tt + h, ss])
            tt, ss = tt[ok], ss[ok]
        return tt.astype(np.int64), ss.astype(np.int64)

    def gather(self, tt, ss, nidx, nvalid):
        """Return X (B,C,L) normalised inputs, M (B,C,L) observed mask, Y (B,24) truth, nv (B,C) valid."""
        ch = nidx[ss]                                              # (B,C)
        tw = tt[:, None] - np.arange(L - 1, -1, -1)[None, :]       # (B,L)
        X = self.Zin[tw[:, None, :], ch[:, :, None]]
        M = self.obs[tw[:, None, :], ch[:, :, None]]
        Y = self.Zt[tt[:, None] + HOR[None, :], ss[:, None]]
        return X, M, Y, nvalid[ss]


# ---------------------------------------------------------------- linear routing prior (ARX / Muskingum-type)
def route_features(X):
    """X (B,C,L) normalised values (last index = now). Features: lagged differences of all channels (no absolute level)."""
    now = X[:, :, -1]
    d = [X[:, :, -1 - l] - now for l in RLAGS]
    return np.concatenate(d, axis=1).astype(np.float32)           # (B, C*len(RLAGS))


def nv_expand(nv):
    return np.concatenate([nv] * len(RLAGS), axis=1)


def irls_ridge(A, T, lam, iters=4, s=1.0):
    """Huber-robust ridge (IRLS) multi-output: weights from the row-wise RMS residual."""
    A = A.astype(np.float64); T = T.astype(np.float64); w = np.ones(len(A))
    for _ in range(1 + iters):
        G = (A * w[:, None]).T @ A + lam * len(A) / 1e4 * np.eye(A.shape[1])
        W = np.linalg.solve(G, (A * w[:, None]).T @ T)
        r = np.sqrt(((T - A @ W) ** 2).mean(1)); w = np.where(r < s, 1.0, s / np.maximum(r, 1e-9))
    return W.astype(np.float32)


def fit_route(net, nidx, nvalid, sites, mask_time, lam=3.0):
    """Per-site Huber ridge on route_features -> Y - now (24 horizons). Returns W (S,F+1,24)."""
    C = nidx.shape[1]; F = C * len(RLAGS)
    W = np.zeros((net.S, F + 1, len(HOR)), dtype=np.float32)
    for s in sites:
        tt, ss = net.valid_origins(mask_time, [s])
        if len(tt) < 500: continue
        X, M, Y, nv = net.gather(tt, ss, nidx, nvalid)
        f = route_features(X) * nv_expand(nv)
        A = np.concatenate([f, np.ones((len(f), 1), dtype=np.float32)], axis=1)
        W[s] = irls_ridge(A, Y - X[:, 0, -1:], lam)
    return W


def route_predict(W, X, nv, ss):
    f = route_features(X) * nv_expand(nv)
    A = np.concatenate([f, np.ones((len(f), 1), dtype=np.float32)], axis=1)
    return np.einsum("bf,bfh->bh", A, W[ss])                      # predicted delta from current (normalised)


# ---------------------------------------------------------------- metrics
def mae_phys(pred_z, true_z, ss, sig):
    return np.abs(pred_z - true_z) * sig[ss][:, None]


def nse(pred, obs):
    return 1 - np.sum((pred - obs) ** 2) / np.sum((obs - obs.mean()) ** 2)


def kge(pred, obs):
    r = np.corrcoef(pred, obs)[0, 1]; a = pred.std() / obs.std(); b = pred.mean() / obs.mean() if obs.mean() != 0 else np.nan
    return 1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)


def save_json(o, f):
    def conv(x):
        if isinstance(x, (np.floating, np.integer)): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        raise TypeError(type(x))
    json.dump(o, open(f, "w"), indent=1, default=conv)
