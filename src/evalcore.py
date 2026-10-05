"""Evaluation helpers: accuracy, gradient skill, alerting, cost-loss, conformal."""
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from lib import *


class Ev:
    def __init__(self, net, meta):
        self.net = net; self.meta = meta
        self.thr_minor = meta.minor.values.astype(float)
        self.thr_action = meta.action.values.astype(float)

    def load(self, f):
        d = np.load(f)
        return {k: d[k] for k in d.files}

    def phys(self, z, ss):                                     # normalised -> ft
        return z.astype(np.float32) * self.net.sig[ss][:, None] + self.net.med[ss][:, None]

    def truth(self, tt, ss):
        return self.net.Ht[tt[:, None] + HOR[None, :], ss[:, None]], self.net.Ht[tt, ss]

    # ---------------- accuracy
    def accuracy(self, d, mask_sites=None):
        tt, ss = d["tt"], d["ss"]
        yh, h0 = self.truth(tt, ss); ph = self.phys(d["med"], ss)
        err = np.abs(ph - yh)
        return err, yh, ph, h0

    def site_mae(self, err, ss, hidx):
        df = pd.DataFrame({"s": ss, "e": err[:, hidx].mean(1) if np.ndim(hidx) else err[:, hidx]})
        return df.groupby("s").e.mean()

    # ---------------- alerting
    def alert_data(self, d, level="minor", horizon=24):
        """Returns per-origin arrays for sites having threshold and currently below it."""
        tt, ss = d["tt"], d["ss"]
        thr_all = self.thr_minor if level == "minor" else self.thr_action
        thr = thr_all[ss]
        h0 = self.net.Ht[tt, ss]
        ok = np.isfinite(thr) & (h0 < thr)
        yh, _ = self.truth(tt, ss)
        ph = self.phys(d["med"], ss)
        y = (np.nanmax(yh[:, :horizon], axis=1) >= thr)
        # normalised margin score (higher => more likely exceedance)
        sc = (np.nanmax(ph[:, :horizon], axis=1) - thr) / self.net.sig[ss]
        return dict(ok=ok, y=y, sc=sc, tt=tt, ss=ss, thr=thr, h0=h0, ph=ph)

    def alert_scores(self, a, deltas=np.linspace(-3, 8, 111)):
        ok = a["ok"]; y = a["y"][ok]; sc = a["sc"][ok]
        res = []
        for dl in deltas:
            al = sc >= -dl
            tp = (al & y).sum(); fp = (al & ~y).sum(); fn = (~al & y).sum()
            pod = tp / max(tp + fn, 1); far = fp / max(tp + fp, 1); csi = tp / max(tp + fp + fn, 1)
            res.append((dl, pod, far, csi, tp, fp, fn))
        return pd.DataFrame(res, columns=["delta", "POD", "FAR", "CSI", "tp", "fp", "fn"])

    def ap(self, a):
        ok = a["ok"]
        if a["y"][ok].sum() < 5: return np.nan
        return average_precision_score(a["y"][ok], a["sc"][ok])

    def events(self, a, delta, lead_max=24, level="minor"):
        """Event-level verification. Events = up-crossings of threshold preceded by >=24h below."""
        net = self.net; ev = []
        tt, ss = a["tt"], a["ss"]; sc = a["sc"]; thr_all = self.thr_minor if level == "minor" else self.thr_action
        out = []
        for s in np.unique(ss):
            thr = thr_all[s]
            if not np.isfinite(thr): continue
            h = net.Ht[:, s]
            te = np.where(net.te)[0]
            above = h[te] >= thr
            up = np.where(above[1:] & ~above[:-1])[0] + 1
            cross = te[up]
            # separate events (>=48 h of below before)
            last = -10**9; cl = []
            for c in cross:
                if c - last >= 48: cl.append(c)
                last = c
            m = ss == s
            t_s = tt[m]; sc_s = sc[m]; o = np.argsort(t_s); t_s, sc_s = t_s[o], sc_s[o]
            alert = sc_s >= -delta
            ok = a["ok"][m][o]
            alert = alert & ok
            for c in cl:
                sel = (t_s >= c - lead_max) & (t_s < c)
                if sel.sum() == 0: continue
                ts, al = t_s[sel], alert[sel]
                # earliest origin from which alert stays active until crossing
                lead = 0
                if al[-1]:
                    k = len(al) - 1
                    while k > 0 and al[k - 1]: k -= 1
                    lead = c - ts[k]
                out.append((s, c, lead, al.any()))
            # false alarm episodes: alert runs whose end is not followed by crossing within lead_max
            if len(t_s):
                run = np.diff(np.r_[0, alert.astype(int)]) == 1
                starts = t_s[run]
                cl_arr = np.array(cl) if cl else np.array([10**9])
                for st in starts:
                    if not ((cl_arr > st) & (cl_arr <= st + lead_max)).any():
                        out.append((s, st, -1, True))      # false-alarm episode
        return pd.DataFrame(out, columns=["s", "t", "lead", "alerted_in_window"])


def cost_loss_value(sc, y, rs):
    """Relative economic value V(r) maximised over alert thresholds (Richardson 2000)."""
    o = y.mean(); N = len(y)
    order = np.argsort(-sc); ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(~ys)
    H = tp / max(y.sum(), 1); F = fp / max((~y).sum(), 1)
    out = []
    for r in rs:
        Eclim = min(r, o)
        Efc = F * r * (1 - o) + H * o * r + (1 - H) * o      # per-origin expense in units of L
        Eperf = o * r
        v = (Eclim - Efc.min()) / max(Eclim - Eperf, 1e-12)
        out.append(max(v, 0.0))
    return np.array(out)


# ---------------- conformal
def cqr_calibrate(lo, hi, y, alpha):
    """Per-horizon CQR correction. lo,hi,y: (n,24) normalised units."""
    s = np.maximum(lo - y, y - hi)
    q = np.zeros(s.shape[1])
    for h in range(s.shape[1]):
        v = s[:, h][np.isfinite(s[:, h])]
        n = len(v); q[h] = np.quantile(v, min(1, np.ceil((n + 1) * (1 - alpha)) / n))
    return q


def aci_run(score_lo, score_hi, y, order_t, alpha=0.1, gamma=0.005, delay=0):
    """Adaptive conformal inference on a single time-ordered stream (one site, one horizon).
    Uses rolling calibration set of past nonconformity scores; returns coverage indicators and widths adj."""
    n = len(y); a_t = alpha; cov = np.zeros(n, dtype=bool); qs = np.zeros(n)
    s = np.maximum(score_lo - y, y - score_hi)
    hist = []
    for i in range(n):
        if len(hist) > 50:
            ql = min(max(1 - a_t, 0), 1)
            q = np.quantile(hist[-2000:], ql)
        else:
            q = 0.0
        qs[i] = q; cov[i] = s[i] <= q
        # label available after `delay` steps; feed with delay
        if i - delay >= 0:
            j = i - delay
            hist.append(s[j]); a_t = a_t + gamma * (alpha - (0.0 if s[j] <= qs[j] else 1.0))
    return cov, qs
