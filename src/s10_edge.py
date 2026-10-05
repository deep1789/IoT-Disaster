"""Edge feasibility: latency, size, quantisation effect, MAC count."""
import time, io, numpy as np, torch, pickle
from lib import *; from models import *
torch.set_num_threads(1)
H, meta = load(); net = Net(H, meta); S = H.shape[1]
sel = pickle.load(open("data/proc/prior.pkl", "rb")); nidx, nv = nbr_index(sel, S)
m = Twin(); m.load_state_dict(torch.load("results/models/temporal_full_s0.pt")); m.eval(); W = np.load("results/models/temporal_full_s0_W.npy")
nparam = sum(p.numel() for p in m.parameters())
buf = io.BytesIO(); torch.save(m.state_dict(), buf); size_kb = len(buf.getvalue()) / 1024
q = torch.ao.quantization.quantize_dynamic(Twin().eval() if False else m, {torch.nn.Linear, torch.nn.GRU}, dtype=torch.qint8)
buf = io.BytesIO(); torch.save(q.state_dict(), buf); qsize_kb = len(buf.getvalue()) / 1024
tt = np.full(S, net.index.get_loc(net.index[net.te][5000]), dtype=np.int64); ss = np.arange(S)
def bench(model, bs_sites, reps=30):
    t_, s_ = tt[:bs_sites], ss[:bs_sites]
    run_model(model, net, t_, s_, nidx, nv, W)
    ts = []
    for _ in range(reps):
        a = time.perf_counter(); run_model(model, net, t_, s_, nidx, nv, W); ts.append(time.perf_counter() - a)
    return float(np.median(ts) * 1000)
res = dict(params=nparam, size_kb=size_kb, qsize_kb=qsize_kb, lat_all_sites_ms=bench(m, S), lat_one_site_ms=bench(m, 1), qlat_all_sites_ms=bench(q, S), qlat_one_site_ms=bench(q, 1))
# MACs per forecast: GRU (3 gates) over L steps + MLP + heads
C = 5; hid = 64; macs_gru = L * 3 * (hid * (3 * C + hid)); macs_mlp = (hid + 2 * C + 2) * 128 + 128 * 128 + 3 * 128 * 24; macs_route = 24 * (C * len(RLAGS) + 1)
res.update(macs_per_forecast=macs_gru + macs_mlp + macs_route)
# accuracy impact on test subset (stride 12)
tt2, ss2 = net.valid_origins(net.te & (np.arange(len(net.index)) % 12 == 0), np.arange(S))
a = run_model(m, net, tt2, ss2, nidx, nv, W)[0]; b = run_model(q, net, tt2, ss2, nidx, nv, W)[0]
Y = net.Zt[tt2[:, None] + HOR[None, :], ss2[:, None]]
res.update(nMAE_fp32=float(np.abs(a - Y).mean()), nMAE_int8=float(np.abs(b - Y).mean()), max_diff_ft=float((np.abs(a - b) * net.sig[ss2][:, None]).max()), mean_diff_ft=float((np.abs(a - b) * net.sig[ss2][:, None]).mean()))
save_json(res, "results/edge.json"); print(res)
