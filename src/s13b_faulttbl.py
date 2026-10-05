import json
f = json.load(open("results/faults.json")); g = json.load(open("results/faults_noaug.json"))["results"]; r = f["results"]; det = f["detection"]
LAB = {"clean_0": "Clean", "dropout_0.1": "Dropout 10%", "dropout_0.3": "Dropout 30%", "dropout_0.5": "Dropout 50%", "dropout_0.7": "Dropout 70%", "stuck_0.1": "Stuck-at 10%", "stuck_0.3": "Stuck-at 30%",
       "spike_0.01": "Spikes 1%", "spike_0.03": "Spikes 3%", "bias_0.5": "Bias b=0.5σ", "bias_1.0": "Bias b=1.0σ", "latency_3": "Latency 3 h", "latency_6": "Latency 6 h", "latency_12": "Latency 12 h", "combined_0": "Combined"}
cols = [("persist", r, "Persistence"), ("tf", r, "TF"), ("gru", r, "Vanilla GRU"), ("gru+QC", r, "GRU+QC"), ("noaug", g, "Twin w/o curriculum"), ("full", r, "**Twin**"), ("full+QC", r, "**Twin+QC**")]
for key, nm, fn in (("nMAE", "all", "fault_all"), ("nMAE_active", "active", "fault_active"), ("AP_minor", "AP", "fault_ap")):
    rows = ["| Scenario | " + " | ".join(c[2] for c in cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for s, lab in LAB.items():
        vals = [c[1][s][c[0]][key] for c in cols]
        best = min(vals) if key != "AP_minor" else max(vals)
        rows.append(f"| {lab} | " + " | ".join((f"**{v:.3f}**" if v == best else f"{v:.3f}") for v in vals) + " |")
    open(f"manuscript/tbl_{fn}.md", "w").write("\n".join(rows))
rows = ["| Fault | Precision | Recall | Healthy readings falsely rejected |", "|---|---|---|---|"]
for s in ("stuck_0.1", "stuck_0.3", "spike_0.01", "spike_0.03", "combined_0"): rows.append(f"| {LAB[s]} | {det[s]['precision']:.3f} | {det[s]['recall']:.3f} | {100*det[s]['fp_rate_clean']:.2f}% |")
open("manuscript/tbl_detect.md", "w").write("\n".join(rows)); print(open("manuscript/tbl_fault_all.md").read())
