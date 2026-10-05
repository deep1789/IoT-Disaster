| Comparison | metric | gauges | median Δ (σ) | % gauges improved | p (Wilcoxon) | p (BH-adj.) |
|---|---|---|---|---|---|---|
| PG-FaultTwin vs Persistence | nMAE all | 99 | -0.0386 | 91% | 1.7e-16 | 3.4e-16 |
| PG-FaultTwin vs Persistence | nMAE active | 99 | -0.3739 | 100% | 5.7e-18 | 1.4e-17 |
| PG-FaultTwin vs AR (own gauge) | nMAE all | 99 | -0.0261 | 90% | 5.5e-16 | 9.1e-16 |
| PG-FaultTwin vs AR (own gauge) | nMAE active | 99 | -0.1987 | 100% | 5.7e-18 | 1.4e-17 |
| PG-FaultTwin vs TF (routing prior) | nMAE all | 99 | -0.0142 | 88% | 1.2e-14 | 1.7e-14 |
| PG-FaultTwin vs TF (routing prior) | nMAE active | 99 | -0.0920 | 100% | 5.7e-18 | 1.4e-17 |
| PG-FaultTwin vs LightGBM | nMAE all | 99 | -0.0495 | 100% | 5.7e-18 | 1.4e-17 |
| PG-FaultTwin vs LightGBM | nMAE active | 99 | -0.0207 | 65% | 0.00027 | 0.0003 |
| PG-FaultTwin vs Vanilla GRU | nMAE all | 99 | -0.0019 | 58% | 0.0029 | 0.0029 |
| PG-FaultTwin vs Vanilla GRU | nMAE active | 99 | -0.0473 | 78% | 5.4e-11 | 6.8e-11 |