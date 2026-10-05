| Variant | nMAE | nMAE active | grad. corr. 3-h | AP (minor) | AP (action) | CSI (action) |
|---|---|---|---|---|---|---|
| Full PG-FaultTwin | 0.185 | 0.875 | 0.725 | 0.276 | 0.355 | 0.258 |
| – routing prior | 0.189 | 0.936 | 0.616 | 0.212 | 0.273 | 0.190 |
| – fault curriculum | 0.184 | 0.873 | 0.735 | 0.280 | 0.354 | 0.255 |
| geographic instead of physics-selected neighbours | 0.190 | 0.891 | 0.712 | 0.231 | 0.321 | 0.240 |
| own gauge only (no neighbours) | 0.195 | 0.954 | 0.679 | 0.189 | 0.252 | 0.193 |
| + event-focused sampling (weight 7) | 0.213 | 0.872 | 0.673 | 0.275 | 0.362 | 0.267 |
| Vanilla GRU (all elements removed) | 0.191 | 0.957 | 0.673 | 0.177 | 0.250 | 0.176 |