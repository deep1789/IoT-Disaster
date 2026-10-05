| Held-out region | Method | nMAE | nMAE active | grad. corr. 3-h | AP (action) | CSI (action) |
|---|---|---|---|---|---|---|
| MidAtlantic | TF (routing prior) | 0.162 | 1.196 | 0.681 | 0.196 | 0.167 |
| MidAtlantic | LightGBM | 0.188 | 0.966 | 0.761 | 0.303 | 0.207 |
| MidAtlantic | Vanilla GRU | 0.148 | 0.990 | 0.743 | 0.259 | 0.178 |
| MidAtlantic | **PG-FaultTwin** | 0.147 | 0.968 | 0.760 | 0.305 | 0.212 |
| OhioValley | TF (routing prior) | 0.316 | 1.031 | 0.685 | 0.187 | 0.145 |
| OhioValley | LightGBM | 0.346 | 0.930 | 0.728 | 0.164 | 0.145 |
| OhioValley | Vanilla GRU | 0.294 | 0.911 | 0.726 | 0.164 | 0.142 |
| OhioValley | **PG-FaultTwin** | 0.290 | 0.913 | 0.702 | 0.185 | 0.153 |
| TexasHill | TF (routing prior) | 0.166 | 1.676 | 0.488 | 0.051 | 0.056 |
| TexasHill | LightGBM | 0.222 | 1.434 | 0.400 | 0.033 | 0.058 |
| TexasHill | Vanilla GRU | 0.147 | 1.417 | 0.393 | 0.032 | 0.044 |
| TexasHill | **PG-FaultTwin** | 0.190 | 1.463 | 0.264 | 0.032 | 0.044 |
| UpperMidwest | TF (routing prior) | 0.270 | 1.292 | 0.534 | 0.206 | 0.162 |
| UpperMidwest | LightGBM | 0.267 | 1.054 | 0.588 | 0.281 | 0.194 |
| UpperMidwest | Vanilla GRU | 0.231 | 1.082 | 0.574 | 0.265 | 0.177 |
| UpperMidwest | **PG-FaultTwin** | 0.223 | 1.034 | 0.530 | 0.287 | 0.194 |