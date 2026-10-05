# 4. The PG-FaultTwin framework

## 4.1 Overview and twin architecture

PG-FaultTwin is organised in four layers that mirror a deployable IoT system (@@fig:architecture@@). The **sensing layer** consists of the telemetered gauges. The **edge layer** runs on a gateway at the regional hub and performs causal quality control (QC), gap bookkeeping and, optionally, the forecasting network itself. The **twin layer** maintains a synchronised virtual state of the river network, namely the current stage estimate at every gauge, the confidence in each sensor, and the 24-hour stage trajectory with calibrated intervals. The **decision layer** converts trajectories into rise rates, threshold-crossing probabilities and alerts, using the cost–loss trade-off chosen by the agency. The twin is *synchronised* in the sense that every update ingests the newest readings and re-estimates the state; it is *queryable* because users can request the trajectory, the rise rate or the time to a given threshold at any gauge; and it supports *what-if* analysis because the same network can be driven with hypothetical upstream hydrographs (Section 6.9).

Let there be S gauges, indexed by s, with hourly stage H_s(t). Let t denote the forecast origin (the current hour). The twin forecasts, for every gauge and every horizon h = 1,…,24 h, the future stage Ĥ_s(t+h), a lower and an upper bound, and from these the derived quantities in Section 4.6.

![Architecture of PG-FaultTwin. Sensing layer (gauges), edge layer (causal QC gate, mask bookkeeping, routing prior and GRU residual), twin layer (state, 24-h trajectories with conformal intervals, what-if queries) and decision layer (rise rate, time to threshold, alert margin); the topology-free prior, the fault curriculum and the verification suite are the supporting elements.](figures/fig1_architecture.png)

## 4.2 Normalisation and topology-free physics prior

**Normalisation.** For each gauge, we compute from the training years the median m_s and the scale σ_s, the standard deviation of the 24-hour stage change. The normalised stage is z_s(t) = (H_s(t) − m_s)/σ_s. This scale reflects how much the stage typically moves over the forecast window at that site, so a given normalised error corresponds to a comparable fraction of the local dynamics, whether the gauge is on a flashy headwater creek or a large lowland river. All learning takes place in normalised units, and results are reported both in normalised units and in feet.

**Directional lag structure.** A flood wave leaves a signature in gauges along its path: the rise at an upstream station precedes the rise downstream by a travel time. We extract this signature from the telemetry alone. For each gauge we build an event-scale signal x_s(t), the 3-hour difference of log(1 + H_s − min H_s) smoothed over three hours, which suppresses seasonal baseflow trends and emphasises storm response. For every ordered pair (i, j) of gauges *within the same region* we compute the lagged cross-correlation ρ_ij(τ) = corr(x_i(t), x_j(t+τ)) for τ = 0,…,48 h over the training years. The forward peak ρ*_ij = max_{τ≥1} ρ_ij(τ), attained at the lag τ*_ij, is the strength and lead time of the influence of i on j. The *directionality* is d_ij = ρ*_ij − ρ*_ji, the excess of the forward peak over the reverse peak. A gauge i is a candidate upstream neighbour of j if d_ij > 0 and ρ*_ij > 0.15. Candidates are ranked by ρ*_ij (1 + d_ij) and the top K = 4 are retained. No geographic information is used. The result is a sparse, directed, data-derived graph in which each edge carries a physically meaningful travel time τ*_ij.

**Plausibility check.** Because the topology is estimated, it is essential to verify that the edges are plausible. The distance between connected gauges, divided by τ*, gives an apparent celerity (km/h), which can be compared with the range of flood-wave celerities in natural channels, typically of order one to a few metres per second (about 4 to 11 km/h). We report the distribution of these quantities in Section 6.1, and in Section 6.9 we test whether the trained network's sensitivity to each neighbour is consistent with ρ*.

**Linear multi-lag routing prior.** For each gauge s with channel set C_s (the gauge itself and its up to four neighbours), we define the feature vector φ_s(t) containing, for every channel c ∈ C_s, the current normalised level z_c(t) and the lagged differences z_c(t−ℓ) − z_c(t) for ℓ ∈ {1, 2, 3, 4, 6, 9, 12, 18, 24, 30} h. With five channels this gives 55 features. A ridge regression, fitted separately for every gauge and every horizon h on the training years, predicts the stage change Δ_s(t,h) = z_s(t+h) − z_s(t):

Δ̂^route_s(t,h) = w_{s,h}ᵀ φ_s(t) + b_{s,h}.

This is a multi-lag linear transfer function from upstream and local stage history to future stage change, which includes the Muskingum routing equation (a linear combination of lagged inflow and outflow) as a special case [@cunge1969]. We refer to it as the *routing prior*. It is fast to fit (seconds per network), has a closed-form solution, and is easily audited. When trained with the own-gauge channel only, it becomes an autoregressive baseline (AR); with the physics-selected neighbours it is the TF baseline of Section 5.2.

## 4.3 The neural twin

The routing prior is linear and therefore cannot represent the nonlinear stage–storage relationships, thresholds in overbank flow and the interaction of upstream pulses that characterise the largest events. PG-FaultTwin adds a compact recurrent network that learns a *residual* correction.

**Inputs.** For origin t, the network receives a window of L = 36 hours for each channel c ∈ C_s: the normalised values (missing hours filled by the last observation within the window; a leading gap is filled by the first observation in the window; a fully missing channel is set to zero), and a binary observation mask. Per time step, the sequence feature is [z_c(τ) − z_c(t)]_c, [mask_c(τ)]_c, [3(z_c(τ) − z_c(τ−1))]_c, i.e. level relative to the current level, observation flag and scaled first difference, giving 3C = 15 features per hour. The current anomaly of every channel, a channel validity flag (channels that do not exist for gauges with fewer than four neighbours are masked out), the age in hours since the last real own-gauge observation (normalised by 72) and the mask of the latest own reading form a static vector.

**Network.** A single-layer GRU [@cho2014] with 64 hidden units encodes the sequence. Its final hidden state is concatenated with the static vector and passed through two fully connected layers of 128 units with GELU activations. Three linear heads output (i) a residual r_h to the routing prior and (ii) two non-negative offsets, via softplus, defining the lower and upper quantile estimates. The prediction of the stage change is

Δ̂_s(t,h) = Δ̂^route_s(t,h) + r_h,  lower: Δ̂ − u_h,  upper: Δ̂ + v_h,

and the stage forecast is Ĥ_s(t+h) = m_s + σ_s [z_s(t)^obs + Δ̂_s(t,h)], where z_s(t)^obs is the last observed normalised level of the gauge's own sensor. The output layer of the residual head is initialised to zero so that training begins from the routing prior. The model has 51,208 trainable parameters.

**Loss.** The point head is trained with the Huber loss (δ = 1 in normalised units) [@huber1964], which is robust to the occasional extreme rise, and the interval heads are trained with the pinball loss at quantile levels 0.1 and 0.9 [@koenker1978]:

ℒ = Huber(Δ̂, Δ) + 0.5 [ρ_{0.1}(Δ − Δ̂_lo) + ρ_{0.9}(Δ − Δ̂_hi)].

The target Δ is the true stage at t+h relative to the *input* current level, which means that when the own sensor is stale or missing, the model must learn to correct for the staleness. The loss is averaged over horizons.

**Event-focused sampling.** More than 99% of training origins are quiescent. To prevent the network from learning only the null case we draw training origins with importance sampling: an origin whose maximum absolute stage change at 6, 12 or 24 h exceeds one σ_s receives a weight of 7 relative to a weight of 1 for others. The resulting model is calibrated on an *unweighted* validation year (Section 4.5), which removes the effect of the sampling on the stated coverage.

**Optimisation.** AdamW [@loshchilov2019; @kingma2015] with weight decay 10⁻⁴, a one-cycle learning-rate schedule with maximum 2 × 10⁻³ [@smith2019], batch size 1,024, gradient-norm clipping at 2.0, eight epochs of 200,000 sampled origins. The checkpoint with the lowest normalised mean absolute error on a 60,000-origin validation subsample is retained. One global model is trained for all gauges and regions; it has no gauge-specific parameters, and the routing prior carries the site specificity.

## 4.4 Fault-tolerant training

The sensor-fault curriculum is applied on the fly to each training window. For every sample, independently for each channel, we apply a combination of (a) i.i.d. hourly dropout at a rate drawn uniformly from [0, 0.5] per sample; (b) with probability 0.4, a block outage of random start and a length of 2 to 26 hours; (c) with probability 0.25, a *stale tail*, in which the most recent 0 to 24 hours are lost (this simulates packet delays and a sensor that has just gone offline); (d) with probability 0.06, the complete loss of the channel. Dropped values are replaced as described in Section 4.3, the mask is set accordingly, the routing prior is recomputed from the corrupted window and the regression target is kept at the true future stage. The network therefore learns to use whichever sensors are alive and to treat the mask as informative: a stale upstream gauge should be trusted less than a live one.

**Edge QC gate.** Dropouts are visible; spikes and stuck readings are not. A causal gate runs at the gateway before the model. It maintains for each sensor the last accepted reading and its age. A reading is rejected as a spike if it differs from the last accepted reading by more than 1.5 (1 + age/2) times the largest hourly stage change that gauge exhibited during the training years, allowing the permitted change to grow as the sensor stays out of contact. A reading is rejected as stuck if it repeats the previous reading exactly for at least twelve consecutive hours *while at least half of the sensor's neighbours show activity* (a change larger than 0.02 σ per hour); the neighbour condition prevents the flagging of legitimately steady low-flow conditions. Rejected readings are set to missing, so the network sees them as dropouts. The gate uses no learned parameters and makes no use of future data.

## 4.5 Conformal calibration and adaptive conformal inference

**Split CQR.** For each horizon h, we compute on the validation year (unweighted, natural distribution) the conformity score E = max(Δ̂_lo − Δ, Δ − Δ̂_hi), the signed distance by which the observation lies outside the quantile band [@romano2019]. The correction q̂_h is the ⌈(n+1)(1−α)⌉/n empirical quantile of E. The conformalised interval is [Δ̂_lo − q̂_h, Δ̂_hi + q̂_h] [@lei2018; @angelopoulos2023]. Under exchangeability between calibration and test origins, the coverage is at least 1 − α. We use α = 0.1 for the main intervals.

**State-conditional (Mondrian) calibration.** Marginal coverage can hide poor coverage exactly when it matters, in rising limbs. We therefore also calibrate separately for two states defined from information available at the origin: *rising* (the last three-hour stage change of the gauge exceeds 0.25σ) and *not rising*.

**Adaptive conformal inference.** Because stage is non-stationary, we run ACI [@gibbs2021] online on the test stream for each gauge and horizon: the miscoverage level α_t is updated by α_{t+1} = α_t + γ (α − 1[miss_t]) with γ = 0.005, using rolling scores from the previous 2,000 origins, and each label is revealed only h hours after the forecast is issued, which respects the information structure of real operations [@zaffran2022].

## 4.6 From trajectories to warnings

**Gradient.** The rise rate over the next k hours is ĝ_k = (Ĥ(t+k) − H(t))/k, in feet per hour; the predicted *time to threshold* θ is the smallest h such that Ĥ(t+h) ≥ θ_s, where θ_s is the flood stage of the gauge.

**Alert rule.** An alert is issued at origin t if the gauge is currently below its threshold and the maximum over the next H_w hours of the forecast stage plus a safety margin of δ σ_s exceeds the threshold. The margin δ is the single tuning parameter that trades detection against false alarms. With conformal intervals the rule can alternatively use the calibrated upper bound at miscoverage α_alert, which gives an alert interpretable as "there is at least a (1 − α_alert)-level chance, under the calibration assumptions, that the stage may reach the threshold".

**Value-based selection.** Given a cost–loss ratio r = C/L (the cost of acting divided by the loss avoided by acting) [@murphy1977; @richardson2000], the optimal alert margin differs between users. A relief agency that can pre-position resources cheaply (small r) wants a low threshold; an agency for whom each alert implies a costly evacuation (large r) wants a high one. We therefore evaluate relative economic value as a function of r, and select the margin on the validation year when a single value is needed (Section 5.4).

## 4.7 Variants used for ablation

To attribute performance, we define seven neural variants (Table 3). The *vanilla GRU* is the strongest conventional configuration without any of the proposed elements; each ablation removes exactly one element of the full model.

| Variant | Neighbours | Routing prior (residual learning) | Fault curriculum | Event-focused sampling |
|---|---|---|---|---|
| **Full (PG-FaultTwin)** | physics-selected | yes | yes | yes |
| No routing prior | physics-selected | no | yes | yes |
| No fault curriculum | physics-selected | yes | no | yes |
| Geographic neighbours | 4 nearest in space | yes | yes | yes |
| Own gauge only | none | yes (AR) | yes | yes |
| Uniform sampling | physics-selected | yes | yes | no |
| Vanilla GRU | 4 nearest in space | no | no | no |

*Table 3. Neural variants for ablation.*
