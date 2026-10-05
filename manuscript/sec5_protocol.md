# 5. Experimental protocol

## 5.1 Forecast origins and targets

Forecasts are issued every second hour of the test period (every third hour for the validation year), for each gauge, at horizons h = 1,…,24 h. An origin is scored only if the true stage is available (after the short-gap interpolation of Section 3.3) at all 24 horizons and at the origin. This yields about 1.33 million scored test origins across the 99 gauges in the temporal protocol. The headline accuracy target is the stage Ĥ(t+h) in feet; the gradient target is the average rise rate over the next 3, 6 and 12 hours. Because stage is quiet for most of the time, we also report accuracy on the *active subset*, the origins at which the true stage changes by more than one σ_s within 6, 12 or 24 hours. The proportion of such origins is reported with the results.

## 5.2 Baselines

We compare PG-FaultTwin with the following seven methods, from simple to strong.

1. **Persistence.** The stage remains at its last observed value. This is the standard no-skill benchmark in stage forecasting and is surprisingly hard to beat in mean absolute error.
2. **AR.** The routing regression of Section 4.2 using only the own-gauge channel (a linear autoregressive model on lagged stage differences).
3. **TF (transfer function).** The routing regression with the physics-selected upstream neighbours. This is our linear physics baseline, equivalent to a multi-lag ARX model.
4. **LightGBM.** Gradient-boosted trees [@ke2017] on the same 55 routing features plus the window maximum, minimum and standard deviation of the own gauge and the age of the own reading. Separate models are trained for h ∈ {1, 3, 6, 12, 18, 24} h and the other horizons are obtained by linear interpolation of the predicted stage changes. The training set is 350,000 origins drawn with the same event-focused weights as the neural models; the validation year provides early stopping (600 trees at most, learning rate 0.05, 63 leaves, Huber objective).
5. **Vanilla GRU.** The GRU network of Section 4.3 using the four geographically nearest gauges, no routing prior, no fault curriculum and uniform sampling.
6. The **neural ablations** of Table 3.
7. **Three-seed ensemble of the vanilla GRU**, used for the uncertainty comparison in Section 6.6.

The same hourly data, the same normalisation, the same origins and the same input windows (including the observation mask) are used for all methods; the baselines therefore have the benefit of the mask channel where applicable. Hyper-parameters were chosen on the validation year only, and the test years were not used for any tuning decision. Neural models are trained with three random seeds for the full model and the vanilla GRU, and with one seed for each ablation; we report means and standard deviations over seeds and mark single-seed results as such.

## 5.3 Generalisation protocols

**Temporal.** Train on water years 2017–2022 at all four regions and test on water years 2024–2026.

**Leave-one-region-out (LORO).** For each region in turn, the gauges of the other three regions provide the training data, and the model is tested on the test years of the held-out region, which contributes no training data. The topology-free prior is still computed within the held-out region from its training-period telemetry, because a deployed system would have this telemetry at commissioning; however, no region-specific *model* fitting takes place: the routing prior for held-out gauges is a single pooled ridge regression fitted over the training regions and shared by all gauges, and the neural weights are those trained on the other regions. This tests whether a model can be commissioned in a new region with no local training of predictors.

## 5.4 Metrics

**Accuracy.** Mean absolute error (MAE) and root-mean-square error (RMSE) in feet at selected horizons; the normalised MAE (nMAE) in units of σ_s, which can be averaged across gauges and horizons; the skill score SS = 1 − MAE/MAE_persistence; per-gauge Nash–Sutcliffe efficiency of the stage change forecast [@nash1970] and Kling–Gupta efficiency [@gupta2009]. The relationship between MAE-based and MSE-based comparisons is explored in Section 6.2.

**Gradient.** The error and the correlation of the predicted and observed three-hour rise rates, and the classification of "rapidly rising" origins (true rise rate above 0.5 σ_s per hour).

**Alerting.** The alert rule of Section 4.6 is evaluated for the minor-flood threshold (primary) and the action stage (secondary) using a forecast window of 24 h. At the origin-level, a *hit* is an alert followed by a threshold exceedance within the window, a *false alarm* is an alert not followed by an exceedance, and a *miss* is an exceedance without an alert; only origins below the threshold are scored. We report the average precision (AP; area under the precision–recall curve), which is threshold-free and appropriate for rare events [@davis2006; @saito2015], and, at the margin that maximises the critical success index (CSI) [@schaefer1990], the probability of detection (POD), the false-alarm ratio (FAR) and CSI. Because choosing the margin on the test data would be optimistic, the same selection rule is applied to all methods, so that comparisons are fair although absolute POD/FAR values are slightly optimistic; for the twin with conformal bounds we additionally report operating points selected on the validation year.

**Event-level verification.** An *event* is an up-crossing of the threshold preceded by 48 hours below it (Section 3.4). For each event, the *lead time* is the number of hours between the earliest origin from which an alert is continuously active and the crossing, up to 24 h (zero if there was no alert at the last origin before the crossing). An event is *detected* if an alert is active at the origin immediately before the crossing. A **false-alarm episode** is a maximal run of consecutive alerts that is not followed by a threshold crossing within 24 h of its start; we report the rate per gauge-year. Because origins are two hours apart, lead times have a resolution of two hours.

**Economic value.** Relative economic value V(r) in the cost–loss model, computed from the contingency table at the optimal alert margin for each cost–loss ratio r, normalised so that V = 0 corresponds to the best of "always alert" and "never alert" and V = 1 to perfect information [@richardson2000; @wilks2001].

**Calibration.** For intervals: empirical coverage by horizon, mean interval width in feet, and coverage in strata (rising versus not rising; high versus low stage), and for the online ACI: the long-run coverage and its deviation from the nominal level.

**Statistical inference.** Differences between methods are tested on *per-gauge* summary scores with the two-sided Wilcoxon signed-rank test [@wilcoxon1945], with Benjamini–Hochberg control of the false-discovery rate across the family of comparisons [@benjamini1995]. For quantities pooled over origins we give 95% bootstrap intervals computed by resampling gauges with replacement [@efron1979]; where autocorrelation matters we use the stationary bootstrap [@politis1994] and the Diebold–Mariano test with the Harvey–Leybourne–Newbold small-sample correction [@diebold1995; @harvey1997] on squared-error loss differentials. Where only 71 minor-flood events are available, event-level estimates carry wide intervals, and we say so.

## 5.5 Fault-injection experiments

We corrupt the *inputs* of the test years (the targets are always the clean truth) according to six fault families (14 settings in total), each applied independently to every sensor.

- **Dropout** (block outages with a mean length of eight hours): 10, 30, 50 and 70% of readings missing.
- **Stuck-at (flat-line)** segments (mean length 24 hours): 10 and 30% of readings.
- **Spikes**: 1% and 3% of readings receive an additive offset of 5 to 20 σ_s with random sign.
- **Calibration bias**: 30% of the sensors receive a constant offset drawn from N(0, b²σ_s²) with b = 0.5 and 1.0.
- **Latency**: all data delayed by 3, 6 or 12 hours.
- **Combined**: 20% dropout, 10% stuck and 1% spikes simultaneously.

We evaluate persistence, the TF baseline, the vanilla GRU and the full twin, with and without the edge QC gate where applicable, on every sixth hourly origin of the test period. We report nMAE at all horizons, the active-subset nMAE and the AP for minor-flood alerts. The QC gate itself is evaluated as a detector against the injected labels (precision, recall and the rate of false rejections of healthy readings).

## 5.6 Edge feasibility

On the 4-core CPU used for all experiments (no GPU), we measure the inference latency of the neural component for a batch of one origin per gauge (all 99 gauges), the throughput for large batches, the model size on disk, and the effect of post-training dynamic int8 quantisation of the GRU and linear layers on forecast error. The energy per inference is approximated from the number of multiply–accumulate operations under explicitly stated assumptions and should be read as an order-of-magnitude indication only.

## 5.7 Software and compute

Experiments use Python 3.11, PyTorch [@paszke2019], scikit-learn [@pedregosa2011] and LightGBM [@ke2017] on a single 4-core CPU with 15 GB of memory and no GPU. The complete pipeline (retrieval of 45 million readings, topology-free prior, all trainings and all evaluation) is scripted and takes a few hours on this hardware.
