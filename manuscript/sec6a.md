# 6. Results

All results are for the final configuration of Section 4 and the test years (water years 2024–2026) unless stated otherwise. Neural results for the full twin and the vanilla GRU are means ± standard deviations over three seeds; other methods are deterministic or single-seed as indicated. There are 1,278,451 scored test origins in the temporal protocol; 10.9% of them are *active* (the true stage changes by more than one σ_s within 6, 12 or 24 h). Of the origins below the minor-flood stage, 807 are followed by an exceedance within 24 h (against 2,904 for the action stage), which gives a sense of how rare the target is: about one origin in 1,100.

## 6.1 The topology-free prior recovers a plausible, if resolution-limited, coupling structure

The cross-correlation analysis produced 321 directed edges, an average of 3.2 upstream neighbours per gauge. Seventy-three gauges obtained the full complement of four neighbours, and nine gauges, usually isolated headwater or main-stem stations without a clearly leading partner, obtained none and fall back on the own-gauge routing model (@@fig:map@@). The median edge connects gauges 23 km apart, has a forward peak correlation of 0.43 and a travel time of 2 h (inter-quartile range 1–3 h); 37% of edges have ρ* > 0.5. The directionality statistic d is positive for every selected edge by construction, with a median of 0.13, i.e. the forward peak correlation exceeds the reverse peak by 0.13 on a typical edge, and is largest in the Mid-Atlantic (0.19) and Ohio Valley (0.12), where tributary–main-stem structure is clearest, and smallest in the Upper Midwest (0.07) (Table 4).

![Gauge networks (colour = region; large dots have an official NWS flood stage) and the edges inferred from telemetry alone (arrow width proportional to the forward peak correlation ρ*).](figures/fig2_map.png)

| Region | Edges | Median τ* (h) | Median ρ* | Median distance (km) | Median apparent celerity (km/h) | Median directionality d |
|---|---|---|---|---|---|---|
| Mid-Atlantic | 98 | 2 | 0.55 | 21.1 | 6.2 | 0.19 |
| Ohio Valley | 74 | 2 | 0.51 | 19.9 | 5.7 | 0.12 |
| Texas Hill Country | 60 | 1 | 0.33 | 25.9 | 13.0 | 0.10 |
| Upper Midwest | 89 | 2 | 0.32 | 29.0 | 12.8 | 0.07 |
| **All** | **321** | **2** | **0.43** | **23.3** | **9.8** | **0.13** |

*Table 4. Summary of the inferred upstream neighbourhoods.*

The physical plausibility check gives a mixed but informative picture (@@fig:prior@@). In the two Appalachian-influenced regions, where gauges are well coupled, the median apparent celerities (6.2 and 5.7 km/h, i.e. 1.7 and 1.6 m/s) lie inside the range expected for flood waves in natural channels. In the Texas Hill Country and the Upper Midwest the apparent celerities are about twice as large (12.8–13.0 km/h). Two features of the data explain much of this and also bound what the prior can claim. First, 44.5% of the edges sit at the smallest resolvable lag of one hour: the gauges are close enough that the true travel time is often below the one-hour resolution of our time series, so the quotient of distance and lag is a *lower* bound on celerity for these edges when the true lag is shorter than one hour and an overestimate when the lag is longer, but in either case it is quantised. Second, straight-line distance underestimates the length of a sinuous channel, which inflates celerities. Overall, 28% of the edges have celerities in the classical 3.6–11 km/h range and 66% fall within a factor of two on either side of it (1.8–22 km/h). The correlation between distance and inferred lag is weak (Pearson r = 0.09), which confirms that the prior should be read as a data-derived *coupling structure* that identifies which gauges carry predictive information about which, rather than as a reconstruction of channel hydraulics. Its value is established by the predictive experiments below.

![(a) Distribution of inferred travel times by region; (b) travel time against straight-line distance, with reference celerities of 3 and 11 km/h; (c) distribution of apparent celerities with the classical natural-channel range (3.6–11 km/h) shaded.](figures/fig3_prior.png)

## 6.2 Forecast accuracy

Table 5 and @@fig:accuracy@@ compare the methods on the test years. Several patterns stand out.

@@TBL:acc@@

*Table 5. Forecast accuracy (temporal protocol, test years). nMAE: mean absolute error in units of the gauge-specific scale σ_s, averaged over the 24 horizons; "quiet" and "active" are the complementary subsets defined in Section 5.1. Mean ± s.d. over three seeds for the neural models.*

**The linear routing prior is a strong baseline and beats persistence under active conditions.** Persistence is hard to beat in the average over all origins because stage is quiet most of the time: its nMAE of 0.234 is dominated by the 89% quiet origins, where it scores 0.107. The own-gauge AR model (0.210) improves on it modestly, and the transfer-function (TF) model, which adds the physics-selected neighbours, reaches 0.198 and reduces the active-condition error from 1.273 (persistence) to 0.955. The physics-selected neighbours are therefore informative in themselves: moving from AR to TF reduces the active error by 9% (1.048 → 0.955) without any nonlinearity.

**Tree-based learning does not help in quiet conditions.** LightGBM obtains a good active error (0.921, the best among the non-neural baselines) but a quiet-state error of 0.153, 43% worse than persistence. Regression trees fitted to a heavy-tailed target (the 24-h change) with a Huber loss produce piecewise-constant outputs that jitter around zero in conditions where the best prediction is "nothing changes". Its overall nMAE (0.237) is therefore worse than persistence's. This illustrates why the quiet/active decomposition is needed: a single pooled score would hide the fact that LightGBM is the best non-neural model for floods and the worst for ordinary days.

**The recurrent residual model improves on all baselines where it matters.** PG-FaultTwin has the lowest overall nMAE (0.185 ± 0.000), the lowest active error (0.876 ± 0.001) and the lowest RMSE at 24 h (0.590 ft against 0.622 ft for the vanilla GRU and 0.630 ft for TF), while matching the best quiet-state error (0.100 against 0.098 for the vanilla GRU). Relative to persistence the active-condition error falls by 31% (1.273 → 0.876), and the 24-h MAE by 21% (0.225 → 0.177 ft). Relative to the vanilla GRU, which has identical capacity apart from the proposed elements, the active error falls by 8% (0.950 → 0.876) and the 24-h RMSE by 5%. The paired comparison across gauges (Table 6) shows that the improvements are broad rather than driven by a few gauges, although their size varies by comparator. The full twin has a lower active-condition nMAE than persistence, AR and TF at all 99 gauges, than the vanilla GRU at 78% of gauges (median difference −0.047σ) and than LightGBM at 65% of gauges (median difference −0.021σ); all BH-adjusted p-values are below 5 × 10⁻⁴. For the overall nMAE the advantage is large against persistence, AR, TF and LightGBM (88–100% of gauges) but marginal against the vanilla GRU (58% of gauges, median difference −0.002σ, adjusted p = 0.003), which is consistent with the tie in quiet conditions: the proposed elements pay off in active conditions rather than on ordinary days.

@@TBL:sig@@

*Table 6. Paired per-gauge comparisons of PG-FaultTwin with each baseline (Wilcoxon signed-rank test on the per-gauge mean nMAE, seed-averaged; BH-adjusted across the family). Negative median differences favour PG-FaultTwin.*

The improvement grows with the lead time (@@fig:accuracy@@). At one hour ahead all models are within a few hundredths of σ_s of each other, as expected for a variable with strong persistence; at 24 h the active-condition nMAE is 2.04 for persistence and 1.47 for the full twin, a reduction of 28%.

![Normalised MAE against lead time for (a) all test origins and (b) the active subset. The advantage of the learned, network-aware models widens with lead time.](figures/fig5_accuracy.png)

**A cautionary episode: a stuck sensor can corrupt a learned twin while a physics-style model survives.** Before the dead-sensor screen of Section 3.2 was introduced, an early version of the experiments included a gauge in the Guadalupe basin that reported a constant 16.2 ft from 2023 onward after a median of −0.5 ft in 2016–2022. The neural twin, which used the (now abnormal) absolute level of this neighbour as an input, produced a persistent 2.9σ bias at the downstream gauge in the test years, whereas the routing prior, which uses only lagged *differences*, was unaffected and was accurate at the same gauge. We removed such sensors with the screen and re-ran every experiment from scratch, but the episode is a real-world example of the failure class this paper targets and motivates two design choices: the routing prior deliberately uses no absolute levels, and the fault experiments of Section 6.8 include bias faults.

## 6.3 Gradient skill

The rise rate is the quantity a duty officer watches. Table 7 evaluates the predicted three-hour rise rate derived from the stage trajectory. All learned models achieve a 3-h rise-rate MAE of about 0.011 ft/h overall, a 33% reduction from persistence (0.016 ft/h), but the differences emerge in active conditions, where the error of the full twin (0.0430 ft/h) is 45% below persistence (0.0783), 14% below AR (0.0502), 8% below the vanilla GRU (0.0466) and 3% below TF (0.0444). The correlation between predicted and observed rise rates is 0.714 ± 0.011 for the full twin, essentially tied with TF (0.715) and higher than the vanilla GRU (0.669 ± 0.008), AR (0.660) and LightGBM (0.686). The advantage of the full twin is clearer at the six-hour horizon (0.687, against 0.657 for TF, 0.639 for LightGBM and 0.612 for the vanilla GRU) and in the detection of *rapidly rising* origins, for which the F1 score is 0.532 against 0.509 for TF and 0.478 for the vanilla GRU. LightGBM's F1 (0.530) is comparable, again showing that tree-based learners are competitive at identifying rises even when they are poor in quiet conditions.

@@TBL:grad@@

*Table 7. Gradient (rise-rate) skill. Rapid rise: true 3-h rise exceeding 0.5σ_s (26,140 test origins).*

## 6.4 Flood alert verification

The alert rule of Section 4.6 turns each stage trajectory into a warning decision; Table 8 reports the verification for the minor-flood stage (primary) and the action stage. We start with the threshold-free average precision, which is the area under the precision–recall curve and does not depend on the margin δ. The full twin attains 0.274 ± 0.002 for the minor stage against 0.261 for TF, 0.195 for LightGBM, 0.187 for AR, 0.170 ± 0.007 for the vanilla GRU and 0.086 for persistence; for the action stage the corresponding figures are 0.354 ± 0.002, 0.338, 0.286, 0.242, 0.251 ± 0.004 and 0.119. The physics-selected neighbours and the routing prior are thus the main drivers of alert skill: TF, a linear model, outperforms the nonlinear vanilla GRU by 53% in minor-stage AP, and the full twin adds a further 5% (minor) and 5% (action) over TF.

@@TBL:alert_minor@@

*Table 8a. Verification of alerts for the NWS minor flood stage (68 verified events at 37 gauges; margin δ chosen to maximise CSI for each method; see text).*

@@TBL:alert_action@@

*Table 8b. Verification of alerts for the NWS action stage (245 verified events at 52 gauges).*

At each method's CSI-optimal margin, the full twin detects 57 of the 68 verifiable minor-flood events (84%) with a mean lead time of 5.1 h (6.2 h among detected events) and 0.44 false-alarm episodes per gauge-year, whereas TF detects 58 events at 5.9 h but with 0.62 false-alarm episodes, and the vanilla GRU detects 54 events with 3.1 h lead and 0.34 false-alarm episodes. For the action stage, the full twin has the lowest false-alarm ratio (0.351 against 0.404 for TF and 0.616 for the vanilla GRU) and the fewest false-alarm episodes (1.02 per gauge-year against 1.28 for TF and 1.46 for the vanilla GRU), at a detection rate of 209 of 245 events (85%) and a lead of 5.7 h. In other words, the full twin delivers detection and lead time comparable to the strong linear baseline, with about a quarter to a third fewer false alarms: 29% fewer episodes for minor floods and 20% fewer for the action stage.

@@fig:alerts@@ shows the precision–recall curves, the performance diagram and the lead-time distributions: the full twin's precision–recall curve lies above or on that of TF over most of the range and well above all other methods. Three caveats apply. First, the absolute skill is modest, with CSI values near 0.24–0.26 and POD near 0.3 at the CSI-optimal margin. This is not surprising: the information set consists solely of stage telemetry, no rainfall, so that rises driven by rainfall that has not yet reached the gauges cannot be anticipated, and for the rapidly responding headwater gauges a 24-h window contains events that start from a dry channel. Second, the margin δ was selected on the test period for every method, because the validation year contains only two minor-flood events and cannot support selection; this gives all methods the same optimistic advantage, and the threshold-free AP is not affected. Third, the numbers of events are small (68 minor-flood events), and differences of one or two events in the detection counts should not be over-interpreted; the gauge-level bootstrap in Section 6.5 quantifies this uncertainty.

![(a) Precision–recall curves; (b) performance diagram (POD against success ratio, markers at each method's CSI-optimal margin); (c) event lead times.](figures/fig6_alerts.png)

Fig. @@num:case@@ (Section 6.11) shows one such alert for a real event, including its limits.
