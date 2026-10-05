TITLE: A Fault-Tolerant, Physics-Guided Digital Twin for Flood Stage-Rise and Gradient Early Warning from IoT River-Gauge Networks, with Conformal Uncertainty and Cost-Aware Alerting
CENTER: *[Author names and affiliations to be added]*
CENTER: Manuscript prepared for the Engineering Applications of Artificial Intelligence special issue "IoT in Disaster Management" (VSI: EAAI_IoT Disaster Mgmt)

# Abstract

Flood early warning depends on telemetered gauge networks whose sensors fail, drift and go silent during storms, yet learned flood models are mostly evaluated on curated, fault-free records. We present PG-FaultTwin, a digital twin for 1–24 h stage-rise and flood-threshold forecasting that (i) infers upstream neighbours and travel times from telemetry alone, without river-network data; (ii) corrects a linear multi-lag routing prior with a 51k-parameter recurrent residual network trained under a sensor-fault curriculum with explicit observation masks; (iii) screens inputs with a causal edge quality-control gate; and (iv) issues conformalised intervals and cost-aware alerts. We evaluate on 99 USGS gauges in four US regions (44.8 million raw readings, ten water years) with official NWS flood stages, a chronological split, leave-one-region-out transfer and 14 injected fault settings. Against persistence the twin lowers active-condition error by 31%; against an identically sized vanilla GRU it lowers it by 8% and raises minor-flood average precision from 0.170 to 0.274. Its edge over a strong linear routing baseline is small but resolved (+0.015 AP, 95% interval 0.002–0.028; 20–29% fewer false-alarm episodes). The twin stays ahead of persistence up to 70% data loss, but spikes are damaging without the gate, which costs 15% of alert precision on clean data. State-conditional conformal calibration restores 0.89 coverage on rising limbs (0.79–0.82 unconditionally). Transfer to an unseen region keeps 16–25% of the gain over persistence but fails in flash-flood terrain. A 99-gauge forecast takes 5 ms on one CPU thread.

**Keywords:** Disaster management; IoT; Digital twin; Early warning; AI & ML algorithms; Flood forecasting; Sensor fault tolerance; Conformal prediction

# Highlights

- Topology-free physics prior: upstream gauges and lead times inferred from telemetry alone.
- Fault-curriculum twin stays ahead of persistence up to 70% sensor data loss.
- Causal edge QC gate is essential against spikes but costs 15% alert precision.
- State-conditional conformal calibration restores 0.89 coverage on rising limbs.
- Open 99-gauge benchmark, official flood stages, leave-one-region-out transfer test.

PAGEBREAK


# 1. Introduction

River floods remain one of the most frequent and costly natural hazards, and the evidence that their human toll is growing is now difficult to dispute. Satellite observations indicate that the share of the world's population living in flood-exposed areas has risen substantially in recent decades, largely because settlement has expanded into floodplains faster than hazard has changed (Tellman et al., 2021). At the same time, the physical drivers of flooding are shifting in regionally heterogeneous ways: climate change both increases and decreases river floods depending on the regime, season and catchment (Blöschl et al., 2019; Kundzewicz et al., 2014), and disastrous floods emerge from the coincidence of meteorological extremes, antecedent catchment state and exposure (Merz et al., 2021). For the engineer responsible for a regional warning service, the practical implication is that the hazard cannot be managed through structural protection alone. A complementary and much cheaper lever is time: a forecast that reaches people and operators a few hours before the water does allows evacuation, asset protection, and the pre-positioning of response resources. The economic case for this lever is well documented. Hydro-meteorological services, early warning and evacuation are among the most cost-effective measures for reducing disaster losses (Hallegatte, 2012), benefit–cost analyses of operational flood warning systems regularly return favourable ratios (Carsell et al., 2004; Islam et al., 2024; Pappenberger et al., 2015; Rai et al., 2020), and the value of a warning depends strongly on how its uncertainty is communicated and used (Verkade and Werner, 2011).

The Internet of Things (IoT) has changed what is technically possible at the sensing end of this chain. Dense networks of telemetered gauges, low-power radios, gateways and cloud back-ends now deliver observations of river stage every few minutes from thousands of locations (Atzori et al., 2010; Gubbi et al., 2013), and the disaster-management literature has repeatedly argued that these capabilities should be organised into end-to-end architectures that connect sensing, communication, analytics and decision support (Basha et al., 2008; Poslad et al., 2015; Ray et al., 2017). A parallel development is the *digital twin*, a continuously updated virtual counterpart of a physical system that is used to monitor, forecast and test interventions (Fuller et al., 2020; Grieves and Vickers, 2017; Tao et al., 2019). Earth-system scientists have proposed twins at planetary scale (Bauer et al., 2021; Bauer et al., 2021), and recent work has begun to instantiate flood-specific twins coupling observations with hydrologic–hydrodynamic models or graph networks (M. C. Rápalo et al., 2024; Roudbari et al., 2024). Machine learning has meanwhile become a serious operational tool in hydrology. Recurrent networks trained on many catchments outperform catchment-calibrated conceptual models in a range of settings (Kratzert et al., 2018; Kratzert et al., 2019; Kratzert et al., 2019), they extend to extreme events and ungauged basins (Frame et al., 2022; Nearing et al., 2024), and they have been deployed in regional operational flood-warning frameworks (Nevo et al., 2022).

Despite this progress, three gaps stand between published models and a warning service that a regional agency could operate on its own IoT gauge network.

**Gap 1: the target of the forecast.** Most hydrological learning work reports skill on discharge or stage as a time series, summarised by Nash–Sutcliffe or Kling–Gupta efficiency (Gupta et al., 2009; Knoben et al., 2019; Nash and Sutcliffe, 1970). These aggregate measures are dominated by the long stretches of quiescent flow and say little about the quantity a warning duty officer needs: *how fast is the river rising, and when will it cross the threshold at which damage starts?* The rise rate (the stage gradient) and the time to threshold are the operational variables, and the metrics that matter are event-based (probability of detection, false-alarm ratio, critical success index, lead time) and economic rather than squared-error based (Richardson, 2000; Roebber, 2009; Schaefer, 1990).

**Gap 2: the sensors are assumed to be perfect.** Nearly all of the studies cited above are trained and evaluated on curated archives. Real IoT deployments, by contrast, produce dropouts, stuck readings, spikes, calibration drift and delayed packets at appreciable rates (Ni et al., 2009; Sharma et al., 2010; Teh et al., 2020). A model that has never seen a missing input cannot be expected to degrade gracefully when a headwater gauge goes offline in the first hour of a storm, which is exactly when its information is most valuable. Sensor-fault handling is usually treated as a separate pre-processing stage, so the downstream model's robustness is never quantified.

**Gap 3: the uncertainty is not decision-grade.** Deep ensembles, Monte-Carlo dropout and quantile heads give uncertainty estimates (Gal and Ghahramani, 2016; Klotz et al., 2022; Lakshminarayanan et al., 2017), but their calibration is rarely verified under the temporal and spatial shifts that characterise deployment, and it is typically not connected to the alerting rule. Distribution-free conformal prediction offers finite-sample coverage guarantees under exchangeability (Angelopoulos and Bates, 2023; Lei et al., 2018; Romano et al., 2019), and its extensions to time series and distribution shift (Barber et al., 2023; Gibbs and Candès, 2021; Zaffran et al., 2022) are promising, yet their value for *threshold-crossing alerts* on river networks has not been examined systematically. Warning agencies must also manage the false-alarm ("cry-wolf") problem (Barnes et al., 2007; LeClerc and Joslyn, 2015; Trainor et al., 2015), which calls for a principled way to trade detection against false alarms.

A further practical obstacle concerns *topology*. Graph-based hydrological models require a river network, i.e. the knowledge of which gauge lies upstream of which, together with reach lengths. In many data-poor regions such information is incomplete, outdated or not digitised at the resolution of the gauge network. Hydraulic routing, from the classical Muskingum formulation (Cunge, 1969; Todini, 2007) to modern hybrid models, encodes the physics of flood-wave propagation but requires parameters that must be calibrated reach by reach. It would be valuable if the essential physical content of routing (that disturbances travel downstream with a finite celerity, so upstream gauges carry predictive information at lead times set by travel time) could be extracted from the telemetry itself.

## 1.1 Contribution

This paper addresses the three gaps within a single, deployable framework which we call **PG-FaultTwin**, a fault-tolerant, physics-guided digital twin for stage-rise and gradient early warning on IoT gauge networks. Its contributions are as follows.

1. **A gradient-first forecasting target and decision-centred evaluation protocol.** The twin forecasts the full 1–24 h stage trajectory and derives from it the rise rate and the threshold-crossing probability. We evaluate with event-based verification against official National Weather Service flood stages (probability of detection, false-alarm ratio, critical success index, lead time, false-alarm episodes per site-year), with relative economic value across the cost–loss spectrum, and with conventional accuracy measures reported separately for quiescent and active conditions.
2. **A topology-free physics prior.** Upstream–downstream relationships, travel times and flood-wave celerities are estimated from lagged cross-correlation of the gauge telemetry alone. They define (i) which upstream gauges feed each forecast point and (ii) a linear multi-lag routing model, an ARX/Muskingum-type transfer function, whose prediction the neural component corrects. We check that the inferred celerities are physically plausible and that the trained network's reliance on neighbours agrees with the inferred travel times.
3. **Fault-tolerant learning.** A stochastic sensor-fault curriculum (random and block dropouts, stale tails, lost channels) is applied during training together with explicit observation-mask channels, and a causal, lightweight edge quality-control gate screens spikes and stuck readings. We quantify robustness under six fault families (14 settings), including dropout rates up to 70%, flat-lining, spikes, calibration bias and packet latency of up to 12 h.
4. **Decision-grade uncertainty.** Quantile heads are conformalised (CQR) on a held-out year, and adaptive conformal inference is run online. We evaluate coverage by horizon, by hydrological regime and under spatial transfer, and we convert the calibrated upper bound into an alerting rule whose false-alarm/detection trade-off is explicit.
5. **A large-scale, fully open benchmark with two generalisation protocols.** We assemble 99 USGS telemetered gauges in four hydro-climatically different regions of the United States (Mid-Atlantic, Ohio Valley, Texas Hill Country, Upper Midwest), covering ten water years and about 44.8 million raw 15-minute readings. Evaluation uses a strict chronological split and a leave-one-region-out protocol that tests transfer to a region never seen in training.
6. **Edge feasibility.** We report parameter counts, latency and memory on a CPU-only device, together with the effect of post-training dynamic quantisation, to show that the forecasting component can be hosted on a gateway.

The remainder of the paper is organised as follows. Section 2 reviews related work. Section 3 describes the data and the study design. Section 4 presents the framework. Section 5 details the experimental protocol. Section 6 reports results. Section 7 discusses implications, limitations and the path to operations, and Section 8 concludes.


# 2. Related work

## 2.1 IoT architectures for disaster management and flood monitoring

The idea that networked sensors can transform disaster response is almost as old as the sensor-network field itself. Early deployments such as the Honduran flood-monitoring system of Basha and co-authors combined in-situ gauges with model-based detection to raise warnings in remote basins with intermittent connectivity (Basha et al., 2008). Subsequent surveys organise the field by the functional layers of an IoT stack (perception, network, processing and application) and by disaster phase (mitigation, preparedness, response and recovery) (Ray et al., 2017), and propose semantic, interoperable architectures that fuse heterogeneous sources for crisis management (Poslad et al., 2015). The general IoT literature supplies the enabling technologies: ubiquitous identification and sensing, low-power wide-area connectivity, and cloud platforms (Atzori et al., 2010; Gubbi et al., 2013). Two themes are particularly relevant to the present paper. The first is *edge computing*, which moves computation to gateways close to the sensors to reduce latency and bandwidth and to maintain operation when backhaul links fail during the very events they are meant to monitor (Satyanarayanan, 2017; Shi et al., 2016). The second is *tiny machine learning*, which pushes inference onto microcontroller-class hardware and has been surveyed extensively for IoT settings (Abadade et al., 2023; Dutta and Bharali, 2021; Rajapakse et al., 2023). Most IoT-for-disaster papers, however, evaluate the communication or architectural layer and treat analytics as a black box. Where analytics are examined, evaluation often consists of a single station or a short record, and robustness to sensor faults, which is the defining characteristic of field IoT, is seldom quantified.

## 2.2 Sensor data quality and fault handling

Field-deployed environmental sensors fail in characteristic ways. Ni and colleagues catalogued the fault types encountered in real sensor-network datasets, including short-duration spikes, constant ("stuck-at") readings, noise and calibration offsets (Ni et al., 2009), and Sharma and co-authors showed that these faults are prevalent in real datasets and that simple detectors based on rate-of-change, flat-lining and cross-sensor consistency catch a large part of them (Sharma et al., 2010). A systematic review of sensor data quality confirms that quality is multi-dimensional (accuracy, completeness, timeliness, consistency) and that most work treats it as a pre-processing step (Teh et al., 2020). General anomaly-detection techniques (Chandola et al., 2009; Liu et al., 2008) can be applied, but they usually require labelled or representative data and do not account for the hydrological context in which a sudden rise can be either a spike or a genuine flash flood. A separate line of work handles missing values by learning imputations or by modelling missingness directly inside recurrent networks (Cao et al., 2018; Che et al., 2018; Du et al., 2023). Such methods are typically evaluated on imputation error rather than on the downstream forecasting and alerting skill, and they presuppose that faults have been flagged. Our approach differs in two respects: faults are injected into the *forecasting* task and quantified in terms of alert skill, and the network is trained with the mask as an explicit input so that it learns to rely on whichever sensors are alive.

## 2.3 Machine learning for streamflow and flood forecasting

Reviews of deep learning in hydrology document rapid growth over the past decade (Mosavi et al., 2018; Sit et al., 2020; Tripathy and Mishra, 2024; Zounemat-Kermani et al., 2020). Long short-term memory networks (Hochreiter and Schmidhuber, 1997) trained jointly on hundreds of catchments reproduce rainfall–runoff behaviour better than calibrated conceptual models, and the benefit comes from learning regional regularities that single-catchment models cannot access (Kratzert et al., 2018; Kratzert et al., 2019; Kratzert et al., 2019). Variants handle several time scales with one network (Gauch et al., 2021), analyse what hydrological concepts the hidden states encode (Lees et al., 2022), quantify predictive uncertainty (Klotz et al., 2022), and extend to extreme events (Frame et al., 2022) and to ungauged watersheds at global scale (Nearing et al., 2024). Operational deployments have started to appear: a regional framework based on LSTMs and linear models with inundation mapping is in service for flood warnings (Nevo et al., 2022). Attention-based and transformer models have been applied to streamflow (Lim et al., 2021; Rasiya Koya and Roy, 2024), and interpretable spatio-temporal attention networks have been proposed for flood forecasting with upstream stations (Ding et al., 2020). Within the general time-series literature, the strongest recent baselines include temporal convolution (Bai et al., 2018), basis-expansion networks (Oreshkin et al., 2020), probabilistic autoregressive RNNs (Salinas et al., 2020), patch-based transformers (Nie et al., 2023), and long-sequence transformers (Zhou et al., 2021), although simple linear models can match or exceed transformers in long-horizon forecasting benchmarks (Zeng et al., 2023). This finding is consistent with our experience that, at 1–24 h lead times along a river network, much of the predictable signal is captured by linear lagged relationships between gauges, and that the value of nonlinear learning lies in the rapidly rising, rare cases.

Most of these studies use precipitation or reanalysis forcings as the primary input. A telemetry-only forecast, one that uses only upstream gauge measurements, is a weaker information set for catchments with short response times, because it cannot anticipate rainfall that has not yet reached the channel. It is nevertheless an important operating regime: it remains available when the weather feed is delayed or absent, it is the natural design for sensor networks in data-poor regions, and it provides a lower bound on skill that precipitation forcing can then improve. We return to this point in Section 7.

## 2.4 Graph-based and network-aware models

River networks are naturally graphs, and graph neural networks have therefore been applied to streamflow. Graph convolutional networks (Kipf and Welling, 2017) and diffusion-convolutional recurrent networks (Li et al., 2018), first developed for traffic forecasting, motivate hydrological variants such as graph convolutional GRUs for hourly streamflow (Sit et al., 2021), directed-graph networks for multi-step daily streamflow (Liu et al., 2022), and graph networks coupled with digital-twin visualisation for flood forecasting (Roudbari et al., 2024). Graph WaveNet showed that an adaptive adjacency matrix can be learned from data when the graph is unknown (Wu et al., 2019). Those methods either require the network topology to be supplied or learn a dense latent adjacency with no physical constraint. In contrast, we estimate a *sparse, directed* neighbourhood from the lag structure of the telemetry, which is physically interpretable (the lag at maximum correlation is a travel time) and can be audited against known geography.

## 2.5 Physics-guided and hybrid learning

Physics-informed and physics-guided learning combines scientific knowledge with data-driven models (Karniadakis et al., 2021; Raissi et al., 2019; Reichstein et al., 2019; Willard et al., 2023). In hydrology, the debate has emphasised that process knowledge remains valuable even when networks are highly accurate (Nearing et al., 2021), and concrete proposals include mass-conserving recurrent architectures (Hoedt et al., 2021) and physics-informed learning for river forecasting (Bhasme et al., 2022). Muskingum-type routing is the classical way to encode wave attenuation and lag between river sections (Cunge, 1969), and mass-conservative formulations have been derived (Todini, 2007). Our physics prior is deliberately modest: a data-estimated multi-lag linear transfer function acting on upstream gauges, which is the structure implied by linear routing, with the neural network learning only the *residual*. The design has three advantages: it guarantees that the network starts from a physically sensible baseline, it improves data efficiency, and it provides a transparent baseline that operators already understand.

## 2.6 Digital twins for floods and the environment

Since its introduction in product-lifecycle management (Grieves and Vickers, 2017), the digital twin has been adopted in manufacturing (Tao et al., 2019) and, increasingly, in infrastructure and environmental systems (Fuller et al., 2020). Earth-system twins promise interactive, observation-driven simulation at planetary scales (Bauer et al., 2021; Bauer et al., 2021). At the regional scale, twin-like flood systems have been demonstrated that couple hydrologic and hydrodynamic models with observation streams in data-scarce regions (M. C. Rápalo et al., 2024) and that attach graph-based forecasting to visual analytics (Roudbari et al., 2024). These contributions focus on architecture and on forecasting accuracy in selected basins. What is largely missing is a quantitative treatment of the *twin's behaviour under imperfect observation*, namely how state estimates and alerts degrade when the physical layer misbehaves, and a calibrated link between the twin's uncertainty and the decision to alert. We adopt the functional definition of a twin as a synchronised, queryable virtual model that supports monitoring, prediction and what-if analysis, and we examine the three properties that matter for an IoT deployment: synchronisation robustness, calibrated prediction and decision support.

## 2.7 Uncertainty quantification and conformal prediction

Probabilistic forecasts should be evaluated by proper scoring rules and by calibration and sharpness (BRIER, 1950; Gneiting and Raftery, 2007; Gneiting et al., 2007; Murphy, 1993). Quantile regression (Koenker and Bassett, 1978) provides a direct route to prediction intervals but offers no guarantee of finite-sample coverage. Split conformal prediction wraps any predictor to give marginal coverage under exchangeability (Lei et al., 2018), and conformalised quantile regression adapts interval width to local difficulty (Romano et al., 2019); a recent monograph-length introduction is available (Angelopoulos and Bates, 2023). River stage data are not exchangeable: they are autocorrelated, non-stationary and subject to regime change. Adaptive conformal inference adjusts the target miscoverage online to maintain long-run coverage under shift (Gibbs and Candès, 2021), extended with aggregation or expert methods for time series (Xu and Xie, 2021; Zaffran et al., 2022), and a general theory describes the cost of non-exchangeability (Barber et al., 2023). In hydrology, uncertainty estimation for deep rainfall–runoff models has been compared across approaches (Klotz et al., 2022), but guarantees for event detection have received little attention. We contribute an empirical study of conformal intervals for gradient and threshold alerts in a multi-region setting.

## 2.8 Forecast verification and the value of warnings

Verification of warnings differs from verification of continuous forecasts. The probability of detection, false-alarm ratio and critical success index summarise a 2×2 contingency table (Schaefer, 1990), and the performance diagram displays them jointly with bias (Roebber, 2009). For rare events, precision–recall analysis is more informative than ROC analysis (Davis and Goadrich, 2006; Saito and Rehmsmeier, 2015). Economic value in the cost–loss framework quantifies the benefit of a forecast to users whose protective cost is a fraction of the avoidable loss (Murphy, 1977; Richardson, 2000; Wilks, 2001), and has been used to compare flood forecasting systems (Pappenberger et al., 2015; Verkade and Werner, 2011). Social-science research shows that false alarms erode public response (Barnes et al., 2007; LeClerc and Joslyn, 2015; Trainor et al., 2015), although the relationship is nuanced, and that warning systems must be judged as socio-technical systems (Basher, 2006; Sorensen, 2000). Operational river forecasters in turn face a distinctive set of challenges including data assimilation, uncertainty communication and the maintenance of observation networks (Pagano et al., 2014). The verification suite of this paper builds on these strands: event-level POD/FAR/CSI and lead time, false-alarm episodes per site-year, precision–recall summaries and relative economic value.

## 2.9 Positioning

Table 1 summarises how the present work relates to representative strands of the literature along the dimensions that matter for an IoT-based flood twin. The comparison is qualitative and based on the published scope of the cited works rather than on a re-implementation.

| Strand (representative works) | Learned from telemetry | Topology-free | Sensor faults quantified | Calibrated uncertainty | Event/economic verification | Multi-region transfer |
|---|---|---|---|---|---|---|
| IoT architectures (Basha et al., 2008; Poslad et al., 2015; Ray et al., 2017) | – | – | partial | – | partial | – |
| LSTM rainfall–runoff and flood models (Frame et al., 2022; Kratzert et al., 2018; Nevo et al., 2022) | yes | n/a (catchment-based) | – | partial | partial | yes |
| Graph/attention flood models (Ding et al., 2020; Liu et al., 2022; Sit et al., 2021) | yes | – (graph supplied) | – | – | – | – |
| Digital-twin flood systems (M. C. Rápalo et al., 2024; Roudbari et al., 2024) | partial | – | – | – | partial | – |
| Conformal/ensemble UQ in hydrology (Klotz et al., 2022) | yes | n/a | – | yes | – | partial |
| **This work (PG-FaultTwin)** | yes | yes | yes | yes (conformal) | yes | yes |

*Table 1. Qualitative positioning of this work. "–" means that the dimension is not a focus of the cited strand; "partial" indicates limited treatment.*


# 3. Study design and data

## 3.1 Design principles

Three principles guided the design of the study. First, **every data source is open, real and reproducible**: no simulated hydrographs are used, and all observations are retrieved from public web services. Second, **the information set is restricted to what an IoT gauge network itself delivers**, namely stage (water level) telemetry. This choice isolates the contribution of network-aware learning and of fault tolerance from that of weather forcing, and it corresponds to the minimal deployable configuration for a data-poor basin. Third, **evaluation is split by time and by space**: the primary protocol is strictly chronological, and a second protocol withholds an entire hydro-climatic region from training.

## 3.2 Sensor network

The sensor layer is the telemetered stream-gauge network operated by the U.S. Geological Survey (USGS). Each station reports stage at a nominal 15-minute interval through satellite or cellular telemetry, which makes it a representative large-scale IoT deployment. We queried the USGS National Water Information System (NWIS) instantaneous-values service for stage (parameter 00065) and discharge (parameter 00060) over ten water years, from 1 October 2016 to 30 September 2026. To construct networks with enough hydrologically coupled stations, we selected four regions that differ in climate, relief and flood-generating mechanisms: the **Mid-Atlantic** (Monocacy, Catoctin and upper Potomac gauges in Maryland and northern Virginia), the **Ohio Valley** (headwater tributaries of the Monongahela in the Appalachian Plateau of West Virginia, i.e. the Ohio River basin), the **Texas Hill Country** (Guadalupe, Blanco and Cibolo basins, a well-known flash-flood region with thin soils and steep terrain), and the **Upper Midwest** (low-gradient, rain- and snowmelt-driven rivers of central Iowa such as the Raccoon and Des Moines basins). Within each region we located the densest cluster (centred at 39.3°N 77.4°W, 39.2°N 80.2°W, 29.8°N 98.1°W and 41.7°N 93.4°W respectively) of active telemetered stream stations that are also reference points of the National Water Prediction Service of the National Weather Service (NWS), and we retained the 32 nearest stations to the cluster centre. This produced 128 candidate stations.

Stations were screened for data availability and for dead sensors. A **dead-sensor screen** treats any 180-day window in which the hourly stage varies by no more than 0.05 ft as a stuck or failed sensor and blanks it. The screen is deliberately conservative, because Texas creeks are legitimately flat for weeks in dry spells; it nevertheless identified five stations with flat-lined stretches of six months or longer. Two of them (in the Guadalupe basin) are instructive: from 2023 one station reports a constant 16.2 ft, against a 2016–2022 median of about −0.5 ft, a failure that would pass any range check yet contaminated the forecasts of its downstream neighbour in a preliminary run of our experiments (see Section 6.2). Stations whose stage coverage after this screen was below 85% of the ten-year hourly record were discarded, leaving **99 gauges** (Mid-Atlantic 29, Ohio Valley 23, Texas Hill Country 19, Upper Midwest 28) from the 128 candidates. The retained stations hold 44.8 million raw 15-minute readings. After hourly aggregation (Section 3.3) the mean coverage is 98.9% (minimum 88.8%), which still amounts to 4,625 natural gaps; the median gap is a single hour, but 21% of gaps exceed six hours and 10% last at least 22 hours (Table 2). These natural outages form the baseline level of fault the models must tolerate even before any synthetic fault is injected.

| Region | Gauges | Mean coverage | Gauges with NWS minor flood stage | Gauges with action stage | Median stage (ft) | Typical 24-h stage change σ (ft) |
|---|---|---|---|---|---|---|
| Mid-Atlantic | 29 | 0.989 | 23 | 27 | 2.20 | 0.35 |
| Ohio Valley | 23 | 0.993 | 15 | 17 | 5.20 | 0.67 |
| Texas Hill Country | 19 | 0.973 | 7 | 7 | 4.03 | 0.34 |
| Upper Midwest | 28 | 0.994 | 22 | 24 | 5.37 | 0.59 |
| **All** | **99** | **0.989** | **67** | **75** | | |

*Table 2. Sensor network summary. σ is the regional median of the per-gauge standard deviation of 24-h stage change in the training period, used for normalisation.*

## 3.3 Pre-processing and quality control

Raw readings were parsed with their reported time zone and converted to UTC. Physically implausible stage values (below −20 ft or above 300 ft) were removed. A conservative spike filter removed isolated excursions: a reading is deleted when it changes by more than twelve robust standard deviations of the station's 15-min differences (and at least 1 ft) in one direction and returns in the next step, which eliminates single-sample electronic noise without suppressing sustained rises. Readings were aggregated to hourly means. We retained hourly resolution for modelling because the 1–24 h warning horizon of interest does not require sub-hourly states, because it reduces computational cost by a factor of four, and because the cross-correlation analysis of Section 4.2 is better conditioned at this scale; the lead-time resolution of event verification is accordingly two hours (Section 5.4).

The "truth" series used for supervision and verification closes gaps of at most six hours by linear interpolation; longer gaps remain missing and the corresponding forecast origins are excluded from scoring. Model *inputs*, in contrast, never use interpolation: missing readings are represented by an observation-mask channel together with last-observation-carried-forward values (capped at 72 h), so that the model sees exactly what a real-time system would see.

## 3.4 Flood thresholds and event definition

Warning-relevant events are defined using the official flood categories published by the NWS through the National Water Prediction Service API. For each station the *action stage* (the level at which agencies begin to prepare) and the *minor flood stage* (the level at which minimal property damage begins) are retrieved where defined. Sixty-seven of the 99 gauges have a minor flood stage and 75 have an action stage. A **flood event** is an up-crossing of the threshold that is preceded by at least 48 hours below it. Thresholds are used only for labelling and verification; they are never used as model inputs. Because flood stages are rare by construction, the numbers of events are modest, which is why we report both thresholds and quantify uncertainty by bootstrap (Section 5.5): in the training period (water years 2017–2022) there are 220 minor-flood events and 625 action-stage events, in the validation year (water year 2023, a dry year in these regions) there are only 2 and 16, and in the test period (water years 2024–2026) there are **69 minor-flood events at 37 gauges** and **249 action-stage events at 52 gauges**. The validation year therefore cannot be used to tune alert thresholds on minor-flood events, and we say explicitly where this limits the analysis.

## 3.5 Splits

For the primary, *temporal* protocol the first six water years (1 October 2016 to 30 September 2022) form the training set, water year 2023 forms the validation set (early stopping, hyper-parameter selection, conformal calibration), and the final three water years (1 October 2023 to 30 September 2026) form the test set. All normalisation statistics (per-gauge median and scale), the topology-free physics prior and every fitted routing coefficient are computed from the training years only. For the *leave-one-region-out* protocol (Section 5.3) the models are trained on the gauges of three regions and tested on the test years of the fourth region.

## 3.6 Ethical and reproducibility note

All data are public-domain government data, and no personal data are involved. All steps, from the station inventory query to the figures, are scripted (see Data and code availability), and the retrieval date (4 October 2026) is recorded because provisional USGS data are subject to later revision.


# 4. The PG-FaultTwin framework

## 4.1 Overview and twin architecture

PG-FaultTwin is organised in four layers that mirror a deployable IoT system (Fig. 1). The **sensing layer** consists of the telemetered gauges. The **edge layer** runs on a gateway at the regional hub and performs causal quality control (QC), gap bookkeeping and, optionally, the forecasting network itself. The **twin layer** maintains a synchronised virtual state of the river network, namely the current stage estimate at every gauge, the confidence in each sensor, and the 24-hour stage trajectory with calibrated intervals. The **decision layer** converts trajectories into rise rates, threshold-crossing probabilities and alerts, using the cost–loss trade-off chosen by the agency. The twin is *synchronised* in the sense that every update ingests the newest readings and re-estimates the state; it is *queryable* because users can request the trajectory, the rise rate or the time to a given threshold at any gauge; and it supports *what-if* analysis because the same network can be driven with hypothetical upstream hydrographs (Section 6.9).

Let there be S gauges, indexed by s, with hourly stage H_s(t). Let t denote the forecast origin (the current hour). The twin forecasts, for every gauge and every horizon h = 1,…,24 h, the future stage Ĥ_s(t+h), a lower and an upper bound, and from these the derived quantities in Section 4.6.

![Figure 1. Architecture of PG-FaultTwin. Sensing layer (gauges), edge layer (causal QC gate, mask bookkeeping, routing prior and GRU residual), twin layer (state, 24-h trajectories with conformal intervals, what-if queries) and decision layer (rise rate, time to threshold, alert margin); the topology-free prior, the fault curriculum and the verification suite are the supporting elements.](figures/fig1_architecture.png)

## 4.2 Normalisation and topology-free physics prior

**Normalisation.** For each gauge, we compute from the training years the median m_s and the scale σ_s, the standard deviation of the 24-hour stage change. The normalised stage is z_s(t) = (H_s(t) − m_s)/σ_s. This scale reflects how much the stage typically moves over the forecast window at that site, so a given normalised error corresponds to a comparable fraction of the local dynamics, whether the gauge is on a flashy headwater creek or a large lowland river. All learning takes place in normalised units, and results are reported both in normalised units and in feet.

**Directional lag structure.** A flood wave leaves a signature in gauges along its path: the rise at an upstream station precedes the rise downstream by a travel time. We extract this signature from the telemetry alone. For each gauge we build an event-scale signal x_s(t), the 3-hour difference of log(1 + H_s − min H_s) smoothed over three hours, which suppresses seasonal baseflow trends and emphasises storm response. For every ordered pair (i, j) of gauges *within the same region* we compute the lagged cross-correlation ρ_ij(τ) = corr(x_i(t), x_j(t+τ)) for τ = 0,…,48 h over the training years. The forward peak ρ*_ij = max_{τ≥1} ρ_ij(τ), attained at the lag τ*_ij, is the strength and lead time of the influence of i on j. The *directionality* is d_ij = ρ*_ij − ρ*_ji, the excess of the forward peak over the reverse peak. A gauge i is a candidate upstream neighbour of j if d_ij > 0 and ρ*_ij > 0.15. Candidates are ranked by ρ*_ij (1 + d_ij) and the top K = 4 are retained. No geographic information is used. The result is a sparse, directed, data-derived graph in which each edge carries a physically meaningful travel time τ*_ij.

**Plausibility check.** Because the topology is estimated, it is essential to verify that the edges are plausible. The distance between connected gauges, divided by τ*, gives an apparent celerity (km/h), which can be compared with the range of flood-wave celerities in natural channels, typically of order one to a few metres per second (about 4 to 11 km/h). We report the distribution of these quantities in Section 6.1, and in Section 6.9 we test whether the trained network's sensitivity to each neighbour is consistent with ρ*.

**Linear multi-lag routing prior.** For each gauge s with channel set C_s (the gauge itself and its up to four neighbours), we define the feature vector φ_s(t) containing, for every channel c ∈ C_s, the current normalised level z_c(t) and the lagged differences z_c(t−ℓ) − z_c(t) for ℓ ∈ {1, 2, 3, 4, 6, 9, 12, 18, 24, 30} h. With five channels this gives 55 features. A ridge regression, fitted separately for every gauge and every horizon h on the training years, predicts the stage change Δ_s(t,h) = z_s(t+h) − z_s(t):

Δ̂^route_s(t,h) = w_{s,h}ᵀ φ_s(t) + b_{s,h}.

This is a multi-lag linear transfer function from upstream and local stage history to future stage change, which includes the Muskingum routing equation (a linear combination of lagged inflow and outflow) as a special case (Cunge, 1969). We refer to it as the *routing prior*. It is fast to fit (seconds per network), has a closed-form solution, and is easily audited. When trained with the own-gauge channel only, it becomes an autoregressive baseline (AR); with the physics-selected neighbours it is the TF baseline of Section 5.2.

## 4.3 The neural twin

The routing prior is linear and therefore cannot represent the nonlinear stage–storage relationships, thresholds in overbank flow and the interaction of upstream pulses that characterise the largest events. PG-FaultTwin adds a compact recurrent network that learns a *residual* correction.

**Inputs.** For origin t, the network receives a window of L = 36 hours for each channel c ∈ C_s: the normalised values (missing hours filled by the last observation within the window; a leading gap is filled by the first observation in the window; a fully missing channel is set to zero), and a binary observation mask. Per time step, the sequence feature is [z_c(τ) − z_c(t)]_c, [mask_c(τ)]_c, [3(z_c(τ) − z_c(τ−1))]_c, i.e. level relative to the current level, observation flag and scaled first difference, giving 3C = 15 features per hour. The current anomaly of every channel, a channel validity flag (channels that do not exist for gauges with fewer than four neighbours are masked out), the age in hours since the last real own-gauge observation (normalised by 72) and the mask of the latest own reading form a static vector.

**Network.** A single-layer GRU (Cho et al., 2014) with 64 hidden units encodes the sequence. Its final hidden state is concatenated with the static vector and passed through two fully connected layers of 128 units with GELU activations. Three linear heads output (i) a residual r_h to the routing prior and (ii) two non-negative offsets, via softplus, defining the lower and upper quantile estimates. The prediction of the stage change is

Δ̂_s(t,h) = Δ̂^route_s(t,h) + r_h,  lower: Δ̂ − u_h,  upper: Δ̂ + v_h,

and the stage forecast is Ĥ_s(t+h) = m_s + σ_s [z_s(t)^obs + Δ̂_s(t,h)], where z_s(t)^obs is the last observed normalised level of the gauge's own sensor. The output layer of the residual head is initialised to zero so that training begins from the routing prior. The model has 51,208 trainable parameters.

**Loss.** The point head is trained with the Huber loss (δ = 1 in normalised units) (Huber, 1964), which is robust to the occasional extreme rise, and the interval heads are trained with the pinball loss at quantile levels 0.1 and 0.9 (Koenker and Bassett, 1978):

ℒ = Huber(Δ̂, Δ) + 0.5 [ρ_{0.1}(Δ − Δ̂_lo) + ρ_{0.9}(Δ − Δ̂_hi)].

The target Δ is the true stage at t+h relative to the *input* current level, which means that when the own sensor is stale or missing, the model must learn to correct for the staleness. The loss is averaged over horizons.

**Event-focused sampling.** More than 99% of training origins are quiescent. To prevent the network from learning only the null case we draw training origins with importance sampling: an origin whose maximum absolute stage change at 6, 12 or 24 h exceeds one σ_s receives a weight of 7 relative to a weight of 1 for others. The resulting model is calibrated on an *unweighted* validation year (Section 4.5), which removes the effect of the sampling on the stated coverage.

**Optimisation.** AdamW (Kingma and Ba, 2015; Loshchilov and Hutter, 2019) with weight decay 10⁻⁴, a one-cycle learning-rate schedule with maximum 2 × 10⁻³ (Smith and Topin, 2019), batch size 1,024, gradient-norm clipping at 2.0, eight epochs of 200,000 sampled origins. The checkpoint with the lowest normalised mean absolute error on a 60,000-origin validation subsample is retained. One global model is trained for all gauges and regions; it has no gauge-specific parameters, and the routing prior carries the site specificity.

## 4.4 Fault-tolerant training

The sensor-fault curriculum is applied on the fly to each training window. For every sample, independently for each channel, we apply a combination of (a) i.i.d. hourly dropout at a rate drawn uniformly from [0, 0.5] per sample; (b) with probability 0.4, a block outage of random start and a length of 2 to 26 hours; (c) with probability 0.25, a *stale tail*, in which the most recent 0 to 24 hours are lost (this simulates packet delays and a sensor that has just gone offline); (d) with probability 0.06, the complete loss of the channel. Dropped values are replaced as described in Section 4.3, the mask is set accordingly, the routing prior is recomputed from the corrupted window and the regression target is kept at the true future stage. The network therefore learns to use whichever sensors are alive and to treat the mask as informative: a stale upstream gauge should be trusted less than a live one.

**Edge QC gate.** Dropouts are visible; spikes and stuck readings are not. A causal gate runs at the gateway before the model. It maintains for each sensor the last accepted reading and its age. A reading is rejected as a spike if it differs from the last accepted reading by more than 1.5 (1 + age/2) times the largest hourly stage change that gauge exhibited during the training years, allowing the permitted change to grow as the sensor stays out of contact. A reading is rejected as stuck if it repeats the previous reading exactly for at least twelve consecutive hours *while at least half of the sensor's neighbours show activity* (a change larger than 0.02 σ per hour); the neighbour condition prevents the flagging of legitimately steady low-flow conditions. Rejected readings are set to missing, so the network sees them as dropouts. The gate uses no learned parameters and makes no use of future data.

## 4.5 Conformal calibration and adaptive conformal inference

**Split CQR.** For each horizon h, we compute on the validation year (unweighted, natural distribution) the conformity score E = max(Δ̂_lo − Δ, Δ − Δ̂_hi), the signed distance by which the observation lies outside the quantile band (Romano et al., 2019). The correction q̂_h is the ⌈(n+1)(1−α)⌉/n empirical quantile of E. The conformalised interval is [Δ̂_lo − q̂_h, Δ̂_hi + q̂_h] (Angelopoulos and Bates, 2023; Lei et al., 2018). Under exchangeability between calibration and test origins, the coverage is at least 1 − α. We use α = 0.1 for the main intervals.

**State-conditional (Mondrian) calibration.** Marginal coverage can hide poor coverage exactly when it matters, in rising limbs. We therefore also calibrate separately for two states defined from information available at the origin: *rising* (the last three-hour stage change of the gauge exceeds 0.25σ) and *not rising*.

**Adaptive conformal inference.** Because stage is non-stationary, we run ACI (Gibbs and Candès, 2021) online on the test stream for each gauge and horizon: the miscoverage level α_t is updated by α_{t+1} = α_t + γ (α − 1[miss_t]) with γ = 0.005, using rolling scores from the previous 2,000 origins, and each label is revealed only h hours after the forecast is issued, which respects the information structure of real operations (Zaffran et al., 2022).

## 4.6 From trajectories to warnings

**Gradient.** The rise rate over the next k hours is ĝ_k = (Ĥ(t+k) − H(t))/k, in feet per hour; the predicted *time to threshold* θ is the smallest h such that Ĥ(t+h) ≥ θ_s, where θ_s is the flood stage of the gauge.

**Alert rule.** An alert is issued at origin t if the gauge is currently below its threshold and the maximum over the next H_w hours of the forecast stage plus a safety margin of δ σ_s exceeds the threshold. The margin δ is the single tuning parameter that trades detection against false alarms. With conformal intervals the rule can alternatively use the calibrated upper bound at miscoverage α_alert, which gives an alert interpretable as "there is at least a (1 − α_alert)-level chance, under the calibration assumptions, that the stage may reach the threshold".

**Value-based selection.** Given a cost–loss ratio r = C/L (the cost of acting divided by the loss avoided by acting) (Murphy, 1977; Richardson, 2000), the optimal alert margin differs between users. A relief agency that can pre-position resources cheaply (small r) wants a low threshold; an agency for whom each alert implies a costly evacuation (large r) wants a high one. We therefore evaluate relative economic value as a function of r, and select the margin on the validation year when a single value is needed (Section 5.4).

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


# 5. Experimental protocol

## 5.1 Forecast origins and targets

Forecasts are issued every second hour of the test period (every third hour for the validation year), for each gauge, at horizons h = 1,…,24 h. An origin is scored only if the true stage is available (after the short-gap interpolation of Section 3.3) at all 24 horizons and at the origin. This yields about 1.33 million scored test origins across the 99 gauges in the temporal protocol. The headline accuracy target is the stage Ĥ(t+h) in feet; the gradient target is the average rise rate over the next 3, 6 and 12 hours. Because stage is quiet for most of the time, we also report accuracy on the *active subset*, the origins at which the true stage changes by more than one σ_s within 6, 12 or 24 hours. The proportion of such origins is reported with the results.

## 5.2 Baselines

We compare PG-FaultTwin with the following seven methods, from simple to strong.

1. **Persistence.** The stage remains at its last observed value. This is the standard no-skill benchmark in stage forecasting and is surprisingly hard to beat in mean absolute error.
2. **AR.** The routing regression of Section 4.2 using only the own-gauge channel (a linear autoregressive model on lagged stage differences).
3. **TF (transfer function).** The routing regression with the physics-selected upstream neighbours. This is our linear physics baseline, equivalent to a multi-lag ARX model.
4. **LightGBM.** Gradient-boosted trees (Ke et al., 2017) on the same 55 routing features plus the window maximum, minimum and standard deviation of the own gauge and the age of the own reading. Separate models are trained for h ∈ {1, 3, 6, 12, 18, 24} h and the other horizons are obtained by linear interpolation of the predicted stage changes. The training set is 350,000 origins drawn with the same event-focused weights as the neural models; the validation year provides early stopping (600 trees at most, learning rate 0.05, 63 leaves, Huber objective).
5. **Vanilla GRU.** The GRU network of Section 4.3 using the four geographically nearest gauges, no routing prior, no fault curriculum and uniform sampling.
6. The **neural ablations** of Table 3.
7. **Three-seed ensemble of the vanilla GRU**, used for the uncertainty comparison in Section 6.6.

The same hourly data, the same normalisation, the same origins and the same input windows (including the observation mask) are used for all methods; the baselines therefore have the benefit of the mask channel where applicable. Hyper-parameters were chosen on the validation year only, and the test years were not used for any tuning decision. Neural models are trained with three random seeds for the full model and the vanilla GRU, and with one seed for each ablation; we report means and standard deviations over seeds and mark single-seed results as such.

## 5.3 Generalisation protocols

**Temporal.** Train on water years 2017–2022 at all four regions and test on water years 2024–2026.

**Leave-one-region-out (LORO).** For each region in turn, the gauges of the other three regions provide the training data, and the model is tested on the test years of the held-out region, which contributes no training data. The topology-free prior is still computed within the held-out region from its training-period telemetry, because a deployed system would have this telemetry at commissioning; however, no region-specific *model* fitting takes place: the routing prior for held-out gauges is a single pooled ridge regression fitted over the training regions and shared by all gauges, and the neural weights are those trained on the other regions. This tests whether a model can be commissioned in a new region with no local training of predictors.

## 5.4 Metrics

**Accuracy.** Mean absolute error (MAE) and root-mean-square error (RMSE) in feet at selected horizons; the normalised MAE (nMAE) in units of σ_s, which can be averaged across gauges and horizons; the skill score SS = 1 − MAE/MAE_persistence; per-gauge Nash–Sutcliffe efficiency of the stage change forecast (Nash and Sutcliffe, 1970) and Kling–Gupta efficiency (Gupta et al., 2009). The relationship between MAE-based and MSE-based comparisons is explored in Section 6.2.

**Gradient.** The error and the correlation of the predicted and observed three-hour rise rates, and the classification of "rapidly rising" origins (true rise rate above 0.5 σ_s per hour).

**Alerting.** The alert rule of Section 4.6 is evaluated for the minor-flood threshold (primary) and the action stage (secondary) using a forecast window of 24 h. At the origin-level, a *hit* is an alert followed by a threshold exceedance within the window, a *false alarm* is an alert not followed by an exceedance, and a *miss* is an exceedance without an alert; only origins below the threshold are scored. We report the average precision (AP; area under the precision–recall curve), which is threshold-free and appropriate for rare events (Davis and Goadrich, 2006; Saito and Rehmsmeier, 2015), and, at the margin that maximises the critical success index (CSI) (Schaefer, 1990), the probability of detection (POD), the false-alarm ratio (FAR) and CSI. Because choosing the margin on the test data would be optimistic, the same selection rule is applied to all methods, so that comparisons are fair although absolute POD/FAR values are slightly optimistic; for the twin with conformal bounds we additionally report operating points selected on the validation year.

**Event-level verification.** An *event* is an up-crossing of the threshold preceded by 48 hours below it (Section 3.4). For each event, the *lead time* is the number of hours between the earliest origin from which an alert is continuously active and the crossing, up to 24 h (zero if there was no alert at the last origin before the crossing). An event is *detected* if an alert is active at the origin immediately before the crossing. A **false-alarm episode** is a maximal run of consecutive alerts that is not followed by a threshold crossing within 24 h of its start; we report the rate per gauge-year. Because origins are two hours apart, lead times have a resolution of two hours.

**Economic value.** Relative economic value V(r) in the cost–loss model, computed from the contingency table at the optimal alert margin for each cost–loss ratio r, normalised so that V = 0 corresponds to the best of "always alert" and "never alert" and V = 1 to perfect information (Richardson, 2000; Wilks, 2001).

**Calibration.** For intervals: empirical coverage by horizon, mean interval width in feet, and coverage in strata (rising versus not rising; high versus low stage), and for the online ACI: the long-run coverage and its deviation from the nominal level.

**Statistical inference.** Differences between methods are tested on *per-gauge* summary scores with the two-sided Wilcoxon signed-rank test (Wilcoxon, 1945), with Benjamini–Hochberg control of the false-discovery rate across the family of comparisons (Benjamini and Hochberg, 1995). For quantities pooled over origins we give 95% bootstrap intervals computed by resampling gauges with replacement (Efron, 1979); where autocorrelation matters we use the stationary bootstrap (Politis and Romano, 1994) and the Diebold–Mariano test with the Harvey–Leybourne–Newbold small-sample correction (Diebold and Mariano, 1995; Harvey et al., 1997) on squared-error loss differentials. Where only 71 minor-flood events are available, event-level estimates carry wide intervals, and we say so.

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

Experiments use Python 3.11, PyTorch (Paszke et al., 2019), scikit-learn (Pedregosa et al., 2011) and LightGBM (Ke et al., 2017) on a single 4-core CPU with 15 GB of memory and no GPU. The complete pipeline (retrieval of 45 million readings, topology-free prior, all trainings and all evaluation) is scripted and takes a few hours on this hardware.


# 6. Results

All results are for the final configuration of Section 4 and the test years (water years 2024–2026) unless stated otherwise. Neural results for the full twin and the vanilla GRU are means ± standard deviations over three seeds; other methods are deterministic or single-seed as indicated. There are 1,278,451 scored test origins in the temporal protocol; 10.9% of them are *active* (the true stage changes by more than one σ_s within 6, 12 or 24 h). Of the origins below the minor-flood stage, 807 are followed by an exceedance within 24 h (against 2,904 for the action stage), which gives a sense of how rare the target is: about one origin in 1,100.

## 6.1 The topology-free prior recovers a plausible, if resolution-limited, coupling structure

The cross-correlation analysis produced 321 directed edges, an average of 3.2 upstream neighbours per gauge. Seventy-three gauges obtained the full complement of four neighbours, and nine gauges, usually isolated headwater or main-stem stations without a clearly leading partner, obtained none and fall back on the own-gauge routing model (Fig. 2). The median edge connects gauges 23 km apart, has a forward peak correlation of 0.43 and a travel time of 2 h (inter-quartile range 1–3 h); 37% of edges have ρ* > 0.5. The directionality statistic d is positive for every selected edge by construction, with a median of 0.13, i.e. the forward peak correlation exceeds the reverse peak by 0.13 on a typical edge, and is largest in the Mid-Atlantic (0.19) and Ohio Valley (0.12), where tributary–main-stem structure is clearest, and smallest in the Upper Midwest (0.07) (Table 4).

![Figure 2. Gauge networks (colour = region; large dots have an official NWS flood stage) and the edges inferred from telemetry alone (arrow width proportional to the forward peak correlation ρ*).](figures/fig2_map.png)

| Region | Edges | Median τ* (h) | Median ρ* | Median distance (km) | Median apparent celerity (km/h) | Median directionality d |
|---|---|---|---|---|---|---|
| Mid-Atlantic | 98 | 2 | 0.55 | 21.1 | 6.2 | 0.19 |
| Ohio Valley | 74 | 2 | 0.51 | 19.9 | 5.7 | 0.12 |
| Texas Hill Country | 60 | 1 | 0.33 | 25.9 | 13.0 | 0.10 |
| Upper Midwest | 89 | 2 | 0.32 | 29.0 | 12.8 | 0.07 |
| **All** | **321** | **2** | **0.43** | **23.3** | **9.8** | **0.13** |

*Table 4. Summary of the inferred upstream neighbourhoods.*

The physical plausibility check gives a mixed but informative picture (Fig. 3). In the two Appalachian-influenced regions, where gauges are well coupled, the median apparent celerities (6.2 and 5.7 km/h, i.e. 1.7 and 1.6 m/s) lie inside the range expected for flood waves in natural channels. In the Texas Hill Country and the Upper Midwest the apparent celerities are about twice as large (12.8–13.0 km/h). Two features of the data explain much of this and also bound what the prior can claim. First, 44.5% of the edges sit at the smallest resolvable lag of one hour: the gauges are close enough that the true travel time is often below the one-hour resolution of our time series, so the quotient of distance and lag is a *lower* bound on celerity for these edges when the true lag is shorter than one hour and an overestimate when the lag is longer, but in either case it is quantised. Second, straight-line distance underestimates the length of a sinuous channel, which inflates celerities. Overall, 28% of the edges have celerities in the classical 3.6–11 km/h range and 66% fall within a factor of two on either side of it (1.8–22 km/h). The correlation between distance and inferred lag is weak (Pearson r = 0.09), which confirms that the prior should be read as a data-derived *coupling structure* that identifies which gauges carry predictive information about which, rather than as a reconstruction of channel hydraulics. Its value is established by the predictive experiments below.

![Figure 3. (a) Distribution of inferred travel times by region; (b) travel time against straight-line distance, with reference celerities of 3 and 11 km/h; (c) distribution of apparent celerities with the classical natural-channel range (3.6–11 km/h) shaded.](figures/fig3_prior.png)

## 6.2 Forecast accuracy

Table 5 and Fig. 4 compare the methods on the test years. Several patterns stand out.

| Method | nMAE (all) | nMAE quiet | nMAE active | MAE h=6 (ft) | MAE h=24 (ft) | RMSE h=24 (ft) | nRMSE active |
|---|---|---|---|---|---|---|---|
| Persistence | 0.234 | 0.107 | 1.273 | 0.087 | 0.225 | 0.716 | 2.142 |
| AR (own gauge) | 0.210 | 0.107 | 1.048 | 0.069 | 0.208 | 0.663 | 1.934 |
| TF (routing prior) | 0.198 | 0.106 | 0.955 | 0.064 | 0.196 | 0.630 | 1.827 |
| LightGBM | 0.237 | 0.153 | 0.921 | 0.070 | 0.247 | 0.645 | 1.827 |
| Vanilla GRU | 0.191 ± 0.001 | 0.098 ± 0.001 | 0.950 ± 0.007 | 0.063 ± 0.001 | 0.186 ± 0.001 | 0.622 ± 0.004 | 1.852 ± 0.009 |
| **PG-FaultTwin** | 0.185 ± 0.000 | 0.100 ± 0.001 | 0.876 ± 0.001 | 0.062 ± 0.000 | 0.177 ± 0.001 | 0.590 ± 0.001 | 1.756 ± 0.004 |

*Table 5. Forecast accuracy (temporal protocol, test years). nMAE: mean absolute error in units of the gauge-specific scale σ_s, averaged over the 24 horizons; "quiet" and "active" are the complementary subsets defined in Section 5.1. Mean ± s.d. over three seeds for the neural models.*

**The linear routing prior is a strong baseline and beats persistence under active conditions.** Persistence is hard to beat in the average over all origins because stage is quiet most of the time: its nMAE of 0.234 is dominated by the 89% quiet origins, where it scores 0.107. The own-gauge AR model (0.210) improves on it modestly, and the transfer-function (TF) model, which adds the physics-selected neighbours, reaches 0.198 and reduces the active-condition error from 1.273 (persistence) to 0.955. The physics-selected neighbours are therefore informative in themselves: moving from AR to TF reduces the active error by 9% (1.048 → 0.955) without any nonlinearity.

**Tree-based learning does not help in quiet conditions.** LightGBM obtains a good active error (0.921, the best among the non-neural baselines) but a quiet-state error of 0.153, 43% worse than persistence. Regression trees fitted to a heavy-tailed target (the 24-h change) with a Huber loss produce piecewise-constant outputs that jitter around zero in conditions where the best prediction is "nothing changes". Its overall nMAE (0.237) is therefore worse than persistence's. This illustrates why the quiet/active decomposition is needed: a single pooled score would hide the fact that LightGBM is the best non-neural model for floods and the worst for ordinary days.

**The recurrent residual model improves on all baselines where it matters.** PG-FaultTwin has the lowest overall nMAE (0.185 ± 0.000), the lowest active error (0.876 ± 0.001) and the lowest RMSE at 24 h (0.590 ft against 0.622 ft for the vanilla GRU and 0.630 ft for TF), while matching the best quiet-state error (0.100 against 0.098 for the vanilla GRU). Relative to persistence the active-condition error falls by 31% (1.273 → 0.876), and the 24-h MAE by 21% (0.225 → 0.177 ft). Relative to the vanilla GRU, which has identical capacity apart from the proposed elements, the active error falls by 8% (0.950 → 0.876) and the 24-h RMSE by 5%. The paired comparison across gauges (Table 6) shows that the improvements are broad rather than driven by a few gauges, although their size varies by comparator. The full twin has a lower active-condition nMAE than persistence, AR and TF at all 99 gauges, than the vanilla GRU at 78% of gauges (median difference −0.047σ) and than LightGBM at 65% of gauges (median difference −0.021σ); all BH-adjusted p-values are below 5 × 10⁻⁴. For the overall nMAE the advantage is large against persistence, AR, TF and LightGBM (88–100% of gauges) but marginal against the vanilla GRU (58% of gauges, median difference −0.002σ, adjusted p = 0.003), which is consistent with the tie in quiet conditions: the proposed elements pay off in active conditions rather than on ordinary days.

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

*Table 6. Paired per-gauge comparisons of PG-FaultTwin with each baseline (Wilcoxon signed-rank test on the per-gauge mean nMAE, seed-averaged; BH-adjusted across the family). Negative median differences favour PG-FaultTwin.*

The improvement grows with the lead time (Fig. 4). At one hour ahead all models are within a few hundredths of σ_s of each other, as expected for a variable with strong persistence; at 24 h the active-condition nMAE is 2.04 for persistence and 1.47 for the full twin, a reduction of 28%.

![Figure 4. Normalised MAE against lead time for (a) all test origins and (b) the active subset. The advantage of the learned, network-aware models widens with lead time.](figures/fig5_accuracy.png)

**A cautionary episode: a stuck sensor can corrupt a learned twin while a physics-style model survives.** Before the dead-sensor screen of Section 3.2 was introduced, an early version of the experiments included a gauge in the Guadalupe basin that reported a constant 16.2 ft from 2023 onward after a median of −0.5 ft in 2016–2022. The neural twin, which used the (now abnormal) absolute level of this neighbour as an input, produced a persistent 2.9σ bias at the downstream gauge in the test years, whereas the routing prior, which uses only lagged *differences*, was unaffected and was accurate at the same gauge. We removed such sensors with the screen and re-ran every experiment from scratch, but the episode is a real-world example of the failure class this paper targets and motivates two design choices: the routing prior deliberately uses no absolute levels, and the fault experiments of Section 6.8 include bias faults.

## 6.3 Gradient skill

The rise rate is the quantity a duty officer watches. Table 7 evaluates the predicted three-hour rise rate derived from the stage trajectory. All learned models achieve a 3-h rise-rate MAE of about 0.011 ft/h overall, a 33% reduction from persistence (0.016 ft/h), but the differences emerge in active conditions, where the error of the full twin (0.0430 ft/h) is 45% below persistence (0.0783), 14% below AR (0.0502), 8% below the vanilla GRU (0.0466) and 3% below TF (0.0444). The correlation between predicted and observed rise rates is 0.714 ± 0.011 for the full twin, essentially tied with TF (0.715) and higher than the vanilla GRU (0.669 ± 0.008), AR (0.660) and LightGBM (0.686). The advantage of the full twin is clearer at the six-hour horizon (0.687, against 0.657 for TF, 0.639 for LightGBM and 0.612 for the vanilla GRU) and in the detection of *rapidly rising* origins, for which the F1 score is 0.532 against 0.509 for TF and 0.478 for the vanilla GRU. LightGBM's F1 (0.530) is comparable, again showing that tree-based learners are competitive at identifying rises even when they are poor in quiet conditions.

| Method | 3-h rise-rate MAE (ft/h) | 3-h rise-rate MAE active | corr. 3-h | corr. 6-h | rapid-rise F1 |
|---|---|---|---|---|---|
| Persistence | 0.0162 | 0.0783 | -0.007 | -0.006 | 0.000 |
| AR (own gauge) | 0.0115 | 0.0502 | 0.660 | 0.571 | 0.402 |
| TF (routing prior) | 0.0110 | 0.0444 | 0.715 | 0.657 | 0.509 |
| LightGBM | 0.0114 | 0.0447 | 0.686 | 0.639 | 0.530 |
| Vanilla GRU | 0.0109 ± 0.0001 | 0.0466 ± 0.0006 | 0.669 ± 0.008 | 0.612 ± 0.010 | 0.478 ± 0.009 |
| **PG-FaultTwin** | 0.0109 ± 0.0001 | 0.0430 ± 0.0003 | 0.714 ± 0.011 | 0.687 ± 0.005 | 0.532 ± 0.001 |

*Table 7. Gradient (rise-rate) skill. Rapid rise: true 3-h rise exceeding 0.5σ_s (26,140 test origins).*

## 6.4 Flood alert verification

The alert rule of Section 4.6 turns each stage trajectory into a warning decision; Table 8 reports the verification for the minor-flood stage (primary) and the action stage. We start with the threshold-free average precision, which is the area under the precision–recall curve and does not depend on the margin δ. The full twin attains 0.274 ± 0.002 for the minor stage against 0.261 for TF, 0.195 for LightGBM, 0.187 for AR, 0.170 ± 0.007 for the vanilla GRU and 0.086 for persistence; for the action stage the corresponding figures are 0.354 ± 0.002, 0.338, 0.286, 0.242, 0.251 ± 0.004 and 0.119. The physics-selected neighbours and the routing prior are thus the main drivers of alert skill: TF, a linear model, outperforms the nonlinear vanilla GRU by 53% in minor-stage AP, and the full twin adds a further 5% (minor) and 5% (action) over TF.

| Method | AP | POD | FAR | CSI | events detected / total | mean lead (h) | mean lead of detected (h) | false-alarm episodes per gauge-year |
|---|---|---|---|---|---|---|---|---|
| Persistence | 0.086 | 0.180 | 0.838 | 0.093 | 56 / 68 | 3.07 | 3.73 | 0.58 |
| AR (own gauge) | 0.187 | 0.253 | 0.594 | 0.184 | 61 / 68 | 4.28 | 4.77 | 0.74 |
| TF (routing prior) | 0.261 | 0.322 | 0.515 | 0.240 | 58 / 68 | 5.94 | 6.97 | 0.62 |
| LightGBM | 0.195 | 0.183 | 0.553 | 0.149 | 53 / 68 | 3.07 | 3.94 | 0.32 |
| Vanilla GRU | 0.170 ± 0.007 | 0.174 ± 0.009 | 0.618 ± 0.038 | 0.135 ± 0.002 | 54 / 68 | 3.07 ± 0.19 | 3.89 ± 0.09 | 0.34 ± 0.04 |
| **PG-FaultTwin** | 0.274 ± 0.002 | 0.294 ± 0.014 | 0.435 ± 0.033 | 0.239 ± 0.004 | 57 / 68 | 5.14 ± 0.29 | 6.17 ± 0.12 | 0.44 ± 0.06 |

*Table 8a. Verification of alerts for the NWS minor flood stage (68 verified events at 37 gauges; margin δ chosen to maximise CSI for each method; see text).*

| Method | AP | POD | FAR | CSI | events detected / total | mean lead (h) | mean lead of detected (h) | false-alarm episodes per gauge-year |
|---|---|---|---|---|---|---|---|---|
| Persistence | 0.119 | 0.258 | 0.809 | 0.123 | 190 / 245 | 5.02 | 6.47 | 1.91 |
| AR (own gauge) | 0.242 | 0.280 | 0.584 | 0.201 | 211 / 245 | 5.25 | 6.10 | 1.77 |
| TF (routing prior) | 0.338 | 0.321 | 0.404 | 0.263 | 213 / 245 | 6.16 | 7.08 | 1.28 |
| LightGBM | 0.286 | 0.261 | 0.550 | 0.198 | 192 / 245 | 5.00 | 6.38 | 1.31 |
| Vanilla GRU | 0.251 ± 0.004 | 0.246 ± 0.004 | 0.616 ± 0.005 | 0.176 ± 0.003 | 192 / 245 | 4.78 ± 0.13 | 6.09 ± 0.16 | 1.46 ± 0.02 |
| **PG-FaultTwin** | 0.354 ± 0.002 | 0.298 ± 0.001 | 0.351 ± 0.023 | 0.256 ± 0.003 | 209 / 245 | 5.74 ± 0.03 | 6.74 ± 0.02 | 1.02 ± 0.06 |

*Table 8b. Verification of alerts for the NWS action stage (245 verified events at 52 gauges).*

At each method's CSI-optimal margin, the full twin detects 57 of the 68 verifiable minor-flood events (84%) with a mean lead time of 5.1 h (6.2 h among detected events) and 0.44 false-alarm episodes per gauge-year, whereas TF detects 58 events at 5.9 h but with 0.62 false-alarm episodes, and the vanilla GRU detects 54 events with 3.1 h lead and 0.34 false-alarm episodes. For the action stage, the full twin has the lowest false-alarm ratio (0.351 against 0.404 for TF and 0.616 for the vanilla GRU) and the fewest false-alarm episodes (1.02 per gauge-year against 1.28 for TF and 1.46 for the vanilla GRU), at a detection rate of 209 of 245 events (85%) and a lead of 5.7 h. In other words, the full twin delivers detection and lead time comparable to the strong linear baseline, with about a quarter to a third fewer false alarms: 29% fewer episodes for minor floods and 20% fewer for the action stage.

Fig. 5 shows the precision–recall curves, the performance diagram and the lead-time distributions: the full twin's precision–recall curve lies above or on that of TF over most of the range and well above all other methods. Three caveats apply. First, the absolute skill is modest, with CSI values near 0.24–0.26 and POD near 0.3 at the CSI-optimal margin. This is not surprising: the information set consists solely of stage telemetry, no rainfall, so that rises driven by rainfall that has not yet reached the gauges cannot be anticipated, and for the rapidly responding headwater gauges a 24-h window contains events that start from a dry channel. Second, the margin δ was selected on the test period for every method, because the validation year contains only two minor-flood events and cannot support selection; this gives all methods the same optimistic advantage, and the threshold-free AP is not affected. Third, the numbers of events are small (68 minor-flood events), and differences of one or two events in the detection counts should not be over-interpreted; the gauge-level bootstrap in Section 6.5 quantifies this uncertainty.

![Figure 5. (a) Precision–recall curves; (b) performance diagram (POD against success ratio, markers at each method's CSI-optimal margin); (c) event lead times.](figures/fig6_alerts.png)

Fig. 10 (Section 6.11) shows one such alert for a real event, including its limits.


## 6.5 Economic value and statistical uncertainty of the alert skill

Relative economic value (Fig. 6) shows how the skill translates into decision value across user types. For users with a low cost–loss ratio (C/L ≤ 0.02), i.e. those for whom taking protective action is cheap relative to the loss avoided, PG-FaultTwin and TF are close (V = 0.459 and 0.463 at C/L = 0.005; 0.367 and 0.376 at 0.02) and both are well above the vanilla GRU (0.405 and 0.290) and persistence (0.351 and 0.245). As the cost–loss ratio rises, the full twin pulls ahead of TF: V = 0.205 against 0.177 at C/L = 0.3, 0.119 against 0.084 at 0.5, and 0.074 against 0.046 at 0.7. Averaged over C/L ≤ 0.3, the values are 0.324 (twin) and 0.326 (TF), against 0.228 (vanilla GRU), 0.247 (LightGBM), 0.270 (AR) and 0.170 (persistence). The practical reading is that the twin is at least as valuable as the linear baseline for every user type, and more valuable to the users who can least afford false alarms, which are the users whose decisions (for example evacuations) are costly.

![Figure 6. Relative economic value V against the cost–loss ratio C/L (maximised over alert margins for each ratio).](figures/fig7_value.png)

The number of independent flood events is the limiting factor for any claim about alerts, so we quantify it with a gauge-level bootstrap (200 resamples of gauges with replacement; for the neural models the three seeds are averaged within each resample). For the minor flood stage, the AP of the full twin is 0.277 (95% interval 0.216–0.346), against 0.262 (0.201–0.321) for TF, 0.199 (0.150–0.250) for LightGBM, 0.174 (0.129–0.217) for the vanilla GRU and 0.091 (0.056–0.128) for persistence. The *paired* differences, which are far more precise than the marginal intervals because the same gauges and events are resampled, are +0.015 (0.002 to 0.028) over TF, +0.079 (0.037 to 0.114) over LightGBM, +0.104 (0.057 to 0.163) over the vanilla GRU and +0.186 (0.128 to 0.258) over persistence. For the action stage, the corresponding paired differences are +0.016 (0.006 to 0.030), +0.068 (0.039 to 0.098), +0.101 (0.064 to 0.141) and +0.229 (0.165 to 0.294). The advantage over TF is therefore real but small (about 5% in relative terms), while the advantages over the learned baselines are large.

## 6.6 Ablation: what makes the difference

Table 9 removes one element at a time from the full twin (single seed per variant, so differences of the order of the seed-to-seed standard deviation, 0.001–0.011 depending on the metric, should be disregarded).

| Variant | nMAE | nMAE active | grad. corr. 3-h | AP (minor) | AP (action) | CSI (action) |
|---|---|---|---|---|---|---|
| Full PG-FaultTwin | 0.185 | 0.875 | 0.725 | 0.276 | 0.355 | 0.258 |
| – routing prior | 0.189 | 0.936 | 0.616 | 0.212 | 0.273 | 0.190 |
| – fault curriculum | 0.184 | 0.873 | 0.735 | 0.280 | 0.354 | 0.255 |
| geographic instead of physics-selected neighbours | 0.190 | 0.891 | 0.712 | 0.231 | 0.321 | 0.240 |
| own gauge only (no neighbours) | 0.195 | 0.954 | 0.679 | 0.189 | 0.252 | 0.193 |
| + event-focused sampling (weight 7) | 0.213 | 0.872 | 0.673 | 0.275 | 0.362 | 0.267 |
| Vanilla GRU (all elements removed) | 0.191 | 0.957 | 0.673 | 0.177 | 0.250 | 0.176 |

*Table 9. Ablation of PG-FaultTwin (temporal protocol, seed 0; the first row and the last row repeat seed 0 of the main comparison).*

**The routing prior is the most valuable element.** Removing it (a GRU with physics-selected neighbours and the fault curriculum, but no linear base) raises the active nMAE from 0.875 to 0.936, lowers the rise-rate correlation from 0.725 to 0.616 and drops the minor-stage AP from 0.276 to 0.212 (−23%). Residual learning on a linear base therefore delivers most of the benefit of the physics-guided design. It does so while also making the model robust: a GRU alone must learn the linear relationships from scratch and does so imperfectly.

**Physics-selected neighbours beat geographic neighbours.** Replacing the four neighbours selected by the lag analysis with the four geographically nearest gauges (with the routing prior recomputed for them) leaves the overall error almost unchanged (0.190 against 0.185) but lowers the minor-stage AP from 0.276 to 0.231 and the action-stage AP from 0.355 to 0.321. Proximity is a weaker predictor of hydrological coupling than the directional lagged correlation, which presumably separates a gauge on the same river from a gauge on a neighbouring catchment. Using the own gauge only (no neighbours) is the weakest of the physics-based configurations (active nMAE 0.954; minor AP 0.189), confirming that upstream information is essential.

**The fault curriculum is free on clean data.** Removing it changes essentially nothing in the clean test years (nMAE 0.184, active 0.873, minor AP 0.280), so its benefit, established in Section 6.8, comes at no cost in nominal operation. The share of corrupted windows (35%) and the corruption intensities were fixed during a preliminary development phase on the validation year, in which a strongly corrupting curriculum was tried first; we did not run a separate sweep of the curriculum intensity, and a systematic study of it is left for future work.

**Event-focused sampling is a trade-off, not a free gain.** Over-sampling rising situations (weight 7 relative to quiet ones) leaves the active error unchanged (0.872) and raises the action-stage AP marginally (0.362 against 0.355) but increases the overall error from 0.185 to 0.213, because the model becomes biased towards predicting change on ordinary days (quiet nMAE 0.132 against 0.100). This is the reason why uniform sampling is part of the final configuration. A user whose only objective is detection of action-stage events could prefer the weighted variant; calibrated decisions, however, require unbiased quiet-state forecasts.

**All elements together.** The full model improves the vanilla GRU by 8% in active nMAE (0.957 → 0.875), 8% in rise-rate correlation (0.673 → 0.725), 56% in minor-stage AP (0.177 → 0.276) and 42% in action-stage CSI (0.176 → 0.258). The ablations show that these gains are due mainly to the routing prior and the physics-selected neighbours; the fault curriculum contributes robustness rather than nominal accuracy.


## 6.7 Spatial generalisation: leave-one-region-out

The leave-one-region-out (LORO) protocol asks whether a model trained in other regions can be commissioned in a new one without local training of predictors. Table 10 compares, region by region, the temporal protocol (all four regions seen in training; same test origins) with the LORO protocol (the region withheld); persistence is a protocol-independent reference.

| Region (action-stage positives) | Method | Protocol | nMAE | nMAE active | rise-rate corr. | AP (action) |
|---|---|---|---|---|---|---|
| Mid-Atlantic (766) | Persistence | temporal | 0.171 | 1.288 | -0.006 | 0.116 |
| Mid-Atlantic (766) | TF (routing prior) | temporal | 0.154 | 0.956 | 0.782 | 0.440 |
| Mid-Atlantic (766) | TF (routing prior) | LORO | 0.162 | 1.196 | 0.681 | 0.196 |
| Mid-Atlantic (766) | Vanilla GRU | temporal | 0.139 | 0.925 | 0.752 | 0.291 |
| Mid-Atlantic (766) | Vanilla GRU | LORO | 0.148 | 0.990 | 0.743 | 0.259 |
| Mid-Atlantic (766) | **PG-FaultTwin** | temporal | 0.139 | 0.859 | 0.794 | 0.445 |
| Mid-Atlantic (766) | **PG-FaultTwin** | LORO | 0.147 | 0.968 | 0.760 | 0.305 |
| Ohio Valley (539) | Persistence | temporal | 0.338 | 1.116 | -0.005 | 0.071 |
| Ohio Valley (539) | TF (routing prior) | temporal | 0.270 | 0.801 | 0.781 | 0.288 |
| Ohio Valley (539) | TF (routing prior) | LORO | 0.316 | 1.031 | 0.685 | 0.187 |
| Ohio Valley (539) | Vanilla GRU | temporal | 0.270 | 0.833 | 0.752 | 0.187 |
| Ohio Valley (539) | Vanilla GRU | LORO | 0.294 | 0.911 | 0.726 | 0.164 |
| Ohio Valley (539) | **PG-FaultTwin** | temporal | 0.257 | 0.760 | 0.805 | 0.292 |
| Ohio Valley (539) | **PG-FaultTwin** | LORO | 0.290 | 0.913 | 0.702 | 0.185 |
| Texas Hill Country (84) | Persistence | temporal | 0.156 | 1.744 | -0.004 | 0.021 |
| Texas Hill Country (84) | TF (routing prior) | temporal | 0.153 | 1.533 | 0.526 | 0.060 |
| Texas Hill Country (84) | TF (routing prior) | LORO | 0.166 | 1.676 | 0.488 | 0.051 |
| Texas Hill Country (84) | Vanilla GRU | temporal | 0.141 | 1.379 | 0.395 | 0.030 |
| Texas Hill Country (84) | Vanilla GRU | LORO | 0.147 | 1.417 | 0.393 | 0.032 |
| Texas Hill Country (84) | **PG-FaultTwin** | temporal | 0.144 | 1.381 | 0.497 | 0.045 |
| Texas Hill Country (84) | **PG-FaultTwin** | LORO | 0.190 | 1.463 | 0.264 | 0.032 |
| Upper Midwest (1515) | Persistence | temporal | 0.266 | 1.349 | -0.010 | 0.149 |
| Upper Midwest (1515) | TF (routing prior) | temporal | 0.214 | 1.010 | 0.681 | 0.330 |
| Upper Midwest (1515) | TF (routing prior) | LORO | 0.270 | 1.292 | 0.534 | 0.206 |
| Upper Midwest (1515) | Vanilla GRU | temporal | 0.213 | 1.010 | 0.631 | 0.273 |
| Upper Midwest (1515) | Vanilla GRU | LORO | 0.231 | 1.082 | 0.574 | 0.265 |
| Upper Midwest (1515) | **PG-FaultTwin** | temporal | 0.199 | 0.910 | 0.652 | 0.351 |
| Upper Midwest (1515) | **PG-FaultTwin** | LORO | 0.223 | 1.034 | 0.530 | 0.287 |

*Table 10. Temporal versus leave-one-region-out results by region (test years; neural temporal results are means of three seeds; LORO results are single-seed). Rise-rate corr.: correlation of predicted and observed 3-h rise rates. AP: average precision for action-stage alerts.*

Four findings emerge.

**Region matters more than method.** Predictability differs widely between regions. In the Texas Hill Country, only 4.7% of origins are active and there are 84 action-stage positives, against 19.2% and 539 in the Ohio Valley, and flash-flood rises begin within hours of rainfall that the telemetry cannot see; the persistence active error is 1.74σ there against 1.12σ in the Ohio Valley, and the best action-stage AP in the temporal protocol is 0.06 (TF) against 0.44 in the Mid-Atlantic. A pooled score therefore hides large regional differences, and the Texas result should be read as a statement about the information content of upstream stage alone in a flash-flood regime (Section 7.3).

**In the temporal protocol the full twin is best or tied in every region.** It has the lowest active nMAE in the Mid-Atlantic (0.859, against 0.925 for the vanilla GRU and 0.956 for TF), the Ohio Valley (0.760 against 0.833 and 0.801) and the Upper Midwest (0.910 against 1.010 for both), and ties with the vanilla GRU in Texas (1.381 against 1.379). It also has the best action-stage AP in the Mid-Atlantic (0.445), the Ohio Valley (0.292) and the Upper Midwest (0.351).

**Transfer to an unseen region costs accuracy but retains most of the gain over persistence.** Under LORO, the active nMAE of the full twin increases by 0.08–0.15σ relative to the temporal protocol (to 0.968, 0.913, 1.463 and 1.034 in the four regions), yet remains 25%, 18%, 16% and 23% below persistence. These models never saw a single training sample from the held-out region, and the routing prior was a pooled regression shared by all gauges, so the result shows that much of the learned structure (how a rise at a leading gauge propagates to a lagging one) is transferable across these regions.

**The advantage of the proposed twin over the baselines shrinks under transfer and reverses in Texas.** In the Mid-Atlantic and Upper Midwest the full twin remains best in action-stage AP (0.305 and 0.287, against 0.259 and 0.265 for the vanilla GRU and 0.196 and 0.206 for TF) and in active nMAE (0.968 and 1.034). In the Ohio Valley it ties with the vanilla GRU (active nMAE 0.913 against 0.911) and with TF in AP (0.185 against 0.187). In the Texas Hill Country, the full twin transfers worse than the vanilla GRU (active nMAE 1.463 against 1.417; rise-rate correlation 0.264 against 0.393). The most plausible explanation is that the physics-selected neighbourhoods and routing coefficients, which are well conditioned in the three regions with slower, better-coupled dynamics, are poorly conditioned in a flash-flood regime where the inferred lags sit at the one-hour resolution floor (Table 4) and where a model trained elsewhere imposes a coupling structure that does not apply. We conclude that the proposed twin can be commissioned in a new region of similar dynamics without local model training, but that regions with fundamentally different response times need local training of at least the routing prior and probably higher-resolution telemetry.

## 6.8 Robustness to sensor faults


We corrupt the inputs of the test years according to the regimes of Section 5.5 and evaluate on every twelfth hourly origin (about 730 origins per gauge-year; the clean-data row therefore differs slightly from the main comparison, which used every second hour). Tables 11 and 12 report the overall and active-state errors, Table 13 the alert skill and Fig. 7 summarises the main trends. The twin trained without the fault curriculum is included to isolate the contribution of the curriculum; "+QC" denotes the same model behind the edge quality-control gate of Section 4.4.

| Scenario | Persistence | TF | Vanilla GRU | GRU+QC | Twin w/o curriculum | **Twin** | **Twin+QC** |
|---|---|---|---|---|---|---|---|
| Clean | 0.235 | 0.202 | 0.193 | 0.193 | **0.186** | 0.188 | 0.188 |
| Dropout 10% | 0.245 | 0.216 | 0.210 | 0.210 | 0.203 | **0.201** | 0.201 |
| Dropout 30% | 0.261 | 0.242 | 0.240 | 0.241 | 0.233 | **0.226** | 0.226 |
| Dropout 50% | 0.278 | 0.266 | 0.270 | 0.270 | 0.259 | **0.252** | 0.252 |
| Dropout 70% | 0.294 | 0.290 | 0.302 | 0.303 | **0.285** | 0.287 | 0.286 |
| Stuck-at 10% | 0.253 | 0.226 | 0.215 | 0.216 | 0.212 | 0.213 | **0.210** |
| Stuck-at 30% | 0.294 | 0.280 | 0.264 | 0.266 | 0.268 | 0.268 | **0.263** |
| Spikes 1% | 0.360 | 1.005 | 0.371 | **0.253** | 0.960 | 0.958 | 0.339 |
| Spikes 3% | 0.605 | 2.390 | 0.661 | **0.356** | 2.264 | 2.270 | 0.611 |
| Bias b=0.5σ | 0.316 | 0.285 | 0.276 | 0.276 | **0.270** | 0.271 | 0.271 |
| Bias b=1.0σ | 0.376 | 0.347 | 0.333 | 0.333 | **0.329** | 0.331 | 0.331 |
| Latency 3 h | 0.274 | 0.236 | 0.228 | 0.228 | **0.219** | 0.220 | 0.220 |
| Latency 6 h | 0.307 | 0.269 | 0.261 | 0.261 | **0.252** | 0.253 | 0.253 |
| Latency 12 h | 0.366 | 0.333 | 0.321 | 0.321 | **0.314** | 0.314 | 0.314 |
| Combined | 0.549 | 1.257 | 0.543 | **0.417** | 1.194 | 1.173 | 0.592 |

*Table 11. Overall nMAE under injected sensor faults (best value per row in bold).*

| Scenario | Persistence | TF | Vanilla GRU | GRU+QC | Twin w/o curriculum | **Twin** | **Twin+QC** |
|---|---|---|---|---|---|---|---|
| Clean | 1.284 | 0.983 | 0.974 | 0.975 | **0.897** | 0.900 | 0.903 |
| Dropout 10% | 1.319 | 1.046 | 1.040 | 1.041 | 0.966 | **0.948** | 0.952 |
| Dropout 30% | 1.366 | 1.148 | 1.151 | 1.151 | 1.082 | **1.034** | 1.034 |
| Dropout 50% | 1.424 | 1.241 | 1.259 | 1.260 | 1.182 | **1.111** | 1.118 |
| Dropout 70% | 1.480 | 1.348 | 1.383 | 1.384 | 1.296 | 1.206 | **1.204** |
| Stuck-at 10% | 1.330 | 1.060 | 1.047 | 1.050 | 0.981 | 0.983 | **0.968** |
| Stuck-at 30% | 1.448 | 1.247 | 1.222 | 1.232 | 1.177 | 1.176 | **1.142** |
| Spikes 1% | 1.400 | 1.693 | 1.155 | 1.019 | 1.592 | 1.588 | **0.991** |
| Spikes 3% | 1.620 | 2.959 | 1.428 | **1.082** | 2.821 | 2.807 | 1.144 |
| Bias b=0.5σ | 1.312 | 1.021 | 1.012 | 1.013 | **0.939** | 0.941 | 0.943 |
| Bias b=1.0σ | 1.384 | 1.114 | 1.092 | 1.093 | **1.028** | 1.031 | 1.034 |
| Latency 3 h | 1.437 | 1.156 | 1.147 | 1.148 | 1.069 | **1.067** | 1.068 |
| Latency 6 h | 1.543 | 1.304 | 1.284 | 1.285 | 1.219 | **1.216** | 1.222 |
| Latency 12 h | 1.703 | 1.547 | 1.504 | 1.503 | 1.470 | 1.467 | **1.467** |
| Combined | 1.627 | 2.035 | 1.419 | **1.282** | 1.944 | 1.899 | 1.307 |

*Table 12. Active-state nMAE under injected sensor faults (best value per row in bold).*

| Scenario | Persistence | TF | Vanilla GRU | GRU+QC | Twin w/o curriculum | **Twin** | **Twin+QC** |
|---|---|---|---|---|---|---|---|
| Clean | 0.088 | 0.288 | 0.182 | 0.180 | **0.301** | 0.294 | 0.250 |
| Dropout 10% | 0.066 | 0.189 | 0.118 | 0.116 | 0.196 | **0.208** | 0.172 |
| Dropout 30% | 0.043 | 0.116 | 0.062 | 0.057 | 0.114 | **0.119** | 0.093 |
| Dropout 50% | 0.039 | 0.089 | 0.043 | 0.043 | 0.083 | **0.106** | 0.082 |
| Dropout 70% | 0.029 | 0.068 | 0.028 | 0.028 | 0.068 | **0.085** | 0.072 |
| Stuck-at 10% | 0.066 | 0.246 | 0.129 | 0.141 | **0.247** | 0.244 | 0.205 |
| Stuck-at 30% | 0.037 | **0.102** | 0.056 | 0.055 | 0.102 | 0.102 | 0.092 |
| Spikes 1% | 0.031 | 0.011 | 0.038 | **0.102** | 0.011 | 0.011 | 0.089 |
| Spikes 3% | 0.013 | 0.006 | 0.014 | **0.068** | 0.006 | 0.006 | 0.040 |
| Bias b=0.5σ | 0.091 | 0.291 | 0.187 | 0.185 | **0.302** | 0.298 | 0.255 |
| Bias b=1.0σ | 0.084 | 0.282 | 0.174 | 0.171 | **0.293** | 0.286 | 0.244 |
| Latency 3 h | 0.027 | 0.112 | 0.046 | 0.046 | 0.110 | **0.113** | 0.101 |
| Latency 6 h | 0.018 | **0.083** | 0.024 | 0.024 | 0.075 | 0.080 | 0.076 |
| Latency 12 h | 0.017 | **0.025** | 0.015 | 0.015 | 0.024 | 0.024 | 0.019 |
| Combined | 0.011 | 0.009 | 0.013 | 0.019 | 0.009 | 0.010 | **0.027** |

*Table 13. Average precision for minor-flood alerts under injected sensor faults (best value per row in bold; the numbers of positives are about 6 times smaller than in the main comparison because of the coarser origin grid, so absolute values are noisier than in Section 6.4).*

![Figure 7. Robustness to sensor faults (active-state nMAE): (a) block dropouts, (b) telemetry latency, (c) overall nMAE under spikes without and with the edge QC gate (bars capped at 2.5).](figures/fig9_faults.png)

**Missing data, stale data and bias: the twin is the most robust model, and its advantage persists up to heavy loss.** Under block dropouts, every model degrades, as it must, since the information content of the input falls. The full twin stays below persistence in active-state error at all loss levels, from 0.948 against 1.319 at 10% loss to 1.206 against 1.480 at 70% (−18% at the most severe level), and has the lowest active-state error of all models at every loss level. The vanilla GRU deteriorates faster than the linear TF baseline and at 50% loss is worse than TF (active nMAE 1.259 against 1.241); at 70% loss its overall nMAE (0.302) is higher than that of persistence (0.294), i.e. a network that has not seen missing data can be worse than doing nothing when much of the data are missing. The twin's overall nMAE at 70% loss (0.287) is still below persistence. Under telemetry delays of 3, 6 and 12 hours the twin's active error is lowest or tied (1.067, 1.216, 1.467 against 1.437, 1.543, 1.703 for persistence), but the margin over the vanilla GRU shrinks with the delay (−7.0%, −5.3%, −2.5%), as would be expected when the input has become uninformative about the short lead times. Calibration bias of 0.5σ or 1.0σ on 30% of the sensors, which no gate can detect without a reference, increases the twin's active error by 4.6% and 14.6% over its clean value (0.941 and 1.031 from 0.900), a degradation similar to that of the other learned models (the vanilla GRU: +3.9% and +12.1%) and comparable in relative terms to that of persistence (1.312 and 1.384 from 1.284), whose absolute active error remains far above the twin's.

**The fault curriculum is targeted at data loss, and it is only useful there.** Comparing the twin with and without the curriculum shows that the curriculum reduces the active error under dropout (1.082 → 1.034 at 30%, 1.182 → 1.111 at 50%, 1.296 → 1.206 at 70%; −4% to −7%) and raises the minor-flood AP at 50% loss from 0.083 to 0.106 (+28%; Table 13). For all other fault types (stuck, spikes, bias, latency) the two models are indistinguishable, which is a reasonable outcome, since the curriculum contains no such faults. The curriculum costs nothing under clean conditions (overall nMAE 0.188 against 0.186 in this subset and 0.185 against 0.184 in the main comparison). Taken together with the ablation of Section 6.6, the curriculum is thus a low-cost insurance against data loss. It is not a general defence against faulty data.

**Spikes are the dominant threat to the physics-guided models, and the edge gate is essential.** With 1% spikes, the overall nMAE of the full twin and of the TF baseline increase from 0.188 and 0.202 to 0.958 and 1.005, several times higher than persistence (0.360). The mechanism is clear: a linear routing model multiplies a spurious jump at an upstream or local gauge by its learned coefficients and extrapolates a flood that does not exist. The vanilla GRU, which saturates inputs through its non-linearities, is much less affected (0.371). The causal gate of Section 4.4 removes the problem for most of the spikes: the gated twin's nMAE returns to 0.339 (1% spikes) and 0.611 (3% spikes), and its active error to 0.991 and 1.144, better than persistence at 1% spikes (1.400) and 3% (1.620). The gate has precision of 0.55 and 0.80 and recall of 0.64 for 1% and 3% spikes (Table 14) and rejects 0.5% of healthy readings. Two caveats are important. First, the vanilla GRU behind the same gate is *more accurate* than the gated twin under spike faults (nMAE 0.253 and 0.356; active error 1.019 and 1.082; AP 0.102 and 0.068 against 0.089 and 0.040): the 36% of spikes that escape the gate (mostly the smaller ones, whose jump is below 1.5 times the largest hourly change of the gauge) are amplified by the routing prior. The robustness gained from the physics-guided design under data loss is thus paid for with a vulnerability to undetected spikes, which a deployment must address, for example through a tighter gate, a robust (median-based) preprocessing step, or a cross-sensor consistency check. Second, the gate is not free in clean conditions. It rejects a few genuine extreme rises, which is visible as a 15% reduction in the minor-flood AP of the twin (0.294 → 0.250) although the overall nMAE is unchanged (0.188). The gate threshold (1.5 times the largest hourly change observed in training) is deliberately not tuned on the test data, and a rise exceeding the historical record at a gauge is exactly the case in which an automatic rule is least reliable. It is therefore preferable to *flag* than to *discard* such readings, and to confirm them against neighbouring gauges before rejection.

| Fault | Precision | Recall | Healthy readings falsely rejected |
|---|---|---|---|
| Stuck-at 10% | 0.795 | 0.161 | 0.46% |
| Stuck-at 30% | 0.940 | 0.129 | 0.35% |
| Spikes 1% | 0.553 | 0.638 | 0.52% |
| Spikes 3% | 0.795 | 0.640 | 0.51% |
| Combined | 0.877 | 0.159 | 0.27% |

*Table 14. Performance of the edge QC gate as a fault detector on the test years (reading level).*

**Stuck sensors are only partially detectable, but the undetected part is mostly benign.** The stuck-at detector has high precision (0.80 and 0.94) but low recall (0.16 and 0.13). By design it flags a reading as stuck only after twelve identical readings *and* only when at least half of the neighbouring sensors show activity, so a stuck sensor during quiet weather is not flagged. This is the right trade-off for forecasting, because a constant reading during quiet conditions causes little harm: with 10% and 30% stuck readings the ungated twin's active error rises moderately (0.983 and 1.176 from 0.900), and the gate lowers it further (0.968 and 1.142). The consequence is nonetheless that a stuck sensor which freezes at the start of a flood is caught only if the gauges around it keep changing, which is the typical situation in a storm.

**Combined faults.** With 20% dropout, 10% stuck readings and 1% spikes simultaneously, the ungated twin fails (nMAE 1.173; active 1.899), as do the TF baseline (1.257) and, less strongly, the vanilla GRU (0.543), whereas with the gate the twin attains 0.592 (active 1.307), close to persistence (0.549; active 1.627) and above the gated vanilla GRU (0.417; active 1.282). Under this compound stress, no model is clearly better than persistence in overall error, and the minor-flood AP of every method falls to 0.01–0.03, i.e. a combination of undetected data faults removes essentially all alert skill. We interpret this as a limit of what can be asked of an unsupervised gate and as an argument for field maintenance, redundancy and sensor-health monitoring as part of any operational twin.

## 6.9 Calibrated uncertainty

The quantile heads (10% and 90%) form a nominal 80% interval for every horizon. On the test years it covers 83% of the outcomes at one hour and 79.7–80.0% from the sixth hour onward, so the heads are well calibrated at the level they were trained for, but they offer no guidance for other levels and no guarantee. To obtain intervals at an operator-chosen level we calibrated the heads with split CQR on the (unweighted) validation year.

**Marginal coverage.** After calibration, the empirical coverage at the nominal 90% level (α = 0.1) is 0.897 at one hour and 0.886 at 24 h (mean over horizons 0.889), against 0.80 for the raw heads, whose design level is 80% (Fig. 8a). At the nominal 80% and 95% levels, the coverage at 6 h is 0.793 and 0.944. The small shortfall at long lead times, 0.4 to 1.4 percentage points, is expected under a temporal shift: the calibration year (water year 2023) was dry, with a smaller proportion of high flows than the test years, so exchangeability is only approximately satisfied. A symmetric split-conformal interval around the median forecast attains similar marginal coverage (0.896 to 0.879) but is 23% wider at 24 h (0.664 ft against 0.540 ft for CQR, and 0.279 ft against 0.215 ft at 6 h), i.e. CQR is sharper at equal coverage because its width adapts to the state.

**Coverage by regime: where it fails.** Marginal coverage conceals regime dependence (Fig. 8b). For origins in a *rising* state (stage up by more than 0.25σ in the last three hours, 4.0% of test origins), CQR covers only 79–82% at the nominal 90%, and the symmetric split interval only 39–51%. State-conditional (Mondrian) calibration, with separate corrections for rising and not-rising origins, restores 0.888–0.892 coverage in the rising state at no cost in marginal coverage (0.889) and at a width within 2% of CQR (0.547 ft against 0.540 ft at 24 h). Since the rising state is determined by information known at the forecast origin, Mondrian calibration is implementable in operation and should be preferred to marginal calibration for warning systems.

A harder case remains. When coverage is evaluated on the *realised* active origins (the outcome moves by more than 1σ within the horizon, 12.4% of the origins), none of the calibrations achieves the nominal level: coverage is 0.82 at one hour and falls to 0.72 at 6 h and 0.54 at 24 h for CQR (0.83, 0.73 and 0.56 for Mondrian), while it is 0.91–0.93 on the complementary quiet origins. This set is defined by the outcome, so it is not a conditional-coverage guarantee that a method could be expected to provide, but it shows how marginal calibration works: high coverage on the numerous quiet origins compensates for under-coverage on the rare large changes, which are the ones that matter for flood warning. An interval should therefore be read as a statement about the typical day and not about the extreme event.

![Figure 8. Calibration of the 90% intervals. (a) Marginal coverage by lead time; (b) coverage in the rising state and on realised-active origins; (c) mean interval width.](figures/fig10_conformal.png)

**Per-gauge variability and online adaptation.** Coverage varies across gauges: at 6 h the 10th and 90th percentiles of per-gauge coverage are 0.80 and 0.95 (0.81 and 0.94 at 24 h) for α = 0.1. Adaptive conformal inference, run online per gauge on a random sample of 30 gauges with realistic label delays, reduces this spread nearly to zero. At 6 h its mean coverage is 0.9001 (standard deviation across gauges 0.0007) and at 24 h 0.9000 (0.0009), whereas a static per-gauge calibration reaches 0.893 ± 0.029 and 0.881 ± 0.023. The mean absolute deviation from the nominal level thus falls from 0.021–0.025 to 0.0006–0.0007, a roughly 35-fold reduction. This is the property that an operational system needs: a long-run miss rate that does not depend on how well the calibration period resembles the present.

**From interval to alert.** Using the conformalised *upper* bound as the alert trigger (alert if the upper bound exceeds the threshold at any horizon) is not competitive with the margin rule of Section 6.4. With α_alert = 0.4, 0.2, 0.1 and 0.05, POD rises slightly from 0.373 to 0.388, but the false-alarm ratio is 0.76–0.79, giving CSI of 0.171 to 0.158, against a CSI of 0.239 for the margin rule. The upper bound is calibrated for coverage of all outcomes on all days; it is not designed for the rare event of crossing a particular threshold, for which a margin chosen for the purpose is more efficient. We take this as evidence that the conformal layer is best used for communicating uncertainty (the interval shown with the forecast) and for monitoring the health of the twin (online coverage) rather than as the trigger itself.

## 6.10 Physics consistency of the learned twin

Does the trained network rely on the neighbours that the topology-free prior identified, and does that reliance match the physical meaning of the lag analysis? We remove each selected neighbour in turn (by masking its channel and routing features for the gauge in question) and measure the increase of the active-state nMAE on the test years, over all 321 edges (Fig. 9).

![Figure 9. Increase in active-state nMAE when a single inferred upstream neighbour is removed from the input: (a) against the inferred peak correlation ρ*, (b) by neighbour rank, (c) by inferred travel time.](figures/fig11_physics.png)

Removing a neighbour increases the error for 80% of the edges (mean +0.037σ). The importance of an edge is positively rank-correlated with its inferred peak correlation (Spearman ρ = 0.34, p = 4 × 10⁻¹⁰) and with its directionality d (ρ = 0.23, p = 3 × 10⁻⁵), and negatively with the neighbour's rank in the selection (ρ = −0.37, p = 5 × 10⁻¹²: the mean increase falls from 0.062σ for the first-ranked to 0.013σ for the fourth-ranked neighbour). Importance is also higher for neighbours with longer inferred lead times (ρ = 0.32 with τ*, p = 4 × 10⁻⁹; 0.061σ for edges with τ* > 2 h against 0.024σ for τ* ≤ 2 h), which is what one expects physically: a gauge that leads by many hours supplies information that cannot be obtained from the gauge's own recent history, while a gauge that leads by one hour is nearly redundant with it. These correlations are moderate and not deterministic, which is expected because importance also depends on whether the neighbour is available and on what the other neighbours provide, but they are consistent with the interpretation that the network has learned to use the structure that the prior encodes rather than an arbitrary correlate.

A what-if experiment probes the twin as a simulator. For each gauge we selected the 2% of test windows with the largest absolute six-hour change at its first-ranked upstream neighbour and rescaled the neighbour's deviation from its window start by 0.5, 1.0, 1.5 and 2.0, leaving all other inputs unchanged. The predicted 6-h stage change at the downstream gauge increases monotonically with the scale, from 0.011σ to 0.110σ, 0.204σ and 0.303σ on average (the 12-h change goes from −0.19σ to +0.16σ), and the response at the individual gauge is monotone over the four scales in 69% of the gauges. The linear routing component alone responds monotonically by construction (0.11σ to 0.55σ at 12 h); the neural residual thus slightly reduces the sensitivity to the upstream signal (the 12-h response per unit of scale is 0.24σ against 0.29σ for the routing component alone), and in about one-third of gauges the full response is non-monotone, which signals that the neural correction can override the linear structure in sparse regions of the input space. This is a limitation of unconstrained residual learning that monotonicity-constrained architectures could address.

## 6.11 Edge feasibility

The forecasting component consists of 51,208 parameters (204 kB in 32-bit floating point), and one forecast requires about 0.58 million multiply–accumulate operations. On one thread of the CPU used for all experiments, a full network update for all 99 gauges takes 5.2 ms including window assembly and the routing prior (1.7 ms for a single gauge), which is orders of magnitude below the hourly update interval and compatible with gateway-class hardware. Post-training dynamic quantisation of the linear and recurrent layers to 8 bit reduces the size to 60 kB (3.4×) and increases the normalised error from 0.1880 to 0.1915 (+1.9%; mean absolute change in the forecast 0.010 ft, maximum 1.4 ft), but does not reduce the latency in this software stack on this CPU (5.0 ms) because the cost is dominated by the data movement and not by the arithmetic. If a hypothetical energy cost of 1 nJ per multiply–accumulate is assumed for a microcontroller-class device (an assumption, not a measurement), each forecast needs about 0.6 mJ and a full hourly update of the network about 58 mJ, a small fraction of a typical duty cycle's energy budget. The memory and compute demands of the neural component are thus modest enough for the edge layer to retain operation when the backhaul link fails during an event, which is the situation in which a centralised twin is blind.

![Figure 10. Illustrative event (selected as the test-period minor-flood event with the largest exceedance among those with at least 8 h of lead time; it is not a random draw). The Potomac River at Point of Rocks, 14 May 2025. Top: its two highest-ranked upstream neighbours. Bottom: observed stage and the twin's 24-h forecasts (lines) with the raw 10–90% quantile band (shaded) issued 15, 9 and 5 hours before the stage crossed the minor flood stage. The forecast issued 15 h ahead sees no flood; the forecast issued 9 h ahead already reaches the threshold.](figures/fig4_case.png)

The case study (Fig. 10) illustrates a typical successful alert and its limits. The Potomac River at Point of Rocks rose from about 4 ft to above the 16-ft minor flood stage within about ten hours, following a sharp rise at two upstream gauges. The forecast issued 15 hours before the crossing, when neither the gauge nor its neighbours had yet responded, shows no flood, as it must in a telemetry-only system; the forecast issued 9 hours before the crossing, shortly after the upstream gauges began to rise, brings the stage to the threshold about one to two hours after the real crossing, and the forecast issued 5 hours before is accurate for the rising limb and slightly overestimates the later peak. The usable warning is therefore about 9 hours in this case, consistent with the median lead times of 5–6 hours in Table 8.


# 7. Discussion

## 7.1 What the results say

**The gains are real, consistent and concentrated where they matter, but they are incremental over a strong linear baseline.** Across 99 gauges and 1.28 million test origins, PG-FaultTwin reduced the error under active conditions by 31% relative to persistence and by 8% relative to the vanilla GRU, improved rise-rate skill and raised the average precision of minor-flood alerts from 0.170 (vanilla GRU) to 0.274 (+61%). Its edge over the *linear routing* model built from the same physics-selected neighbours is smaller and should be stated plainly: 10% lower 24-h MAE (0.177 against 0.196 ft), 8% lower active error (0.876 against 0.955), a rise-rate correlation that is statistically a tie, and a minor-flood AP that is 0.015 higher (95% interval 0.002 to 0.028). In operational terms the twin's advantage over TF materialises mostly as *fewer false alarms at the same detection*: 29% fewer false-alarm episodes for minor floods and 20% for the action stage, and a higher economic value for users with large cost–loss ratios (V = 0.119 against 0.084 at C/L = 0.5). A reader who needs a minimal, auditable system could deploy the TF baseline and obtain most of the benefit; the neural residual then buys additional precision, graceful degradation under data loss and a probabilistic output, at a small computational price (Section 6.11).

**Most of the value comes from the physics-selected structure, not from model capacity.** The ablations show that removing the routing prior costs more than any other change (−23% in minor-stage AP), that replacing the lag-selected neighbours by geographically nearest gauges costs 16%, and that the vanilla GRU, with the same recurrent capacity but none of the structure, is the weakest neural model in the alert metrics (AP 0.170) and loses to the linear TF baseline in alerting despite being nonlinear. This is in line with the recent argument that process knowledge remains valuable even when flexible learners are available (Nearing et al., 2021), and with the finding that simple linear models are strong baselines in time-series forecasting (Zeng et al., 2023). In our setting the explanation is physical: over 1–24 h lead times along a river, the dominant predictable signal is the propagation of an existing disturbance, which is approximately linear in the lagged upstream changes, whereas the nonlinear component, which carries the residual gains, matters in the rare large events.

**A cheap, topology-free prior is enough to find the useful neighbours.** The lag analysis uses no GIS layer, yet the neighbours it selects are more informative than the geographically nearest ones, and the trained network's reliance on each neighbour correlates with the inferred strength, direction and lead time of the link (Section 6.10). The apparent celerities are plausible in two regions and inflated in two others, mostly because the one-hour resolution is coarse relative to travel times between nearby gauges. We therefore do not claim to recover river hydraulics. The claim is the more modest one that the telemetry alone contains the information needed to choose which sensors to trust for which forecast point, which is precisely what an operator commissioning a network in a data-poor basin needs.

**Fault tolerance is multi-faceted, and no single mechanism covers it.** The results separate three classes of fault with different remedies. Missing and stale data are handled by masked inputs and the fault curriculum: the twin stays ahead of persistence in active-state error up to 70% data loss, while the vanilla GRU, never trained on gaps, falls below persistence in overall error at that level. Gross errors (spikes, and by extension flat-lined sensors in dynamic conditions) cannot be handled by the learned model, and for the physics-guided model they are *more* harmful than for the vanilla GRU because the linear routing coefficients amplify them; an explicit causal gate is indispensable, but it is imperfect (recall 0.64 for spikes, 0.13–0.16 for stuck readings) and has a price in clean conditions (a 15% reduction in AP from rejecting some genuine extremes). Slow faults such as calibration bias are not detectable without an independent reference and degrade all models by similar amounts (+5% to +15% in the active error of the twin). The practical implication is that a robust twin needs *layers*: a gate at the edge, a model that understands missingness, and sensor-health analytics and maintenance upstream of both. It also suggests that the preference for the simplest defence (data imputation followed by an unchanged forecaster) deserves scrutiny, since the unchanged forecaster here was fragile in exactly the cases in which imputation does not help.

**Uncertainty must be conditioned on the state to be useful for warnings.** Marginal conformal calibration delivered the nominal coverage (0.889 at the 90% level), but the intervals under-covered the situation that matters most, rising limbs (0.79–0.82), and Mondrian calibration on an observable state fixed this at no loss of sharpness (0.89). Beyond that, no calibration based on information available at the origin could prevent the intervals from missing a large fraction of large realised changes (coverage 0.54 at 24 h on realised-active origins). Adaptive conformal inference removed the dependence of the long-run miss rate on the calibration period (0.900 ± 0.001 against 0.88–0.89 ± 0.02–0.03). These results suggest a division of labour: the *margin rule* triggers alerts, the conformalised interval communicates the uncertainty, and online coverage tracking monitors the health of the twin.

**A cautionary tale about data.** The most instructive single result was not produced by a model: it was the discovery in the public telemetry of sensors that report constant values for years while still being counted as "available". One of them sat 16 ft above its historical median; none of them triggers a range check. Standard coverage filters retain such sensors, and a model with access to absolute levels can be corrupted by them without any visible sign in the training or validation metrics, as our preliminary run showed (Section 6.2). Any benchmark built on operational telemetry should include a dead-sensor screen; any operational twin should include sensor-health monitoring.

## 7.2 Implications for deployment

Five recommendations follow from the evidence, intended for agencies operating an IoT gauge network and considering a learned twin.

1. **Start with the linear routing model and add the residual later.** The TF model is cheap, transparent and strong; it also generalises, with a pooled coefficient set, to a new region (Table 10). Its principal weakness, vulnerability to spikes, is shared with the full twin and requires the same gate.
2. **Design the edge layer around the mask.** Treat each reading as present, absent or suspect, give the model the mask, and train it with realistic gap patterns. This costs nothing on clean data (Section 6.6) and avoids the deterioration to below-persistence skill that we observed when a standard network meets 70% missing data.
3. **Use state-conditional calibration and online coverage monitoring.** Report intervals with Mondrian CQR conditioned on a rising/not-rising flag, and track empirical coverage online with ACI; use the coverage trace as a health indicator.
4. **Choose the alert margin on the basis of the users' cost–loss ratios.** The value curves of Fig. 6 show that the best model depends on the user: for users with a very low cost–loss ratio, TF and the twin are equivalent, whereas for users with a high ratio, the twin's precision matters. A single operating point cannot serve both.
5. **Commission locally where dynamics are fast.** In the Texas Hill Country, the transferred model did not outperform the vanilla GRU or the TF baseline, and all methods had low alert skill. For flash-flood regimes, local training of at least the routing prior, finer time resolution and rainfall inputs are preconditions for useful warnings.

## 7.3 Limitations

We list the limitations that we consider most important, in approximately decreasing order of consequence.

*Telemetry-only information.* The twin sees only stage. It cannot anticipate rises driven by rainfall that has not yet reached a gauge, which caps the skill for headwater and flash-flood gauges and explains the modest absolute POD (about 0.3) at the CSI-optimal margin. Coupling with radar or satellite precipitation, numerical weather prediction and, in the language of the call for this special issue, satellite constellations such as CubeSats would extend the lead time. We deliberately did not include such forcings, to isolate the contributions of this paper, and we expect the absolute skill figures to improve when they are added, though the relative advantage of the proposed elements may change.

*Event numbers and test-set choices.* The test period contains 68 verifiable minor-flood events at 37 gauges and 245 action-stage events at 52 gauges. The paired bootstrap shows that the differences between the twin and the learned baselines are well resolved, but the difference to TF is small, and event-level quantities (lead time, detection counts) are noisy. The alert margin δ was chosen on the test period for every method because the validation year contained two minor-flood events; this is optimistic in absolute terms for all methods alike, and we have reported the threshold-free AP and the value curves, which do not depend on it, as the primary measures.

*The development path.* The final configuration was chosen after a number of decisions made on validation data: the form of the routing prior (robust fit on differences only, preferred to a least-squares fit with level terms, which was worse than persistence on validation), the use of uniform rather than event-focused sampling, and the curriculum intensity. The dead-sensor screen, however, was motivated by inspecting *test-period* errors of a preliminary run in which one gauge was responsible for an implausible share of the error, and the experiments were then repeated from scratch on the cleaned data. The final test results are therefore not entirely uncontaminated by the test data in the sense that the data screening was prompted by them, although the screen is generic (a rule on stage variability) and applies to all methods, and the test data were not used for any model-selection or hyper-parameter decision. We recommend that independent replication on a new period or region be treated as the real confirmation.

*Scope of regions and sensors.* All data come from the United States, where gauge networks are comparatively dense and well maintained. The topology-free prior depends on neighbouring gauges being within a few tens of kilometres; in sparser networks, fewer neighbours would be found, and the benefit of the physics-selected structure would diminish. The cross-region experiment covers four regions, only one of which (Texas) is a flash-flood regime, and one cannot generalise from a single negative case; but the case shows that the method has a limit.

*Resolution.* Hourly aggregation limits the resolution of travel times and lead times to one and two hours respectively. For small, fast catchments, 15-minute modelling is needed.

*Fault models.* Our fault regimes are synthetic and independent between sensors; real faults are correlated (a storm can disable several gauges at once, and communication outages affect sensors that share a gateway), and bias drift is gradual. The gate does not include drift detection and handles stuck sensors conservatively. The results for compound faults are a warning about what happens when several things go wrong, and not an estimate of the real-world frequency of such cases.

*Uncertainty.* Conformal guarantees rely on exchangeability, which our time series violate; the dry calibration year produced a small marginal shortfall, and coverage on realised-active origins is poor. The ACI results were evaluated on a random subset of 30 gauges.

*Model selection.* Neural results are based on three seeds for the main models and a single seed for ablations, and we did not tune the architecture beyond a first configuration; a systematic hyper-parameter search, better sequence encoders or graph architectures might improve all neural methods, including the baselines.

## 7.4 Relation to the other themes of the special issue

The framework was developed and tested for river stage, but its components are not specific to rivers, and we mention possible transfers with the proviso that they are untested hypotheses. *Digital twins for landslides* share the structure of sparse sensors with intermittent faults, a causal driver (rainfall, pore pressure) and lagged downstream effects, so that a mask-aware, residual-on-physics model and a state-conditional conformal layer could be used with slope-stability prior models. *Structural health monitoring* of bridges, dams or canals suffers from the same sensor fault classes (stuck, spike, drift), for which the gate-plus-mask design and the evidence that gross errors need explicit screening are directly relevant; here, the "physics prior" would be a modal or influence-line model. *CubeSat networks* can be viewed as a sensor of high latency and low revisit frequency; in our framework such an observation is a stale channel with an age flag, which the model has learned to weight. *Post-disaster resource management* is addressed only at the interface: the cost–loss analysis (Section 6.5) is the quantitative bridge from calibrated warnings to decisions about the pre-positioning and routing of rescue teams and machinery, since each class of resource has its own cost–loss ratio.

## 7.5 Future work

Four directions appear most promising. First, adding precipitation and weather-model forcings to the twin, with the same mask-based treatment of unavailable inputs, to extend lead times beyond what upstream telemetry can offer. Second, replacing the unconstrained residual by a monotonicity-constrained or mass-conserving architecture (Hoedt et al., 2021), to remove the non-monotone responses seen in the what-if analysis (Section 6.10) and make the twin safer for scenario analysis. Third, a learned or cross-sensor fault detector for drift and stuck sensors that is evaluated on real fault logs from an operating agency. Fourth, a prospective, real-time evaluation over a flood season, with an operator-in-the-loop study of how calibrated intervals and cost-aware margins affect decisions.


# 8. Conclusions

We presented PG-FaultTwin, a fault-tolerant, physics-guided digital twin for 1–24 h stage-rise and flood-threshold early warning on IoT gauge networks, and evaluated it on 99 telemetered USGS gauges in four US regions (44.8 million raw readings; water years 2017–2026) with official NWS flood stages as event definitions. The twin identifies its upstream neighbours and their lead times from telemetry alone, forecasts with a linear multi-lag routing prior corrected by a 51k-parameter recurrent network trained under a sensor-fault curriculum, protects itself with a causal edge QC gate, and delivers conformalised intervals and cost-aware alerts. The principal conclusions are as follows.

1. **Accuracy and warning skill.** Against persistence, PG-FaultTwin reduced the error in active conditions by 31% and the 24-h MAE by 21%; against a vanilla GRU of identical capacity, by 8% (active error) and 5% (24-h RMSE). The average precision for minor-flood alerts rose from 0.170 (vanilla GRU) and 0.086 (persistence) to 0.274, and the paired gauge-level bootstrap resolves these gains clearly. The advantage over a strong linear routing baseline is small (+0.015 AP, 95% interval 0.002–0.028; 20–29% fewer false-alarm episodes at similar detection), and the baseline deserves to be the first choice of any agency.
2. **Physics-selected structure is the main driver.** The routing prior and the lag-selected neighbours account for most of the gain (−23% and −16% AP when removed); the neural residual contributes in the rare, large events. The neighbours selected from lag structure outperform geographic neighbours, and the network's use of them tracks the inferred strength and lead time of the link.
3. **Fault tolerance requires layers.** Mask-aware training keeps the twin ahead of persistence in active-state error up to 70% data loss, whereas an unprepared network falls below persistence in overall error. Gross errors need an explicit gate, because the physics-guided model amplifies spikes: without the gate, 1% spikes raised the twin's error from 0.188 to 0.958 nMAE; with it, to 0.339. The gate costs 15% of AP in clean conditions, and it catches only 64% of spikes and 13–16% of stuck readings.
4. **Uncertainty must be conditioned and monitored.** Marginal conformal calibration gave 0.889 coverage at the 90% level but 0.79–0.82 on rising limbs; state-conditional calibration restored 0.89 and online adaptive inference stabilised the long-run miss rate (0.900 ± 0.001). Coverage on realised large changes remained poor (0.54 at 24 h).
5. **Transfer has limits.** A model trained in three regions retained 16–25% of its gain over persistence in the held-out region but lost its advantage over the baselines in the flash-flood Texas Hill Country, where every method had low alert skill; local commissioning is advisable there.
6. **Edge deployment is feasible.** A forecast for all 99 gauges takes 5 ms on one CPU thread with a 204 kB (60 kB int8) model.

A by-product of independent importance is the finding that public telemetry contains sensors that report constant values for years, undetected by range checks and counted as available by coverage filters; benchmarks and operational twins should screen for them. The study's main limitations are the restriction to stage telemetry (no rainfall), the modest number of independent flood events, and the fact that all data are from the United States; they point to the extension to multi-source forcing and prospective operational evaluation outlined in Section 7.5.


# Data and code availability

All data used in this study are public. Stage and discharge telemetry were obtained from the U.S. Geological Survey National Water Information System instantaneous-values web service (waterservices.usgs.gov), station metadata from the NWIS site service, and official flood stages from the NOAA National Weather Service National Water Prediction Service API (api.water.noaa.gov). Data were retrieved on 4 October 2026; provisional data may be revised by the originating agencies. The complete pipeline (station inventory, download, quality control, topology-free prior, all models, fault injection, conformal calibration, edge benchmark, figures, and this manuscript's tables) is provided in the accompanying repository (github.com/deep1789/IoT-Disaster) with a run order documented in `PLAN.md`; every number in the Results section is generated by these scripts and stored in `results/`.

# Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper. *[To be confirmed by the authors.]*

# Declaration of generative AI and AI-assisted technologies in the writing process

*[Template for the authors to review and adapt before submission, as required by the publisher's policy.]* During the preparation of this work, an AI coding and writing assistant was used to help write analysis code, run the computational experiments and draft the manuscript text. The authors reviewed and edited the content as needed and take full responsibility for the content of the publication.

# CRediT authorship contribution statement

*[To be completed by the authors.]* Conceptualization; Methodology; Software; Validation; Formal analysis; Investigation; Data curation; Writing – original draft; Writing – review & editing; Visualization.

# Acknowledgements

We thank the U.S. Geological Survey and the National Weather Service for operating and openly distributing the monitoring networks on which this study depends.


# References

Abadade, Y., Temouden, A., Bamoumen, H., Benamar, N., Chtouki, Y. & Hafid, A. S. (2023). A Comprehensive Survey on TinyML. IEEE Access, 11, 96892-96922. https://doi.org/10.1109/access.2023.3294111

Angelopoulos, A. N. & Bates, S. (2023). Conformal Prediction: A Gentle Introduction. Foundations and Trends in Machine Learning, 16(4), 494-591. https://doi.org/10.1561/9781638281597

Atzori, L., Iera, A. & Morabito, G. (2010). The Internet of Things: A survey. Computer Networks, 54(15), 2787-2805. https://doi.org/10.1016/j.comnet.2010.05.010

Bai, S., Kolter, J. Z. & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271.

Barber, R. F., Candès, E. J., Ramdas, A. & Tibshirani, R. J. (2023). Conformal prediction beyond exchangeability. The Annals of Statistics, 51(2). https://doi.org/10.1214/23-aos2276

Barnes, L. R., Gruntfest, E. C., Hayden, M. H., Schultz, D. M. & Benight, C. (2007). False Alarms and Close Calls: A Conceptual Model of Warning Accuracy. Weather and Forecasting, 22(5), 1140-1147. https://doi.org/10.1175/waf1031.1

Basha, E. A., Ravela, S. & Rus, D. (2008). Model-based monitoring for early warning flood detection. Proceedings of the 6th ACM conference on Embedded network sensor systems, 295-308. https://doi.org/10.1145/1460412.1460442

Basher, R. (2006). Global early warning systems for natural hazards: systematic and people-centred. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 364(1845), 2167-2182. https://doi.org/10.1098/rsta.2006.1819

Bauer, P., Dueben, P. D., Hoefler, T., Quintino, T., Schulthess, T. C. & Wedi, N. P. (2021). The digital revolution of Earth-system science. Nature Computational Science, 1(2), 104-113. https://doi.org/10.1038/s43588-021-00023-0

Bauer, P., Stevens, B. & Hazeleger, W. (2021). A digital twin of Earth for the green transition. Nature Climate Change, 11(2), 80-83. https://doi.org/10.1038/s41558-021-00986-y

Benjamini, Y. & Hochberg, Y. (1995). Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. Journal of the Royal Statistical Society Series B: Statistical Methodology, 57(1), 289-300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

Bhasme, P., Vagadiya, J. & Bhatia, U. (2022). Enhancing predictive skills in physically-consistent way: Physics Informed Machine Learning for hydrological processes. Journal of Hydrology, 615, 128618. https://doi.org/10.1016/j.jhydrol.2022.128618

Blöschl, G., Hall, J., Viglione, A., Perdigão, R. A. P., Parajka, J., Merz, B., et al. (2019). Changing climate both increases and decreases European river floods. Nature, 573(7772), 108-111. https://doi.org/10.1038/s41586-019-1495-6

BRIER, G. W. (1950). VERIFICATION OF FORECASTS EXPRESSED IN TERMS OF PROBABILITY. Monthly Weather Review, 78(1), 1-3. https://doi.org/10.1175/1520-0493(1950)078<0001:vofeit>2.0.co;2

Cao, W., Wang, D., Li, J., Zhou, H., Li, L. & Li, Y. (2018). BRITS: Bidirectional recurrent imputation for time series. Advances in Neural Information Processing Systems 31 (NeurIPS).

Carsell, K. M., Pingel, N. D. & Ford, D. T. (2004). Quantifying the Benefit of a Flood Warning System. Natural Hazards Review, 5(3), 131-140. https://doi.org/10.1061/(asce)1527-6988(2004)5:3(131)

Chandola, V., Banerjee, A. & Kumar, V. (2009). Anomaly detection: A survey. ACM Computing Surveys, 41(3), 1-58. https://doi.org/10.1145/1541880.1541882

Che, Z., Purushotham, S., Cho, K., Sontag, D. & Liu, Y. (2018). Recurrent Neural Networks for Multivariate Time Series with Missing Values. Scientific Reports, 8(1). https://doi.org/10.1038/s41598-018-24271-9

Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., et al. (2014). Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1724-1734. https://doi.org/10.3115/v1/d14-1179

Cunge, J. A. (1969). On The Subject Of A Flood Propagation Computation Method (Musklngum Method). Journal of Hydraulic Research, 7(2), 205-230. https://doi.org/10.1080/00221686909500264

Davis, J. & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. Proceedings of the 23rd international conference on Machine learning - ICML '06, 233-240. https://doi.org/10.1145/1143844.1143874

Diebold, F. X. & Mariano, R. S. (1995). Comparing Predictive Accuracy. Journal of Business & Economic Statistics, 13(3), 253-263. https://doi.org/10.1080/07350015.1995.10524599

Ding, Y., Zhu, Y., Feng, J., Zhang, P. & Cheng, Z. (2020). Interpretable spatio-temporal attention LSTM model for flood forecasting. Neurocomputing, 403, 348-359. https://doi.org/10.1016/j.neucom.2020.04.110

Du, W., Côté, D. & Liu, Y. (2023). SAITS: Self-attention-based imputation for time series. Expert Systems with Applications, 219, 119619. https://doi.org/10.1016/j.eswa.2023.119619

Dutta, D. L. & Bharali, S. (2021). TinyML Meets IoT: A Comprehensive Survey. Internet of Things, 16, 100461. https://doi.org/10.1016/j.iot.2021.100461

Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife. The Annals of Statistics, 7(1). https://doi.org/10.1214/aos/1176344552

Frame, J. M., Kratzert, F., Klotz, D., Gauch, M., Shalev, G., Gilon, O., et al. (2022). Deep learning rainfall–runoff predictions of extreme events. Hydrology and Earth System Sciences, 26(13), 3377-3392. https://doi.org/10.5194/hess-26-3377-2022

Fuller, A., Fan, Z., Day, C. & Barlow, C. (2020). Digital Twin: Enabling Technologies, Challenges and Open Research. IEEE Access, 8, 108952-108971. https://doi.org/10.1109/access.2020.2998358

Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. Proceedings of the 33rd International Conference on Machine Learning (ICML), PMLR, 48, 1050-1059.

Gauch, M., Kratzert, F., Klotz, D., Nearing, G., Lin, J. & Hochreiter, S. (2021). Rainfall–runoff prediction at multiple timescales with a single Long Short-Term Memory network. Hydrology and Earth System Sciences, 25(4), 2045-2062. https://doi.org/10.5194/hess-25-2045-2021

Gibbs, I. & Candès, E. (2021). Adaptive conformal inference under distribution shift. Advances in Neural Information Processing Systems 34 (NeurIPS).

Gneiting, T. & Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. Journal of the American Statistical Association, 102(477), 359-378. https://doi.org/10.1198/016214506000001437

Gneiting, T., Balabdaoui, F. & Raftery, A. E. (2007). Probabilistic Forecasts, Calibration and Sharpness. Journal of the Royal Statistical Society Series B: Statistical Methodology, 69(2), 243-268. https://doi.org/10.1111/j.1467-9868.2007.00587.x

Grieves, M. & Vickers, J. (2017). Digital Twin: Mitigating Unpredictable, Undesirable Emergent Behavior in Complex Systems. Transdisciplinary Perspectives on Complex Systems, 85-113. https://doi.org/10.1007/978-3-319-38756-7_4

Gubbi, J., Buyya, R., Marusic, S. & Palaniswami, M. (2013). Internet of Things (IoT): A vision, architectural elements, and future directions. Future Generation Computer Systems, 29(7), 1645-1660. https://doi.org/10.1016/j.future.2013.01.010

Gupta, H. V., Kling, H., Yilmaz, K. K. & Martinez, G. F. (2009). Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling. Journal of Hydrology, 377(1-2), 80-91. https://doi.org/10.1016/j.jhydrol.2009.08.003

Hallegatte, S. (2012). A Cost Effective Solution to Reduce Disaster Losses in Developing Countries: Hydro-Meteorological Services, Early Warning, and Evacuation. World Bank Policy Research Working Paper 6058. https://doi.org/10.1596/1813-9450-6058

Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the equality of prediction mean squared errors. International Journal of Forecasting, 13(2), 281-291. https://doi.org/10.1016/s0169-2070(96)00719-4

Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735

Hoedt, P.-J., Kratzert, F., Klotz, D., Halmich, C., Holzleitner, M., Nearing, G., et al. (2021). MC-LSTM: Mass-conserving LSTM. Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR, 139.

Huber, P. J. (1964). Robust Estimation of a Location Parameter. The Annals of Mathematical Statistics, 35(1), 73-101. https://doi.org/10.1214/aoms/1177703732

Islam, A. R. M. T., Mamun, A. A., Rahman, M. N., Akter, M. Y., Chisty, M. A., Alam, G. M., et al. (2024). A cost-benefit analysis of flood early warning system: Evidence from lower Brahmaputra River Basin, Bangladesh. International Journal of Disaster Risk Reduction, 104, 104380. https://doi.org/10.1016/j.ijdrr.2024.104380

Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S. & Yang, L. (2021). Physics-informed machine learning. Nature Reviews Physics, 3(6), 422-440. https://doi.org/10.1038/s42254-021-00314-5

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems 30 (NeurIPS), 3146-3154.

Kingma, D. P. & Ba, J. (2015). Adam: A method for stochastic optimization. International Conference on Learning Representations (ICLR).

Kipf, T. N. & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. International Conference on Learning Representations (ICLR).

Klotz, D., Kratzert, F., Gauch, M., Keefe Sampson, A., Brandstetter, J., Klambauer, G., et al. (2022). Uncertainty estimation with deep learning for rainfall–runoff modeling. Hydrology and Earth System Sciences, 26(6), 1673-1693. https://doi.org/10.5194/hess-26-1673-2022

Knoben, W. J. M., Freer, J. E. & Woods, R. A. (2019). Technical note: Inherent benchmark or not? Comparing Nash–Sutcliffe and Kling–Gupta efficiency scores. Hydrology and Earth System Sciences, 23(10), 4323-4331. https://doi.org/10.5194/hess-23-4323-2019

Koenker, R. & Bassett, G. (1978). Regression Quantiles. Econometrica, 46(1), 33. https://doi.org/10.2307/1913643

Kratzert, F., Klotz, D., Brenner, C., Schulz, K. & Herrnegger, M. (2018). Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks. Hydrology and Earth System Sciences, 22(11), 6005-6022. https://doi.org/10.5194/hess-22-6005-2018

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S. & Nearing, G. S. (2019). Toward Improved Predictions in Ungauged Basins: Exploiting the Power of Machine Learning. Water Resources Research, 55(12), 11344-11354. https://doi.org/10.1029/2019wr026065

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S. & Nearing, G. (2019). Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets. Hydrology and Earth System Sciences, 23(12), 5089-5110. https://doi.org/10.5194/hess-23-5089-2019

Kundzewicz, Z. W., Kanae, S., Seneviratne, S. I., Handmer, J., Nicholls, N., Peduzzi, P., et al. (2014). Flood risk and climate change: global and regional perspectives. Hydrological Sciences Journal, 59(1), 1-28. https://doi.org/10.1080/02626667.2013.857411

Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in Neural Information Processing Systems 30 (NeurIPS).

LeClerc, J. & Joslyn, S. (2015). The Cry Wolf Effect and Weather‐Related Decision Making. Risk Analysis, 35(3), 385-395. https://doi.org/10.1111/risa.12336

Lees, T., Reece, S., Kratzert, F., Klotz, D., Gauch, M., De Bruijn, J., et al. (2022). Hydrological concept formation inside long short-term memory (LSTM) networks. Hydrology and Earth System Sciences, 26(12), 3079-3101. https://doi.org/10.5194/hess-26-3079-2022

Lei, J., G’Sell, M., Rinaldo, A., Tibshirani, R. J. & Wasserman, L. (2018). Distribution-Free Predictive Inference for Regression. Journal of the American Statistical Association, 113(523), 1094-1111. https://doi.org/10.1080/01621459.2017.1307116

Li, Y., Yu, R., Shahabi, C. & Liu, Y. (2018). Diffusion convolutional recurrent neural network: Data-driven traffic forecasting. International Conference on Learning Representations (ICLR).

Lim, B., Arık, S. Ö., Loeff, N. & Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. International Journal of Forecasting, 37(4), 1748-1764. https://doi.org/10.1016/j.ijforecast.2021.03.012

Liu, F. T., Ting, K. M. & Zhou, Z. H. (2008). Isolation Forest. 2008 Eighth IEEE International Conference on Data Mining, 413-422. https://doi.org/10.1109/icdm.2008.17

Liu, Y., Hou, G., Huang, F., Qin, H., Wang, B. & Yi, L. (2022). Directed graph deep neural network for multi-step daily streamflow forecasting. Journal of Hydrology, 607, 127515. https://doi.org/10.1016/j.jhydrol.2022.127515

Loshchilov, I. & Hutter, F. (2019). Decoupled weight decay regularization. International Conference on Learning Representations (ICLR).

M. C. Rápalo, L., Gomes Jr, M. N. & Mendiondo, E. M. (2024). Developing an open-source flood forecasting system adapted to data-scarce regions: A digital twin coupled with hydrologic-hydrodynamic simulations. Journal of Hydrology, 644, 131929. https://doi.org/10.1016/j.jhydrol.2024.131929

Merz, B., Blöschl, G., Vorogushyn, S., Dottori, F., Aerts, J. C. J. H., Bates, P., et al. (2021). Causes, impacts and patterns of disastrous river floods. Nature Reviews Earth & Environment, 2(9), 592-609. https://doi.org/10.1038/s43017-021-00195-3

Mosavi, A., Ozturk, P. & Chau, K. W. (2018). Flood Prediction Using Machine Learning Models: Literature Review. Water, 10(11), 1536. https://doi.org/10.3390/w10111536

Murphy, A. H. (1977). The Value of Climatological, Categorical and Probabilistic Forecasts in the Cost-Loss Ratio Situation. Monthly Weather Review, 105(7), 803-816. https://doi.org/10.1175/1520-0493(1977)105<0803:tvocca>2.0.co;2

Murphy, A. H. (1993). What Is a Good Forecast? An Essay on the Nature of Goodness in Weather Forecasting. Weather and Forecasting, 8(2), 281-293. https://doi.org/10.1175/1520-0434(1993)008<0281:wiagfa>2.0.co;2

Nash, J. & Sutcliffe, J. (1970). River flow forecasting through conceptual models part I — A discussion of principles. Journal of Hydrology, 10(3), 282-290. https://doi.org/10.1016/0022-1694(70)90255-6

Nearing, G. S., Kratzert, F., Sampson, A. K., Pelissier, C. S., Klotz, D., Frame, J. M., et al. (2021). What Role Does Hydrological Science Play in the Age of Machine Learning?. Water Resources Research, 57(3). https://doi.org/10.1029/2020wr028091

Nearing, G., Cohen, D., Dube, V., Gauch, M., Gilon, O., Harrigan, S., et al. (2024). Global prediction of extreme floods in ungauged watersheds. Nature, 627(8004), 559-563. https://doi.org/10.1038/s41586-024-07145-1

Nevo, S., Morin, E., Gerzi Rosenthal, A., Metzger, A., Barshai, C., Weitzner, D., et al. (2022). Flood forecasting with machine learning models in an operational framework. Hydrology and Earth System Sciences, 26(15), 4013-4032. https://doi.org/10.5194/hess-26-4013-2022

Ni, K., Ramanathan, N., Chehade, M. N. H., Balzano, L., Nair, S., Zahedi, S., et al. (2009). Sensor network data fault types. ACM Transactions on Sensor Networks, 5(3), 1-29. https://doi.org/10.1145/1525856.1525863

Nie, Y., Nguyen, N. H., Sinthong, P. & Kalagnanam, J. (2023). A time series is worth 64 words: Long-term forecasting with transformers. International Conference on Learning Representations (ICLR).

Oreshkin, B. N., Carpov, D., Chapados, N. & Bengio, Y. (2020). N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. International Conference on Learning Representations (ICLR).

Pagano, T. C., Wood, A. W., Ramos, M. H., Cloke, H. L., Pappenberger, F., Clark, M. P., et al. (2014). Challenges of Operational River Forecasting. Journal of Hydrometeorology, 15(4), 1692-1707. https://doi.org/10.1175/jhm-d-13-0188.1

Pappenberger, F., Cloke, H. L., Parker, D. J., Wetterhall, F., Richardson, D. S. & Thielen, J. (2015). The monetary benefit of early flood warnings in Europe. Environmental Science & Policy, 51, 278-291. https://doi.org/10.1016/j.envsci.2015.04.016

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems 32 (NeurIPS), 8024-8035.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.

Politis, D. N. & Romano, J. P. (1994). The Stationary Bootstrap. Journal of the American Statistical Association, 89(428), 1303-1313. https://doi.org/10.1080/01621459.1994.10476870

Poslad, S., Middleton, S. E., Chaves, F., Tao, R., Necmioglu, O. & Bugel, U. (2015). A Semantic IoT Early Warning System for Natural Environment Crisis Management. IEEE Transactions on Emerging Topics in Computing, 3(2), 246-257. https://doi.org/10.1109/tetc.2015.2432742

Rai, R. K., van den Homberg, M. J., Ghimire, G. P. & McQuistan, C. (2020). Cost-benefit analysis of flood early warning system in the Karnali River Basin of Nepal. International Journal of Disaster Risk Reduction, 47, 101534. https://doi.org/10.1016/j.ijdrr.2020.101534

Raissi, M., Perdikaris, P. & Karniadakis, G. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707. https://doi.org/10.1016/j.jcp.2018.10.045

Rajapakse, V., Karunanayake, I. & Ahmed, N. (2023). Intelligence at the Extreme Edge: A Survey on Reformable TinyML. ACM Computing Surveys, 55(13s), 1-30. https://doi.org/10.1145/3583683

Rasiya Koya, S. & Roy, T. (2024). Temporal Fusion Transformers for streamflow Prediction: Value of combining attention with recurrence. Journal of Hydrology, 637, 131301. https://doi.org/10.1016/j.jhydrol.2024.131301

Ray, P. P., Mukherjee, M. & Shu, L. (2017). Internet of Things for Disaster Management: State-of-the-Art and Prospects. IEEE Access, 5, 18818-18835. https://doi.org/10.1109/access.2017.2752174

Reichstein, M., Camps-Valls, G., Stevens, B., Jung, M., Denzler, J., Carvalhais, N., et al. (2019). Deep learning and process understanding for data-driven Earth system science. Nature, 566(7743), 195-204. https://doi.org/10.1038/s41586-019-0912-1

Richardson, D. S. (2000). Skill and relative economic value of the ECMWF ensemble prediction system. Quarterly Journal of the Royal Meteorological Society, 126(563), 649-667. https://doi.org/10.1002/qj.49712656313

Roebber, P. J. (2009). Visualizing Multiple Measures of Forecast Quality. Weather and Forecasting, 24(2), 601-608. https://doi.org/10.1175/2008waf2222159.1

Romano, Y., Patterson, E. & Candès, E. (2019). Conformalized quantile regression. Advances in Neural Information Processing Systems 32 (NeurIPS).

Roudbari, N. S., Punekar, S. R., Patterson, Z., Eicker, U. & Poullis, C. (2024). From data to action in flood forecasting leveraging graph neural networks and digital twin visualization. Scientific Reports, 14(1). https://doi.org/10.1038/s41598-024-68857-y

Saito, T. & Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

Salinas, D., Flunkert, V., Gasthaus, J. & Januschowski, T. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. International Journal of Forecasting, 36(3), 1181-1191. https://doi.org/10.1016/j.ijforecast.2019.07.001

Satyanarayanan, M. (2017). The Emergence of Edge Computing. Computer, 50(1), 30-39. https://doi.org/10.1109/mc.2017.9

Schaefer, J. T. (1990). The Critical Success Index as an Indicator of Warning Skill. Weather and Forecasting, 5(4), 570-575. https://doi.org/10.1175/1520-0434(1990)005<0570:tcsiaa>2.0.co;2

Sharma, A. B., Golubchik, L. & Govindan, R. (2010). Sensor faults: Detection methods and prevalence in real-world datasets. ACM Transactions on Sensor Networks, 6(3), 1-39. https://doi.org/10.1145/1754414.1754419

Shi, W., Cao, J., Zhang, Q., Li, Y. & Xu, L. (2016). Edge Computing: Vision and Challenges. IEEE Internet of Things Journal, 3(5), 637-646. https://doi.org/10.1109/jiot.2016.2579198

Sit, M., Demiray, B. Z., Xiang, Z., Ewing, G. J., Sermet, Y. & Demir, I. (2020). A comprehensive review of deep learning applications in hydrology and water resources. Water Science and Technology, 82(12), 2635-2670. https://doi.org/10.2166/wst.2020.369

Sit, M., Demiray, B. & Demir, I. (2021). Short-term hourly streamflow prediction with graph convolutional GRU networks. arXiv preprint arXiv:2107.07039.

Smith, L. N. & Topin, N. (2019). Super-convergence: very fast training of neural networks using large learning rates. Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications, 36. https://doi.org/10.1117/12.2520589

Sorensen, J. H. (2000). Hazard Warning Systems: Review of 20 Years of Progress. Natural Hazards Review, 1(2), 119-125. https://doi.org/10.1061/(asce)1527-6988(2000)1:2(119)

Tao, F., Zhang, H., Liu, A. & Nee, A. Y. C. (2019). Digital Twin in Industry: State-of-the-Art. IEEE Transactions on Industrial Informatics, 15(4), 2405-2415. https://doi.org/10.1109/tii.2018.2873186

Teh, H. Y., Kempa-Liehr, A. W. & Wang, K. I. K. (2020). Sensor data quality: a systematic review. Journal of Big Data, 7(1). https://doi.org/10.1186/s40537-020-0285-1

Tellman, B., Sullivan, J. A., Kuhn, C., Kettner, A. J., Doyle, C. S., Brakenridge, G. R., et al. (2021). Satellite imaging reveals increased proportion of population exposed to floods. Nature, 596(7870), 80-86. https://doi.org/10.1038/s41586-021-03695-w

Todini, E. (2007). A mass conservative and water storage consistent variable parameter Muskingum-Cunge approach. Hydrology and Earth System Sciences, 11(5), 1645-1659. https://doi.org/10.5194/hess-11-1645-2007

Trainor, J. E., Nagele, D., Philips, B. & Scott, B. (2015). Tornadoes, Social Science, and the False Alarm Effect. Weather, Climate, and Society, 7(4), 333-352. https://doi.org/10.1175/wcas-d-14-00052.1

Tripathy, K. P. & Mishra, A. K. (2024). Deep learning in hydrology and water resources disciplines: concepts, methods, applications, and research directions. Journal of Hydrology, 628, 130458. https://doi.org/10.1016/j.jhydrol.2023.130458

Verkade, J. S. & Werner, M. G. F. (2011). Estimating the benefits of single value and probability forecasting for flood warning. Hydrology and Earth System Sciences, 15(12), 3751-3765. https://doi.org/10.5194/hess-15-3751-2011

Wilcoxon, F. (1945). Individual Comparisons by Ranking Methods. Biometrics Bulletin, 1(6), 80. https://doi.org/10.2307/3001968

Wilks, D. S. (2001). A skill score based on economic value for probability forecasts. Meteorological Applications, 8(2), 209-219. https://doi.org/10.1017/s1350482701002092

Willard, J., Jia, X., Xu, S., Steinbach, M. & Kumar, V. (2023). Integrating Scientific Knowledge with Machine Learning for Engineering and Environmental Systems. ACM Computing Surveys, 55(4), 1-37. https://doi.org/10.1145/3514228

Wu, Z., Pan, S., Long, G., Jiang, J. & Zhang, C. (2019). Graph WaveNet for Deep Spatial-Temporal Graph Modeling. Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, 1907-1913. https://doi.org/10.24963/ijcai.2019/264

Xu, C. & Xie, Y. (2021). Conformal prediction interval for dynamic time-series. Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR, 139.

Zaffran, M., Féron, O., Goude, Y., Josse, J. & Dieuleveut, A. (2022). Adaptive conformal predictions for time series. Proceedings of the 39th International Conference on Machine Learning (ICML), PMLR, 162.

Zeng, A., Chen, M., Zhang, L. & Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting?. Proceedings of the AAAI Conference on Artificial Intelligence, 37(9), 11121-11128. https://doi.org/10.1609/aaai.v37i9.26317

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. Proceedings of the AAAI Conference on Artificial Intelligence, 35(12), 11106-11115. https://doi.org/10.1609/aaai.v35i12.17325

Zounemat-Kermani, M., Matta, E., Cominola, A., Xia, X., Zhang, Q., Liang, Q., et al. (2020). Neurocomputing in surface water hydrology and hydraulics: A review of two decades retrospective, current status and future prospects. Journal of Hydrology, 588, 125085. https://doi.org/10.1016/j.jhydrol.2020.125085