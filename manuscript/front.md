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
