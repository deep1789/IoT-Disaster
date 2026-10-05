# IoT-Disaster: PG-FaultTwin (fault-tolerant, physics-guided flood digital twin)

Manuscript for the EAAI special issue "IoT in Disaster Management" (VSI: EAAI_IoT Disaster Mgmt).
Built output: `manuscript/manuscript.docx` (and `.pdf`); markdown sources in `manuscript/`.

## Reproduce (4-core CPU, 15 GB RAM, no GPU)
Run from the repo root with `PYTHONPATH=src`:
1. `src/s01_inventory.py` station inventory (USGS NWIS site service + NOAA NWPS) 
2. `src/s02_download.py` 10 water years of 15-min stage/discharge; `src/s03a_thresholds.py` official flood stages
3. `src/s03_preprocess.py` QC, hourly aggregation, dead-sensor screen, coverage filter
4. `src/s04_stats.py` dataset statistics and topology-free prior diagnostics
5. `src/queueA.sh`, `src/queueB.sh` all models (`src/run_exp.py <protocol> <variant> <seed>`)
6. `src/s06_eval.py`, `src/s13_tables.py`, `src/s14_boot.py`, `src/s15_region.py` metrics, tables, bootstrap, regional comparison
7. `src/s07_faults.py`, `src/s07b_faults_noaug.py` fault injection; `src/s08_conformal.py`; `src/s10_edge.py`; `src/s11_attr.py`
8. `src/s09*_figs.py`, `src/s13b_faulttbl.py`, `src/s12_build.py` (docx), `src/s12b_pdf.py` (pdf)

Raw data and trained models are not committed (re-downloadable / re-trainable). Numeric outputs are in `results/`.
