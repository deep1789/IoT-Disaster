export PYTHONPATH=src NT=2 EP=8 NPE=200000
for s in 0 1 2; do python3 src/run_exp.py temporal full $s; done
python3 src/run_exp.py temporal noroute 0; python3 src/run_exp.py temporal noaug 0
for r in MidAtlantic OhioValley TexasHill UpperMidwest; do python3 src/run_exp.py loro:$r full 0; done
