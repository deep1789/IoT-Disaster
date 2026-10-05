export PYTHONPATH=src NT=2 EP=8 NPE=200000
for s in 0 1 2; do python3 src/run_exp.py temporal gru $s; done
python3 src/run_exp.py temporal geo 0; python3 src/run_exp.py temporal own 0; python3 src/run_exp.py temporal evsamp 0
for r in MidAtlantic OhioValley TexasHill UpperMidwest; do python3 src/run_exp.py loro:$r gru 0; python3 src/run_exp.py loro:$r tf 0; python3 src/run_exp.py loro:$r lgbm 0; done
