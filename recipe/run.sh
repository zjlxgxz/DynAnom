#/bin/bash

#### Graph-level anomaly experiment
python -u  ./src/algorithms/DynAnom/exp_DynAnom_graph_level.py -dataset_name darpa -detect_anomaly -track_mode active -top_index 100 -make_multi_graph -dim 1024 -push_threshold 0.0001
python -u  ./src/algorithms/DynAnom/exp_DynAnom_graph_level.py -dataset_name enron -detect_anomaly -track_mode active  -push_threshold 0.0001  -top_index 100 -make_multi_graph

#### Node-level anomaly localization experiment 
python -u ./src/algorithms/DynAnom/exp_DynAnom_nodel_level.py -dataset_name darpa -detect_anomaly -track_mode definitive -top_index 200  -make_multi_graph -dim 1024 -push_threshold 0.0001
python -u ./src/algorithms/DynAnom/exp_DynAnom_nodel_level.py -dataset_name eucore_inject_1 -detect_anomaly -track_mode definitive -top_index 200  -make_multi_graph -dim 1024 -push_threshold 0.0001
python -u ./src/algorithms/DynAnom/exp_DynAnom_nodel_level.py -dataset_name eucore_inject_3 -detect_anomaly -track_mode definitive -top_index 200  -make_multi_graph -dim 1024 -push_threshold 0.0001

#### For Case-study 
python -u -m cProfile -s time ./src/algorithms/DynAnom/exp_DynAnom_case_study.py -dataset_name PEGraph-1980-2022-degcut-5 -write_snapshot_ppvs -track_mode definitive -alpha 0.7 -push_threshold 0.0 -make_multi_graph 
python -u -m cProfile -s time ./src/algorithms/DynAnom/exp_DynAnom_case_study.py -dataset_name PEGraph-2000-2022-degcut-5 -write_snapshot_ppvs -track_mode definitive -alpha 0.5 -push_threshold 0.0 -make_multi_graph
