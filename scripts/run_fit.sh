#!/bin/bash
python crossval-swarm.py A TB --n_folds=1 --avg_img
python crossval-swarm.py A 12-005 --n_folds=1 --avg_img
python crossval-swarm.py A 51-009 --n_folds=1 --avg_img
python crossval-swarm.py A 52-001 --n_folds=1 --avg_img
#
python crossval-swarm.py B TB --n_folds=1 --avg_img
python crossval-swarm.py B 12-005 --n_folds=1 --avg_img
python crossval-swarm.py B 51-009 --n_folds=1 --avg_img
python crossval-swarm.py B 52-001 --n_folds=1 --avg_img
#
python crossval-swarm.py C TB --n_folds=1 --avg_img
python crossval-swarm.py C 12-005 --n_folds=1 --avg_img
python crossval-swarm.py C 51-009 --n_folds=1 --avg_img
python crossval-swarm.py C 52-001 --n_folds=1 --avg_img
#
python crossval-swarm.py D TB --n_folds=1 --avg_img
python crossval-swarm.py D 12-005 --n_folds=1 --avg_img
python crossval-swarm.py D 51-009 --n_folds=1 --avg_img
python crossval-swarm.py D 52-001 --n_folds=1 --avg_img
#
#
python crossval-swarm.py A TB --n_folds=1 --avg_img --adjust_bias
python crossval-swarm.py A 12-005 --n_folds=1 --avg_img --adjust_bias
python crossval-swarm.py A 51-009 --n_folds=1 --avg_img --adjust_bias
python crossval-swarm.py A 52-001 --n_folds=1 --avg_img --adjust_bias
#
python crossval-swarm.py B TB --n_folds=1 --avg_img --adjust_bias
python crossval-swarm.py B 12-005 --n_folds=1 --avg_img --adjust_bias
python crossval-swarm.py B 51-009 --n_folds=1 --avg_img --adjust_bias
python crossval-swarm.py B 52-001 --n_folds=1 --avg_img --adjust_bias
#
python crossval-swarm.py C TB --n_folds=1 --avg_img --adjust_bias
python crossval-swarm.py C 12-005 --n_folds=1 --avg_img --adjust_bias
python crossval-swarm.py C 51-009 --n_folds=1 --avg_img --adjust_bias
python crossval-swarm.py C 52-001 --n_folds=1 --avg_img --adjust_bias
#
python crossval-swarm.py D TB --n_folds=1 --avg_img --adjust_bias
python crossval-swarm.py D 12-005 --n_folds=1 --avg_img --adjust_bias
python crossval-swarm.py D 51-009 --n_folds=1 --avg_img --adjust_bias
python crossval-swarm.py D 52-001 --n_folds=1 --avg_img --adjust_bias
