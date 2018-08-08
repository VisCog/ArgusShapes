#!/bin/bash
# Fit the scoreboard model:
python crossval-swarm.py A S1 --n_folds=1 --n_jobs=-1
python crossval-swarm.py A S2 --n_folds=1 --n_jobs=-1
python crossval-swarm.py A S3 --n_folds=1 --n_jobs=-1
python crossval-swarm.py A S4 --n_folds=1 --n_jobs=-1

# Fit the axon map model:
python crossval-swarm.py C S1 --n_folds=1 --n_jobs=-1
python crossval-swarm.py C S2 --n_folds=1 --n_jobs=-1
python crossval-swarm.py C S3 --n_folds=1 --n_jobs=-1
python crossval-swarm.py C S4 --n_folds=1 --n_jobs=-1
