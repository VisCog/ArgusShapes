#!/bin/bash
# Fit the scoreboard model:
python crossval-swarm.py Scoreboard S1 --n_folds=1 --n_jobs=-1
python crossval-swarm.py Scoreboard S2 --n_folds=1 --n_jobs=-1
python crossval-swarm.py Scoreboard S3 --n_folds=1 --n_jobs=-1
python crossval-swarm.py Scoreboard S4 --n_folds=1 --n_jobs=-1

# Fit the axon map model:
python crossval-swarm.py AxonMap S1 --n_folds=1 --n_jobs=-1
python crossval-swarm.py AxonMap S2 --n_folds=1 --n_jobs=-1
python crossval-swarm.py AxonMap S3 --n_folds=1 --n_jobs=-1
python crossval-swarm.py AxonMap S4 --n_folds=1 --n_jobs=-1
