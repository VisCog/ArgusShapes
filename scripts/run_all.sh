#!/bin/bash
python crossval-swarm.py D2 52-001 --w_scale=20 --w_rot=0 --w_dice=80 --n_folds=-1 --avg_img
python crossval-swarm.py D2 51-009 --w_scale=20 --w_rot=0 --w_dice=80 --n_folds=-1 --avg_img
python crossval-swarm.py C2 TB --w_scale=20 --w_rot=0 --w_dice=80 --n_folds=-1 --avg_img
python crossval-swarm.py C2 12-005 --w_scale=20 --w_rot=0 --w_dice=80 --n_folds=-1 --avg_img
python crossval-swarm.py C2 52-001 --w_scale=20 --w_rot=0 --w_dice=80 --n_folds=-1 --avg_img
python crossval-swarm.py C2 51-009 --w_scale=20 --w_rot=0 --w_dice=80 --n_folds=-1 --avg_img
