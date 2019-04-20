#!/bin/bash
# Run each fold separately:
models="Scoreboard AxonMap"
for model in ${models}
do
	for fold in {0..11}
	do
		python crossval-swarm.py ${model} S1 --n_folds=-1 --idx_fold=${fold} --n_jobs=1
	done
done

for model in ${models}
do
	for fold in {0..21}
	do
		python crossval-swarm.py ${model} S2 --n_folds=-1 --idx_fold=${fold} --n_jobs=1
	done
done

for model in ${models}
do
	for fold in {0..17}
	do
		python crossval-swarm.py ${model} S3 --n_folds=-1 --idx_fold=${fold} --n_jobs=1
	done
done

for model in ${models}
do
	for fold in {0..27}
	do
		python crossval-swarm.py ${model} S4 --n_folds=-1 --idx_fold=${fold} --n_jobs=1
	done
done
