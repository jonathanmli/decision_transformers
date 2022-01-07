#!/bin/bash
STEPS=10000
for ENVI in hopper halfcheetah walker2d
do 
for DATASET in medium medium-replay expert
do
for SEED in 1 2 3 4 5 6 7 8 9 10
do 
sbatch -p gpu -c1 -C 11g gym_job.sh $ENVI $DATASET $SEED $STEPS
done 
done
done