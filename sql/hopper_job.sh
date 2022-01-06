#!/bin/bash
eval "$($HOME/mc3/bin/conda 'shell.bash' 'hook')"
conda activate ml2
python3 jonathans_experiment.py --max_iters 10 --env hopper --seed 1 --num_steps_per_iter 10000 -lr 0.0001 --warmup_steps 10000 --dataset expert --model_type dt 
