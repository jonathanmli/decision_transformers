#!/bin/bash
eval "$($HOME/mc3/bin/conda 'shell.bash' 'hook')"
conda activate ml
python3 jonathans_experiment.py --env hopper --num_steps_per_iter 100000 -lr 0.0001 --warmup_steps 100000 --dataset medium --model_type dt 
