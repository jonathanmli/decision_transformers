#!/bin/bash
eval "$($HOME/mc3/bin/conda 'shell.bash' 'hook')"
conda activate ml
python3 jonathans_experiment.py --env hopper --num_steps_per_iter 1000 -lr 0.001 --warmup_steps 1000 --dataset medium --model_type dt 