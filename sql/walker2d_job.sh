#!/bin/bash
eval "$($HOME/mc3/bin/conda 'shell.bash' 'hook')"
conda activate ml
python3 jonathans_experiment.py --env walker2d --num_steps_per_iter 10000 -lr 0.0001 --warmup_steps 10000 --dataset medium --model_type dt 