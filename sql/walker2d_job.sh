#!/bin/bash
eval "$($HOME/mc3/bin/conda 'shell.bash' 'hook')"
conda activate ml2
python3 jonathans_experiment.py --env walker2d --max_iters 1 --seed 1 --num_steps_per_iter 40000 -lr 0.0001 --warmup_steps 40000 --dataset expert --model_type dt 