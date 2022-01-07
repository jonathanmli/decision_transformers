#!/bin/bash
eval "$($HOME/mc3/bin/conda 'shell.bash' 'hook')"
conda activate ml2
python3 /share/data/mei-work/jolly/decision_transformers/sql/jonathans_experiment.py --max_iters 1 --env $1 --dataset $2 --seed $3 --num_steps_per_iter $4 -lr 0.0001 --warmup_steps $4 --model_type dt 
