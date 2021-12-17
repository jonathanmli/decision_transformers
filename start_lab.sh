#!/usr/bin/env bash

eval "$($HOME/mc3/bin/conda 'shell.bash' 'hook')"
conda activate ml
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home-nfs/doctorduality/.mujoco/mujoco210/bin
unset XDG_RUNTIME_DI
export NODEIP=$(hostname -i)
export NODEPORT=$(( $RANDOM + 1024 ))
echo $NODEIP:$NODEPORT
jupyter-lab --ip=$NODEIP --port=$NODEPORT --no-browser
