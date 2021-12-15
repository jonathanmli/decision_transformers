#!/usr/bin/env zsh

module load python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jmzhl/.mujoco/mujoco210/bin
conda activate jjj
IPAD=$(/sbin/ip route get 8.8.8.8 | awk '{print $NF;exit}')
jupyter-lab --no-browser --ip=$IPAD
