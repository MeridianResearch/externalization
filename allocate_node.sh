#!/bin/bash

# Default time is 1 hour if no argument is provided
HOURS=${1:-1}

srun --partition=dev-g \
     --nodes=1 \
     --gpus-per-node=1 \
     --time=$(printf "%02d:00:00" "$HOURS") \
     --account=project_465001340 \
     --pty bash -c "
         ml use /appl/local/csc/modulefiles/
         module load pytorch/2.7
         exec bash
     "