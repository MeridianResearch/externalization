srun --partition=dev-g --nodes=1 --gpus-per-node=1 --time=01:00:00 --account=project_465001340 --pty bash -c "
    ml use /appl/local/csc/modulefiles/
    module load pytorch/2.7
    exec bash
"