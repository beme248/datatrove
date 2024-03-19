#!/bin/bash
#SBATCH --job-name=document-stats
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --time=12:00:00

# create the log dir
DIR=$(realpath .)
mkdir -p $DIR/logs

# call the project main with all the arguments
command="
conda init
conda activate /users/bmessmer/conda-envs/datatrove
python scripts/document_count_estimation.py
"

# submit the job to slurm
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
srun -u --output=$DIR/logs/%x_%j_$DATETIME.log bash -c "${command}"