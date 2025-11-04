#!/bin/bash --login
#SBATCH --job-name=reservoir_compute
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --mail-user=dallin.seyfried@mathematics.byu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=1

echo "Running SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "RHO_P_THIN_SET=$RHO_P_THIN_SET"
echo "PARAM_SET=$PARAM_SET"
echo "PARAM_NAME=$PARAM_NAME"
echo "PARAM_VALUE=$PARAM_VALUE"

# Load Conda properly in non-interactive shell
# source ~/anaconda3/etc/profile.d/conda.sh
# 10G for 200, 20G for 400, 30G for 600, 40G for 800 (about 16 - need to up time by a lot), 50G not enough for 1000
conda activate reservoir

module load openmpi/4.1.6-fgmxkt2
export OMPI_MCA_gds=^shmem2

CHUNK_SIZE=10
START=$((SLURM_ARRAY_TASK_ID))
END=$((START + CHUNK_SIZE - 1))

for (( i=START; i<=END; i++ )); do
    echo $i
    export ID_TO_PROCESS=$i
    python3 main.py -r "$RHO_P_THIN_SET" -p "$PARAM_VALUE" -p_name "$PARAM_NAME" -p_set "$PARAM_SET"
done