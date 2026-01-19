#!/bin/bash --login

#SBATCH --time=00:30:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=2048M   # memory per CPU core
#SBATCH -J "Reservoir_Visualization"   # job name
#SBATCH --output=./results/visualization.log
#SBATCH --mail-user=dallin.seyfried@mathematics.byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mamba activate reservoir
python3 utils/visualization.py -r $RHO_P_THIN_SET -p $PARAM_VALUE -p_name $PARAM_NAME -p_set $PARAM_SET