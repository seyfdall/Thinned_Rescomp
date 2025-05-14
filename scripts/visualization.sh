#!/bin/bash --login

#SBATCH --time=0:10:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=1024M   # memory per CPU core
#SBATCH -J "Reservoir_Visualization"   # job name
#SBATCH --output=./results/visualization.txt
#SBATCH --mail-user=dallin.seyfried@mathematics.byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
cd /nobackup/autodelete/usr/seyfdall/network_theory/thinned_rescomp
mamba activate reservoir
python3 utils/visualization.py