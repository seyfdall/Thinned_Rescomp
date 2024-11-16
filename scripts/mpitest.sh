#!/bin/bash --login

#SBATCH --time=00:20:00   # walltime
#SBATCH --ntasks=81   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=1024M   # memory per CPU core
#SBATCH -J "Reservoir_Gridsearch"   # job name
#SBATCH --output=./results/mpi_gridsearch_test.txt
#SBATCH --mail-user=dseyfr99@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --qos=test

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
cd /nobackup/autodelete/usr/seyfdall/network_theory/thinned_rescomp
mamba activate reservoir
# module load mpi/openmpi-1.10.7_gnu4.8.5
export MPICC=$(which mpicc)
mpirun -np 81 python3 rescomp_gridsearch.py