#!/bin/bash --login

#SBATCH --time=00:25:00   # walltime
#SBATCH --ntasks=171   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=2048M   # memory per CPU core # 8192, 6144, 4096, 2048
#SBATCH -J "Reservoir_Gridsearch"   # job name
#SBATCH --output=./results/mpi_gridsearch_test.txt
#SBATCH --mail-user=dallin.seyfried@mathematics.byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --qos=test


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
cd /nobackup/autodelete/usr/seyfdall/network_theory/thinned_rescomp
conda activate reservoir
module load openmpi/4.1.6-fgmxkt2
export OMPI_MCA_gds=^shmem2
mpirun -np 171 python3 main.py