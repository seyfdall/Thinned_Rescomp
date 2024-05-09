#hello.py

# System paths to mpi4py
# pip - /home/seyfdall/.conda/envs/reservoir/lib/python3.12/site-packages
# mamba reservoir - /home/seyfdall/.conda/envs/reservoir

# Module avail

from mpi4py import MPI
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

print("Hello world! I'm process number {}.".format(RANK))