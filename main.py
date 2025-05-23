import itertools
import numpy as np
from mpi4py import MPI


import sys
import os
sys.path.insert(0, os.path.abspath('/nobackup/autodelete/usr/seyfdall/network_theory/thinned_rescomp/utils/'))
import utils.helper as helper
import utils.driver as driver

"""
Main Method to call the Gridsearch
"""

def main():
    # rho_p_thin_prod, erdos_possible_combinations = helper.gridsearch_parameter_setup()
    rho_p_thin_prod, erdos_possible_combinations = helper.simple_params()

    n, m = rho_p_thin_prod.shape

    if n == 1:
        rho, p_thin = rho_p_thin_prod[0]
    else:
        # Setup the parallelization
        SIZE = MPI.COMM_WORLD.Get_size()
        if SIZE != n:
            print(f"Number of processes expected: {n}, received: {SIZE}")
            return()
        
        # Split the Erdos_c exploration according to RANK
        RANK = MPI.COMM_WORLD.Get_rank()
        print(RANK)

        rho, p_thin = rho_p_thin_prod[RANK]

    results_path = '/nobackup/autodelete/usr/seyfdall/network_theory/thinned_rescomp/'
    driver.rescomp_parallel_uniform_gridsearch_h5(
        erdos_possible_combinations, 
        rho,
        p_thin,
        draw_count=100000, 
        hdf5_file_path=f'{results_path}results/erdos_results', 
        tf=1200
    )


if __name__ == "__main__":
    main()