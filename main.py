import itertools
import numpy as np
from mpi4py import MPI


import sys
import os
sys.path.insert(0, os.path.abspath('/mnt/c/Users/dseyf/SeniorLabs/Research/Network_Theory/thinned_rescomp/utils/'))
import utils.helper as helper
import utils.driver as driver

"""
Main Method to call the Gridsearch
"""

def main():
    rho_p_thin_prod, erdos_possible_combinations = helper.gridsearch_parameter_setup()

    rho_p_thin_prod = np.array([[2.,0.1]])
    erdos_possible_combinations = [[50, 4, 0.5, 0.5, 1.0]]

    n, m = rho_p_thin_prod.shape

    rhos = [2.,3.]
    p_thins = [0.1,0.5]
    rho_p_thin_prod = list(itertools.product(rhos, p_thins))
    n, m = len(rhos), len(p_thins)

    if n == 1 and m == 2:
        rho, p_thin = rho_p_thin_prod[0]

    else:
        # Setup the parallelization
        SIZE = MPI.COMM_WORLD.Get_size()
        if SIZE != n*m:
            print(f"Number of processes expected: {n*m}, received: {SIZE}")
            return()
        
        # Split the Erdos_c exploration according to RANK
        RANK = MPI.COMM_WORLD.Get_rank()
        print(RANK)

        rho, p_thin = rho_p_thin_prod[RANK]

    results_path = '/mnt/c/Users/dseyf/SeniorLabs/Research/Network_Theory/thinned_rescomp/'
    driver.rescomp_parallel_gridsearch_h5(
        erdos_possible_combinations, 
        rho,
        p_thin,
        draw_count=2, 
        hdf5_file_path=f'{results_path}results/erdos_results', 
        tf=21000
    )


if __name__ == "__main__":
    main()