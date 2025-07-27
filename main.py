from mpi4py import MPI
import argparse
from datetime import date


import sys
import os
sys.path.insert(0, os.path.abspath(f'{os.getcwd()}/utils/'))
import utils.helper as helper
import utils.driver as driver
import utils.visualization as viz

"""
Main Method to call the Gridsearch
"""

def main():
    param, param_name, param_set = helper.parse_arguments()

    rho_p_thin_prod, erdos_possible_combinations = helper.generate_params(
        param=param, 
        param_name=param_name
    )

    n, _ = rho_p_thin_prod.shape

    if n == 1:
        rho, p_thin = rho_p_thin_prod[0]
    else:
        job_id_number = int(os.getenv('ID_TO_PROCESS'))
        print(job_id_number)
        rho, p_thin = rho_p_thin_prod[job_id_number]

    cwd = os.getcwd()
    results_path = f'{cwd}/results/{param_name}/{param}/{param_set}/'

    driver.rescomp_parallel_uniform_gridsearch_h5(
        erdos_possible_combinations, 
        rho,
        p_thin,
        draw_count=100000, 
        hdf5_file_path=results_path, 
        tf=1200
    )


if __name__ == "__main__":
    main()