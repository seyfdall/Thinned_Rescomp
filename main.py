import sys
import os
sys.path.insert(0, os.path.abspath(f'{os.getcwd()}/utils/'))
import utils.helper as helper
import utils.driver as driver

"""
Main Method to call the Gridsearch
"""

def main():
    rho_p_thin_set, param, param_name, param_set = helper.parse_arguments()

    rho_p_thin_prod, erdos_possible_combinations = helper.generate_params(
        rho_p_thin_set,
        param=param, 
        param_name=param_name,
        param_set=param_set
    )

    n, _ = rho_p_thin_prod.shape

    if n == 1:
        rho, p_thin = rho_p_thin_prod[0]
    else:
        job_id_number = int(os.getenv('ID_TO_PROCESS'))
        print(job_id_number)
        rho, p_thin = rho_p_thin_prod[job_id_number]
        
    home = os.path.expanduser("~")
    results_path = f'{home}/nobackup/autodelete/results/{param_name}/{param}/{param_set}/{rho_p_thin_set}/'

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