"""
Import Statements
"""

import rescomp as rc
import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence
import time
# Set seed for reproducibility
# np.random.seed(1)
import traceback
import logging
import importlib

"""
Import Inhouse Rescomp
"""
import sys
import os
sys.path.insert(0, os.path.abspath('/nobackup/autodelete/usr/seyfdall/network_theory/rescomp_package/rescomp'))
import ResComp
import chaosode


"""
Import Helper functions
"""
from metrics import consistency_analysis, vpt_time, div_metric_tests
from file_io import HDF5FileHandler, GroupIOHandler, create_rescomp_datasets_template, generate_rescomp_means
from helper import get_orbit

"""
Perform the Gridsearch
"""

def rescomp_parallel_gridsearch_h5(
        erdos_possible_combinations, 
        rho,
        p_thin,
        system='lorenz', 
        draw_count=50, 
        tf=144000, 
        hdf5_file_path="results/erdos_results.h5"
    ):
    """ Run the gridsearch over possible combinations """

    # GET TRAINING AND TESTING SIGNALS
    t_train, U_train, t_test, U_test = get_orbit(duration=50, system=system, switch=10)
    tol = 5.
    
    # Parameters
    t0 = time.time()
    # Create a new file_handler
    file_handler = HDF5FileHandler(hdf5_file_path, rho=rho, p_thin=p_thin) # , network="erdos_c", network_param=erdos_c, gamma=gamma, sigma=sigma, alpha=alpha)
    with file_handler:

        # Cycle through every combination
        for i in range(len(erdos_possible_combinations)):

            n, erdos_c, gamma, sigma, alpha = erdos_possible_combinations[i]

            # Run the reservoir on this parameter set draw_count number of times
            for j in range(draw_count):

                # Check time and break if out of time
                t1 = time.time()
                if t1 - t0 > tf:
                    print("Break in Combo")
                    file_handler.save_attrs()
                    file_handler.close_file()
                    return

                # Template for datasets
                datasets = create_rescomp_datasets_template()

                    
                try:
                    # Generate thinned networks
                    mean_degree = erdos_c*(1-p_thin)
                    if mean_degree < 0.0:
                        mean_degree = 0.0
                    
                    res_thinned = ResComp.ResComp(res_sz=n, mean_degree=mean_degree, 
                                                ridge_alpha=alpha, spect_rad=rho, sigma=sigma, 
                                                gamma=gamma, map_initial='activ_f')       

                    # Compute Consistency Metric
                    # First replica run
                    r0_1 = np.random.uniform(-1., 1., n)
                    states_1 = res_thinned.internal_state_response(t_train, U_train, r0_1)

                    # Second replica run
                    r0_2 = np.random.uniform(-1., 1., n)
                    states_2 = res_thinned.internal_state_response(t_train, U_train, r0_2)

                    cap = consistency_analysis(states_1, states_2)[0]

                    # Train the matrix         
                    res_thinned.train(t_train, U_train)

                    # Forecast and compute the vpt along with diversity metrics
                    U_pred = res_thinned.predict(t_test, r0=res_thinned.r0, return_states=True)[0]
                    error = np.linalg.norm(U_test - U_pred, axis=1)
                    vpt = vpt_time(t_test, U_test, U_pred, vpt_tol=tol)
                    divs = div_metric_tests(res_thinned.states)

                    # t_curr = time_comp(t_curr, f"Predict, vpt, divs thinned")

                    # Store results
                    datasets['div_pos'].append(divs[0])
                    datasets['div_der'].append(divs[1])
                    datasets['pred'].append(U_pred)
                    datasets['err'].append(error)
                    datasets['vpt'].append(vpt)
                    datasets['consistency_correlation'].append(cap)

                    mean_attrs = generate_rescomp_means(datasets)

                    # t_curr = time_comp(t_curr, f"Store results thinned")

                except ArpackNoConvergence: # Occasionally sparse linalg eigs isn't able to converge
                    j = j-1
                    print("ArpackNoConvergence Error Caught")
                    continue
                except OverflowError: # Solving for W_out hits overflow errors with high spectral radius and high p_thin
                    j = j-1
                    print("Overflow Error Caught")
                    continue
                except ValueError as err:
                    j = j-1
                    print(rho, p_thin, erdos_c, str(err))
                    print(n, erdos_c*(1-p_thin), erdos_c*(1-p_thin) / n)
                    traceback.print_exc()  # This will print the stack trace
                    continue
                except Exception as e:
                    j = j-1
                    print("General Error")
                    logging.error(traceback.format_exc())
                    continue


                # Get the current group and save the data
                group_handler = file_handler.get_group_handler(f"param_set_{j}", n=n, erdos_c=erdos_c, gamma=gamma, sigma=sigma, alpha=alpha)
                group_handler.add_attrs(**mean_attrs)
                group_handler.add_datasets(**datasets)
                group_handler.save_data()


        file_handler.save_attrs()