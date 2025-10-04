"""
Import Statements
"""

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence
import time
import traceback
import logging
import sys
import os
import signal

"""
Import Inhouse Rescomp
"""
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "rescomp", "rescomp")))
import ResComp
"""
Import Helper functions
"""
from metrics import vpt_time, div_metric_tests, consistency_analysis_pearson
from file_io import HDF5FileHandler, create_rescomp_datasets_template, generate_rescomp_means
from helper import get_orbit


# Global flag to stop gracefully
stop_now = False

def handle_sigterm(signum, frame):
    global stop_now
    print("Received SIGTERM (job cancelled). Cleaning up...")
    stop_now = True

# def handle_usr1(signum, frame):
#     global stop_now
#     print("Received SIGUSR1 (timeout warning). Preparing to stop...")
#     stop_now = True


# Register signal handlers
signal.signal(signal.SIGTERM, handle_sigterm)  # scancel
# signal.signal(signal.SIGUSR1, handle_usr1)     # timeout warning (needs #SBATCH --signal)


def drive_reservoir_analysis(
        tol,
        t_train,
        t_test,
        U_train,
        U_test,
        rho,
        p_thin,
        param_set
    ):
    """Inner for loop work here - run a single reservoir and perform analysis"""

    print("param_set:", param_set)

    n, erdos_c, gamma, sigma, alpha = param_set

    # Template for datasets
    datasets = create_rescomp_datasets_template()

    # Generate thinned networks
    mean_degree = erdos_c*(1-p_thin)
    if mean_degree < 0.0:
        mean_degree = 0.0
    
    # TODO: Run Correlation tests with diversities with VPT try averaging around p_thin squishing things down
    res_thinned = ResComp.ResComp(res_sz=n, mean_degree=mean_degree, 
                                ridge_alpha=alpha, spect_rad=rho, sigma=sigma, 
                                gamma=gamma, map_initial='activ_f')       

    print("First Replica Run")
    # Compute Consistency Metric
    # First replica run
    r0_1 = np.random.uniform(-1., 1., n)
    states_1 = res_thinned.internal_state_response(t_train, U_train, r0_1)

    print("Second Replica Run")
    # Second replica run
    # r0_2 = np.random.uniform(0., 1., n) * np.sign(r0_1)
    r0_2 = np.random.uniform(-1., 1., n)
    states_2 = res_thinned.internal_state_response(t_train, U_train, r0_2)

    cap = consistency_analysis_pearson(states_1.T, states_2.T)

    print("Train")
    # Train the matrix         
    res_thinned.train(t_train, U_train)

    print("Forecast and predict")
    # Forecast and compute the vpt along with diversity metrics
    U_pred = res_thinned.predict(t_test, r0=res_thinned.r0, return_states=True)[0]
    error = np.linalg.norm(U_test - U_pred, axis=1)
    vpt = vpt_time(t_test, U_test, U_pred, vpt_tol=tol)
    divs = div_metric_tests(res_thinned.states)

    # t_curr = time_comp(t_curr, f"Predict, vpt, divs thinned")
    print("Divs:", divs)

    # Store results
    datasets['div_pos'].append(divs[0])
    datasets['div_der'].append(divs[1])
    datasets['div_spect'].append(divs[2])
    datasets['div_rank'].append(divs[3])
    datasets['pred'].append(U_pred)
    datasets['err'].append(error)
    datasets['vpt'].append(vpt)
    datasets['consistency_correlation'].append(cap)

    mean_attrs = generate_rescomp_means(datasets)

    print("Mean_attrs:", mean_attrs)

    return mean_attrs, datasets


"""
Uniform Sampling Gridsearch
"""

def rescomp_parallel_uniform_gridsearch_h5(
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
    file_handler = HDF5FileHandler(hdf5_file_path, rho=rho, p_thin=p_thin)
    with file_handler:

        # Run the reservoir on this parameter set draw_count number of times
        for i in range(draw_count):

            # Check time and break if out of time
            t1 = time.time()
            if (t1 - t0 > tf) or stop_now:
                print("Break in Combo. Signal received?", stop_now)
                file_handler.save_attrs()
                file_handler.close_file()
                return

            param_set_index = np.random.choice(len(erdos_possible_combinations))
            n, erdos_c, gamma, sigma, alpha = erdos_possible_combinations[param_set_index]

            try:
                mean_attrs, datasets = drive_reservoir_analysis(tol, t_train, t_test, U_train, 
                                                                    U_test, rho, p_thin, erdos_possible_combinations[param_set_index])

            except ArpackNoConvergence as e: # Occasionally sparse linalg eigs isn't able to converge
                i = i-1
                tb = e.__traceback__
                print("ArpackNoConvergence Error Caught")
                print("\nFormatted Traceback:")
                traceback.print_tb(tb)
                print(f"Exception message: {e.args[0]}")
                continue
            except OverflowError: # Solving for W_out hits overflow errors with high spectral radius and high p_thin
                i = i-1
                print("Overflow Error Caught")
                continue
            except ValueError as err:
                i = i-1
                print(rho, p_thin, erdos_c, str(err))
                print(n, erdos_c*(1-p_thin), erdos_c*(1-p_thin) / n)
                traceback.print_exc()  # This will print the stack trace
                continue
            except Exception as e:
                i = i-1
                print("General Error")
                logging.error(traceback.format_exc())
                continue

            # Get the current group and save the data
            group_handler = file_handler.get_group_handler(f"set_{i}", n=n, erdos_c=erdos_c, gamma=gamma, sigma=sigma, alpha=alpha)
            group_handler.add_attrs(**mean_attrs)
            # group_handler.add_datasets(**datasets) # Caution: High Storage requirement to store datasets and generally not required for analysis
            group_handler.save_data()


        file_handler.save_attrs()