"""
Import Statements
"""

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence
import time
import traceback
import logging
import signal

"""
Import Helper functions
"""
from file_io import HDF5FileHandler
from helper import get_orbit
from reservoir_workflows import run_single_reservoir_analysis


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
        network_type,
        rho,
        p_thin,
        param_set
    ):
    """Inner for loop work here - run a single reservoir and perform analysis"""
    run_result = run_single_reservoir_analysis(
        tol=tol,
        t_train=t_train,
        t_test=t_test,
        U_train=U_train,
        U_test=U_test,
        network_type=network_type,
        rho=rho,
        p_thin=p_thin,
        param_set=param_set,
    )

    return run_result.mean_attrs, run_result.datasets


"""
Uniform Sampling Gridsearch
"""

def rescomp_parallel_uniform_gridsearch_h5(
        network_type,
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
                mean_attrs, datasets = drive_reservoir_analysis(
                    tol, 
                    t_train, 
                    t_test, 
                    U_train, 
                    U_test,
                    network_type,
                    rho, 
                    p_thin, 
                    erdos_possible_combinations[param_set_index]
                )

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