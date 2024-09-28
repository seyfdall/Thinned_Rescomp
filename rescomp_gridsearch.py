"""
Import Statements
"""

import rescomp as rc
import numpy as np
from scipy.interpolate import CubicSpline
from scipy import integrate, sparse
from scipy.stats import pearsonr
from scipy.sparse.linalg import eigs, ArpackNoConvergence
from scipy.sparse import coo_matrix
import math 
import networkx as nx
import itertools
import csv
import time
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [20, 5]
# Set seed for reproducibility
# np.random.seed(1)
from math import comb
import h5py
from mpi4py import MPI
import traceback
import logging


"""
Debugging Functions
"""
def time_comp(t0, message):
    t1 = time.time()
    print(f"{message}: {round(t1 - t0, 2)}")
    return t1



"""
Automating Tests
"""

def nrmse(true, pred):
        """ Normalized root mean square error. (A metric for measuring difference in orbits)
        Parameters:
            Two mxn arrays. Axis zero is assumed to be the time axis (i.e. there are m time steps)
        Returns:
            err (ndarray): Error at each time value. 1D array with m entries
        """
        # sig = np.std(true, axis=0)
        # err = np.linalg.norm((true-pred) / sig, axis=1, ord=2)
        err = np.linalg.norm((true-pred), axis=1, ord=2) # Just regular 2-norm
        return err

def valid_prediction_index(err, tol):
    """First index i where err[i] > tol. err is assumed to be 1D and tol is a float. If err is never greater than tol, then len(err) is returned."""
    mask = np.logical_or(err > tol, ~np.isfinite(err))
    if np.any(mask):
        return np.argmax(mask)
    return len(err)

def vpt_time(ts, Uts, pre, vpt_tol=5.):
    """
    Valid prediction time for a specific instance.
    """
    err = nrmse(Uts, pre)
    idx = valid_prediction_index(err, vpt_tol)
    if idx == 0:
        vptime = 0.
    else:
        vptime = ts[idx-1] - ts[0]
    return vptime

def div_metric_tests(preds, T, n):
    """ Compute Diversity scores of predictions
    """
    # Take the derivative of the pred_states
    res_deriv = np.gradient(preds[:T], axis=0)

    # Run the metric for the old and new diversity scores
    div_pos = 0
    div_der = 0
    for i in range(n):
        for j in range(n):
            div_pos += np.sum(np.abs(np.abs(preds[:T, i]) - np.abs(preds[:T, j])))
            div_der += np.sum(np.abs(res_deriv[:T, i] - res_deriv[:T, j]))
    denom = T*comb(n,2)
    div_pos = div_pos / denom
    div_der = div_der / denom

    return div_pos, div_der

def remove_edges(A,n_edges):
    """ Randomly removes 'n_edges' edges from a sparse matrix 'A'
    """
    B = A.copy().todok() # - - - - - - - -  set A as copy

    keys = list(B.keys()) # - - - - remove edges
   
    remove_idx = np.random.choice(range(len(keys)),size=n_edges, replace=False)
    remove = [keys[i] for i in remove_idx]
    for e in remove:
        B[e] = 0
    return B

def pearson_consistency_metric(states, states_perturbed):
    """ Compute the consistency metric for predicted states based on the Echo State Paper - Pearson Correlation Coefficient 
        Parameters:
        ----------
        states: ndarray(T,n)
            States using unperturbed initial state r0. T = time, n = number of nodes

        states_perturbed: ndarray(T,n)
            Perturbed states using perturbed initial state r0. T = time, n = number of nodes

        Returns:
        --------
        aggregated_pearson_correlation_coeff: float 
            The aggregated pearson correlation coefficient between the two response states
    """
    if len(states.shape) == 1:
        return pearsonr(states, states_perturbed)[0]
    else:
        T, n = states.shape
        gammas = np.zeros(n)
        for i in range(n):
            gammas[i] = pearsonr(states[:,i], states_perturbed[:,i])[0]
        # pearsonr is not defined for constant state vectors so nan will be returned - remove these from the array
        aggregated_pearson_correlation_coeff = np.mean(gammas[np.isfinite(gammas)])
        return aggregated_pearson_correlation_coeff



"""
Gridsearch Parameter Setup
"""

def gridsearch_uniform_dict_setup():
    # Topological Parameters
    # ns = [500, 1500, 2500]
    ns = [500]
    p_thins = np.concatenate((np.arange(0, 0.8, 0.1), np.arange(0.8, 1.01, 0.02)))
    # p_thins = [0.1, 0.5]

    # Model Specific Parameters
    # erdos_renyi_c = [0.5, 1, 2, 3, 4]
    erdos_renyi_c = [4]
    # random_digraph_c = [.5,1,2,3,4]
    # random_geometric_c = [.5,1,2,3,4]
    # barabasi_albert_m = [1,2]
    # watts_strogatz_k = [2,4]
    # watt_strogatz_q = [.01,.05,.1]

    # Reservoir Computing Parameters
    gammas = [0.1,0.5,1,2,5,10,25,50]
    rhos = [0.1,0.9,1.0,1.1,2.0,5.0,10.0,25.0,50.0]
    # rhos = [5.0, 10.0]
    sigmas = [1e-3,5e-3,1e-2,5e-2,.14,.4,.7,1,10]
    alphas = [1e-8,1e-6,1e-4,1e-2,1]

    erdos_possible_combinations = list(itertools.product(ns, erdos_renyi_c, gammas, sigmas, alphas))
    # digraph_possible_combinations = list(itertools.product(ns, p_thins, random_digraph_c, gammas, rhos, sigmas, alphas))
    # geometric_possible_combinations = list(itertools.product(ns, p_thins, random_geometric_c, gammas, rhos, sigmas, alphas))
    # barabasi_possible_combinations = list(itertools.product(ns, p_thins, barabasi_albert_m, gammas, rhos, sigmas, alphas))
    # strogatz_possible_combinations = list(itertools.product(ns, p_thins, watts_strogatz_k, watt_strogatz_q, gammas, rhos, sigmas, alphas))
    return rhos, p_thins, erdos_possible_combinations


"""
Perform the Gridsearch
"""

def rescomp_parallel_gridsearch_uniform_thinned_h5(
        erdos_possible_combinations, 
        system='lorenz', 
        draw_count=10000, 
        tf=144000, 
        hdf5_file="results/erdos_results.h5", 
        rho=0.1, 
        p_thin=0.0
    ):
    """ Run the gridsearch over possible combinations
    """

    # GET TRAINING AND TESTING SIGNALS
    t, U = rc.orbit('lorenz', duration=50)
    u = CubicSpline(t, U)
    t_train = t[:4000]
    U_train = u(t[:4000])
    t_test = t[4000:]
    U_test = u(t[4000:])
    eps = 1e-5
    tol = 5.

    # Parameters
    t0 = time.time()

    # Create a new HDF5 file
    with h5py.File(hdf5_file, 'w') as file:
        for i in range(draw_count):
            param_set_index = np.random.choice(len(erdos_possible_combinations))
            param_set = erdos_possible_combinations[param_set_index]
            # print(f"Rank: {MPI.COMM_WORLD.Get_rank()}, combination: {i} / {draw_count}")

            # Check time and break if out of time
            t1 = time.time()
            if t1 - t0 > tf:
                print("Break in Combo")
                return

            # Setup initial conditions
            n, erdos_c, gamma, sigma, alpha = param_set

            div_der_thinned = [] 
            div_pos_thinned = [] 
            div_der_connected = [] 
            vpt_thinned = []
            pred_thinned = []
            err_thinned = []
            consistency_correlation_thinned = []

                
            try:
                # Generate thinned networks
                mean_degree = erdos_c*(1-p_thin)
                if mean_degree < 0.0:
                    mean_degree = 0.0
                print("Here", rho, p_thin)
                res_thinned = rc.ResComp(res_sz=n, mean_degree=mean_degree, 
                                         ridge_alpha=alpha, spect_rad=rho, sigma=sigma, 
                                         gamma=gamma, map_initial='activ_f')                
                res_thinned.train(t_train, U_train)

                # t_curr = time_comp(t_curr, f"Train Thinned")

                # Calculate Consistency Metric
                r0 = res_thinned.initial_condition(U_train[0])
                r0_perturbed = r0 + np.random.multivariate_normal(np.zeros(n), np.eye(n)*eps)
                states = res_thinned.internal_state_response(t_train, U_train, r0)
                states_perturbed = res_thinned.internal_state_response(t_train, U_train, r0_perturbed)
                consistency_correlation = pearson_consistency_metric(states, states_perturbed)

                # t_curr = time_comp(t_curr, f"States and Consistency Thinned")

                # Forecast and compute the vpt along with diversity metrics
                U_pred, pred_states = res_thinned.predict(t_test, r0=res_thinned.r0, return_states=True)
                error = np.linalg.norm(U_test - U_pred, axis=1)
                vpt = vpt_time(t_test, U_test, U_pred, vpt_tol=tol)
                divs = div_metric_tests(pred_states, T=len(t_test), n=n)

                # t_curr = time_comp(t_curr, f"Predict, vpt, divs thinned")

                # Store results
                div_pos_thinned.append(divs[0])
                div_der_thinned.append(divs[1])
                pred_thinned.append(U_pred)
                err_thinned.append(error)
                vpt_thinned.append(vpt)
                consistency_correlation_thinned.append(consistency_correlation)

                # t_curr = time_comp(t_curr, f"Store results thinned")

            except ArpackNoConvergence: # Occasionally sparse linalg eigs isn't able to converge
                i = i-1
                if f"param_set_{i}" in file:
                    del file[f"param_set_{i}"]
                print("ArpackNoConvergence Error Caught")
                continue
            except OverflowError: # Solving for W_out hits overflow errors with high spectral radius and high p_thin
                i = i-1
                if f"param_set_{i}" in file:
                    del file[f"param_set_{i}"]
                print("Overflow Error Caught")
                continue
            except ValueError as err:
                i = i-1
                if f"param_set_{i}" in file:
                    del file[f"param_set_{i}"]
                print(rho, p_thin, erdos_c, str(err))
                print(n, erdos_c*(1-p_thin), erdos_c*(1-p_thin) / n)
                traceback.print_exc()  # This will print the stack trace
                continue
            except Exception as e:
                i = i-1
                if f"param_set_{i}" in file:
                    del file[f"param_set_{i}"]
                print("General Error")
                # logging.error(traceback.format_exc())
                continue


            # t_curr = time_comp(t_curr, "Create Group")
            group = file.create_group(f"param_set_{i}")
            group.attrs['n'] = n
            group.attrs['p_thin'] = p_thin
            group.attrs['erdos_c'] = erdos_c
            group.attrs['gamma'] = gamma
            group.attrs['rho'] = rho
            group.attrs['sigma'] = sigma
            group.attrs['alpha'] = alpha

            # Store datasets
            group.create_dataset('div_der_thinned', data=div_der_thinned)
            group.create_dataset('div_pos_thinned', data=div_pos_thinned)
            group.create_dataset('div_der_connected', data=div_der_connected)
            group.create_dataset('vpt_thinned', data=vpt_thinned)
            group.create_dataset('pred_thinned', data=pred_thinned)
            group.create_dataset('err_thinned', data=err_thinned)
            group.create_dataset('consistency_thinned', data=consistency_correlation_thinned)

            # t_curr = time_comp(t_curr, f"Store datasets")


            # Store Means
            group.attrs['div_der_thinned'] = np.mean(div_der_thinned)
            group.attrs['div_pos_thinned'] = np.mean(div_pos_thinned)
            group.attrs['div_der_connected'] = np.mean(div_der_connected)
            group.attrs['mean_vpt_thinned'] = np.mean(vpt_thinned)
            group.attrs['mean_pred_thinned'] = np.mean(pred_thinned)
            group.attrs['mean_err_thinned'] = np.mean(err_thinned)
            group.attrs['mean_consistency_thinned'] = np.mean(consistency_correlation_thinned)



"""
Main Method
"""

if __name__ == "__main__":
    rhos, p_thins, erdos_possible_combinations = gridsearch_uniform_dict_setup()

    n, m = len(rhos), len(p_thins)

    rho_p_thin_prod = list(itertools.product(rhos, p_thins))

    # Setup the parallelization
    SIZE = MPI.COMM_WORLD.Get_size()
    if SIZE != n*m:
        print(f"Number of processes expected: {n*m}, received: {SIZE}")
        exit()
    
    # Split the Erdos_c exploration according to RANK
    RANK = MPI.COMM_WORLD.Get_rank()

    rho, p_thin = rho_p_thin_prod[RANK]
    results_path = '/nobackup/autodelete/usr/seyfdall/network_theory/thinned_rescomp/'
    rescomp_parallel_gridsearch_uniform_thinned_h5(
        erdos_possible_combinations, 
        draw_count=10000, 
        hdf5_file=f'{results_path}results/erdos_results_rho={round(rho,2)}_p_thin={round(p_thin,2)}.h5', 
        rho=rho, 
        p_thin=p_thin,
        tf=210000
    )