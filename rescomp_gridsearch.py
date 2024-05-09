"""
Import Statements
"""

import rescomp as rc
import numpy as np
from scipy.interpolate import CubicSpline
from scipy import integrate, sparse
from scipy.stats import pearsonr
import math 
import networkx as nx
import itertools
import csv
import time
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [20, 5]
# Set seed for reproducibility
np.random.seed(1)
from math import comb
import h5py
from mpi4py import MPI


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
        sig = np.std(true, axis=0)
        err = np.linalg.norm((true-pred) / sig, axis=1, ord=2)
        return err

def valid_prediction_index(err, tol):
    """First index i where err[i] > tol. err is assumed to be 1D and tol is a float. If err is never greater than tol, then len(err) is returned."""
    mask = np.logical_or(err > tol, ~np.isfinite(err))
    if np.any(mask):
        return np.argmax(mask)
    return len(err)

def wa_vptime(ts, Uts, pre, vpt_tol=5.):
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
    div = 0
    old_div = 0
    for i in range(n):
        for j in range(n):
            div += np.sum(np.abs(np.abs(preds[:T, i]) - np.abs(preds[:T, j])))
            old_div += np.sum(np.abs(res_deriv[:T, i] - res_deriv[:T, j]))
    div = div / (T*comb(n,2))
    old_div = old_div / (T*comb(n,2))

    return div, old_div

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
        aggregated_pearson_correlation_coeff = np.mean(gammas)
        return aggregated_pearson_correlation_coeff



"""
Gridsearch Parameter Setup
"""

def gridsearch_dict_setup():
    # Topological Parameters
    ns = [500, 1500, 2500]
    # ns = [500]
    p_thins = np.concatenate((np.arange(0, 0.8, 0.1), np.arange(0.8, 1.01, 0.02)))

    # Model Specific Parameters
    # erdos_renyi_c = [.5,1,2,3,4]
    random_digraph_c = [.5,1,2,3,4]
    random_geometric_c = [.5,1,2,3,4]
    barabasi_albert_m = [1,2]
    watts_strogatz_k = [2,4]
    watt_strogatz_q = [.01,.05,.1]

    # Reservoir Computing Parameters
    gammas = [0.1,0.5,1,2,5,10,25,50]
    rhos = [0.1,0.9,1.0,1.1,2.0,5.0,10.0,25.0,50.0]
    sigmas = [1e-3,5e-3,1e-2,5e-2,.14,.4,.7,1,10]
    alphas = [1e-8,1e-6,1e-4,1e-2,1]

    erdos_possible_combinations = list(itertools.product(ns, p_thins, gammas, rhos, sigmas, alphas))
    # digraph_possible_combinations = list(itertools.product(ns, p_thins, random_digraph_c, gammas, rhos, sigmas, alphas))
    # geometric_possible_combinations = list(itertools.product(ns, p_thins, random_geometric_c, gammas, rhos, sigmas, alphas))
    # barabasi_possible_combinations = list(itertools.product(ns, p_thins, barabasi_albert_m, gammas, rhos, sigmas, alphas))
    # strogatz_possible_combinations = list(itertools.product(ns, p_thins, watts_strogatz_k, watt_strogatz_q, gammas, rhos, sigmas, alphas))

    return erdos_possible_combinations



"""
Perform the Gridsearch
"""

def rescomp_parallel_gridsearch_h5(erdos_possible_combinations, system='lorenz', iterations=30, tf=85000, hdf5_file="results/erdos_results.h5", erdos_c=0.5):
    """ Run the gridsearch over possible combinations
    """

    # Train Test Split the system
    duration = 45
    dt = 0.01
    trainper = 0.88889
    batchsize = int(duration / dt * trainper)
    t_train, U_train, t_test, U_test = rc.train_test_orbit(system, duration=duration, dt=dt, trainper=trainper)
    eps = 1e-5
    tol = 5.

    # Parameters
    t0 = time.time()
    # t_curr = t0
    combination_length = len(erdos_possible_combinations)

    # Create a new HDF5 file
    with h5py.File(hdf5_file, 'w') as file:
        for i, param_set in enumerate(erdos_possible_combinations):
            print(f"Rank: {MPI.COMM_WORLD.Get_rank()}, combination: {i} / {combination_length}")

            # Check time and break if out of time
            t1 = time.time()
            if t1 - t0 > tf:
                print("Break in Combo")
                return

            # Setup initial conditions
            n, p_thin, gamma, rho, sigma, alpha = param_set

            # t_curr = time_comp(t_curr, "Create Group")
            group = file.create_group(f"param_set_{n}_{p_thin}_{erdos_c}_{gamma}_{rho}_{sigma}_{alpha}")
            group.attrs['n'] = n
            group.attrs['p_thin'] = p_thin
            group.attrs['erdos_c'] = erdos_c
            group.attrs['gamma'] = gamma
            group.attrs['rho'] = rho
            group.attrs['sigma'] = sigma
            group.attrs['alpha'] = alpha

            div_old_thinned = [] 
            div_new_thinned = [] 
            div_old_connected = [] 
            div_new_connected = [] 
            vpt_thinned = []
            vpt_connected = []
            pred_thinned = []
            pred_connected = []
            err_thinned = []
            err_connected = []
            consistency_correlation_thinned = []
            consistency_correlation_connected = []

            for iter in range(iterations):

                # t_curr = time_comp(t_curr, f"Iteration: {iter}")

                # Run the connected network
                A_connected, num_edges = rc.erdos(n, erdos_c)
                A_connected = A_connected * rho / np.max(np.abs(np.linalg.eigvals(A_connected.todense())))
                res = rc.ResComp(A_connected, res_sz=n, ridge_alpha=alpha, spect_rad=rho, sigma=sigma, gamma=gamma, batchsize=batchsize)
                res.train(t_train, U_train)

                # t_curr = time_comp(t_curr, f"Train Connected")

                # Calculate Consistency Metric
                r0 = res.initial_condition(U_train[0])
                r0_perturbed = r0 + np.random.multivariate_normal(np.zeros(n), np.eye(n)*eps)
                states = res.internal_state_response(t_train, U_train, r0)
                states_perturbed = res.internal_state_response(t_train, U_train, r0_perturbed)
                consistency_correlation = pearson_consistency_metric(states, states_perturbed)

                # t_curr = time_comp(t_curr, f"States and Consistency Connected")

                # Forecast and compute the vpt along with diversity metrics
                U_pred, pred_states = res.predict(t_test, r0=r0, return_states=True)
                error = np.linalg.norm(U_test - U_pred, axis=1)
                vpt = wa_vptime(t_test, U_test, U_pred, vpt_tol=tol)
                divs = div_metric_tests(pred_states, T=batchsize, n=n)

                # t_curr = time_comp(t_curr, f"Predict, vpt, divs connected")

                # Store results
                div_new_connected.append(divs[0])
                div_old_connected.append(divs[1])
                pred_connected.append(U_pred)
                err_connected.append(error)
                vpt_connected.append(vpt)
                consistency_correlation_connected.append(consistency_correlation)

                # t_curr = time_comp(t_curr, f"Store results connected")


                # Run the thinned network
                A_thinned = remove_edges(A_connected, int(p_thin * num_edges)).todense() # Convert this back to a digraph thingy
                A_thinned = A_thinned * rho / np.max(np.abs(np.linalg.eigvals(A_thinned)))
                res = rc.ResComp(A_thinned, res_sz=n, ridge_alpha=alpha, spect_rad=rho, sigma=sigma, gamma=gamma, batchsize=batchsize)
                res.train(t_train, U_train)

                # t_curr = time_comp(t_curr, f"Train Thinned")

                # Calculate Consistency Metric
                r0 = res.initial_condition(U_train[0])
                r0_perturbed = r0 + np.random.multivariate_normal(np.zeros(n), np.eye(n)*eps)
                states = res.internal_state_response(t_train, U_train, r0)
                states_perturbed = res.internal_state_response(t_train, U_train, r0_perturbed)
                consistency_correlation = pearson_consistency_metric(states, states_perturbed)

                # t_curr = time_comp(t_curr, f"States and Consistency Thinned")

                # Forecast and compute the vpt along with diversity metrics
                U_pred, pred_states = res.predict(t_test, r0=r0, return_states=True)
                error = np.linalg.norm(U_test - U_pred, axis=1)
                vpt = wa_vptime(t_test, U_test, U_pred, vpt_tol=tol)
                divs = div_metric_tests(pred_states, T=batchsize, n=n)

                # t_curr = time_comp(t_curr, f"Predict, vpt, divs thinned")

                # Store results
                div_new_thinned.append(divs[0])
                div_old_thinned.append(divs[1])
                pred_thinned.append(U_pred)
                err_thinned.append(error)
                vpt_thinned.append(vpt)
                consistency_correlation_thinned.append(consistency_correlation)

                # t_curr = time_comp(t_curr, f"Store results thinned")

            # Store datasets
            group.create_dataset('div_old_thinned', data=div_old_thinned)
            group.create_dataset('div_new_thinned', data=div_new_thinned)
            group.create_dataset('div_old_connected', data=div_old_connected)
            group.create_dataset('div_new_connected', data=div_new_connected)
            group.create_dataset('vpt_thinned', data=vpt_thinned)
            group.create_dataset('vpt_connected', data=vpt_connected)
            group.create_dataset('pred_thinned', data=pred_thinned)
            group.create_dataset('pred_connected', data=pred_connected)
            group.create_dataset('err_thinned', data=err_thinned)
            group.create_dataset('err_connected', data=err_connected)
            group.create_dataset('consistency_thinned', data=consistency_correlation_thinned)
            group.create_dataset('consistency_connected', data=consistency_correlation_connected)

            # t_curr = time_comp(t_curr, f"Store datasets")


            # Store Means
            group.attrs['mean_pred_thinned'] = np.mean(pred_thinned)
            group.attrs['mean_pred_connected'] = np.mean(pred_connected)
            group.attrs['mean_err_thinned'] = np.mean(err_thinned)
            group.attrs['mean_err_connected'] = np.mean(err_connected)
            group.attrs['mean_consistency_thinned'] = np.mean(consistency_correlation_thinned)
            group.attrs['mean_consistency_connected'] = np.mean(consistency_correlation_connected)
    


"""
Main Method
"""

if __name__ == "__main__":
    erdos_possible_combinations = gridsearch_dict_setup()
    c_list = [.5,1,2,3,4]

    # rescomp_parallel_gridsearch_h5(erdos_possible_combinations, iterations=10, hdf5_file=f'test_erdos_results_0.h5', erdos_c=c_list[0])

    # Setup the parallelization
    SIZE = MPI.COMM_WORLD.Get_size()
    if SIZE != 5:
        print(f"Number of processes expected: 5, received: {SIZE}")
        exit()
    
    # Split the Erdos_c exploration according to RANK
    RANK = MPI.COMM_WORLD.Get_rank()

    rescomp_parallel_gridsearch_h5(erdos_possible_combinations, iterations=10, hdf5_file=f'results/erdos_results_{RANK}.h5', erdos_c=c_list[RANK])