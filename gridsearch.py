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
Automating Tests
"""

def div_metric_tests(preds, T, n):
    """ Compute Diversity scores of predictions
    """
    # Take the derivative of the pred_states
    res_deriv = np.gradient(preds[:T], axis=0)

    # Run the metric for the old and new diversity scores
    div = 0
    old_div = 0
    for i in range(n):
        for j in range(i+1, n):
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
Reservoir Computing Helper Functions
"""

def train_and_drive(A, n, gamma, rho, sigma, W_in, u, U_train, t, consistency_check=True, eps=1e-4):
    """ Train and Drive the reservoir computer
    """
    # ODE IVP definition and numerical solution
    drdt = lambda r, t : gamma * (-r + np.tanh(rho * A @ r + sigma * W_in @ u(t)))
    r0 = np.random.rand(n)
    states = integrate.odeint(drdt, r0, t[:9000])

    # Training step. Project training data onto reservoir states.  See https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/13/lecture-13.pdf
    W_out =  U_train.T @ states @ np.linalg.inv(states.T @ states )

    # Prediction ODE IVP definition and solution
    trained_drdt = lambda r, t : gamma * (-r + np.tanh(rho * A @ r + sigma * W_in @ W_out @ r))
    r0_pred = states[-1, :]
    pred_states = integrate.odeint(trained_drdt, r0_pred, t[9000:])

    # Map reservoir states onto the dynamical system space
    U_pred = W_out @ pred_states.T

    consistency_correlation = None
    if consistency_check:
        r0_perturbed = r0 + np.random.multivariate_normal(np.zeros(n), np.eye(n)*eps)
        states_perturbed = integrate.odeint(drdt, r0_perturbed, t[:9000])
        consistency_correlation = pearson_consistency_metric(states, states_perturbed)

    return pred_states, U_pred, consistency_correlation



"""
Gridsearch Parameter Setup
"""

def gridsearch_dict_setup():
    # Topological Parameters
    # ns = [500, 1500, 2500]
    ns = [20]
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
    rhos = [0.1,0.9,1,1.1,2,5,10,25,50]
    sigmas = [1e-3,5e-3,1e-2,5e-2,.14,.4,.7,1,10]
    alphas = [1e-8,1e-6,1e-4,1e-2,1]

    erdos_possible_combinations = list(itertools.product(ns, p_thins, gammas, rhos, sigmas, alphas))
    # digraph_possible_combinations = list(itertools.product(ns, p_thins, random_digraph_c, gammas, rhos, sigmas, alphas))
    # geometric_possible_combinations = list(itertools.product(ns, p_thins, random_geometric_c, gammas, rhos, sigmas, alphas))
    # barabasi_possible_combinations = list(itertools.product(ns, p_thins, barabasi_albert_m, gammas, rhos, sigmas, alphas))
    # strogatz_possible_combinations = list(itertools.product(ns, p_thins, watts_strogatz_k, watt_strogatz_q, gammas, rhos, sigmas, alphas))

    return erdos_possible_combinations



"""
Training Data
"""

def lorenz_attractor(duration=100):
    """ Run a Lorenz Attractor for duration steps
    """
    t, U = rc.orbit("lorenz", duration=duration)
    return t, U



"""
Perform the Gridsearch
"""

def run_gridsearch_csv(erdos_possible_combinations, t, U, iterations=30, tf=86400, csv_file="erdos_results.csv"):
    """ Run the gridsearch over possible combinations
    """

    # Interpolate data
    u = CubicSpline(t, U)
    U_train = u(t[:9000])

    # Parameters
    T = 1000
    epsilon = 5
    test_t = t[9000:]
    t0 = time.time()

    # Writing results to csv file
    with open(csv_file, 'w', newline='') as file:
        fieldnames = ['n', 'p_thin', 'c', 'gamma', 'spect_rad', 'sigma', 'ridge_alpha', 
                      'div_old_thinned', 'div_new_thinned', 'div_old_connected',
                      'div_new_connected', 'vpt_thinned', 'vpt_connected','pred_thinned',
                      'pred_connected', 'err_thinned', 'err_connected', 'mean_pred_thinned',
                      'mean_pred_connected', 'mean_err_thinned', 'mean_err_connected']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write header
        writer.writeheader()

        for combo in erdos_possible_combinations:
            # Check time and break if out of time
            t1 = time.time()
            if t1 - t0 > tf:
                print("Break in Combo")
                return

            # Setup initial conditions
            n, p_thin, erdos_c, gamma, rho, sigma, alpha = combo
            row = {
                'n': n, 
                'p_thin': p_thin, 
                'c': erdos_c, 
                'gamma': gamma, 
                'spect_rad': rho, 
                'sigma': sigma, 
                'ridge_alpha': alpha,
                'div_old_thinned': [], 
                'div_new_thinned': [], 
                'div_old_connected': [], 
                'div_new_connected': [], 
                'vpt_thinned': [], 
                'vpt_connected': [],
                'pred_thinned': [],
                'pred_connected': [],
                'err_thinned': [],
                'err_connected': [],
                'mean_pred_thinned': 0,
                'mean_pred_connected': 0,
                'mean_err_thinned': 0,
                'mean_err_connected': 0
            }

            for iter in range(iterations):
                print("Here")
                # Check time and break if out of time
                t1 = time.time()
                if t1 - t0 > tf:
                    print("Break in iterations")
                    return

                # Fixed random matrix
                W_in = np.random.rand(n, 3) - .5


                # Connected Matrix
                conn_prob = erdos_c / n + 1

                # Adjacency Matrix with Directed Erdos-Renyi adjacency matrix
                A_connected = nx.erdos_renyi_graph(n,conn_prob,directed=True)
                num_edges = len(A_connected.edges)
                A_connected = sparse.dok_matrix(nx.adjacency_matrix(A_connected).T)

                pred_states_connected, U_pred_connected = train_and_drive(A_connected, n, gamma, rho, sigma, W_in, u, U_train, t)

                # Compute VPT Score for comparison
                comp_array = np.sqrt((U_pred_connected.T - u(test_t))**2)
                row['vpt_connected'].append(np.argmax(comp_array > epsilon))


                # Thinned matrix
                thin_prob = erdos_c * (1-p_thin) / n + 1

                # Adjacency matrix with zero edges
                A_thinned = remove_edges(A_connected, int((p_thin * num_edges)))

                pred_states_thinned, U_pred_thinned = train_and_drive(A_thinned, n, gamma, rho, sigma, W_in, u, U_train, t)

                # Compute VPT Score for comparison epsilon set to 5
                comp_array = np.sqrt((U_pred_thinned.T - u(test_t))**2)
                row['vpt_thinned'].append(np.argmax(comp_array > epsilon))

                # Compute diversity metrics
                connected_divs = div_metric_tests(pred_states_connected, T, n)
                thinned_divs = div_metric_tests(pred_states_thinned, T, n)
                

                # Store diversity metrics
                row['div_new_connected'].append(connected_divs[0])
                row['div_old_connected'].append(connected_divs[1])
                row['div_new_thinned'].append(thinned_divs[0])
                row['div_old_thinned'].append(thinned_divs[1])

                # Store predictions and errors
                # TODO: Are these the correct values?
                row['pred_thinned'].append(U_pred_thinned)
                row['pred_connected'].append(U_pred_connected)
                row['err_thinned'].append(np.sqrt((U_pred_thinned.T - u(test_t))**2))
                row['err_connected'].append(np.sqrt((U_pred_connected.T - u(test_t))**2))

            # Store Means
            row['mean_pred_thinned'] = np.mean(row['pred_thinned'])
            row['mean_pred_connected'] = np.mean(row['pred_connected'])
            row['mean_err_thinned'] = np.mean(row['err_thinned'])
            row['mean_err_thinned'] = np.mean(row['err_connected'])


            writer.writerow(row)


def run_gridsearch_h5(erdos_possible_combinations, t, U, iterations=30, tf=300, hdf5_file="erdos_results.h5"):
    """ Run the gridsearch over possible combinations
    """

    # TODO: Implement this with mpi4py to take advantage of parallelism

    # Interpolate data
    u = CubicSpline(t, U)
    U_train = u(t[:9000])

    # Parameters
    T = 1000
    epsilon = 5
    test_t = t[9000:]
    t0 = time.time()

    # Create a new HDF5 file
    with h5py.File(hdf5_file, 'w') as file:
        for param_set in erdos_possible_combinations:
            # Check time and break if out of time
            t1 = time.time()
            if t1 - t0 > tf:
                print("Break in Combo")
                return

            # Setup initial conditions
            n, p_thin, erdos_c, gamma, rho, sigma, alpha = param_set

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

            for iter in range(iterations):
                print("Here")
                # Check time and break if out of time
                t1 = time.time()
                if t1 - t0 > tf:
                    print("Break in iterations")
                    return

                # Fixed random matrix
                W_in = np.random.rand(n, 3) - .5


                # Connected Matrix
                conn_prob = erdos_c / n + 1

                # Adjacency Matrix with Directed Erdos-Renyi adjacency matrix
                A_connected = nx.erdos_renyi_graph(n,conn_prob,directed=True)
                num_edges = len(A_connected.edges)
                A_connected = sparse.dok_matrix(nx.adjacency_matrix(A_connected).T)

                pred_states_connected, U_pred_connected = train_and_drive(A_connected, n, gamma, rho, sigma, W_in, u, U_train, t)

                # Compute VPT Score for comparison
                comp_array = np.sqrt((U_pred_connected.T - u(test_t))**2)
                vpt_connected.append(np.argmax(comp_array > epsilon))


                # Thinned matrix
                thin_prob = erdos_c * (1-p_thin) / n + 1

                # Adjacency matrix with zero edges
                A_thinned = remove_edges(A_connected, int((p_thin * num_edges)))

                pred_states_thinned, U_pred_thinned = train_and_drive(A_thinned, n, gamma, rho, sigma, W_in, u, U_train, t)

                # Compute VPT Score for comparison epsilon set to 5
                comp_array = np.sqrt((U_pred_thinned.T - u(test_t))**2)
                vpt_thinned.append(np.argmax(comp_array > epsilon))

                # Compute diversity metrics
                connected_divs = div_metric_tests(pred_states_connected, T, n)
                thinned_divs = div_metric_tests(pred_states_thinned, T, n)
                

                # Store diversity metrics
                div_new_connected.append(connected_divs[0])
                div_old_connected.append(connected_divs[1])
                div_new_thinned.append(thinned_divs[0])
                div_old_thinned.append(thinned_divs[1])

                # Store predictions and errors
                # TODO: Are these the correct values?
                pred_thinned.append(U_pred_thinned)
                pred_connected.append(U_pred_connected)
                err_thinned.append(np.sqrt((U_pred_thinned.T - u(test_t))**2))
                err_connected.append(np.sqrt((U_pred_connected.T - u(test_t))**2))

            # Store arrays
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

            # Store Means
            group.attrs['mean_pred_thinned'] = np.mean(pred_thinned)
            group.attrs['mean_pred_connected'] = np.mean(pred_connected)
            group.attrs['mean_err_thinned'] = np.mean(err_thinned)
            group.attrs['mean_err_connected'] = np.mean(err_connected)


def run_parallel_gridsearch_h5(erdos_possible_combinations, t, U, iterations=30, tf=300, hdf5_file="erdos_results.h5", erdos_c=0.5):
    """ Run the gridsearch over possible combinations
    """

    # Interpolate data
    u = CubicSpline(t, U)
    U_train = u(t[:9000])

    # Parameters
    T = 1000
    epsilon = 5.0
    test_t = t[9000:]
    t0 = time.time()
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
                # Check time and break if out of time
                t1 = time.time()
                if t1 - t0 > tf:
                    print("Break in iterations")
                    return

                # Fixed random matrix
                W_in = np.random.rand(n, 3) - .5


                # Connected Matrix
                conn_prob = erdos_c / n + 1

                # Adjacency Matrix with Directed Erdos-Renyi adjacency matrix
                A_connected = nx.erdos_renyi_graph(n,conn_prob,directed=True)
                num_edges = len(A_connected.edges)
                A_connected = sparse.dok_matrix(nx.adjacency_matrix(A_connected).T)

                pred_states_connected, U_pred_connected, cc_conn = train_and_drive(A_connected, n, gamma, rho, sigma, W_in, u, U_train, t)

                # Compute VPT and Consistency Scores for comparison
                comp_array = np.sqrt((U_pred_connected.T - u(test_t))**2)
                vpt_connected.append(np.argmin(comp_array > epsilon))
                consistency_correlation_connected.append(cc_conn)


                # Thinned matrix
                thin_prob = erdos_c * (1-p_thin) / n + 1

                # Adjacency matrix with zero edges
                A_thinned = remove_edges(A_connected, int((p_thin * num_edges)))

                pred_states_thinned, U_pred_thinned, cc_thin = train_and_drive(A_thinned, n, gamma, rho, sigma, W_in, u, U_train, t)

                # Compute VPT and Consistency Scores for comparison epsilon set to 5
                comp_array = np.sqrt((U_pred_thinned.T - u(test_t))**2)
                vpt_thinned.append(np.argmin(comp_array > epsilon))
                consistency_correlation_thinned.append(cc_thin)

                # Compute diversity metrics
                connected_divs = div_metric_tests(pred_states_connected, T, n)
                thinned_divs = div_metric_tests(pred_states_thinned, T, n)
                

                # Store diversity metrics
                div_new_connected.append(connected_divs[0])
                div_old_connected.append(connected_divs[1])
                div_new_thinned.append(thinned_divs[0])
                div_old_thinned.append(thinned_divs[1])

                # Store predictions and errors
                # TODO: Are these the correct values?
                pred_thinned.append(U_pred_thinned)
                pred_connected.append(U_pred_connected)
                err_thinned.append(np.sqrt((U_pred_thinned.T - u(test_t))**2))
                err_connected.append(np.sqrt((U_pred_connected.T - u(test_t))**2))

            # Store arrays
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
    t, U = lorenz_attractor(duration=100)
    c_list = [.5,1,2,3,4]

    # Setup the parallelization
    SIZE = MPI.COMM_WORLD.Get_size()
    if SIZE != 5:
        print(f"Number of processes expected: 5, received: {SIZE}")
        exit()
    
    # Split the Erdos_c exploration according to RANK
    RANK = MPI.COMM_WORLD.Get_rank()
    run_parallel_gridsearch_h5(erdos_possible_combinations, t, U, iterations=1, hdf5_file= f'erdos_results_{RANK}.h5', erdos_c=c_list[RANK])