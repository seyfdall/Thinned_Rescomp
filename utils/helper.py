import numpy as np
import itertools
import argparse

from scipy.interpolate import CubicSpline

"""
Import Inhouse Rescomp
"""
import sys
import os
sys.path.insert(0, os.path.abspath('/nobackup/autodelete/usr/seyfdall/network_theory/rescomp_package/rescomp/'))
import ResComp
import chaosode

"""
Read in Parameters
"""

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--param', type=float, help="list of parameters to range over")
    parser.add_argument('-p_name', '--param_name', type=str, help="parameter name to fix")
    parser.add_argument('-p_set', '--param_set', type=str, help="parameter set name")
    
    args = parser.parse_args()
    return args.param, args.param_name, args.param_set



"""
Gridsearch Parameter Setup
"""

def gridsearch_parameter_setup():
    # Topological Parameters
    # ns = [500, 1500, 2500]
    ns = [500]
    rhos = [0.1,0.9,1.0,1.1,2.0,5.0,10.0,25.0,50.0]
    # rhos = np.arange(0.1,1,0.1)
    p_thins = np.concatenate((np.arange(0, 0.8, 0.1), np.arange(0.8, 1.01, 0.02)))


    # Model Specific Parameters
    erdos_renyi_c = [4]
    # random_digraph_c = [.5,1,2,3,4]
    # random_geometric_c = [.5,1,2,3,4]
    # barabasi_albert_m = [1,2]
    # watts_strogatz_k = [2,4]
    # watt_strogatz_q = [.01,.05,.1]

    # Reservoir Computing Parameters
    gammas = [0.1,0.5,1,2,5,10,25,50]
    sigmas = [1e-3,5e-3,1e-2,5e-2,.14,.4,.7,1,10]
    alphas = [1e-8,1e-6,1e-4,1e-2,1]

    rho_p_thin_prod = np.array(list(itertools.product(rhos, p_thins)))

    erdos_possible_combinations = list(itertools.product(ns, erdos_renyi_c, gammas, sigmas, alphas))
    # digraph_possible_combinations = list(itertools.product(ns, random_digraph_c, gammas, sigmas, alphas))
    # geometric_possible_combinations = list(itertools.product(ns, random_geometric_c, gammas, sigmas, alphas))
    # barabasi_possible_combinations = list(itertools.product(ns, barabasi_albert_m, gammas, sigmas, alphas))
    # strogatz_possible_combinations = list(itertools.product(ns, watts_strogatz_k, watt_strogatz_q, gammas, sigmas, alphas))

    return rho_p_thin_prod, erdos_possible_combinations #, digraph_possible_combinations, geometric_possible_combinations, barabasi_possible_combinations, strogatz_possible_combinations


def simple_params():
    # Topological Parameters
    ns = [50]

    # Model Specific Parameters
    erdos_renyi_c = [4]

    # Reservoir Computing Parameters

    rhos = [0.1,0.9,1.0,1.1,2.0,5.0,10.0,25.0,50.0]
    p_thins = np.concatenate((np.arange(0, 0.8, 0.1), np.arange(0.8, 1.01, 0.02)))

    # Great Params (VPT > 1): Sigma: 5e-2, 4e-2, 3e-2, 2e-2, gamma: 50, 
    gammas = [50] # Good ones: 60, 55, 51, 45, 40, 25, 5
    sigmas = [1e-3,5e-3,1e-2,5e-2,.14,.4,.7,1,10] # Good ones: 5e-2, 4e-2, 3e-2
    alphas = [1e-8,1e-6,1e-4,1e-2,1] # Good ones: 1e-8, 1e-4, 1e-6

    rho_p_thin_prod = np.array(list(itertools.product(rhos, p_thins)))

    erdos_possible_combinations = list(itertools.product(ns, erdos_renyi_c, gammas, sigmas, alphas))

    return rho_p_thin_prod, erdos_possible_combinations


# TODO: Show Diversity/Consistency helps VPT
# TODO: Try visualizations of rho to p_thin=0, diversity to p_thin=0, and consistency to p_thin=0
# TODO: Then create full blown regular plots to show how to create more diversity to create more VPT
# TODO: See example 1 / 2 in the current paper
# TODO: Try all diversity ranks including Trevor's new one
# TODO: Update parameters to be a csv file (param_set_1, etc.)

def load_rho_pthin():
    rhos = [0.1,1.0,1.5,2.0,5.0,10.0,25.0,50.0]
    p_thins = [0.0,0.25,0.50,0.75,0.95]
    return rhos, p_thins


def generate_params(param, param_name):

    rhos, p_thins = load_rho_pthin()

    # Reservoir Computing Parameters
    parameters = {
        'n': [50],
        'rho': rhos,
        'p_thin': p_thins,
        'erdos_renyi_c': [4],
        'gamma': [0.1,0.5,1,2,5,10,25,50],
        'sigma': [1e-3,5e-3,1e-2,5e-2,.14,.4,.7,1,10],
        'alpha': [1e-8,1e-6,1e-4,1e-2,1]
    }

    # Adjust for input
    parameters[param_name] = [param]

    rho_p_thin_prod = np.array(
        list(
            itertools.product(
                parameters['rho'], 
                parameters['p_thin']
            )
        )
    )

    erdos_possible_combinations = list(
        itertools.product(
            parameters['n'], 
            parameters['erdos_renyi_c'], 
            parameters['gamma'], 
            parameters['sigma'], 
            parameters['alpha']
        )
    )

    return rho_p_thin_prod, erdos_possible_combinations


"""
Get the system orbit
"""

def get_orbit(duration=50, system="lorenz", switch=10):
    test_train_switch = (duration-switch)*100
    t, U = chaosode.orbit(system, duration=duration)
    u = CubicSpline(t, U)
    t_train = t[:test_train_switch]
    U_train = u(t_train)
    t_test = t[test_train_switch:]
    U_test = u(t_test)

    return t_train, U_train, t_test, U_test


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
