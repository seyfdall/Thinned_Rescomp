import numpy as np
import itertools
import argparse
import pandas as pd

from scipy.interpolate import CubicSpline

"""
Import Inhouse Rescomp
"""
import sys
import os
sys.path.insert(0, os.path.abspath('/home/seyfdall/network_theory/rescomp/rescomp'))
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

def load_rho_pthin():
    rhos = [0.1,1.0,1.5,2.0,5.0,10.0,25.0,50.0]
    p_thins = [0.0,0.25,0.50,0.75,0.95]
    return rhos, p_thins


def generate_params(param, param_name, param_set):

    rhos, p_thins = load_rho_pthin()

    # Great Params (VPT > 1): Sigma: 5e-2, 4e-2, 3e-2, 2e-2, gamma: 50, 
    # Good gammas: 60, 55, 51, 45, 40, 25, 5
    # Good sigmas: 5e-2, 4e-2, 3e-2
    # Good alphas: 1e-8, 1e-4, 1e-6

    # Reservoir Computing Parameters
    df = pd.read_csv(f'./utils/param_sets/{param_set}')
    param_dict = {col: df[col].dropna().tolist() for col in df.columns}
    rhos_p_thin_dict = {'rho': rhos, 'p_thin': p_thins}
    parameters = param_dict | rhos_p_thin_dict

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

    possible_combinations = list(
        itertools.product(
            parameters['n'], 
            parameters['erdos_renyi_c'], 
            parameters['gamma'], 
            parameters['sigma'], 
            parameters['alpha']
        )
    )

    return rho_p_thin_prod, possible_combinations


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
