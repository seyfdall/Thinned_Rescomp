import numpy as np
import itertools
import argparse
import json
import networkx as nx
import scipy.sparse as sparse

from scipy.interpolate import CubicSpline

"""
Import Inhouse Rescomp
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "rescomp", "rescomp")))
import ResComp
import chaosode

"""
Read in Parameters
"""

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n_type', '--network-type', type=str, help="type of network (must be listed in helper.py)")
    parser.add_argument('-r', '--rho-p-thin-set', type=str, help="list of rhos and p_thins to range over")
    parser.add_argument('-p', '--param', type=float, help="list of parameters to range over")
    parser.add_argument('-p_name', '--param-name', type=str, help="parameter name to fix")
    parser.add_argument('-p_set', '--param-set', type=str, help="parameter set name")
    
    args = parser.parse_args()
    return args.network_type, args.rho_p_thin_set, args.param, args.param_name, args.param_set



"""
Gridsearch Parameter Setup
"""


def generate_params(rho_p_thin_set, param, param_name, param_set):

    # Great Params (VPT > 1): Sigma: 5e-2, 4e-2, 3e-2, 2e-2, gamma: 50, 
    # Good gammas: 60, 55, 51, 45, 40, 25, 5
    # Good sigmas: 5e-2, 4e-2, 3e-2
    # Good alphas: 1e-8, 1e-4, 1e-6

    # Reservoir Computing Parameters
    rhos_p_thin_dict = {}
    with open(f'./utils/rho_p_thin_sets/{rho_p_thin_set}.json') as f:
        rhos_p_thin_dict = json.load(f)

    param_dict = {}
    with open(f'./utils/param_sets/{param_set}.json') as f:
        param_dict = json.load(f)

    parameters = param_dict | rhos_p_thin_dict

    # Ensure ints are ints
    parameters['n'] = [int(i) for i in parameters['n']]

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


"""
Network design
"""

def undirected_erdos(n, p):
    """
    """

    A = nx.to_scipy_sparse_array(nx.erdos_renyi_graph(n, p), format="lil", dtype=float)
    # Remove self edges
    for i in range(n):
            A[i,i] = 0.0
    # Add one loop to ensure positive spectral radius
    if n > 1:
        A[0, 1] = 1
        A[1, 0] = 1
    return A


def directed_erdos(n, p):
    """
    """
    A = sparse.csr_matrix((np.random.rand(n, n) < p).astype(float))
    
    # Add one loop to ensure positive spectral radius
    if n > 1:
        A[0, 1] = 1
        A[1, 0] = 1
    return A


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


def spectral_rad(A):
    """ Compute spectral radius via max radius of the strongly connected components """
    g = nx.DiGraph(A.T)
    if sparse.issparse(A):
        A = A.copy().todok()
    scc = nx.strongly_connected_components(g)
    rad = 0
    for cmp in scc:
        # If the component is one node, spectral radius is the edge weight of it's self loop
        if len(cmp) == 1:
            i = cmp.pop()
            max_eig = A[i,i]
        else:
            # Compute spectral radius of strongly connected components
            adj = nx.adjacency_matrix(nx.subgraph(g,cmp))
            max_eig = np.max(np.abs(np.linalg.eigvals(adj.T.toarray())))
        if max_eig > rad:
            rad = max_eig
    return rad


def scale_spect_rad(A, rho):
    """ Scales the spectral radius of the reservoir
    """
    curr_rho = spectral_rad(A)
    if not np.isclose(curr_rho,0, 1e-8):
        A *= rho/curr_rho
    else:
        print("Spectral radius of reservoir is close to zero. Edge weights will not be scaled")

    # Convert to csr if sparse
    if sparse.issparse(A):
        A = A.tocsr()
    
    return A


def create_network(params, network_type, rho):
    """
    """

    # Create the network

    A = None
    if network_type == "undirected_erdos":
        A = undirected_erdos(*params)
    elif network_type == "directed_erdos":
        A = directed_erdos(*params)
    else:
        print(f"Network type: {network_type} not supported.")

    # Rescale the network 
    A = scale_spect_rad(A, rho)

    return A
    
