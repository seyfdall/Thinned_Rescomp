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
Main Method
"""

if __name__ == "__main__":
    # Example of how to read in data from an h5df file in Python
    with h5py.File('results_old/erdos_results_0.h5', 'r') as file:
        print(file.keys())

