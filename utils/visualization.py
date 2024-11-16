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
np.random.seed(1)
from math import comb
import h5py
from mpi4py import MPI
from glob import glob
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator



def create_system_plot(values, ax, title, threshold, p_thins=[], rhos=[]):
    """
    
    """

    cmap = 'viridis'
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(values[np.isfinite(values)]))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=min([np.max(values[np.isfinite(values)]), threshold]))
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Create evenly spaced grid for plotting
    x_even = np.linspace(0, 1, len(p_thins))  # evenly spaced x-coordinates
    y_even = np.linspace(0, 1, len(rhos))  # evenly spaced y-coordinates

    # X, Y = np.meshgrid(p_thins, rhos)
    ax.pcolormesh(x_even, y_even, values, shading='nearest', norm=norm, cmap=cmap)
    # ax.pcolormesh(X, Y, values, shading='nearest', cmap=cmap)
    
    ax.set_title(title)
    ax.set_xlabel('p_thin')
    ax.set_ylabel('rho')
    ax.set_xticks(x_even)
    ax.set_xticklabels(p_thins)  # map to original uneven x-values
    ax.set_yticks(y_even)
    ax.set_yticklabels(rhos)  # map to original uneven y-values
    ax.set_aspect('equal', adjustable='box')  # Ensure square cells

    
    # Format the ticks to show 1 decimal place and set them as labels
    ticks = ax.get_xticks()
    ax.set_xticklabels([f'{tick:.1f}' for tick in ticks])

    # Reduce visible x-axis labels
    labels = ax.get_xticklabels()
    for i, label in enumerate(labels):
        if i % 3 != 0:  # Show every 3rd label
            label.set_visible(False)
    
    plt.colorbar(mappable=sm, ax=ax)


def create_plots(mean_values=[], thresholds=[10,10,10,10], cutoff=False):
    fig, axs = plt.subplots(1,4, figsize=(14,3.2))

    titles = ['VPT', 'Div_Pos', 'Div_Der', 'Consistency']

    for i in range(len(mean_values)):
        if cutoff:
            mean_values[i][mean_values[i] > thresholds[i]] = 0
        create_system_plot(mean_values[i], axs[i], titles[i], thresholds[i])

    plt.tight_layout()
    plt.show()