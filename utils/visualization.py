import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [20, 5]
import matplotlib
from file_io import get_system_data, remove_system_data
from helper import parse_arguments, load_rho_pthin
from pathlib import Path
import argparse
import os



def create_system_plot(values, ax, title, p_thins=[], rhos=[]):
    """
    
    """

    cmap = 'viridis'
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(values[np.isfinite(values)]))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=min([np.max(values[np.isfinite(values)])]))
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


def create_plots(
        mean_values, 
        thresholds, 
        titles,
        cutoff,
        param_name,
        param,
        param_set,
        rhos=[],
        p_thins=[]
    ):
    save_path = f'{os.getcwd()}/results/{param_name}/{param}/{param_set}/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    num_plots = len(mean_values)
    num_rows = int(np.sqrt(num_plots))
    num_cols = int(num_plots / num_rows) + 1
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*3.5,num_rows*3.2))

    for i in range(len(mean_values)):
        if cutoff:
            mean_values[i][mean_values[i] > thresholds[i]] = 0
        create_system_plot(mean_values[i], axs.flatten()[i], titles[i], p_thins=p_thins, rhos=rhos)

    plt.suptitle(f'{param_name}: {param}, {param_set}')
    plt.tight_layout()
    plt.savefig(f"{save_path}mean_plots.png")


if __name__ == "__main__":
    """
    Post-Processing Visual Analysis on results
    """
    param, param_name, param_set = parse_arguments()

    home = os.path.expanduser("~")
    results_path = f'{home}/nobackup/autodelete/results/{param_name}/{param}/{param_set}/'

    rhos, p_thins = load_rho_pthin()
    mean_values = get_system_data(p_thins, rhos, results_path)
    create_plots(
        mean_values, 
        [3, 10, 10, 10, 10, 10], 
        ['VPT', 'Div_Pos', 'Div_Der', 'Div_Spect', 'Div_Rank', 'Consistency'], 
        False, 
        param_name,
        param,
        param_set,
        rhos, 
        p_thins
    )

    # Delete unnecessary files:
    # remove_system_data(results_path)

    # path = '/nobackup/autodelete/usr/seyfdall/network_theory/thinned_rescomp/results_single_param_tests/'
    # create_multi_run_plots(path, rhos, p_thins)