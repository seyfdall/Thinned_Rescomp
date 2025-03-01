import rescomp as rc
import numpy as np
import itertools
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [20, 5]
from matplotlib import pyplot as plt
import matplotlib
from file_io import get_system_data
import helper



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
        mean_values=[], 
        thresholds=[10,10,10,10], 
        cutoff=False,
        results_path='/mnt/c/Users/dseyf/SeniorLabs/Research/Network_Theory/thinned_rescomp/results/',
        rhos=[],
        p_thins=[]
    ):
    fig, axs = plt.subplots(1,4, figsize=(14,3.2))

    titles = ['VPT', 'Div_Pos', 'Div_Der', 'Consistency']

    for i in range(len(mean_values)):
        if cutoff:
            mean_values[i][mean_values[i] > thresholds[i]] = 0
        create_system_plot(mean_values[i], axs[i], titles[i], p_thins=p_thins, rhos=rhos)

    plt.tight_layout()
    plt.savefig(f"{results_path}mean_plots.png")


if __name__ == "__main__":
    """
    Post-Processing Visual Analysis on results
    """
    rho_p_thin_prod, erdos_possible_combinations = helper.gridsearch_parameter_setup()
    # erdos_possible_combinations = [[50, 4, 0.5, 0.5, 1.0]]
    # n, m = rho_p_thin_prod.shape
    # rhos = [2.,3.]
    # p_thins = [0.1,0.5]
    # rho_p_thin_prod = list(itertools.product(rhos, p_thins))
    rhos = [0.1,0.9,1.0,1.1,2.0,5.0,10.0,25.0,50.0]
    p_thins = np.concatenate((np.arange(0, 0.8, 0.1), np.arange(0.8, 1.01, 0.02)))
    results_path = '/nobackup/autodelete/usr/seyfdall/network_theory/thinned_rescomp/results/'
    mean_values = get_system_data(p_thins, rhos, results_path)
    create_plots(mean_values, [3, 10, 10, 10], False, results_path, rhos, p_thins)