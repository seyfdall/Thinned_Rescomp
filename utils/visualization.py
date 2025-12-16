import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [20, 5]
import matplotlib
from file_io import get_system_data, remove_system_data
from helper import parse_arguments, load_rho_pthin
from pathlib import Path
import argparse
import os
import pandas as pd
import seaborn as sns
from cycler import cycler


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


def create_correlation_plots(mean_values, save_path, rhos, p_thins, method="pearson"):
    metrics = {
        "VPT": mean_values[0],
        "Div_Pos": mean_values[1],
        "Div_Der": mean_values[2],
        "Div_Spect": mean_values[3],
        "Div_Rank": mean_values[4],
        "Consistency": mean_values[5],
        "Giant_Diam": mean_values[6],
        "Largest_Diam": mean_values[7],
        "Average_Diam": mean_values[8]
    }

    # Flatten and build dataframe
    df = pd.DataFrame({name: mat.flatten() for name, mat in metrics.items()})

    # Global correlation matrix
    global_corr = df.corr(method)

    plt.figure(figsize=(6,4))
    sns.heatmap(global_corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Global Correlation Across All Metrics")
    plt.tight_layout()
    plt.savefig(f"{save_path}{method}_global_correlation_plot.png")

    # Row Correlations (along rho)
    num_plots = len(rhos)
    num_rows = int(np.sqrt(num_plots))
    num_cols = int(num_plots / num_rows) + 1
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*5,num_rows*4))
    axes = axs.flatten()

    for i in range(next(iter(metrics.values())).shape[0]):
        row_df = pd.DataFrame({name: mat[i,:] for name, mat in metrics.items()})
        ax = axes[i]
        sns.heatmap(row_df.corr(method), annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title(f"Rho: {rhos[i]} Correlations")
    plt.tight_layout()
    plt.savefig(f"{save_path}{method}_rho_correlation_plot.png")


    # Column Correlations (along p_thin)
    num_plots = len(p_thins)
    num_rows = int(np.sqrt(num_plots))
    num_cols = int(num_plots / num_rows) + 1
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*5,num_rows*4))
    axes = axs.flatten()

    for j in range(next(iter(metrics.values())).shape[1]):
        row_df = pd.DataFrame({name: mat[:,j] for name, mat in metrics.items()})
        ax = axes[j]
        sns.heatmap(row_df.corr(method), annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title(f"P_thin: {p_thins[j]} Correlations")
    plt.tight_layout()
    plt.savefig(f"{save_path}{method}_p_thins_correlation_plot.png")


def create_correlation_line_plots(mean_values, save_path, rhos, p_thins, p_thin_cs, method="pearson"):
    metrics = {
        "VPT": mean_values[0],
        "Div_Pos": mean_values[1],
        "Div_Der": mean_values[2],
        "Div_Spect": mean_values[3],
        "Div_Rank": mean_values[4],
        "Consistency": mean_values[5],
        "Giant_Diam": mean_values[6],
        "Largest_Diam": mean_values[7],
        "Average_Diam": mean_values[8]
    }

    row_cors = []
    for i in range(next(iter(metrics.values())).shape[0]):
        row_df = pd.DataFrame({name: mat[i,:] for name, mat in metrics.items()})
        row_cors.append(row_df.corr(method))

    col_cors = []
    for j in range(next(iter(metrics.values())).shape[1]):
        row_df = pd.DataFrame({name: mat[:,j] for name, mat in metrics.items()})
        col_cors.append(row_df.corr(method))

    # Generate index positions for equal spacing
    rho_indices = range(len(rhos))
    p_thin_indices = range(len(p_thins))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*6,1*5))
    for key in metrics.keys():
        if key == 'VPT':
            continue
        ax1.plot(rho_indices, [row_cors[i].loc[key, 'VPT'] for i in range(len(rhos))], label=f"{key}")
        ax2.plot(p_thin_indices, [col_cors[i].loc[key, 'VPT'] for i in range(len(p_thins))], label=f"{key}")

    # Plot c=1 line (not exact because of shifting and interpolation but close)
    p_thin_index = np.interp(p_thin_cs[0], p_thins, range(len(p_thins)))
    ax2.axvline(x=p_thin_index, linestyle='--', linewidth=1, label="c=1")
    p_thin_index = np.interp(p_thin_cs[1], p_thins, range(len(p_thins)))
    ax2.axvline(x=p_thin_index, linestyle='--', color='r', linewidth=1, label="c=1.5")
    
    # Format tick labels nicely (max 2 decimals)
    rho_labels = [f"{x:.2f}".rstrip('0').rstrip('.') for x in rhos]
    pthin_labels = [f"{x:.2f}".rstrip('0').rstrip('.') for x in p_thins]

    # Set evenly spaced tick marks with correct labels
    ax1.set_xticks(rho_indices)
    ax1.set_xticklabels(rho_labels, rotation=45, ha='right')
    ax2.set_xticks(p_thin_indices)
    ax2.set_xticklabels(pthin_labels, rotation=45, ha='right')
        
    ax1.set_title("Rho Correlation with VPT")
    ax1.set_xlabel("Rho")
    ax1.set_ylabel(f"{method} correlation")
    ax1.legend()

    ax2.set_title("P_thin Correlation with VPT")
    ax2.set_xlabel("P_thin")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}{method}_correlation_line_plots.png")


def create_column_linear_plots(mean_values, save_path, rhos, p_thins, titles):

    def _normalize_minmax(y):
        return (y - np.min(y)) / (np.max(y) - np.min(y))

    num_plots = len(p_thins)
    num_rows = int(np.sqrt(num_plots))
    num_cols = int(num_plots / num_rows) + 1
    

    rho_indices = range(len(rhos))
    rho_labels = [f"{x:.2f}".rstrip('0').rstrip('.') for x in rhos]

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*5,num_rows*4))
    axes = axs.flatten()

    linestyles = ['-', '--']
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']

    for j in range(len(p_thins)):
        for i in range(len(mean_values)):
            ax = axes[j]
            ax.plot(
                rho_indices, 
                _normalize_minmax(mean_values[i][:,j]), 
                label=titles[i], 
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)]
            )

            ax.set_xlabel("Rho")
            ax.set_xticks(rho_indices)
            ax.set_xticklabels(rho_labels)
            ax.set_title(f'Normalized Standard Plots: p_thin = {round(p_thins[j],2)}')
            ax.legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}p_thin_normalized_plots.png")


    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*5,num_rows*4))
    axes = axs.flatten()

    linestyles = ['-', '--']
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']

    for j in range(len(p_thins)):
        for i in range(len(mean_values)):
            ax = axes[j]
            ax.plot(
                rho_indices, 
                np.abs(_normalize_minmax(mean_values[i][:,j])-_normalize_minmax(mean_values[0][:,j])), 
                label=titles[i], 
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)]
            )

            ax.set_xlabel("Rho")
            ax.set_xticks(rho_indices)
            ax.set_xticklabels(rho_labels)
            ax.set_title(f'L1 Error with VPT: p_thin = {round(p_thins[j],2)}')
            ax.legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}p_thin_normalized_l1_err_plots.png")


    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*5,num_rows*4))
    axes = axs.flatten()

    linestyles = ['-', '--']
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']

    for j in range(len(p_thins)):
        for i in range(len(mean_values)):
            ax = axes[j]
            ax.plot(
                rho_indices, 
                np.cumsum(np.abs(_normalize_minmax(mean_values[i][:,j])-_normalize_minmax(mean_values[0][:,j]))), 
                label=titles[i], 
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)]
            )

            ax.set_xlabel("Rho")
            ax.set_xticks(rho_indices)
            ax.set_xticklabels(rho_labels)
            ax.set_title(f'Cumulative L1 Error with VPT: p_thin = {round(p_thins[j],2)}')
            ax.legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}p_thin_normalized_cumulative_l1_err_plots.png")


def create_plots(
        mean_values, 
        thresholds, 
        titles,
        cutoff,
        rho_p_thin_set,
        param_name,
        param,
        param_set,
        rhos=[],
        p_thins=[]
    ):
    save_path = f'{os.getcwd()}/results/{param_name}/{param}/{param_set}/{rho_p_thin_set}/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    num_plots = len(mean_values)
    num_rows = int(np.sqrt(num_plots))
    num_cols = int(num_plots / num_rows) + 1
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*3.5,num_rows*3.2))

    for i in range(len(mean_values)):
        if cutoff:
            mean_values[i][mean_values[i] > thresholds[i]] = 0
        create_system_plot(mean_values[i], axs.flatten()[i], titles[i], p_thins=p_thins, rhos=rhos)

    plt.suptitle(f'{param_name}: {param}, {param_set}, {rho_p_thin_set}')
    plt.tight_layout()
    plt.savefig(f"{save_path}mean_plots.png")

    df = pd.read_csv(f'./utils/param_sets/{param_set}.csv')
    c = df['erdos_renyi_c'][0]
    p_thin_cs = [(1-1./c), (1-1.5/c)]
    print(f"\np_thin_cs: {p_thin_cs} \n")

    create_correlation_plots(mean_values, save_path, rhos, p_thins)
    create_correlation_line_plots(mean_values, save_path, rhos, p_thins, p_thin_cs)
    create_column_linear_plots(mean_values, save_path, rhos, p_thins, titles)


if __name__ == "__main__":
    """
    Post-Processing Visual Analysis on results
    """
    rho_p_thin_set, param, param_name, param_set = parse_arguments()

    home = os.path.expanduser("~")
    results_path = f'{home}/nobackup/autodelete/results/{param_name}/{param}/{param_set}/{rho_p_thin_set}/'

    rhos, p_thins = load_rho_pthin(rho_p_thin_set)
    mean_values = get_system_data(p_thins, rhos, results_path)
    create_plots(
        mean_values, 
        [3, 10, 10, 10, 10, 10, 10, 10, 10], 
        ['VPT', 'Div_Pos', 'Div_Der', 'Div_Spect', 'Div_Rank', 'Consistency', 'Giant Diameter', 'Largest Diameter', 'Average Diameter'], 
        False, 
        rho_p_thin_set,
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