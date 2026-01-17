import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [20, 5]
import matplotlib
from file_io import get_average_system_metrics
from helper import parse_arguments, load_rho_pthin
from pathlib import Path
import argparse
import os
import pandas as pd
import seaborn as sns
from cycler import cycler
from mpl_toolkits.axes_grid1 import make_axes_locatable


def create_system_plot(values, ax, title, p_thins, rhos, label_step=4):
    values = np.asarray(values)
    p_thins = np.asarray(p_thins)
    rhos = np.asarray(rhos)

    if values.shape != (len(rhos), len(p_thins)):
        raise ValueError("values must have shape (len(rhos), len(p_thins))")

    norm = matplotlib.colors.Normalize(vmin=0, vmax=np.nanmax(values))

    x = np.arange(len(p_thins))
    y = np.arange(len(rhos))

    mesh = ax.pcolormesh(
        x, y, values,
        shading="nearest",
        cmap="viridis",
        norm=norm
    )

    ax.set(
        title=title,
        xlabel="p_thin",
        ylabel="rho",
        xticks=x,
        yticks=y,
        xticklabels=[f"{p:.2f}".rstrip('0').rstrip('.') for p in p_thins],
        yticklabels=[f"{r:.2f}".rstrip('0').rstrip('.') for r in rhos],
    )

    for i, lbl in enumerate(ax.get_xticklabels()):
        lbl.set_visible(i % label_step == 0)

    # ---- external colorbar ----
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(mesh, cax=cax)


def create_correlation_plots(metrics, save_path, rhos, p_thins, method="pearson"):

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


def create_correlation_line_plots(metrics, save_path, rhos, p_thins, p_thin_cs, c, method="pearson"):

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
        if key == 'mean_vpt':
            continue
        ax1.plot(rho_indices, [row_cors[i].loc[key, 'mean_vpt'] for i in range(len(rhos))], label=f"{key}")
        ax2.plot(p_thin_indices, [col_cors[i].loc[key, 'mean_vpt'] for i in range(len(p_thins))], label=f"{key}")

    # Plot the max diameter line
    for i, p_thin_c in enumerate(p_thin_cs):
        p_thin_index = np.interp(p_thin_c, p_thins, range(len(p_thins)))
        ax2.axvline(x=p_thin_index, linestyle='--', linewidth=1, label=f"c={round(c*(1-p_thin_c),1)}")
    
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


def create_column_linear_plots(metrics, save_path, rhos, p_thins, titles):

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
        for i, attr in enumerate(metrics):
            ax = axes[j]
            ax.plot(
                rho_indices, 
                _normalize_minmax(metrics[attr][:,j]), 
                label=titles[i], 
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)]
            )

            ax.set_xlabel("Rho")
            ax.set_xticks(rho_indices)
            ax.set_xticklabels(rho_labels)
            ax.set_title(f'Normalized Standard Plots: p_thin = {round(p_thins[j],3)}')
            ax.legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}p_thin_normalized_plots.png")


    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*5,num_rows*4))
    axes = axs.flatten()

    linestyles = ['-', '--']
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']

    for j in range(len(p_thins)):
        for i, attr in enumerate(metrics):
            ax = axes[j]
            ax.plot(
                rho_indices, 
                np.abs(_normalize_minmax(metrics[attr][:,j])-_normalize_minmax(metrics['mean_vpt'][:,j])), 
                label=titles[i], 
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)]
            )

            ax.set_xlabel("Rho")
            ax.set_xticks(rho_indices)
            ax.set_xticklabels(rho_labels)
            ax.set_title(f'L1 Error with VPT: p_thin = {round(p_thins[j],3)}')
            ax.legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}p_thin_normalized_l1_err_plots.png")


    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*5,num_rows*4))
    axes = axs.flatten()

    linestyles = ['-', '--']
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']

    for j in range(len(p_thins)):
        for i, attr in enumerate(metrics):
            ax = axes[j]
            ax.plot(
                rho_indices, 
                np.cumsum(np.abs(_normalize_minmax(metrics[attr][:,j])-_normalize_minmax(metrics['mean_vpt'][:,j]))), 
                label=titles[i], 
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)]
            )

            ax.set_xlabel("Rho")
            ax.set_xticks(rho_indices)
            ax.set_xticklabels(rho_labels)
            ax.set_title(f'Cumulative L1 Error with VPT: p_thin = {round(p_thins[j],3)}')
            ax.legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}p_thin_normalized_cumulative_l1_err_plots.png")


def create_diameter_p_thin_plots(metrics, c, save_path, p_thins, titles):
    p_thin_indices = range(len(p_thins))
    p_thin_labels = [f"{x:.3f}".rstrip('0').rstrip('.') for x in p_thins]

    # compute c values from p_thin
    c_values = [c * (1 - p) for p in p_thins]
    c_labels = [f"{c:.2f}".rstrip('0').rstrip('.') for c in c_values]

    fig, ax = plt.subplots(figsize=(8, 8))

    # main plots
    ax.plot(metrics['mean_giant_diam'][0, :], label=titles[0])
    ax.plot(metrics['mean_average_diam'][0, :], label=titles[1])

    # vertical line at max diameter
    argmax_p_thin = np.argmax(metrics['mean_giant_diam'][0, :])
    ax.axvline(x=argmax_p_thin, linestyle='--', linewidth=1, label="max diameter")

    # bottom x-axis: p_thin
    ax.set_xticks(p_thin_indices)
    ax.set_xticklabels(p_thin_labels)
    ax.set_xlabel("p_thin")
    ax.set_ylabel("diameter")
    ax.set_title("Diameters over p_thin")

    # top x-axis: c
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())  # keep alignment
    ax_top.set_xticks(p_thin_indices)
    ax_top.set_xticklabels(c_labels)
    ax_top.set_xlabel(r"$c = max_c(1 - p_{\mathrm{thin}})$")

    for i, lbl in enumerate(ax.get_xticklabels()):
        lbl.set_visible(i % 8 == 0)

    for i, lbl in enumerate(ax_top.get_xticklabels()):
        lbl.set_visible(i % 8 == 0)

    ax.legend()
    fig.tight_layout()

    plt.savefig(f"{save_path}diameter_p_thin_plot.png")
    plt.show()


def create_metric_mean_plots(
        metrics,
        param_name,
        param,
        param_set, 
        p_thins, 
        rhos, 
        save_path
    ):

    num_plots = len(metrics.keys())
    num_rows = int(np.sqrt(num_plots))
    num_cols = int(num_plots / num_rows) + 1
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*12,num_rows*10))

    for i, attr in enumerate(metrics):
        create_system_plot(metrics[attr], axs.flatten()[i], attr, p_thins, rhos)

    plt.suptitle(f'{param_name}: {param}, {param_set}, {rho_p_thin_set}')
    plt.tight_layout()
    plt.savefig(f"{save_path}mean_plots.png")


def create_plots_helper(
        comp_metrics,
        rho_p_thin_set,
        param_name,
        param,
        param_set,
        rhos,
        p_thins
    ):
    save_path = f'{os.getcwd()}/results/{param_name}/{param}/{param_set}/{rho_p_thin_set}/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    diameter_keys = ['mean_average_diam', 'mean_giant_diam']
    focus_keys = ['mean_vpt', 'mean_div_pos', 'mean_div_der', 'mean_div_spect', 
                     'mean_div_spect', 'mean_consistency_correlation']
    
    diameter_metrics = {k: comp_metrics[k] for k in diameter_keys}
    focus_metrics = {k: comp_metrics[k] for k in focus_keys}

    df = pd.read_csv(f'./utils/param_sets/{param_set}.csv')
    c = df['erdos_renyi_c'][0]
    diam_p_thin_argmax = np.argmax(diameter_metrics['mean_giant_diam'][0, :])
    print(f"diameter argmax", diam_p_thin_argmax)
    p_thin_cs = [p_thins[diam_p_thin_argmax]]

    create_metric_mean_plots(focus_metrics, param_name, param, param_set, p_thins, rhos, save_path)
    create_correlation_plots(focus_metrics, save_path, rhos, p_thins)
    create_correlation_line_plots(focus_metrics, save_path, rhos, p_thins, p_thin_cs, c)
    create_diameter_p_thin_plots(diameter_metrics, c, save_path, p_thins, diameter_keys)
    create_column_linear_plots(focus_metrics, save_path, rhos, p_thins, focus_keys)


if __name__ == "__main__":
    """
    Post-Processing Visual Analysis on results
    """
    rho_p_thin_set, param, param_name, param_set = parse_arguments()

    home = os.path.expanduser("~")
    results_path = f'{home}/nobackup/autodelete/results/{param_name}/{param}/{param_set}/{rho_p_thin_set}/'

    rhos, p_thins = load_rho_pthin(rho_p_thin_set)
    comp_metrics = get_average_system_metrics(p_thins, rhos, results_path)
    create_plots_helper(
        comp_metrics, 
        rho_p_thin_set,
        param_name,
        param,
        param_set,
        rhos, 
        p_thins
    )