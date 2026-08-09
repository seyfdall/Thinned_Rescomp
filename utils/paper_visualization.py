import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the 3D projection
from pathlib import Path


def configure_paper_style():
    """Set matplotlib rcParams for publication-quality TeX figures."""
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
        "font.family": "serif",
        'font.sans-serif': ['Computer Modern Roman'],
        'font.serif': ['Computer Modern Roman'],
        'font.size': 30, # TODO: Increase font size to match other plots in paper
    })


def _save_and_show(fig, save_path=None, save_dpi=600):
    """Save figure with publication-quality defaults, then display it."""
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight", facecolor="white")
    plt.show()


def create_system_plot(values, title, p_thins, rhos, save_path=None, label_step=4, suptitle=None):
    values = np.asarray(values)
    p_thins = np.asarray(p_thins)
    rhos = np.asarray(rhos)

    if title == r"$D_\sigma(\bm{r})$":
        z = (values - values.mean()) / values.std()
        values = 1 / (1 + np.exp(-z))  # back into (0,1), now more separated

    if values.shape != (len(rhos), len(p_thins)):
        raise ValueError("values must have shape (len(rhos), len(p_thins))")

    fig, ax = plt.subplots(figsize=(8, 6))
    norm = Normalize(vmin=0, vmax=np.nanmax(values))

    x = np.arange(len(p_thins))
    y = np.arange(len(rhos))

    mesh = ax.pcolormesh(
        x,
        y,
        values,
        shading="nearest",
        cmap="viridis",
        norm=norm,
    )

    ax.set(
        title=title,
        xlabel=r"$\text{p}_{\text{thin}}$",
        ylabel=r"$\rho$",
        xticks=x,
        yticks=y,
        xticklabels=[f"{p:.2f}".rstrip("0").rstrip(".") for p in p_thins],
        yticklabels=[f"{r:.2f}".rstrip("0").rstrip(".") for r in rhos],
    )

    for i, lbl in enumerate(ax.get_xticklabels()):
        lbl.set_visible(i % (5*label_step) == 0)

    for i, lbl in enumerate(ax.get_yticklabels()):
        lbl.set_visible(i % label_step == 0 or i % label_step == 2)

    if suptitle:
        fig.suptitle(suptitle)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(mesh, cax=cax)

    print(save_path)
    _save_and_show(fig, save_path=save_path)


def create_metric_mean_plots(
    metrics,
    param_name,
    param,
    param_set,
    p_thins,
    rhos,
    save_path,
    rho_p_thin_set="",
    label_step=4,
):
    output_dir = Path(save_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    suptitle = f"{param_name}: {param}, {param_set}, {rho_p_thin_set}".strip(", ")
    attrs_to_name = {
        "mean_vpt": "VPT",
        "mean_div_pos": r"$D_s(\bm{r})$",
        "mean_div_der": r"$D_r(\bm{r})$",
        "mean_div_spect": r"$D_\sigma(\bm{r})$",
        "mean_div_rank": r"$D_c(\bm{r})$",
        "mean_consistency_correlation": r"$\Gamma(\bm{r})$"
    }

    for attr, values in metrics.items():
        create_system_plot(
            values,
            title=attrs_to_name[attr],
            p_thins=p_thins,
            rhos=rhos,
            save_path=output_dir / f"{attr}.png",
            label_step=label_step,
        )


def create_correlation_line_plots(
    metrics,
    save_path,
    rhos,
    p_thins,
    p_thin_cs,
    c,
    method="pearson",
    label_step=2,
):
    metric_names = list(metrics.keys())
    sample_shape = next(iter(metrics.values())).shape
    output_dir = Path(save_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_labels = {
        "mean_div_pos": r"$D_s(\bm{r})$",
        "mean_div_der": r"$D_r(\bm{r})$",
        "mean_div_spect": r"$D_\sigma(\bm{r})$",
        "mean_div_rank": r"$D_c(\bm{r})$",
        "mean_consistency_correlation": r"$\Gamma(\bm{r})$",
    }

    def _corr_matrix(series_map):
        stacked = np.asarray([series_map[name] for name in metric_names], dtype=float)
        if stacked.shape[0] < 2:
            return np.eye(stacked.shape[0])
        return np.corrcoef(stacked)

    def _plot_correlation_lines(axis, cors, x_labels, x_axis_name, title):
        x_values = np.arange(len(x_labels))
        vpt_index = metric_names.index("mean_vpt")

        for key in metric_names:
            if key == "mean_vpt":
                continue

            axis.plot(
                x_values,
                [cors[i][metric_names.index(key), vpt_index] for i in range(len(x_labels))],
                label=metric_labels.get(key, key),
            )

        axis.set_xticks(x_values)
        axis.set_xticklabels(x_labels, rotation=45, ha="right")
        if x_axis_name == r"$p_{\mathrm{thin}}$":
            for i, lbl in enumerate(axis.get_xticklabels()):
                lbl.set_visible(i % label_step == 0)

        axis.set_title(title)
        axis.set_xlabel(x_axis_name)
        axis.legend(fontsize=16)
        for i, lbl in enumerate(axis.get_xticklabels()):
            lbl.set_visible(i % 3 == 0)

        axis.tick_params(axis='both', labelsize=20)

    row_cors = []
    for i in range(sample_shape[0]):
        row_cors.append(_corr_matrix({name: mat[i, :] for name, mat in metrics.items()}))

    col_cors = []
    for j in range(sample_shape[1]):
        col_cors.append(_corr_matrix({name: mat[:, j] for name, mat in metrics.items()}))

    rho_labels = [f"{x:.2f}".rstrip("0").rstrip(".") for x in rhos]
    pthin_labels = [f"{x:.2f}".rstrip("0").rstrip(".") for x in p_thins]

    fig, ax = plt.subplots(figsize=(9, 7))
    _plot_correlation_lines(
        ax,
        row_cors,
        rho_labels,
        r"$\rho$",
        r"VPT Correlation over $\rho$",
    )
    fig.tight_layout()
    _save_and_show(fig, save_path=output_dir / f"{method}_rho_correlation_line_plots.png")

    fig, ax = plt.subplots(figsize=(9, 7))
    _plot_correlation_lines(
        ax,
        col_cors,
        pthin_labels,
        r"$p_{\mathrm{thin}}$",
        r"VPT Correlation over $p_{\mathrm{thin}}$",
    )

    # for p_thin_c in p_thin_cs:
    #     p_thin_index = np.interp(p_thin_c, p_thins, range(len(p_thins)))
    #     ax.axvline(
    #         x=p_thin_index,
    #         linestyle="--",
    #         linewidth=1,
    #         label=rf"$c={round(c * (1 - p_thin_c), 1)}$",
    #     )

    ax.axvline(x=60, linestyle="--", linewidth=1, color='black')
    ax.axvline(x=80, linestyle="--", linewidth=1, color='black')

    ax.axvspan(xmin=60, xmax=80, color='gray', alpha=0.3)

    fig.tight_layout()
    _save_and_show(fig, save_path=output_dir / f"{method}_p_thin_correlation_line_plots.png")


def create_column_linear_plots(
    metrics,
    save_path,
    rhos,
    p_thins,
    p_thin=None,
    p_thin_index=None,
    titles=None,
    plot_mode="normalized",
    label_step=2,
):
    # def _normalize_minmax(y):
    #     y = np.asarray(y)
    #     y_min = np.min(y)
    #     y_max = np.max(y)
    #     if np.isclose(y_max, y_min):
    #         return np.zeros_like(y, dtype=float)
    #     return (y - y_min) / (y_max - y_min)

    def _normalize_l2(y):
        return y / np.linalg.norm(y, ord=2)

    p_thins = np.asarray(p_thins)
    rhos = np.asarray(rhos)
    rho_indices = np.arange(len(rhos))
    rho_labels = [f"{x:.2f}".rstrip("0").rstrip(".") for x in rhos]

    if p_thin_index is None:
        if p_thin is None:
            raise ValueError("Either p_thin or p_thin_index must be provided")
        p_thin_index = int(np.argmin(np.abs(p_thins - p_thin)))

    if p_thin_index < 0 or p_thin_index >= len(p_thins):
        raise IndexError("p_thin_index is out of range")

    metric_names = list(metrics.keys())
    if titles is None:
        titles = metric_names

    if len(titles) != len(metric_names):
        raise ValueError("titles must match the number of metrics")

    valid_plot_modes = {"normalized", "l2_error", "cumulative_l2_error"}
    if plot_mode not in valid_plot_modes:
        raise ValueError(f"plot_mode must be one of {sorted(valid_plot_modes)}")

    p_thin_value = float(p_thins[p_thin_index])

    fig, ax = plt.subplots(figsize=(9,7))

    if plot_mode == "normalized":
        plot_title = f"Normalized Standard Plots: p_thin = {p_thin_value:.2f}"
        save_name = "p_thin_normalized_plots"
    elif plot_mode == "l2_error":
        plot_title = f"L2 Error with VPT: p_thin = {p_thin_value:.2f}"
        save_name = "p_thin_normalized_l2_err_plots"
    else:
        # plot_title = f"Cumulative L1 Error with VPT: p_thin = {p_thin_value:.2f}"
        plot_title = f"Cumulative L2 Error with VPT"
        save_name = "p_thin_normalized_cumulative_l2_err_plots"

    cumulative_final_values = {}
    if plot_mode == "cumulative_l2_error":
        for attr in metric_names:
            if attr == "mean_vpt":
                continue
            diff = (
                _normalize_l2(metrics[attr][:, p_thin_index])
                - _normalize_l2(metrics["mean_vpt"][:, p_thin_index])  
            )
            
            cumulative_curve = np.sqrt(np.cumsum(diff**2))
            cumulative_final_values[attr] = float(cumulative_curve[-1])

        final_vals = np.asarray(list(cumulative_final_values.values()), dtype=float)
        cumulative_cmap = plt.get_cmap("plasma")
        if len(final_vals) <= 1 or np.isclose(np.max(final_vals), np.min(final_vals)):
            cumulative_norm = Normalize(vmin=0.0, vmax=1.0)
        else:
            cumulative_norm = Normalize(vmin=float(np.min(final_vals)), vmax=float(np.max(final_vals)*1.1))

    for i, attr in enumerate(metric_names):
        y_values = metrics[attr][:, p_thin_index]
        if plot_mode == "normalized":
            plot_values = _normalize_l2(y_values)
        elif plot_mode == "l1_error":
            if attr == "mean_vpt":
                continue
            plot_values = np.abs(
                _normalize_l2(y_values) - _normalize_l2(metrics["mean_vpt"][:, p_thin_index])
            )
        else:
            if attr == "mean_vpt":
                continue
            plot_values = np.cumsum(
                np.abs(_normalize_l2(y_values) - _normalize_l2(metrics["mean_vpt"][:, p_thin_index]))
            )

        line_kwargs = {}
        if plot_mode == "cumulative_l2_error":
            if len(cumulative_final_values) <= 1:
                line_kwargs["color"] = cumulative_cmap(0.5)
            else:
                line_kwargs["color"] = cumulative_cmap(cumulative_norm(cumulative_final_values[attr]))

        ax.plot(
            rho_indices,
            plot_values,
            label=titles[i],
            **line_kwargs,
        )

    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("Cumulative L2 Error")
    ax.set_xticks(rho_indices)
    ax.set_xticklabels(rho_labels)
    ax.set_title(plot_title)
    ax.tick_params(axis='both', labelsize=21)

    # for i, lbl in enumerate(ax.get_xticklabels()):
    #     lbl.set_visible(i % label_step == 0)

    ax.legend(fontsize=18)
    fig.tight_layout()

    output_dir = Path(save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_and_show(fig, save_path=output_dir / f"{save_name}_p_thin_{p_thin_value:.2f}.png")


def plot_reservoir_processing(reservoir_states, u_true, u_hat, T, t, n, save_path=None, save_dpi=600):
    cmap = plt.get_cmap('plasma')

    initial_vals = reservoir_states[0]
    order = np.argsort(initial_vals)
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.linspace(0, 1, len(initial_vals))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --- Signal ---
    ax1 = axes[0]

    for i, u in enumerate(u_true.T):
        ax1.plot(t[:T], u[:T], color="blue", label="True" if i == 0 else None, alpha=0.7)

    signal_bottom, signal_top = ax1.get_ylim()

    ax1.vlines(x=0, ymin=signal_bottom, ymax=signal_top, color="blue", linestyles="--")
    ax1.scatter(np.zeros(3), u_hat[0], c="blue", s=15)
    ax1.annotate(
        r"$\bm{u}(0)$",
        xy=(0, ax1.get_ylim()[1]),
        xytext=(0, 0),
        textcoords="offset points",
        ha='center',
        va='bottom',
    )

    ax1.vlines(x=T, ymin=signal_bottom, ymax=signal_top, color="blue", linestyles="--")
    ax1.scatter(np.ones(3) * T, u_hat[T], c="blue", s=15)
    ax1.annotate(
        r"$\bm{u}(T)$",
        xy=(T, ax1.get_ylim()[1]),
        xytext=(-5.0, 0),
        textcoords="offset points",
        ha='center',
        va='bottom',
    )

    ax1.set_yticks([0])
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xticks([])
    ax1.spines['bottom'].set_position(('data', 0))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title(
        r"$\bm{u}(t) \in \mathbb{R}^m$",
        y=-0.1,
    )

    # --- Reservoir traces ---
    ax2 = axes[1]

    for r, c in zip(reservoir_states.T, ranks):
        ax2.plot(t[:T], r[:T], color=cmap(c), alpha=0.6)

    ax2.vlines(x=0, ymin=-1, ymax=1, color="black", linestyles="--")
    ax2.scatter(np.zeros(n), reservoir_states[0], c="black", s=15)
    ax2.annotate(
        r"$\bm{r}(0) = \bm{r}_0$",
        xy=(0, ax2.get_ylim()[1]),
        xytext=(0, 0),
        textcoords="offset points",
        ha='center',
        va='bottom',
    )

    ax2.vlines(x=T, ymin=-1, ymax=1, color="black", linestyles="--")
    ax2.scatter(np.ones(n) * T, reservoir_states[T], c="black", s=15)
    ax2.annotate(
        r"$\bm{r}(T)$",
        xy=(T, ax2.get_ylim()[1]),
        xytext=(0, 0),
        textcoords="offset points",
        ha='center',
        va='bottom',
    )

    ax2.set_yticks([-1, 0, 1])
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_xticks([])
    ax2.spines['bottom'].set_position(('data', 0))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title(r"$\bm{r}(t) \in \mathbb{R}^n$", y=-0.1)

    plt.tight_layout()

    fig.subplots_adjust(wspace=0.4)

    bbox0 = ax1.get_position()
    bbox1 = ax2.get_position()

    x0 = bbox0.x1
    x1 = bbox1.x0
    y  = 0.4 * (bbox0.y0 + bbox0.y1)

    x_mid = 0.5 * (x0 + x1)
    half_width = 0.05   # controls arrow length

    plt.annotate(
        "",
        xy=(x_mid + half_width, y),
        xytext=(x_mid - 1.75*half_width, y),
        xycoords="figure fraction",
        textcoords="figure fraction",
        arrowprops=dict(arrowstyle="->", linewidth=2.5),
    )

    # Text (placed above arrow)
    plt.text(
        0.5 * (x0 + x1),   # midpoint
        y + 0.08,          # slightly above
        r"$\mathbf{A} \in \mathbb{R}^{n \times n}$",
        ha="center",
        va="bottom",
        transform=plt.gcf().transFigure,
    )
    _save_and_show(fig, save_path=save_path, save_dpi=save_dpi)


def plot_reservoir_aggregation(reservoir_states, u_true, u_hat, T, t, n, save_path=None, save_dpi=600):
    cmap = plt.get_cmap('plasma')

    initial_vals = reservoir_states[0]
    order = np.argsort(initial_vals)
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.linspace(0, 1, len(initial_vals))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --- Reservoir traces ---
    ax1 = axes[0]

    for r, c in zip(reservoir_states.T, ranks):
        ax1.plot(t[:T], r[:T], color=cmap(c), alpha=0.6)

    ax1.vlines(x=0, ymin=-1, ymax=1, color="black", linestyles="--")
    ax1.scatter(np.zeros(n), reservoir_states[0], c="black", s=15)
    ax1.annotate(
        r"$\bm{r}(0)$",
        xy=(0, ax1.get_ylim()[1]),
        xytext=(0, 0),
        textcoords="offset points",
        ha='center',
        va='bottom',
    )

    ax1.vlines(x=T, ymin=-1, ymax=1, color="black", linestyles="--")
    ax1.scatter(np.ones(n) * T, reservoir_states[T], c="black", s=15)
    ax1.annotate(
        r"$\bm{r}(T)$",
        xy=(T, ax1.get_ylim()[1]),
        xytext=(0, 0),
        textcoords="offset points",
        ha='center',
        va='bottom',
    )

    ax1.set_yticks([-1, 0, 1])
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xticks([])
    ax1.spines['bottom'].set_position(('data', 0))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title(r"$\bm{r}(t) \in \mathbb{R}^n$", y=-0.1)

    # --- Signal prediction ---
    ax2 = axes[1]

    for i, u in enumerate(u_hat.T):
        ax2.plot(t[:T], u[:T], color="orange", label="Predicted" if i == 0 else None, alpha=0.7)

    signal_bottom, signal_top = ax2.get_ylim()

    ax2.vlines(x=0, ymin=signal_bottom, ymax=signal_top, color="orange", linestyles="--")
    ax2.scatter(np.zeros(3), u_hat[0], c="orange", s=15)
    ax2.annotate(
        r"$\bm{\hat u}(0)$",
        xy=(0, ax2.get_ylim()[1]),
        xytext=(0, 0),
        textcoords="offset points",
        ha='center',
        va='bottom',
    )

    ax2.vlines(x=T, ymin=signal_bottom, ymax=signal_top, color="orange", linestyles="--")
    ax2.scatter(np.ones(3) * T, u_hat[T], c="orange", s=15)
    ax2.annotate(
        r"$\bm{\hat u}(T)$",
        xy=(T, ax2.get_ylim()[1]),
        xytext=(-5.0, 0),
        textcoords="offset points",
        ha='center',
        va='bottom',
    )

    ax2.set_yticks([0])
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_xticks([])
    ax2.spines['bottom'].set_position(('data', 0))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title(
        r"$\bm{\hat u}(t) = \bm{W}_{\text{out}}\bm{r}(t) \in \mathbb{R}^m$",
        y=-0.1,
    )

    plt.tight_layout()

    fig.subplots_adjust(wspace=0.4)

    bbox0 = ax1.get_position()
    bbox1 = ax2.get_position()

    x0 = bbox0.x1
    x1 = bbox1.x0
    y  = 0.4 * (bbox0.y0 + bbox0.y1)

    x_mid = 0.5 * (x0 + x1)
    half_width = 0.05   # controls arrow length

    plt.annotate(
        "",
        xy=(x_mid + half_width, y),
        xytext=(x_mid - 1.75*half_width, y),
        xycoords="figure fraction",
        textcoords="figure fraction",
        arrowprops=dict(arrowstyle="->", linewidth=2.5),
    )

    # Text (placed above arrow)
    plt.text(
        0.5 * (x0 + x1),   # midpoint
        y + 0.08,          # slightly above
        r"$\mathbf{W}_{\mathrm{out}} \in \mathbb{R}^{m \times n}$",
        ha="center",
        va="bottom",
        transform=plt.gcf().transFigure,
    )

    _save_and_show(fig, save_path=save_path, save_dpi=save_dpi)


def plot_reservoir_prediction(reservoir_states, u_true, u_hat, T, t, n, save_path=None, save_dpi=600):
    cmap = plt.get_cmap('plasma')

    initial_vals = reservoir_states[0]
    order = np.argsort(initial_vals)
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.linspace(0, 1, len(initial_vals))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --- Reservoir traces ---
    ax1 = axes[0]

    for r, c in zip(reservoir_states.T, ranks):
        ax1.plot(t[T:] - t[T], r[T:], color=cmap(c), alpha=0.6)

    ax1.vlines(x=0, ymin=-1, ymax=1, color="black", linestyles="--")
    ax1.scatter(np.zeros(n), reservoir_states[T], c="black", s=15)
    ax1.annotate(
        r"$\bm{r}(T) = \bm{\hat r}(T)$",
        xy=(0, ax1.get_ylim()[1]),
        xytext=(0, 0),
        textcoords="offset points",
        ha='center',
        va='bottom',
    )

    ax1.set_yticks([-1, 0, 1])
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xticks([])
    ax1.spines['bottom'].set_position(('data', 0))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title(r"$\bm{r}(t) \in \mathbb{R}^n$", y=-0.1)

    # --- Signal prediction ---
    ax2 = axes[1]

    for i, u in enumerate(u_hat.T):
        ax2.plot(t[T:] - t[T], u[T:], color="orange", label="Predicted" if i == 0 else None, alpha=0.7)

    signal_bottom, signal_top = ax2.get_ylim()

    ax2.vlines(x=0, ymin=signal_bottom, ymax=signal_top, color="orange", linestyles="--")
    ax2.scatter(np.zeros(3), u_hat[T], c="orange", s=15)
    ax2.annotate(
        r"$\bm{\hat u}(T)$",
        xy=(0, ax2.get_ylim()[1]),
        xytext=(0, 0),
        textcoords="offset points",
        ha='center',
        va='bottom',
    )

    ax2.set_yticks([0])
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_xticks([])
    ax2.spines['bottom'].set_position(('data', 0))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title(
        r"$\bm{\hat u}(t) \in \mathbb{R}^m$",
        y=-0.1,
    )

    plt.tight_layout()

    fig.subplots_adjust(wspace=0.4)

    bbox0 = ax1.get_position()
    bbox1 = ax2.get_position()

    x0 = bbox0.x1
    x1 = bbox1.x0
    y  = 0.4 * (bbox0.y0 + bbox0.y1)
    half_height = 0.5 * (bbox0.y0 + bbox0.y1) / 2.0

    x_mid = 0.5 * (x0 + x1)
    half_width = 0.05   # controls arrow length

    plt.annotate(
        "",
        xy=(x_mid + half_width, y - half_height),
        xytext=(x_mid - 1.75*half_width, y - half_height),
        xycoords="figure fraction",
        textcoords="figure fraction",
        arrowprops=dict(
            arrowstyle="<-", 
            linewidth=2.5,
            connectionstyle="arc3,rad=0.2"
        ),
    )

    # Text (placed above arrow)
    plt.text(
        0.5 * (x0 + x1),   # midpoint
        y + 0.08 - half_height,          # slightly above
        r"$\mathbf{A} \in \mathbb{R}^{m \times n}$",
        ha="center",
        va="bottom",
        transform=plt.gcf().transFigure,
    )

    plt.annotate(
        "",
        xy=(x_mid + half_width, y + half_height),
        xytext=(x_mid - 1.75*half_width, y + half_height),
        xycoords="figure fraction",
        textcoords="figure fraction",
        arrowprops=dict(
            arrowstyle="->", 
            linewidth=2.5,
            connectionstyle="arc3,rad=-0.2"
        ),
    )

    # Text (placed below arrow)
    plt.text(
        0.5 * (x0 + x1),   # midpoint
        y - 0.05 + half_height,          # slightly above
        r"$\mathbf{W}_{\mathrm{out}} \in \mathbb{R}^{n \times n}$",
        ha="center",
        va="bottom",
        transform=plt.gcf().transFigure,
    )

    _save_and_show(fig, save_path=save_path, save_dpi=save_dpi)


def plot_reservoir_response(reservoir_states, u_true, u_hat, T, t, vpt, n, save_path=None, save_dpi=600):
    """Two-panel figure: coloured reservoir node traces (top) and signal prediction (bottom)."""
    cmap = plt.get_cmap('plasma')

    initial_vals = reservoir_states[0]
    order = np.argsort(initial_vals)
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.linspace(0, 1, len(initial_vals))

    fig, axes = plt.subplots(2, 1, figsize=(18, 9))

    # --- Reservoir traces ---
    ax1 = axes[0]

    for r, c in zip(reservoir_states.T, ranks):
        ax1.plot(t, r, color=cmap(c), alpha=0.6)

    ax1.vlines(x=0, ymin=-1, ymax=1, color="black", linestyles="--")
    ax1.scatter(np.zeros(n), reservoir_states[0], c="black", s=15)
    ax1.annotate(
        r"$\bm{r}(0)$",
        xy=(0, ax1.get_ylim()[1]),
        xytext=(0, 0),
        textcoords="offset points",
        ha='center',
        va='bottom',
    )

    ax1.vlines(x=T, ymin=-1, ymax=1, color="black", linestyles="--")
    ax1.scatter(np.ones(n) * T, reservoir_states[T], c="black", s=15)
    ax1.annotate(
        r"$\bm{\hat r}(T) = \bm{r}(T)$",
        xy=(T, ax1.get_ylim()[1]),
        xytext=(0, 0),
        textcoords="offset points",
        ha='center',
        va='bottom',
    )

    ax1.set_yticks([-1, 0, 1])
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xticks([])
    ax1.spines['bottom'].set_position(('data', 0))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title(r"Reservoir Response: $\bm{r}(t), \bm{\hat r}(t) \in \mathbb{R}^n$", y=-0.2)

    # --- Signal prediction ---
    ax2 = axes[1]

    for i, u in enumerate(u_true.T):
        ax2.plot(t, u, color="blue", label="True" if i == 0 else None, alpha=0.7)

    for i, u in enumerate(u_hat.T):
        ax2.plot(t, u, color="orange", label="Predicted" if i == 0 else None, alpha=0.7)

    signal_bottom, signal_top = ax2.get_ylim()

    ax2.vlines(x=0, ymin=signal_bottom, ymax=signal_top, color="black", linestyles="--")
    ax2.scatter(np.zeros(3), u_hat[0], c="black", s=15)
    ax2.annotate(
        r"$\bm{\hat u}(0)$",
        xy=(0, ax2.get_ylim()[1]),
        xytext=(0, 0),
        textcoords="offset points",
        ha='center',
        va='bottom',
    )


    ax2.vlines(x=T, ymin=signal_bottom, ymax=signal_top, color="black", linestyles="--")
    ax2.scatter(np.ones(3) * T, u_hat[T], c="black", s=15)
    ax2.annotate(
        r"$\bm{\hat u}(T)$",
        xy=(T, ax2.get_ylim()[1]),
        xytext=(-20.0, 0),
        textcoords="offset points",
        ha='center',
        va='bottom',
    )


    vpt_x = vpt * 100 + T
    if vpt_x - T > 50:
        ax2.vlines(x=vpt_x, ymin=signal_bottom, ymax=signal_top, color="black", linestyles="--")
        ax2.annotate(
            r"$\bm{\hat u}(T^*)$",
            xy=(vpt_x, ax2.get_ylim()[1]),
            xytext=(20.0, 0),
            textcoords="offset points",
            ha='center',
            va='bottom',
        )

    ax2.set_yticks([0])
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_xticks([])
    ax2.spines['bottom'].set_position(('data', 0))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title(
        r"Predicted Signal: $\bm{\hat u}(t) = \bm{W}_{\text{out}} \bm{\hat r} (t) \in \mathbb{R}^m$",
        y=-0.2,
    )

    fig.subplots_adjust(hspace=0.5)
    _save_and_show(fig, save_path=save_path, save_dpi=save_dpi)


def plot_replica_pair(replica_states_1, replica_states_2, t_train, n, save_path=None, save_dpi=600):
    """Two-panel coloured trace plot of each replica's state trajectories."""
    cmap = plt.get_cmap('plasma')

    fig, axes = plt.subplots(2, 1, figsize=(18, 7))

    for ax, states, label in [
        (axes[0], replica_states_1, r"$\bm{r}(0)$"),
        (axes[1], replica_states_2, r"$\bm{r}'(0)$"),
    ]:
        initial_vals = states[0]
        order = np.argsort(initial_vals)
        ranks = np.empty(len(order), dtype=float)
        ranks[order] = np.linspace(0, 1, len(initial_vals))

        for r, c in zip(states.T, ranks):
            ax.plot(t_train, r, color=cmap(c), alpha=0.6)

        ax.vlines(x=0, ymin=-1, ymax=1, color="black", linestyles="--")
        ax.scatter(np.zeros(n), states[0], c="black", s=15)
        ax.annotate(
            label,
            xy=(0, ax.get_ylim()[1]),
            xytext=(0, 0),
            textcoords="offset points",
            ha='center',
            va='bottom',
        )

        ax.set_yticks([-1, 0, 1])
        ax.set_xticks([])
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_title(r"Replica Test 1", y=-0.1)
    axes[1].set_title(r"Replica Test 2", y=-0.25)
    fig.subplots_adjust(hspace=0.25)
    _save_and_show(fig, save_path=save_path, save_dpi=save_dpi)


def plot_replica_convergence(replica_states_1, replica_states_2, t_train, n, tail=50, conv_tol=1e-3, save_path=None, save_dpi=600):
    """Overlay of 15 sampled node pairs coloured by whether they converge."""
    converged = np.array([
        np.linalg.norm(replica_states_1[-tail:, i] - replica_states_2[-tail:, i]) < conv_tol
        for i in range(n)
    ])

    fig, ax = plt.subplots(figsize=(8, 4))
    subset = np.random.choice(n, 15, replace=False)
    cutoff = 60

    for i in subset:
        c1, c2 = ("green", "blue") if converged[i] else ("red", "orange")
        ax.plot(t_train[:cutoff], replica_states_1[:cutoff, i], color=c1, alpha=0.4)
        ax.plot(t_train[:cutoff], replica_states_2[:cutoff, i], color=c2, alpha=0.4)

    ax.vlines(x=0, ymin=-1, ymax=1, color="black", linestyles="--")
    ax.scatter(np.zeros(len(subset)), replica_states_1[0, subset], c="black", s=15)
    ax.scatter(np.zeros(len(subset)), replica_states_2[0, subset], c="black", s=15)
    ax.annotate(
        r"$\bm{\phi}(0)$, $\bm{\psi}(0)$",
        xy=(0.01, ax.get_ylim()[1]),
        xytext=(0, 0),
        textcoords="offset points",
        ha='center',
        va='bottom',
        fontsize=20
    )

    ax.set_yticks([-1, 0, 1])
    ax.set_xticks([])
    ax.tick_params(axis='both', labelsize=20)
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ax.hlines(y=[-0.8], xmin=[t_train[0]], xmax=[t_train[cutoff-1]], colors=['black'], linestyles=["--"])

    # legend_elements = [
    #     Line2D([0], [0], color="green",  linestyle="-", label="Replica 1 (converged)"),
    #     Line2D([0], [0], color="blue",   linestyle="-", label="Replica 2 (converged)"),
    #     Line2D([0], [0], color="red",    linestyle="-", label="Replica 1 (diverged)"),
    #     Line2D([0], [0], color="orange", linestyle="-", label="Replica 2 (diverged)"),
    # ]
    # ax.legend(handles=legend_elements, loc="center right", fontsize=20)
    ax.set_title(r"Stable Reservoir", y=-0.1, fontsize=20)
    _save_and_show(fig, save_path=save_path, save_dpi=save_dpi)


def plot_reservoir_heatmap(reservoir_states, u_true, u_hat, T, t, save_path=None, save_dpi=600):
    """Heatmap of z-scored reservoir states (top) with plain signal comparison (bottom)."""
    fig, ax = plt.subplots(figsize=(18, 5))

    Rnorm = (reservoir_states - reservoir_states.mean(axis=0)) / reservoir_states.std(axis=0)
    im = ax.imshow(Rnorm.T, aspect="auto", origin="lower")
    ax.axvline(0, color="black", linestyle="--")
    ax.axvline(T, color="black", linestyle="--")
    ax.set_ylabel("Reservoir Node Index")
    ax.set_title(r"Reservoir Response: $\mathbf{r}(t) \in \mathbb{R}^n$")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Node Value")

    plt.tight_layout()
    _save_and_show(fig, save_path=save_path, save_dpi=save_dpi)


def plot_lorenz_attractor(U_test, U_hat_pred, save_path=None, save_dpi=600):
    """3D phase-space plot of true vs RC-predicted Lorenz trajectory."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot(*U_test.T, color="blue", label="True")
    ax.plot(*U_hat_pred.T, color="orange", label="RC")
    ax.set_title("Lorenz Attractor Prediction")
    plt.legend(fontsize=20)
    _save_and_show(fig, save_path=save_path, save_dpi=save_dpi)
