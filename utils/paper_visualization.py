import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
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

    fig, axes = plt.subplots(2, 1, figsize=(18, 7))

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
    plt.legend(loc="center right", fontsize=20)
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

    fig, ax = plt.subplots(figsize=(18, 5))
    subset = np.random.choice(n, 15, replace=False)

    for i in subset:
        c1, c2 = ("green", "blue") if converged[i] else ("red", "orange")
        ax.plot(t_train, replica_states_1[:, i], color=c1, alpha=0.6)
        ax.plot(t_train, replica_states_2[:, i], color=c2, alpha=0.6, linestyle="--")

    ax.vlines(x=0, ymin=-1, ymax=1, color="black", linestyles="--")
    ax.scatter(np.zeros(len(subset)), replica_states_1[0, subset], c="black", s=15)
    ax.scatter(np.zeros(len(subset)), replica_states_2[0, subset], c="black", s=15)
    ax.annotate(
        r"$\bm{r}(0)$, $\bm{r}'(0)$",
        xy=(0.01, ax.get_ylim()[1]),
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

    legend_elements = [
        Line2D([0], [0], color="green",  linestyle="-", label="Replica 1 (converged)"),
        Line2D([0], [0], color="blue",   linestyle="--", label="Replica 2 (converged)"),
        Line2D([0], [0], color="red",    linestyle="-", label="Replica 1 (diverged)"),
        Line2D([0], [0], color="orange", linestyle="--", label="Replica 2 (diverged)"),
    ]
    ax.legend(handles=legend_elements, loc="center right", fontsize=20)
    ax.set_title(r"Replica Test", y=-0.1)
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
