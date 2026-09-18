import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.integrate import odeint

from mmfm.data import u_sine, u_vdp, u_waves
from mmfm.utils import color_picker, create_plot_grid


def plot_results_mmfm(
    X,
    y,
    t,
    trajectory,
    idx_plot,
    n_classes,
    save=False,
    title=False,
    filepath=None,
    s=10,
    ncols=None,
    plot_ode=None,
    paper_style=False,
):
    """Plot results of MMFM model."""
    if paper_style:
        params = {
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "legend.title_fontsize": 18,
            "figure.titlesize": 18,
        }
        plt.rcParams.update(params)
    fig, ax, ncols, _ = create_plot_grid(n_classes, ncols=ncols)

    df = pd.DataFrame(X.reshape(-1, 2)).assign(target=y.reshape(-1, 1))
    df.columns = ["x", "y", "target"]

    color_classes = [int(x) for x in range(len([x for x in np.unique(y) if np.isfinite(x)]))]
    colors = color_picker(color_classes)
    non_nan_targets = [x for x in np.unique(y) if np.isfinite(x)]
    for k, c in enumerate(non_nan_targets):
        axidx = ax[k // ncols, k % ncols]
        sns.scatterplot(data=df, x="x", y="y", hue="target", ax=axidx, legend=False, alpha=0.05, palette=colors, s=s)
        sns.scatterplot(data=df[df["target"] == c], x="x", y="y", ax=axidx, color=colors[k], s=15)
        axidx.set_title(f"c={c}")

    condition_to_class = {c: i for i, c in enumerate(sorted([x for x in np.unique(y) if np.isfinite(x)]), 1)}
    marginal_timepoints = [int(x) for x in (np.unique(t) * (len(trajectory) - 1))]
    for n in idx_plot:
        p = condition_to_class[y[n, 0]] - 1
        # Check if p is integer. If yes, cast, if not, raise error
        if not p.is_integer():
            raise ValueError("Target must be an integer.")
        p = int(p)
        if n_classes >= 3:
            axidx = ax[p // ncols, p % ncols]
        elif n_classes == 2:
            axidx = ax[0, p]
        else:
            axidx = ax[p]
        # Draw a line between consecutive marginal_timepoints
        for i in range(len(trajectory) - 1):
            axidx.plot(
                [trajectory[i, n, 0], trajectory[i + 1, n, 0]],
                [trajectory[i, n, 1], trajectory[i + 1, n, 1]],
                color="black",
                lw=2,
            )
        # Highlight start and end marginal_timepoints
        axidx.scatter(
            trajectory[marginal_timepoints, n, 0],
            trajectory[marginal_timepoints, n, 1],
            color="black",
            label="Interpolated",
            s=20,
        )

        if plot_ode is not None:
            y0, c = trajectory[0, n], y[n, 0]
            if plot_ode == "vdp":
                t = np.linspace(0, 10, 101)
                sol = odeint(u_vdp, y0, t, args=(c,))
                axidx.plot(sol[:, 0], sol[:, 1], color="blue", alpha=0.1, lw=5)

            elif plot_ode == "u_sine":
                t = np.linspace(0, 1, 101)
                sol = odeint(u_sine, y0, t, args=(c,))
                axidx.plot(sol[:, 0], sol[:, 1], color="blue", alpha=0.1, lw=5)

            elif plot_ode == "u_waves":
                t = np.linspace(0, 1, 101)
                sol = odeint(u_waves, y0, t, args=(c,))
                axidx.plot(sol[:, 0], sol[:, 1], color="blue", alpha=0.1, lw=5)

            elif plot_ode == "u":
                t = np.linspace(0, 1, 101)
                sol = odeint(u, y0, t, args=(c,))
                axidx.plot(sol[:, 0], sol[:, 1], color="blue", alpha=0.2, lw=7.5)

    plt.suptitle(f"{title}", fontsize=12)
    plt.tight_layout()
    if save:
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()
