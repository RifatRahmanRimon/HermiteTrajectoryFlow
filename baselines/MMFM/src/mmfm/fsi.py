from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import seaborn as sns
from addict import Dict
from scipy.integrate import odeint
from scipy.interpolate import CubicSpline

from mmfm.data import u, u_linear, u_sine, u_vdp, u_waves
from mmfm.utils import color_picker, create_plot_grid


class FSI:
    """Fast & Smooth Interpolation (FSI) model.

    This class implements the Fast & Smooth Interpolation (FSI) model, which is
    extended with conditional Monge maps for the multi-marginal setting. The model
    learns Monge maps between marginals and interpolates between them using cubic splines.

    Reference:
        Chewi, Sinho, et al. "Fast and smooth interpolation on wasserstein space."
        International Conference on Artificial Intelligence and Statistics. PMLR, 2021.
    """

    def __init__(self, conditional):
        self.conditional = conditional
        self.monge_maps = None
        self.mm_from_to = None
        self.splines = None

    def train(self, X, y, t):
        """Derive Monge maps between marginals.

        This class covers a lot of edge cases for the multi-marginal setting regarding
        the organization of the data, since the tensors are of shape
        [samples, timepoints, features] and contain missings if the data is not fully observed.
        """
        monge_maps = Dict()
        mm_from_to = Dict()

        if self.conditional:
            all_classes = sorted([x for x in np.unique(y) if np.isfinite(x)])
        all_times = sorted(np.unique(t))
        self.all_times = all_times

        # Learn Monge maps between marginals
        if not self.conditional:
            for k, time in enumerate(all_times):
                if k == (len(all_times) - 1):
                    break
                ot_emd = ot.da.EMDTransport()
                ot_emd.fit(Xs=X[:, k], Xt=X[:, k + 1])
                monge_maps[time] = ot_emd
                monge_maps[time] = (k, k + 1)

        else:
            for c in all_classes:
                for k, time in enumerate(all_times):
                    if k == (len(all_times) - 1):
                        break
                    idx = y[:, k] == c
                    if not all(np.isnan(y[idx, k])):
                        found_valid_step = False
                        for next_existing_timestep in all_times[k + 1 :]:
                            k_timestep_next = all_times.index(next_existing_timestep)
                            if not (
                                any(np.isnan(y[idx, k_timestep_next]))
                                & np.isnan(X[idx, k_timestep_next]).any()
                            ):
                                found_valid_step = True
                                break
                            # We will have left the previous loop if the final timepoint has only missings

                        # for next_existing_timestep in all_timesteps[k + 1 :]:
                        if found_valid_step:
                            ot_emd = ot.da.EMDTransport(limit_max=1e6)
                            if np.isnan(X[idx, k]).any():
                                print("Nan in X")
                                raise ValueError("Nan in X")
                            if np.isnan(X[idx, k_timestep_next]).any():
                                print("Nan in X")
                                raise ValueError("Nan in X")

                            ot_emd.fit(Xs=X[idx, k], Xt=X[idx, k_timestep_next])
                            monge_maps[time][c] = ot_emd
                            mm_from_to[time][c] = (k, k_timestep_next)
                        else:
                            monge_maps[time][c] = None
                            mm_from_to[time][c] = None
                    else:
                        monge_maps[time][c] = None
                        mm_from_to[time][c] = None

        self.t_anchors = t
        self.monge_maps = monge_maps.to_dict()
        self.is_trained = True
        self.mm_from_to = mm_from_to.to_dict()

    def interpolate_from_x0(self, X, y=None, t_query=None):
        """Interpolate between Monge maps using cubic splines."""
        # Forward X in time and interpolate at t_query
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        if y is not None and not self.conditional:
            raise ValueError("Model is not conditional.")

        # Compute cubic spline interpolation between Monge maps
        Xs = np.nan * np.ones(shape=(X.shape[0], len(self.all_times), X.shape[1]))
        if not self.conditional:
            Xs[:, 0] = X
            for k, time in enumerate(self.all_times, 1):
                if k == len(self.all_times):
                    break
                Xs[:, k] = self.monge_maps[time].transform(Xs[:, k - 1])
        else:
            Xs[:, 0] = X
            for time, v in self.mm_from_to.items():
                if v is None:
                    continue
                for c, from_to_ in v.items():
                    if from_to_ is None:
                        continue
                    from_, to_ = from_to_
                    if c != y:
                        continue
                    # print(f"Time: {time}, c: {c}, from: {from_}, to: {to_}")
                    Xs[:, to_] = self.monge_maps[time][c].transform(Xs[:, from_])

        # Interpolate between Monge map based project using cubic splines
        splines = []
        for i in range(X.shape[0]):
            # build the splines based on top of all time steps I saw during training
            nonnans = [
                idx
                for idx, _ in enumerate(self.all_times)
                if not np.isnan(Xs[i, idx]).any()
            ]
            splines.append(
                CubicSpline(
                    self.t_anchors[i][nonnans],
                    Xs[i, nonnans],
                    axis=0,
                    bc_type="natural",
                )
            )

        self.splines = splines

        if t_query is not None:
            X_interp = self._eval_splines_at_t(t_query, X)
            return X_interp

    def _eval_splines_at_t(self, t_query, X):
        if not isinstance(t_query, int | float):
            raise ValueError("t_query must be a single timepoint.")
        X_interp = np.nan * np.ones(shape=(X.shape[0], X.shape[1]))
        for i, spline in enumerate(self.splines):
            X_interp[i] = spline(t_query)
        return X_interp

    def plot_interpolation(
        self,
        X,
        y,
        t,
        n_classes,
        idx_plot=None,
        title="FSI",
        save=False,
        filename="",
        filepath="",
        coupling="cot",
        s=10,
        ncols=None,
        plot_ode=None,
    ):
        """Plot interpolation between Monge maps using cubic splines."""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        if not self.splines:
            raise ValueError("Model has not been trained yet.")

        df = pd.DataFrame(X.reshape(-1, 2)).assign(target=y.reshape(-1, 1))
        df.columns = ["x", "y", "target"]

        fig, ax, ncols, _ = create_plot_grid(n_classes, ncols=ncols)

        T = 100
        trajectory = np.nan * np.ones(shape=(T, len(idx_plot), 2))
        for idx, sample in enumerate(idx_plot):
            for tx in range(T):
                transport_c = self.interpolate_from_x0(
                    X=X[sample, 0][None, :],
                    y=y[sample, 0] if coupling == "cot" else None,
                    t_query=tx / T,
                )
                trajectory[tx, idx] = transport_c

        color_classes = [
            int(x) for x in range(len([x for x in np.unique(y) if np.isfinite(x)]))
        ]
        colors = color_picker(color_classes)
        non_nan_targets = [x for x in np.unique(y) if np.isfinite(x)]
        for k, c in enumerate(non_nan_targets):
            axidx = ax[k // ncols, k % ncols]

            sns.scatterplot(
                data=df,
                x="x",
                y="y",
                hue="target",
                ax=axidx,
                legend=False,
                alpha=0.01,
                palette=colors,
                s=s,
            )
            sns.scatterplot(
                data=df[df["target"] == c],
                x="x",
                y="y",
                ax=axidx,
                color=colors[k],
                s=10,
            )
            axidx.set_title(f"c={c}")

        condition_to_class = {
            c: i for i, c in enumerate([x for x in np.unique(y) if np.isfinite(x)], 1)
        }
        # marginal_timepoints = np.linspace(0, len(trajectory) - 1, n_marginals).astype(int)
        marginal_timepoints = [int(x) for x in (np.unique(t) * (len(trajectory) - 1))]
        for n, idx in enumerate(idx_plot):
            p = condition_to_class[y[idx, 0]] - 1
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
                s=12.5,
            )

            if plot_ode is not None:
                y0, c = trajectory[0, n], y[idx, 0]
                if plot_ode == "vdp":
                    t = np.linspace(0, 10, 101)
                    sol = odeint(u_vdp, y0, t, args=(c,))
                    axidx.plot(sol[:, 0], sol[:, 1], color="blue", alpha=0.2, lw=7.5)

                elif plot_ode == "u_sine":
                    t = np.linspace(0, 1, 101)
                    sol = odeint(u_sine, y0, t, args=(c,))
                    axidx.plot(sol[:, 0], sol[:, 1], color="blue", alpha=0.2, lw=7.5)

                elif plot_ode == "u_waves":
                    t = np.linspace(0, 1, 101)
                    sol = odeint(u_waves, y0, t, args=(c,))
                    axidx.plot(sol[:, 0], sol[:, 1], color="blue", alpha=0.2, lw=7.5)

                elif plot_ode == "u_linear":
                    t = np.linspace(0, 1, 101)
                    sol = odeint(u_linear, y0, t, args=(c,))
                    axidx.plot(sol[:, 0], sol[:, 1], color="blue", alpha=0.2, lw=7.5)

                elif plot_ode == "u":
                    t = np.linspace(0, 1, 101)
                    sol = odeint(u, y0, t, args=(c,))
                    axidx.plot(sol[:, 0], sol[:, 1], color="blue", alpha=0.2, lw=7.5)

        if title is not None:
            plt.suptitle(title, fontsize=12)
        plt.tight_layout()
        if save:
            file_path = Path(filepath) / f"{filename}.png"
            plt.savefig(file_path)
        plt.show()
        
    def plot_interpolation_beijing(
        self,
        X,
        y,
        t,
        n_classes,
        idx_plot=None,
        title="FSI",
        save=False,
        filename="",
        filepath="",
        coupling="cot",
        s=10,
        ncols=None,
        plot_ode=None,
    ):
        """Plot interpolation between Monge maps using cubic splines."""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        if not self.splines:
            raise ValueError("Model has not been trained yet.")
        
        df = pd.DataFrame(X.reshape(-1, 1)).assign(target=y.reshape(-1, 1), time=t.reshape(-1, 1))
        df.columns = ["y", "target", "x"]

        # df = pd.DataFrame(X.reshape(-1, 2)).assign(target=y.reshape(-1, 1))
        # df.columns = ["x", "y", "target"]

        fig, ax, ncols, _ = create_plot_grid(n_classes, ncols=ncols)

        T = 100
        trajectory = np.nan * np.ones(shape=(T, len(idx_plot), 1))
        for idx, sample in enumerate(idx_plot):
            for tx in range(T):
                transport_c = self.interpolate_from_x0(
                    X=X[sample, 0][None, :],
                    y=y[sample, 0] if coupling == "cot" else None,
                    t_query=tx / T,
                )
                trajectory[tx, idx] = transport_c

        color_classes = [
            int(x) for x in range(len([x for x in np.unique(y) if np.isfinite(x)]))
        ]
        colors = color_picker(color_classes)
        non_nan_targets = [x for x in np.unique(y) if np.isfinite(x)]
        for k, c in enumerate(non_nan_targets):
            axidx = ax[k // ncols, k % ncols]

            sns.scatterplot(
                data=df,
                x="x",
                y="y",
                hue="target",
                ax=axidx,
                legend=False,
                alpha=0.01,
                palette=colors,
                s=s,
            )
            sns.scatterplot(
                data=df[df["target"] == c],
                x="x",
                y="y",
                ax=axidx,
                color=colors[k],
                s=50,
            )
            axidx.set_title(f"c={c}")

        condition_to_class = {
            c: i for i, c in enumerate([x for x in np.unique(y) if np.isfinite(x)], 1)
        }
        # marginal_timepoints = np.linspace(0, len(trajectory) - 1, n_marginals).astype(int)
        marginal_timepoints = [int(x) for x in (np.unique(t) * (len(trajectory) - 1))]
        for n, idx in enumerate(idx_plot):
            p = condition_to_class[y[idx, 0]] - 1
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
                    [i/100, (i+1)/100],
                    [trajectory[i, n, 0], trajectory[i + 1, n, 0]],
                    color="black",
                    lw=2,
                )
            # # Highlight start and end marginal_timepoints
            # axidx.scatter(
            #     trajectory[marginal_timepoints, n, 0],
            #     trajectory[marginal_timepoints, n, 1],
            #     color="black",
            #     label="Interpolated",
            #     s=12.5,
            # )

        if title is not None:
            plt.suptitle(title, fontsize=12)
        plt.tight_layout()
        if save:
            file_path = Path(filepath) / f"{filename}.png"
            plt.savefig(file_path)
        plt.show()

    def save_monge_maps(self, filename):
        """Save Monge maps using cloudpickle."""
        import cloudpickle

        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")

        with open(filename, "wb") as f:
            cloudpickle.dump(self.monge_maps, f)


def fsi_derive_monge_maps(
    X_train: dict, y_train: dict = None, t_train: list = None, conditional: bool = False
):
    """Derive Monge maps between marginals.

    Args:
        X_train (dict): Dictionary containing the training data for each time step.
                        Shape: (n_samples, n_timepoints, n_features)
        y_train (dict): Dictionary containing the training labels for each time step.
                        Shape: (n_samples, n_timepoints)
        t_train (list, np.ndarray): List of time points.
        conditional (bool): Whether to use conditional OT maps.
    """
    monge_maps = Dict()
    if conditional:
        all_classes = sorted([int(x) for x in np.unique(y_train) if np.isfinite(x)])
    all_times = sorted(np.unique(t_train))

    # Learn Monge maps between marginals
    # Barycentric Mapping
    if not conditional:
        for k, time in enumerate(all_times):
            if k == (len(all_times) - 1):
                break
            ot_emd = ot.da.EMDTransport()
            ot_emd.fit(Xs=X_train[:, k], Xt=X_train[:, k + 1])
            monge_maps[time] = ot_emd

    else:
        for c in all_classes:
            for k, time in enumerate(all_times):
                if k == (len(all_times) - 1):
                    break
                idx = y_train[:, k] == c
                if not all(np.isnan(y_train[idx, k])):
                    for next_existing_timestep in all_times[k + 1 :]:
                        k_timestep_next = all_times.index(next_existing_timestep)
                        if not all(np.isnan(y_train[idx, k_timestep_next])):
                            break

                    # for next_existing_timestep in all_timesteps[k + 1 :]:
                    ot_emd = ot.da.EMDTransport(limit_max=1e6)
                    if np.isnan(X_train[idx, k]).any():
                        print("Nan in X_train")
                        raise ValueError("Nan in X_train")
                    if np.isnan(X_train[idx, k_timestep_next]).any():
                        print("Nan in X_train")
                        raise ValueError("Nan in X_train")

                    ot_emd.fit(Xs=X_train[idx, k], Xt=X_train[idx, k_timestep_next])
                    monge_maps[time][c] = ot_emd
                else:
                    monge_maps[time][c] = None

    return monge_maps.to_dict()


def fsi_apply_monge_maps(
    monge_maps, X_test, y_test=None, t_test=None, conditional=False, interpolate=True
):
    """Apply Monge maps to test data.

    Given an initial starting point (and condition), the Monge maps are applied to
    predict the next time points.

    Args:
        monge_maps (dict): Dictionary containing the Monge maps between marginals.
        X_test (np.ndarray): Test data. Shape (n_samples, n_timepoints, n_features).
        y_test (np.ndarray): Test labels. Shape (n_samples, n_timepoints).
        t_test (np.ndarray): Test time points.
        conditional (bool): Whether to use conditional OT maps.
        interpolate (bool): Whether to interpolate missing time points.
    """
    X_test_hat = np.nan * np.ones(
        shape=(X_test.shape[0], len(monge_maps) + 1, X_test.shape[2])
    )
    X_test_hat[:, 0] = X_test[:, 0]

    if conditional:
        idx_classes = y_test[:, 0]
        all_timesteps = list(np.unique(t_test))
        ys = [int(x) for x in np.unique(y_test)]

        for c in ys:
            for k, time in enumerate(monge_maps.keys()):
                if monge_maps[time][c] is None:
                    continue

                skipped_timesteps = []
                for next_existing_timestep in all_timesteps[k + 1 :]:
                    nearest_dict = monge_maps.get(
                        next_existing_timestep,
                        monge_maps[
                            min(
                                monge_maps.keys(),
                                key=lambda k: abs(k - next_existing_timestep),
                            )
                        ],
                    )
                    if nearest_dict.get(c, None) is not None:
                        break
                    else:
                        skipped_timesteps.append(next_existing_timestep)

                if len(skipped_timesteps) == len(all_timesteps[k + 1 :]):
                    next_existing_timestep = 1.0
                    skipped_timesteps = skipped_timesteps[:-1]

                forward_prediction = (
                    monge_maps[time][c]
                    .transform(X_test_hat[:, k][idx_classes == c])
                    .squeeze()
                )
                X_test_hat[
                    idx_classes == c, all_timesteps.index(next_existing_timestep)
                ] = forward_prediction

                # Fill all missing columns with interpolation
                if interpolate:
                    for skipped_timestep in skipped_timesteps:
                        before = all_timesteps[
                            all_timesteps.index(skipped_timestep) - 1
                        ]
                        after = all_timesteps[all_timesteps.index(skipped_timestep) + 1]
                        data_before = X_test_hat[
                            idx_classes == c, all_timesteps.index(before)
                        ]
                        data_after = X_test_hat[
                            idx_classes == c, all_timesteps.index(after)
                        ]
                        interpolation = data_before + (data_after - data_before) / (
                            after - before
                        ) * (skipped_timestep - before)
                        X_test_hat[
                            idx_classes == c, all_timesteps.index(skipped_timestep)
                        ] = interpolation

    else:
        for k, time in enumerate(monge_maps.keys()):
            X_test_hat[:, k + 1] = (
                monge_maps[time].transform(X_test_hat[:, k]).squeeze()
            )

    return X_test_hat


def train_fsi_model(
    X_train, X_test, y_train=None, y_test=None, conditional=False, unbalanced=False
):
    """Wrapper function to train a FSI model.

    Also returns the trajectory of the test data.

    Args:
        X_train (np.ndarray): Training data. Shape (n_samples, n_timepoints, n_features).
        X_test (np.ndarray): Test data. Shape (n_samples, n_timepoints, n_features).
        y_train (np.ndarray): Training labels. Shape (n_samples, n_timepoints).
        y_test (np.ndarray): Test labels. Shape (n_samples, n_timepoints).
        conditional (bool): Whether to use conditional OT maps.
        unbalanced (bool): Whether to use unbalanced OT maps.
    """
    monge_maps = {}
    X_test_hat = {}

    # Learn Monge maps between marginals
    # Barycentric Mapping
    for k in range(len(X_train) - 1):
        if not unbalanced:
            if conditional:
                ot_emd = ot.da.EMDTransport(limit_max=1e6)
                ot_emd.fit(
                    Xs=X_train[k], Xt=X_train[k + 1], ys=y_train[k], yt=y_train[k + 1]
                )
                monge_maps[k] = ot_emd
            else:
                ot_emd = ot.da.EMDTransport()
                ot_emd.fit(Xs=X_train[k], Xt=X_train[k + 1])
                monge_maps[k] = ot_emd
        else:
            if conditional:
                raise NotImplementedError(
                    "Unbalanced OT not implemented for conditional OT"
                )
            else:
                a, b = (
                    np.ones(X_train[k].shape[0]) / X_train[k].shape[0],
                    np.ones(X_train[k + 1].shape[0]) / X_train[k + 1].shape[0],
                )
                M = ot.dist(X_train[k], X_train[k + 1])
                M /= M.max()
                reg = 0.005
                reg_m_kl = 0.05
                reg_m_l2 = 50
                mass = 0.7
                entropic_kl_uot = ot.unbalanced.mm_unbalanced(
                    a, b, M, reg_m_kl, div="kl"
                )
                # entropic_kl_uot = ot.partial.partial_wasserstein(a, b, M, m=0.7)
                monge_maps[k] = entropic_kl_uot

    # Predict next time step for sample
    for k in range(len(monge_maps) + 1):
        if k == 0:
            X_test_hat[0] = X_test[0]
        else:
            if not unbalanced:
                X_test_hat[k] = monge_maps[k - 1].transform(X_test_hat[k - 1]).squeeze()
            else:
                n_samples = X_train[0].shape[0]
                n_features = X_train[0].shape[1]
                # Find closes point in training data and use it's mapping
                X_test_hat[k] = np.zeros((n_samples, n_features))
                for sample in range(n_samples):
                    # Find closest point in training data at time point k-1
                    closest_train_point = np.argmin(
                        np.linalg.norm(
                            X_train[k - 1] - X_test_hat[k - 1][sample], axis=1
                        )
                    )
                    new_target_sample = np.argmin(
                        monge_maps[k - 1][closest_train_point]
                    )
                    X_test_hat[k][sample] = X_train[k][new_target_sample]

    return X_test_hat, monge_maps


# Plots results on test data
def plot_fsi_results(
    df, X_hat, y, idx_plot, title, n_distribution, n_classes, figsize=(10, 5)
):
    """Plot results of FSI model training."""
    fig, ax = plt.subplots(1, n_classes, figsize=figsize, sharex=True, sharey=True)
    if n_classes == 1:
        ax = [ax]

    # Plat background samples
    for k in range(n_classes):
        sns.scatterplot(
            data=df,
            x="x",
            y="y",
            hue="target",
            ax=ax[k],
            legend=False,
            alpha=0.15,
            palette="tab10",
        )
        if n_classes > 1:
            ax[k].set_title(f"c={k + 1}")

    # Plot interpolations
    xs = np.array([x / (n_distribution - 1) for x in range(n_distribution)])
    for n in idx_plot:
        ys = np.array([X_hat[k][n] for k in range(n_distribution)])
        cs = CubicSpline(xs, ys)
        spline = [cs(timepoint) for timepoint in np.linspace(0, 1, 100)]
        color = "black"
        c = 0 if n_classes == 1 else y[0][n] - 1
        ax[c].scatter(ys[:, 0], ys[:, 1], color=color, label="Interpolated", s=50)
        ax[c].plot(
            [s[0] for s in spline],
            [s[1] for s in spline],
            color=color,
            linestyle=":",
            lw=2.5,
            label="Interpolated",
        )
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
