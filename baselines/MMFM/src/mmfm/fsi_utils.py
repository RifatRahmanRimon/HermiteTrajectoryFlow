import matplotlib.pyplot as plt
import numpy as np
import ot
import seaborn as sns
from addict import Dict
from scipy.interpolate import CubicSpline


def fsi_derive_monge_maps(X_train, y_train=None, t_train=None, conditional=False):
    """Derive Monge maps between marginals.

    Parameters
    ----------
    X_train : dict
        Dictionary containing the training data for each time step.
        Shape: (n_samples, n_timepoints, n_features)

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


def fsi_apply_monge_maps(monge_maps, X_test, y_test=None, t_test=None, conditional=False, interpolate=True):
    """Apply Monge maps to test data."""
    X_test_hat = np.nan * np.ones(shape=(X_test.shape[0], len(monge_maps) + 1, X_test.shape[2]))
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
                for next_existing_timestep in all_timesteps[k + 1 :]:  #  -1
                    nearest_dict = monge_maps.get(
                        next_existing_timestep,
                        monge_maps[min(monge_maps.keys(), key=lambda k: abs(k - next_existing_timestep))],
                    )
                    if nearest_dict.get(c, None) is not None:
                        break
                    else:
                        skipped_timesteps.append(next_existing_timestep)

                if len(skipped_timesteps) == len(all_timesteps[k + 1 :]):
                    next_existing_timestep = 1.0
                    skipped_timesteps = skipped_timesteps[:-1]

                forward_prediction = monge_maps[time][c].transform(X_test_hat[:, k][idx_classes == c]).squeeze()
                X_test_hat[idx_classes == c, all_timesteps.index(next_existing_timestep)] = forward_prediction

                # print(skipped_timesteps)
                # Fill all missing columns with interpolation
                if interpolate:
                    for skipped_timestep in skipped_timesteps:
                        before = all_timesteps[all_timesteps.index(skipped_timestep) - 1]
                        after = all_timesteps[all_timesteps.index(skipped_timestep) + 1]
                        data_before = X_test_hat[idx_classes == c, all_timesteps.index(before)]
                        data_after = X_test_hat[idx_classes == c, all_timesteps.index(after)]
                        interpolation = data_before + (data_after - data_before) / (after - before) * (
                            skipped_timestep - before
                        )
                        X_test_hat[idx_classes == c, all_timesteps.index(skipped_timestep)] = interpolation

    else:
        for k, time in enumerate(monge_maps.keys()):
            X_test_hat[:, k + 1] = monge_maps[time].transform(X_test_hat[:, k]).squeeze()

    return X_test_hat


def train_fsi_model(X_train, X_test, y_train=None, y_test=None, conditional=False, unbalanced=False):
    """Train FSI model.

    Parameters
    ----------
    X_train : dict
        Dictionary containing the training data for each time step.
    X_test : dict
        Dictionary containing the test data for each time step.
    y_train : dict
        Dictionary containing the training labels for each time step.
    y_test : dict
        Dictionary containing the test labels for each time step.
    conditional : bool
        Whether to use conditional OT maps.
    """
    monge_maps = {}
    X_test_hat = {}

    # Learn Monge maps between marginals
    # Barycentric Mapping
    for k in range(len(X_train) - 1):
        if not unbalanced:
            if conditional:
                ot_emd = ot.da.EMDTransport(limit_max=1e6)
                ot_emd.fit(Xs=X_train[k], Xt=X_train[k + 1], ys=y_train[k], yt=y_train[k + 1])
                monge_maps[k] = ot_emd
            else:
                ot_emd = ot.da.EMDTransport()
                ot_emd.fit(Xs=X_train[k], Xt=X_train[k + 1])
                monge_maps[k] = ot_emd
        else:
            if conditional:
                raise NotImplementedError("Unbalanced OT not implemented for conditional OT")
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
                entropic_kl_uot = ot.unbalanced.mm_unbalanced(a, b, M, reg_m_kl, div="kl")
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
                    closest_train_point = np.argmin(np.linalg.norm(X_train[k - 1] - X_test_hat[k - 1][sample], axis=1))
                    new_target_sample = np.argmin(monge_maps[k - 1][closest_train_point])
                    X_test_hat[k][sample] = X_train[k][new_target_sample]

    return X_test_hat, monge_maps


# Plots results on test data
def plot_fsi_results(df, X_hat, y, idx_plot, title, n_distribution, n_classes, figsize=(10, 5)):
    """Plot results of FSI model."""
    fig, ax = plt.subplots(1, n_classes, figsize=figsize, sharex=True, sharey=True)
    if n_classes == 1:
        ax = [ax]

    # Plat background samples
    for k in range(n_classes):
        sns.scatterplot(data=df, x="x", y="y", hue="target", ax=ax[k], legend=False, alpha=0.15, palette="tab10")
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
