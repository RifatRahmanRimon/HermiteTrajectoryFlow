import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pathlib import Path
import anndata as ad
import numpy as np
import ot
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import torch
from addict import Dict
from scipy.integrate import odeint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#
# Vector Fields
#
def u(y, t, c):
    """Define the ODE for the cosine vector field."""
    dy = (c + 1) ** 1.75 * np.cos(5 * np.pi * t)
    dydt = [np.array(3.0), dy]
    return dydt


def u_linear(y, t, c):
    """Define the ODE for the cosine vector field."""
    dy = 3 * t + t * c**2 * np.sin(3 * np.pi * t)
    dydt = [5 * t, dy]
    return dydt


def u_waves(y, t, c):
    """Define the ODE for the waves vector field."""
    dy = 2 * c**2 * np.cos(6 * np.pi * t)
    dydt = [np.array(2.0), dy]
    return dydt


def u_strong(y, t, c):
    """Define the ODE for the cosine vector field."""
    dy = (c + 1) ** 3 * np.cos(5 * np.pi * t)
    dydt = [np.array(3.0), dy]
    return dydt


def u_sine(y, t, c):
    """Define the ODE for the sine vector field."""
    dy = (c + 1) * np.cos(2 * np.pi * t)
    dydt = [np.array(2.0), dy]
    return dydt


def u_vdp(y, t, c):
    """Define the ODE for the sine vector field."""
    dx = c * (y[0] - y[0] ** 3 / 3 - y[1])  # y[0]  #
    dy = 1 / c * y[0]  # c * (1 - y[0] ** 2) * y[1] - y[0]  #
    dydt = [dx, dy]
    return dydt


def u_circle(y, t, phis, dimension):
    """Define the ODE for a n-dimensional sphere with radius 1."""
    # t : 1              1
    # phis : n - 2       8
    dydt = []

    # First term is always -sin(t)
    dydt.append(-np.sin(t))

    phis = list(phis)

    # Calculate the remaining terms
    for i in range(1, dimension):  # 1 2 3
        term = np.cos(t)

        if len(phis) == 0:
            dydt.append(term)
            continue

        if i <= dimension - 2:  # 2
            for j in range(0, i - 1):
                term *= np.sin(phis[j])
            term *= np.cos(phis[i - 1])
        elif i == dimension - 1:  # 3
            for j in range(0, i - 2):
                term *= np.sin(phis[j])
            term *= np.sin(phis[i - 2])
        else:
            raise ValueError("Something went wrong")

        dydt.append(term)

    return dydt


#
# Synthetic Data
#
def apply_couplings_no_int(xs, ys, couplings):
    """Apply couplings to xs and ys without assuming integer labels."""
    all_timepoints = sorted(xs.keys())

    # Apply couplings to xs
    for i, time in enumerate(xs.keys()):
        temp_x = xs[time]
        for j in range(i - 1, -1, -1):
            # Convert j to timepoint
            temp_x = couplings[all_timepoints[j]] @ temp_x
        xs[time] = temp_x

        # Apply couplings to ys
        temp_y = ys[time]
        for j in range(i - 1, -1, -1):
            temp_y = couplings[all_timepoints[j]] @ temp_y.float()
        ys[time] = temp_y

    return xs, ys


def couple_samples_no_int(xs, ys):
    """Couple samples from consecutive time points."""
    couplings = {}
    x_timepoints = sorted(xs.keys())
    for current, next in zip(x_timepoints[:-1], x_timepoints[1:], strict=False):
        X_source = xs[current].cpu().detach().numpy()
        X_target = xs[next].cpu().detach().numpy()

        # Barycentric Mapping
        ot_emd = ot.da.EMDTransport()
        ot_emd.fit(Xs=X_source, Xt=X_target)
        couplings[current] = torch.tensor(ot_emd.coupling_)
        if couplings[current].shape[0] != couplings[current].shape[1]:
            raise NotImplementedError("Coupling matrix is not squared")
        else:
            couplings[current] = torch.where(
                couplings[current] != 0, torch.tensor(1.0), couplings[current]
            )

    xs, ys = apply_couplings_no_int(xs, ys, couplings)
    X_train = xs
    y_train = ys

    return X_train, y_train, couplings


def construct_vector_field(
    class_label, timepoint, n_samples, std, y0, vf, off_diagonal
):
    """Generate data for the DGP1 experiment."""
    if vf == "u":
        t = np.linspace(0, 1, 501)
        sol = odeint(u, y0, t, args=(class_label,))
        # Evaluate sol at position timepoint
        sol = sol[int(timepoint * 500)]
    elif vf == "u_waves":
        t = np.linspace(0, 1, 501)
        sol = odeint(u_waves, y0, t, args=(class_label,))
        # Evaluate sol at position timepoint
        sol = sol[int(timepoint * 500)]
    elif vf == "u_strong":
        t = np.linspace(0, 1, 501)
        sol = odeint(u_strong, y0, t, args=(class_label,))
        # Evaluate sol at position timepoint
        sol = sol[int(timepoint * 500)]
    elif vf == "u_sine":
        t = np.linspace(0, 1, 501)
        sol = odeint(u_sine, y0, t, args=(class_label,))
        # Evaluate sol at position timepoint
        sol = sol[int(timepoint * 500)]
    elif vf == "u_vdp":
        t = np.linspace(0, 20, 501)
        sol = odeint(u_vdp, y0, t, args=(class_label,))
        # Evaluate sol at position timepoint
        sol = sol[int(timepoint * 500)]
    elif vf == "u_linear":
        t = np.linspace(0, 1, 501)
        sol = odeint(u_linear, y0, t, args=(class_label,))
        # Evaluate sol at position timepoint
        sol = sol[int(timepoint * 500)]
    else:
        raise ValueError("Unknown vector field")

    # Convert to torch tensor
    sol = torch.tensor(sol, dtype=torch.float32)

    X = torch.distributions.MultivariateNormal(
        loc=sol,
        covariance_matrix=torch.tensor(
            [[std**2, off_diagonal**2], [off_diagonal**2, std**2]]
        ),
    ).sample((n_samples,))

    y = torch.ones(n_samples) * class_label
    t = torch.ones(n_samples) * timepoint

    return X, y, t


#
# Synthetic Data
#
def dgp_schiebinger():
    pass


def dgp_waves(
    data_specs,
    coupling=False,
    batch_size=32,
    dimension=2,
    off_diagonal=0.0,
    vf="u_waves",
    classes_first=True,
):
    """Create data for the DGP1 experiment.

    Parameters
    ----------
    n_classes: int
        Number of classes.
        We start counting classes at 1. Class 0 is reserved for the 'unknown' class.
    """
    if classes_first:
        # If duplicates in data_specs.keys(), raise error
        if len(data_specs.keys()) != len(set(data_specs.keys())):
            raise ValueError("Duplicate timepoints in data_specs")

        # If n_samples_per_t_and_c is integer, we assume to use the same number for all classes and timepoints
        # Combine all timepoints into one list
        all_timepoints = []
        for _, v in data_specs.items():
            if isinstance(v["timepoints"], np.ndarray):
                all_timepoints += list(np.unique(v["timepoints"]))
                is_numpy = True
            else:
                all_timepoints += list(v["timepoints"])
                is_numpy = False
        all_timepoints = sorted(set(all_timepoints))

        X_data = Dict()
        y_data = Dict()
        t_data = Dict()
        if not is_numpy:
            for _, v in data_specs.items():
                for time in v["timepoints"]:
                    c = v["condition"]
                    X, y, t = construct_vector_field(
                        c,
                        time,
                        v["n_samples"],
                        v["std"],
                        v["y0"],
                        vf,
                        off_diagonal=off_diagonal,
                    )
                    if not len(y.unique()) == 1:
                        raise ValueError("More than one class in the data")
                    X_data[time][float(y.unique())] = X
                    y_data[time][float(y.unique())] = y
                    t_data[time][float(y.unique())] = t
                    # Sort the keys for each timepoint
                X_data[time] = {c: X_data[time][c] for c in sorted(X_data[time].keys())}
                y_data[time] = {c: y_data[time][c] for c in sorted(y_data[time].keys())}
                t_data[time] = {c: t_data[time][c] for c in sorted(t_data[time].keys())}
        else:
            for _, v in data_specs.items():
                for time in list(np.unique(v["timepoints"])):
                    c = v["condition"]
                    X, y, t = construct_vector_field(
                        c,
                        time,
                        v["n_samples"],
                        v["std"],
                        v["y0"],
                        vf,
                        off_diagonal=off_diagonal,
                    )
                    if not len(y.unique()) == 1:
                        raise ValueError("More than one class in the data")
                    X_data[time][float(y.unique())] = X
                    y_data[time][float(y.unique())] = y
                    t_data[time][float(y.unique())] = t
                    # Sort the keys for each timepoint
                X_data[time] = {c: X_data[time][c] for c in sorted(X_data[time].keys())}
                y_data[time] = {c: y_data[time][c] for c in sorted(y_data[time].keys())}
                t_data[time] = {c: t_data[time][c] for c in sorted(t_data[time].keys())}

    else:
        all_timepoints = sorted(set(data_specs.keys()))

        X_data = Dict()
        y_data = Dict()
        t_data = Dict()
        for time, specs in data_specs.items():
            if not isinstance(specs["classes"], list | tuple):
                specs["classes"] = [specs["classes"]]
            for c in specs["classes"]:
                X, y, t = construct_vector_field(
                    c,
                    time,
                    specs["n_samples"],
                    specs["std"],
                    None,
                    vf,
                    off_diagonal=off_diagonal,
                )
                if not len(y.unique()) == 1:
                    raise ValueError("More than one class in the data")
                X_data[time][int(y.unique())] = X
                y_data[time][int(y.unique())] = y
                t_data[time][int(y.unique())] = t
                # Sort the keys for each timepoint

    if coupling == "ot":
        X_data = {
            t: torch.cat([X_data[t][c] for c in X_data[t].keys()], dim=0)
            for t in sorted(X_data.keys())
        }
        y_data = {
            t: torch.cat([y_data[t][c] for c in y_data[t].keys()], dim=0)
            for t in sorted(y_data.keys())
        }
        t_data = {
            t: torch.cat([t_data[t][c] for c in t_data[t].keys()], dim=0)
            for t in sorted(t_data.keys())
        }

        max_dimension = max([X_data[t].shape[0] for t in X_data.keys()])
        padded_data_xs, padded_data_ys, padded_data_ts = [], [], []
        for xs, ys, ts in zip(
            X_data.values(), y_data.values(), t_data.values(), strict=False
        ):
            if xs.shape[0] < max_dimension:
                print("- Upsampling and Padding")
                # Instead of padding with missings, we pad with random points from the same timepoint
                idx_sampling = np.random.choice(
                    xs.shape[0], max_dimension - xs.shape[0], replace=True
                )
                padded_data_xs.append(
                    torch.cat([xs, xs[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_data_ys.append(
                    torch.cat([ys, ys[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_data_ts.append(
                    torch.cat([ts, ts[idx_sampling]], dim=0).unsqueeze(1)
                )
            else:
                padded_data_xs.append(xs.unsqueeze(1))
                padded_data_ys.append(ys.unsqueeze(1))
                padded_data_ts.append(ts.unsqueeze(1))

        X_data = {
            t: padded_data_xs[i].squeeze() for i, t in enumerate(sorted(X_data.keys()))
        }
        y_data = {
            t: padded_data_ys[i].squeeze() for i, t in enumerate(sorted(y_data.keys()))
        }

        # Couple samples without labels
        X_data, y_data, _ = couple_samples_no_int(X_data, y_data)

        timepoints = sorted(X_data.keys())
        X_data = torch.cat(
            [X_data[t].unsqueeze(1) for t in sorted(X_data.keys())], dim=1
        )
        y_data = torch.cat(
            [y_data[t].unsqueeze(1) for t in sorted(y_data.keys())], dim=1
        )
        t_data = torch.from_numpy(
            np.repeat(np.array(timepoints)[None, :], X_data.shape[0], axis=0)
        )

        # timepoints = sorted(X_data.keys())
        # X_data = torch.cat([X_data[t].unsqueeze(1) for t in sorted(X_data.keys())], dim=1)
        # y_data = torch.cat([y_data[t].unsqueeze(1) for t in sorted(y_data.keys())], dim=1)
        # t_data = torch.from_numpy(np.repeat(np.array(timepoints)[None, :], X_data.shape[0], axis=0))

    elif coupling == "cot":
        # Apply OT per class/target
        # n_samples = sum([specs["n_samples"] for specs in data_specs.values()])
        all_timepoints = sorted(X_data.keys())
        all_classes = sorted({c for t in X_data.keys() for c in X_data[t].keys()})
        X_data_aligned, y_data_aligned, t_data_aligned = Dict(), Dict(), Dict()
        for c in all_classes:
            xs = {t: X_data[t][c] for t in all_timepoints if c in X_data[t].keys()}
            ys = {t: y_data[t][c] for t in all_timepoints if c in y_data[t].keys()}
            ts = {t: t_data[t][c] for t in all_timepoints if c in t_data[t].keys()}

            X_coupled, y_coupled, _ = couple_samples_no_int(xs, ys)

            for t in X_coupled.keys():
                X_data_aligned[t][c] = X_coupled[t]
                y_data_aligned[t][c] = y_coupled[t]
                t_data_aligned[t][c] = ts[t]

        n_samples_per_class = {c: X_data_aligned[0][c].shape[0] for c in all_classes}
        X_data = torch.cat(
            [
                torch.cat(
                    [
                        X_data_aligned[t].get(
                            c,
                            torch.tensor(
                                np.nan * np.ones((n_samples_per_class[c], 2)),
                                dtype=torch.float32,
                            ),
                        )[:, None, :]
                        for c in all_classes
                    ],
                    dim=0,
                )
                for t in all_timepoints
            ],
            dim=1,
        )

        y_data = torch.cat(
            [
                torch.cat(
                    [
                        y_data_aligned[t].get(
                            c,
                            torch.tensor(
                                np.nan * np.ones(n_samples_per_class[c]),
                                dtype=torch.float32,
                            ),
                        )[:, None]
                        for c in all_classes
                    ],
                    dim=0,
                )
                for t in all_timepoints
            ],
            dim=1,
        )

        t_data = torch.from_numpy(
            np.repeat(
                np.array(sorted(X_data_aligned.keys()))[None, :],
                X_data.shape[0],
                axis=0,
            )
        )

    else:
        X_data = {
            t: torch.cat([X_data[t][c] for c in X_data[t].keys()], dim=0)
            for t in sorted(X_data.keys())
        }
        y_data = {
            t: torch.cat([y_data[t][c] for c in y_data[t].keys()], dim=0)
            for t in sorted(y_data.keys())
        }
        t_data = {
            t: torch.cat([t_data[t][c] for c in t_data[t].keys()], dim=0)
            for t in sorted(t_data.keys())
        }

        max_dimension = max([X_data[t].shape[0] for t in X_data.keys()])
        padded_data_xs, padded_data_ys, padded_data_ts = [], [], []
        for xs, ys, ts in zip(
            X_data.values(), y_data.values(), t_data.values(), strict=False
        ):
            if xs.shape[0] < max_dimension:
                print("- Upsampling and Padding")
                # Instead of padding with missings, we pad with random points from the same timepoint
                idx_sampling = np.random.choice(
                    xs.shape[0], max_dimension - xs.shape[0], replace=True
                )
                padded_data_xs.append(
                    torch.cat([xs, xs[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_data_ys.append(
                    torch.cat([ys, ys[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_data_ts.append(
                    torch.cat([ts, ts[idx_sampling]], dim=0).unsqueeze(1)
                )
            else:
                padded_data_xs.append(xs.unsqueeze(1))
                padded_data_ys.append(ys.unsqueeze(1))
                padded_data_ts.append(ts.unsqueeze(1))
        X_data = torch.cat(padded_data_xs, dim=1)
        y_data = torch.cat(padded_data_ys, dim=1)
        t_data = torch.cat(padded_data_ts, dim=1)

        # Shuffle data in 2nd dimension to make sure we do not have OT couplings
        for t in range(X_data.shape[1]):
            idx = np.random.permutation(X_data.shape[0])
            X_data[:, t] = X_data[idx, t]
            y_data[:, t] = y_data[idx, t]

    # Construct data loaders
    # Each dataloader should return a batch of samples of dimension [n_samples, n_times, 2]
    # and a batch of targets of dimension [n_samples, n_times]
    # and a batch of timepoints of dimension [n_samples, n_times]
    dataset = torch.utils.data.TensorDataset(X_data, y_data, t_data)
    if batch_size is None:
        batch_size = X_data.shape[0]
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # Return as numpy arrays
    X_data = X_data.cpu().detach().numpy()
    y_data = y_data.cpu().detach().numpy()
    t_data = t_data.cpu().detach().numpy()

    return data_loader, X_data, y_data, t_data


def dgp_beijing(
    data_specs,
    coupling=False,
    batch_size=32,
    target="PM2.5",
    start=None,
    end=None,
    n_quarters=None,
    ns_per_t_and_c=50,
):
    """Create data for the beijing experiment."""
    stations = [
        "Aotizhongxin",  # 1
        "Changping",
        "Dingling",
        "Dongsi",
        "Guanyuan",  # 5
        "Gucheng",
        "Huairou",  # X
        "Nongzhanguan",
        "Shunyi",
        "Tiantan",  # 10  # X
        "Wanliu",
        "Wanshouxigong",
    ]
    stations_to_condition = {
        k: v for k, v in zip(stations, range(1, len(stations) + 1))
    }
    condition_to_station = {v: k for k, v in stations_to_condition.items()}
    data_frames = []

    for station in stations:
        file_path = (
            Path("/home/m015k/code/MMFM/data/beijing/PRSA_Data_20130301-20170228")
            / f"PRSA_Data_{station}_20130301-20170228.csv"
        )
        df = pd.read_csv(file_path)
        df["station"] = station
        data_frames.append(df)

    df = pd.concat(data_frames, ignore_index=True)

    # Convert to datetime
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]].assign(hour=0))
    df["quarter"] = df["datetime"].dt.quarter

    if start:
        df = df[df["year"] >= start]
    if end:
        df = df[df["year"] <= end]
    if n_quarters:
        # Filter first n_quarters after start/earlierst time point
        start_year = df["year"].min()
        start_quarter = df[df["year"] == start_year]["quarter"].min()

        year_steps = n_quarters // 4
        quarter_steps = n_quarters % 4

        end_year = start_year + year_steps
        end_quarter = start_quarter + quarter_steps

        df = df[
            (df["year"] < end_year)
            | ((df["year"] == end_year) & (df["quarter"] < end_quarter))
        ]

    # Group by station, year, quarter and calculate means
    df = df.groupby(["station", "year", "month"]).agg({target: "mean"}).reset_index()

    # Create datetime for easier plotting
    df["date"] = df["year"].astype(str) + df["month"].astype(str).str.zfill(2)
    df["timepoint"] = df["date"].rank(method="dense") - 1
    df["timepoint"] = df["timepoint"].astype(int)

    # Filter out stations that are not relevant for training
    for condition, v in data_specs.items():
        station_name = condition_to_station[condition]
        df = df.loc[
            (df["station"] != station_name)
            | (df["station"] == station_name) & (df["timepoint"].isin(v["timepoints"]))
        ]
        
    # Resample each row to have n_samples_per_t_and_c samples 
    # Sample around var with a variance of 0.1
    def generate_new_rows(df, k):
        new_rows = []
        for index, row in df.iterrows():
            # Sample k new values from a normal distribution around the value of d
            new_d_values = np.random.normal(loc=row[target], scale=0.1, size=k)
            for new_d in new_d_values:
                new_row = row.copy()
                new_row[target] = new_d
                new_rows.append(new_row)
        return pd.DataFrame(new_rows)

    df = generate_new_rows(df, ns_per_t_and_c)

    X_data = {}
    y_data = {}
    t_data = {}

    # Normalize timepoints and round to 3 decimals
    df["timepoint"] = df["timepoint"] / df["timepoint"].max()
    df["timepoint"] = df["timepoint"].round(3)

    for t in df["timepoint"].unique():
        # t = int(t)
        X_data[t] = {}
        y_data[t] = {}
        t_data[t] = {}
        for station in stations:
            id = stations_to_condition[station]
            data = df[(df["station"] == station) & (df["timepoint"] == t)][
                [target]
            ].values

            if len(data) == 0:
                # print("No data for", t, station)
                continue

            X_data[t][id] = torch.tensor(data, dtype=torch.float32)
            y_data[t][id] = torch.tensor([id] * data.shape[0], dtype=torch.float32)
            # y_data[t][id] = y_data[t][id].unsqueeze(1)
            t_data[t][id] = torch.tensor([t] * data.shape[0], dtype=torch.float32)
            # t_data[t][id] = t_data[t][id].unsqueeze(1)

            # Sample 100 points at random
            idx = np.random.choice(data.shape[0], ns_per_t_and_c, replace=True)
            X_data[t][id] = X_data[t][id][idx]
            y_data[t][id] = y_data[t][id][idx]
            t_data[t][id] = t_data[t][id][idx]

    # X_data[t][id].shape, y_data[t][id].shape, t_data[t][id].shape

    if coupling == "ot":
        X_data = {
            t: torch.cat([X_data[t][c] for c in X_data[t].keys()], dim=0)
            for t in sorted(X_data.keys())
        }
        y_data = {
            t: torch.cat([y_data[t][c] for c in y_data[t].keys()], dim=0)
            for t in sorted(y_data.keys())
        }
        t_data = {
            t: torch.cat([t_data[t][c] for c in t_data[t].keys()], dim=0)
            for t in sorted(t_data.keys())
        }

        max_dimension = max([X_data[t].shape[0] for t in X_data.keys()])
        padded_data_xs, padded_data_ys, padded_data_ts = [], [], []
        for xs, ys, ts in zip(
            X_data.values(), y_data.values(), t_data.values(), strict=False
        ):
            if xs.shape[0] < max_dimension:
                print("- Upsampling and Padding")
                # Instead of padding with missings, we pad with random points from the same timepoint
                idx_sampling = np.random.choice(
                    xs.shape[0], max_dimension - xs.shape[0], replace=True
                )
                padded_data_xs.append(
                    torch.cat([xs, xs[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_data_ys.append(
                    torch.cat([ys, ys[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_data_ts.append(
                    torch.cat([ts, ts[idx_sampling]], dim=0).unsqueeze(1)
                )
            else:
                padded_data_xs.append(xs.unsqueeze(1))
                padded_data_ys.append(ys.unsqueeze(1))
                padded_data_ts.append(ts.unsqueeze(1))

        X_data = {
            t: padded_data_xs[i].squeeze(-1)
            for i, t in enumerate(sorted(X_data.keys()))
        }
        y_data = {
            t: padded_data_ys[i].squeeze() for i, t in enumerate(sorted(y_data.keys()))
        }

        # Couple samples without labels
        X_data, y_data, _ = couple_samples_no_int(X_data, y_data)

        timepoints = sorted(X_data.keys())
        X_data = torch.cat(
            [X_data[t].unsqueeze(1) for t in sorted(X_data.keys())], dim=1
        )
        y_data = torch.cat(
            [y_data[t].unsqueeze(1) for t in sorted(y_data.keys())], dim=1
        )
        t_data = torch.from_numpy(
            np.repeat(np.array(timepoints)[None, :], X_data.shape[0], axis=0)
        )

        # timepoints = sorted(X_data.keys())
        # X_data = torch.cat([X_data[t].unsqueeze(1) for t in sorted(X_data.keys())], dim=1)
        # y_data = torch.cat([y_data[t].unsqueeze(1) for t in sorted(y_data.keys())], dim=1)
        # t_data = torch.from_numpy(np.repeat(np.array(timepoints)[None, :], X_data.shape[0], axis=0))

    elif coupling == "cot":
        # Apply OT per class/target
        # n_samples = sum([specs["n_samples"] for specs in data_specs.values()])
        all_timepoints = sorted(X_data.keys())
        all_classes = sorted({c for t in X_data.keys() for c in X_data[t].keys()})
        X_data_aligned, y_data_aligned, t_data_aligned = Dict(), Dict(), Dict()
        for c in all_classes:
            xs = {t: X_data[t][c] for t in all_timepoints if c in X_data[t].keys()}
            ys = {t: y_data[t][c] for t in all_timepoints if c in y_data[t].keys()}
            ts = {t: t_data[t][c] for t in all_timepoints if c in t_data[t].keys()}

            X_coupled, y_coupled, _ = couple_samples_no_int(xs, ys)

            for t in X_coupled.keys():
                X_data_aligned[t][c] = X_coupled[t]
                y_data_aligned[t][c] = y_coupled[t]
                t_data_aligned[t][c] = ts[t]

        n_samples_per_class = {c: X_data_aligned[0][c].shape[0] for c in all_classes}
        X_data = torch.cat(
            [
                torch.cat([X_data_aligned[t].get(c, torch.tensor(np.nan * np.ones((n_samples_per_class[c], 1)), dtype=torch.float32))[:, None, :] for c in all_classes], dim=0)
                for t in all_timepoints
            ],
            dim=1,
        )

        y_data = torch.cat(
            [
                torch.cat(
                    [
                        y_data_aligned[t].get(
                            c,
                            torch.tensor(
                                np.nan * np.ones(n_samples_per_class[c]),
                                dtype=torch.float32,
                            ),
                        )[:, None]
                        for c in all_classes
                    ],
                    dim=0,
                )
                for t in all_timepoints
            ],
            dim=1,
        )

        t_data = torch.from_numpy(
            np.repeat(
                np.array(sorted(X_data_aligned.keys()))[None, :],
                X_data.shape[0],
                axis=0,
            )
        )

    else:
        X_data = {
            t: torch.cat([X_data[t][c] for c in X_data[t].keys()], dim=0)
            for t in sorted(X_data.keys())
        }
        y_data = {
            t: torch.cat([y_data[t][c] for c in y_data[t].keys()], dim=0)
            for t in sorted(y_data.keys())
        }
        t_data = {
            t: torch.cat([t_data[t][c] for c in t_data[t].keys()], dim=0)
            for t in sorted(t_data.keys())
        }

        max_dimension = max([X_data[t].shape[0] for t in X_data.keys()])
        padded_data_xs, padded_data_ys, padded_data_ts = [], [], []
        for xs, ys, ts in zip(
            X_data.values(), y_data.values(), t_data.values(), strict=False
        ):
            if xs.shape[0] < max_dimension:
                print("- Upsampling and Padding")
                # Instead of padding with missings, we pad with random points from the same timepoint
                idx_sampling = np.random.choice(
                    xs.shape[0], max_dimension - xs.shape[0], replace=True
                )
                padded_data_xs.append(
                    torch.cat([xs, xs[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_data_ys.append(
                    torch.cat([ys, ys[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_data_ts.append(
                    torch.cat([ts, ts[idx_sampling]], dim=0).unsqueeze(1)
                )
            else:
                padded_data_xs.append(xs.unsqueeze(1))
                padded_data_ys.append(ys.unsqueeze(1))
                padded_data_ts.append(ts.unsqueeze(1))
        X_data = torch.cat(padded_data_xs, dim=1)
        y_data = torch.cat(padded_data_ys, dim=1)
        t_data = torch.cat(padded_data_ts, dim=1)

        # Shuffle data in 2nd dimension to make sure we do not have OT couplings
        for t in range(X_data.shape[1]):
            idx = np.random.permutation(X_data.shape[0])
            X_data[:, t] = X_data[idx, t]
            y_data[:, t] = y_data[idx, t]

    # Construct data loaders
    # Each dataloader should return a batch of samples of dimension [n_samples, n_times, 2]
    # and a batch of targets of dimension [n_samples, n_times]
    # and a batch of timepoints of dimension [n_samples, n_times]
    dataset = torch.utils.data.TensorDataset(X_data, y_data, t_data)
    if batch_size is None:
        batch_size = X_data.shape[0]
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # Return as numpy arrays
    X_data = X_data.cpu().detach().numpy()
    y_data = y_data.cpu().detach().numpy()
    t_data = t_data.cpu().detach().numpy()

    return data_loader, X_data, y_data, t_data


#
# Real-World Data
#
def dgp_iccite(
    hvg,
    subsample_frac,
    use_pca,
    coupling=None,
    batch_size=None,
    n_samples_per_c_in_b=250,
    train_test_split=0.8,
    seed=0,
    preproc=True,
    top_n_effects=None,
    preset=None,
    leave_out_mid=None,
    leave_out_end=None,
):
    """Create data for the ICCITE experiment."""
    # Print all files in data folder
    filename_full_data = f"/home/rohbeckm/code/mmfm/benchmark/dgp_iccite/data/iccite_{hvg}_{use_pca}_{subsample_frac}_{coupling}_{batch_size}_{n_samples_per_c_in_b}_{train_test_split}_{top_n_effects}_{leave_out_mid}_{leave_out_end}_{seed}_{preset}.pt"
    filename_partial_data = f"/home/rohbeckm/code/mmfm/benchmark/dgp_iccite/data/iccite_preproc_{hvg}_{use_pca}.h5ad"
    if preproc & Path(filename_full_data).exists():
        print("Loading FULL preprocessed data")
        data = torch.load(filename_full_data)
        return (
            data["train_loader"],
            data["X_train"],
            data["y_train"],
            data["t_train"],
            data["X_valid"],
            data["y_valid"],
            data["t_valid"],
            data["X_test"],
            data["y_test"],
            data["t_test"],
            data["ps"],
        )

    # # Check if file exists: "data/schiebinger_preproc_{ipsc_timepoint}_{hvg}_{use_pca}.h5ad"
    if preproc & Path(filename_partial_data).exists():
        print("Loading PARTIAL preprocessed data")
        adata = ad.read_h5ad(filename_partial_data)
    else:
        pl.seed_everything(seed)

        if use_pca and hvg:
            raise ValueError("Cannot use PCA and HVG at the same time.")

        adata = sc.read(
            "/home/rohbeckm/code/mmfm/data/icCITE-plex_filtered_top_drugs.h5ad"
        )
        if subsample_frac:
            adata = adata[adata.obs.sample(frac=subsample_frac).index]

        # # Remove all data where dose is != 10 uM
        # # adata.obs["dose"].value_counts()  # 10 uM
        # print(f"Before filtering dosage: {adata.shape}")
        # adata = adata[adata.obs["dose"] == "10 uM"]
        # print(f"After filtering dosage: {adata.shape}")

        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        if use_pca:
            sc.pp.pca(adata, n_comps=use_pca)
            # THIS DOES NOT WORK: adata.X = adata.obsm["X_pca"]
            adata = sc.AnnData(
                X=adata.obsm["X_pca"],
                obs=adata.obs,
                var=pd.DataFrame(
                    index=[f"PC{i+1}" for i in range(adata.obsm["X_pca"].shape[1])]
                ),
                uns=adata.uns,
            )
        elif hvg is not None:
            sc.pp.highly_variable_genes(adata, n_top_genes=hvg)
            adata = adata[:, adata.var["highly_variable"]]

        # Convert age to float timepoints
        adata.obs["timepoint"] = (
            adata.obs["timepoint"].astype(str).str.extract(r"(\d+)").astype(int)
        )
        adata.obs["timepoint"] = (adata.obs["timepoint"] - 24) / 48

        # Print number of samples per timepoint and perturbation
        # dfx = adata.obs.groupby(["timepoint", "target"]).size().reset_index().pivot(index="timepoint", columns="target")

        adata.write(filename_partial_data)

    if preset is not None:
        if preset == "a":
            df_experiment = pd.read_csv(
                "/home/rohbeckm/code/mmfm/benchmark/dgp_iccite/data/experiment_50_20_random.csv"
            )

        elif preset == "b":
            df_experiment = pd.read_csv(
                "/home/rohbeckm/code/mmfm/benchmark/dgp_iccite/data/experiment_70_30_random.csv"
            )

        elif preset == "c":
            df_experiment = pd.read_csv(
                "/home/rohbeckm/code/mmfm/benchmark/dgp_iccite/data/experiment_90_40_random.csv"
            )

        elif preset == "d":
            df_experiment = pd.read_csv(
                "/home/rohbeckm/code/mmfm/benchmark/dgp_iccite/data/experiment_123_50_random.csv"
            )

        else:
            raise ValueError("Unknown preset")

        treatments = df_experiment.loc[df_experiment["train"], "treatment"].unique()
        leave_out_mid = df_experiment.loc[
            df_experiment["leave_out"] == "mid", "treatment"
        ].unique()
        leave_out_end = df_experiment.loc[
            df_experiment["leave_out"] == "end", "treatment"
        ].unique()

        adata = adata[adata.obs["treatment"].isin(treatments)]

        ps = dict(enumerate(adata.obs["treatment"].value_counts().index, 1))
        adata.obs["condition"] = adata.obs["treatment"].map(
            {v: k for k, v in ps.items()}
        )
        adata.obs["condition"] = adata.obs["condition"].astype(int)

    else:
        subset_effects = adata.obs["treatment"].value_counts()[: (top_n_effects + 1)]
        adata = adata[adata.obs["treatment"].isin(subset_effects.index)]

        raise NotImplementedError("No preset implemented for DGP_iccite")

    condition_list = range(1, len(adata.obs["condition"].unique()) + 1)
    all_timepoints = sorted(adata.obs["timepoint"].unique())

    # Sample which conditions to leave out
    X_data = Dict()
    y_data = Dict()
    t_data = Dict()
    X_test = Dict()
    y_test = Dict()
    t_test = Dict()
    for condition in sorted(adata.obs.condition.unique()):
        # We observe condition two only at a shifted point in time
        for timepoint in all_timepoints:
            potential_samples = adata[
                (adata.obs["timepoint"] == timepoint)
                & (adata.obs["condition"] == condition)
            ].obs.index
            potential_test_samples = None
            # Skip if we want to leave out this condition
            # Sample n_samples_per_condition_in_batch samples per condition and timepoint at random
            if len(potential_samples) > n_samples_per_c_in_b:
                idx = np.random.choice(
                    potential_samples, n_samples_per_c_in_b, replace=False
                )
            else:
                # Remove at least one sample for testing
                potential_samples, potential_test_samples = potential_samples[:-1], [
                    potential_samples[-1]
                ]
                idx = np.random.choice(
                    potential_samples, n_samples_per_c_in_b, replace=True
                )

            if potential_test_samples is None:
                potential_test_samples = np.setdiff1d(potential_samples, idx)
            if len(potential_test_samples) > n_samples_per_c_in_b:
                idx_test = np.random.choice(
                    potential_test_samples, n_samples_per_c_in_b, replace=False
                )
            else:
                idx_test = np.random.choice(
                    potential_test_samples, n_samples_per_c_in_b, replace=True
                )

            # if not (
            #     ((ps[condition] in leave_out_end) & (timepoint == 1))
            #     | ((ps[condition] in leave_out_mid) & (timepoint == 0.5))
            # ):
            X_data[timepoint][condition] = np.array(
                adata[adata.obs.index.get_indexer(idx)].X
            )
            y_data[timepoint][condition] = condition * np.ones(
                shape=(X_data[timepoint][condition].shape[0])
            )
            t_data[timepoint][condition] = timepoint * np.ones(
                shape=(X_data[timepoint][condition].shape[0])
            )

            X_test[timepoint][condition] = np.array(
                adata[adata.obs.index.get_indexer(idx_test)].X
            )
            y_test[timepoint][condition] = condition * np.ones(
                shape=(X_test[timepoint][condition].shape[0])
            )
            t_test[timepoint][condition] = timepoint * np.ones(
                shape=(X_test[timepoint][condition].shape[0])
            )

    X_train, y_train, t_train = Dict(), Dict(), Dict()
    X_valid, y_valid, t_valid = Dict(), Dict(), Dict()

    # Split data into train and validation
    for c in condition_list:
        for t in all_timepoints:
            if X_data.get(t, {}).get(c, None) is not None:
                n_samples = X_data[t][c].shape[0]
                n_samples_train_per_c = int(
                    train_test_split * n_samples
                )  # This value always exists
                idx_train = np.random.choice(
                    np.arange(n_samples), size=n_samples_train_per_c, replace=False
                )
                idx_valid = np.setdiff1d(np.arange(n_samples), idx_train)
                X_train[t][c] = torch.tensor(X_data[t][c][idx_train]).float()
                y_train[t][c] = torch.tensor(y_data[t][c][idx_train]).float()
                t_train[t][c] = torch.tensor(t_data[t][c][idx_train]).float()
                X_valid[t][c] = torch.tensor(X_data[t][c][idx_valid]).float()
                y_valid[t][c] = torch.tensor(y_data[t][c][idx_valid]).float()
                t_valid[t][c] = torch.tensor(t_data[t][c][idx_valid]).float()
            if X_test.get(t, {}).get(c, None) is not None:
                X_test[t][c] = torch.tensor(X_test[t][c]).float()
                y_test[t][c] = torch.tensor(y_test[t][c]).float()
                t_test[t][c] = torch.tensor(t_test[t][c]).float()

    # Remove all timepoints and condition from training data which we must not observe
    timepoint = 1.0
    for c_name in leave_out_end:
        # Find condition int from condition name
        c = [k for k, v in ps.items() if v == c_name][0]
        X_train[timepoint].pop(c)
        y_train[timepoint].pop(c)
        t_train[timepoint].pop(c)
    timepoint = 0.5
    for c_name in leave_out_mid:
        # Find condition int from condition name
        c = [k for k, v in ps.items() if v == c_name][0]
        X_train[timepoint].pop(c)
        y_train[timepoint].pop(c)
        t_train[timepoint].pop(c)

    def swap_levels(nested_dict):
        swapped = Dict()
        for outer_key, inner_dict in nested_dict.items():
            for inner_key, value in inner_dict.items():
                swapped[inner_key][outer_key] = value
        return swapped

    if coupling is None:
        X_train = {
            t: torch.cat([X_train[t][c] for c in X_train[t].keys()], dim=0)
            for t in sorted(X_train.keys())
        }
        y_train = {
            t: torch.cat([y_train[t][c] for c in y_train[t].keys()], dim=0)
            for t in sorted(y_train.keys())
        }
        t_train = {
            t: torch.cat([t_train[t][c] for c in t_train[t].keys()], dim=0)
            for t in sorted(t_train.keys())
        }

        max_dimension = max([X_train[t].shape[0] for t in X_train.keys()])
        padded_train_xs, padded_train_ys, padded_train_ts = [], [], []
        for xs, ys, ts in zip(
            X_train.values(), y_train.values(), t_train.values(), strict=False
        ):
            if xs.shape[0] < max_dimension:
                print("- Upsampling and Padding")
                # Instead of padding with missings, we pad with random points from the same timepoint
                idx_sampling = np.random.choice(
                    xs.shape[0], max_dimension - xs.shape[0], replace=True
                )
                padded_train_xs.append(
                    torch.cat([xs, xs[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_train_ys.append(
                    torch.cat([ys, ys[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_train_ts.append(
                    torch.cat([ts, ts[idx_sampling]], dim=0).unsqueeze(1)
                )
            else:
                padded_train_xs.append(xs.unsqueeze(1))
                padded_train_ys.append(ys.unsqueeze(1))
                padded_train_ts.append(ts.unsqueeze(1))

        X_train = {
            t: padded_train_xs[i].squeeze()
            for i, t in enumerate(sorted(X_train.keys()))
        }
        y_train = {
            t: padded_train_ys[i].squeeze()
            for i, t in enumerate(sorted(y_train.keys()))
        }

        timepoints = sorted(X_train.keys())
        X_train = torch.cat(
            [X_train[t].unsqueeze(1) for t in sorted(X_train.keys())], dim=1
        )
        y_train = torch.cat(
            [y_train[t].unsqueeze(1) for t in sorted(y_train.keys())], dim=1
        )
        t_train = torch.from_numpy(
            np.repeat(np.array(timepoints)[None, :], X_train.shape[0], axis=0)
        )

        X_valid = {
            t: torch.cat([X_valid[t][c] for c in X_valid[t].keys()], dim=0)
            for t in sorted(X_valid.keys())
        }
        y_valid = {
            t: torch.cat([y_valid[t][c] for c in y_valid[t].keys()], dim=0)
            for t in sorted(y_valid.keys())
        }
        t_valid = {
            t: torch.cat([t_valid[t][c] for c in t_valid[t].keys()], dim=0)
            for t in sorted(t_valid.keys())
        }

        max_dimension = max([X_valid[t].shape[0] for t in X_valid.keys()])
        padded_valid_xs, padded_valid_ys, padded_valid_ts = [], [], []
        for xs, ys, ts in zip(
            X_valid.values(), y_valid.values(), t_valid.values(), strict=False
        ):
            if xs.shape[0] < max_dimension:
                print("- Upsampling and Padding")
                # Instead of padding with missings, we pad with random points from the same timepoint
                idx_sampling = np.random.choice(
                    xs.shape[0], max_dimension - xs.shape[0], replace=True
                )
                padded_valid_xs.append(
                    torch.cat([xs, xs[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_valid_ys.append(
                    torch.cat([ys, ys[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_valid_ts.append(
                    torch.cat([ts, ts[idx_sampling]], dim=0).unsqueeze(1)
                )
            else:
                padded_valid_xs.append(xs.unsqueeze(1))
                padded_valid_ys.append(ys.unsqueeze(1))
                padded_valid_ts.append(ts.unsqueeze(1))

        X_valid = {
            t: padded_valid_xs[i].squeeze()
            for i, t in enumerate(sorted(X_valid.keys()))
        }
        y_valid = {
            t: padded_valid_ys[i].squeeze()
            for i, t in enumerate(sorted(y_valid.keys()))
        }

        timepoints = sorted(X_valid.keys())
        X_valid = torch.cat(
            [X_valid[t].unsqueeze(1) for t in sorted(X_valid.keys())], dim=1
        )
        y_valid = torch.cat(
            [y_valid[t].unsqueeze(1) for t in sorted(y_valid.keys())], dim=1
        )
        t_valid = torch.from_numpy(
            np.repeat(np.array(timepoints)[None, :], X_valid.shape[0], axis=0)
        )

    elif coupling == "ot":
        X_train = {
            t: torch.cat([X_train[t][c] for c in X_train[t].keys()], dim=0)
            for t in sorted(X_train.keys())
        }
        y_train = {
            t: torch.cat([y_train[t][c] for c in y_train[t].keys()], dim=0)
            for t in sorted(y_train.keys())
        }
        t_train = {
            t: torch.cat([t_train[t][c] for c in t_train[t].keys()], dim=0)
            for t in sorted(t_train.keys())
        }

        max_dimension = max([X_train[t].shape[0] for t in X_train.keys()])
        padded_train_xs, padded_train_ys, padded_train_ts = [], [], []
        for xs, ys, ts in zip(
            X_train.values(), y_train.values(), t_train.values(), strict=False
        ):
            if xs.shape[0] < max_dimension:
                print("- Upsampling and Padding")
                # Instead of padding with missings, we pad with random points from the same timepoint
                idx_sampling = np.random.choice(
                    xs.shape[0], max_dimension - xs.shape[0], replace=True
                )
                padded_train_xs.append(
                    torch.cat([xs, xs[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_train_ys.append(
                    torch.cat([ys, ys[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_train_ts.append(
                    torch.cat([ts, ts[idx_sampling]], dim=0).unsqueeze(1)
                )
            else:
                padded_train_xs.append(xs.unsqueeze(1))
                padded_train_ys.append(ys.unsqueeze(1))
                padded_train_ts.append(ts.unsqueeze(1))

        X_train = {
            t: padded_train_xs[i].squeeze()
            for i, t in enumerate(sorted(X_train.keys()))
        }
        y_train = {
            t: padded_train_ys[i].squeeze()
            for i, t in enumerate(sorted(y_train.keys()))
        }

        # Couple samples without labels
        X_train, y_train, _ = couple_samples_no_int(X_train, y_train)

        timepoints = sorted(X_train.keys())
        X_train = torch.cat(
            [X_train[t].unsqueeze(1) for t in sorted(X_train.keys())], dim=1
        )
        y_train = torch.cat(
            [y_train[t].unsqueeze(1) for t in sorted(y_train.keys())], dim=1
        )
        t_train = torch.from_numpy(
            np.repeat(np.array(timepoints)[None, :], X_train.shape[0], axis=0)
        )

        X_valid = {
            t: torch.cat([X_valid[t][c] for c in X_valid[t].keys()], dim=0)
            for t in sorted(X_valid.keys())
        }
        y_valid = {
            t: torch.cat([y_valid[t][c] for c in y_valid[t].keys()], dim=0)
            for t in sorted(y_valid.keys())
        }
        t_valid = {
            t: torch.cat([t_valid[t][c] for c in t_valid[t].keys()], dim=0)
            for t in sorted(t_valid.keys())
        }

        max_dimension = max([X_valid[t].shape[0] for t in X_valid.keys()])
        padded_valid_xs, padded_valid_ys, padded_valid_ts = [], [], []
        for xs, ys, ts in zip(
            X_valid.values(), y_valid.values(), t_valid.values(), strict=False
        ):
            if xs.shape[0] < max_dimension:
                print("- Upsampling and Padding")
                # Instead of padding with missings, we pad with random points from the same timepoint
                idx_sampling = np.random.choice(
                    xs.shape[0], max_dimension - xs.shape[0], replace=True
                )
                padded_valid_xs.append(
                    torch.cat([xs, xs[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_valid_ys.append(
                    torch.cat([ys, ys[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_valid_ts.append(
                    torch.cat([ts, ts[idx_sampling]], dim=0).unsqueeze(1)
                )
            else:
                padded_valid_xs.append(xs.unsqueeze(1))
                padded_valid_ys.append(ys.unsqueeze(1))
                padded_valid_ts.append(ts.unsqueeze(1))

        X_valid = {
            t: padded_valid_xs[i].squeeze()
            for i, t in enumerate(sorted(X_valid.keys()))
        }
        y_valid = {
            t: padded_valid_ys[i].squeeze()
            for i, t in enumerate(sorted(y_valid.keys()))
        }

        # Couple samples without labels
        X_valid, y_valid, _ = couple_samples_no_int(X_valid, y_valid)

        timepoints = sorted(X_valid.keys())
        X_valid = torch.cat(
            [X_valid[t].unsqueeze(1) for t in sorted(X_valid.keys())], dim=1
        )
        y_valid = torch.cat(
            [y_valid[t].unsqueeze(1) for t in sorted(y_valid.keys())], dim=1
        )
        t_valid = torch.from_numpy(
            np.repeat(np.array(timepoints)[None, :], X_valid.shape[0], axis=0)
        )

    elif coupling == "cot":
        # Apply OT per class/target/label
        all_timepoints = sorted(X_train.keys())
        # Change nested dict structure from X_train[t][c] to X_train[c][t]
        xs = swap_levels(X_train)
        ys = swap_levels(y_train)

        for c in condition_list:
            X_coupled, y_coupled, _ = couple_samples_no_int(xs[c], ys[c])
            for t in all_timepoints:
                if xs.get(c, {}).get(t, None) is not None:
                    X_train[t][c] = X_coupled[t]
                    y_train[t][c] = y_coupled[t]

        # Sum up all samples for each timepoint
        n_samples = sum(X_train[0][c].shape[0] for c in condition_list)
        n_features = X_train[0][1].shape[1]
        X_train_final = np.nan * torch.ones(n_samples, len(all_timepoints), n_features)
        y_train_final = np.nan * torch.ones(n_samples, len(all_timepoints))
        indices_per_class = {
            c: range(
                sum(X_train[0][cc].shape[0] for cc in condition_list if cc < c),
                sum(X_train[0][cc].shape[0] for cc in condition_list if cc <= c),
            )
            for c in condition_list
        }
        for t in all_timepoints:
            for c in condition_list:
                t_idx = all_timepoints.index(t)
                if X_train.get(t, {}).get(c, None) is not None:
                    X_train_final[indices_per_class[c], t_idx] = X_train[t][c]
                    y_train_final[indices_per_class[c], t_idx] = y_train[t][c]

        X_train = X_train_final
        y_train = y_train_final
        t_train = torch.repeat_interleave(
            torch.tensor(all_timepoints)[None, :], n_samples, dim=0
        )

        # Now for validation data

        # Apply OT per class/target/label
        all_timepoints = sorted(X_valid.keys())
        # Change nested dict structure from X_valid[t][c] to X_valid[c][t]
        xs = swap_levels(X_valid)
        ys = swap_levels(y_valid)

        for c in condition_list:
            X_coupled, y_coupled, _ = couple_samples_no_int(xs[c], ys[c])
            for t in all_timepoints:
                if xs.get(c, {}).get(t, None) is not None:
                    X_valid[t][c] = X_coupled[t]
                    y_valid[t][c] = y_coupled[t]

        # Sum up all samples for each timepoint
        n_samples = sum(X_valid[0][c].shape[0] for c in condition_list)
        n_features = X_valid[0][1].shape[1]
        X_valid_final = np.nan * torch.ones(n_samples, len(all_timepoints), n_features)
        y_valid_final = np.nan * torch.ones(n_samples, len(all_timepoints))
        indices_per_class = {
            c: range(
                sum(X_valid[0][cc].shape[0] for cc in condition_list if cc < c),
                sum(X_valid[0][cc].shape[0] for cc in condition_list if cc <= c),
            )
            for c in condition_list
        }
        for t in all_timepoints:
            for c in condition_list:
                t_idx = all_timepoints.index(t)
                if X_valid.get(t, {}).get(c, None) is not None:
                    X_valid_final[indices_per_class[c], t_idx] = X_valid[t][c]
                    y_valid_final[indices_per_class[c], t_idx] = y_valid[t][c]

        X_valid = X_valid_final
        y_valid = y_valid_final
        t_valid = torch.repeat_interleave(
            torch.tensor(all_timepoints)[None, :], n_samples, dim=0
        )

    # Process test data
    # Note that this is not coupled
    X_test = {
        t: torch.cat([X_test[t][c] for c in X_test[t].keys()], dim=0)
        for t in sorted(X_test.keys())
    }
    y_test = {
        t: torch.cat([y_test[t][c] for c in y_test[t].keys()], dim=0)
        for t in sorted(y_test.keys())
    }
    t_test = {
        t: torch.cat([t_test[t][c] for c in t_test[t].keys()], dim=0)
        for t in sorted(t_test.keys())
    }
    X_test = torch.cat([X_test[t].unsqueeze(1) for t in X_test.keys()], dim=1)
    y_test = torch.cat([y_test[t].unsqueeze(1) for t in y_test.keys()], dim=1)
    t_test = torch.cat([t_test[t].unsqueeze(1) for t in t_test.keys()], dim=1)

    dataset = torch.utils.data.TensorDataset(
        X_train.float(), y_train.float(), t_train.float()
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=X_train.shape[0] if batch_size is None else batch_size,
        shuffle=True,
    )

    # Convert to numpy arrays for analysis
    X_train = X_train.numpy().astype(np.float32)
    y_train = y_train.numpy().astype(np.float32)
    t_train = t_train.numpy().astype(np.float32)
    X_valid = X_valid.numpy().astype(np.float32)
    y_valid = y_valid.numpy().astype(np.float32)
    t_valid = t_valid.numpy().astype(np.float32)
    X_test = X_test.numpy().astype(np.float32)
    y_test = y_test.numpy().astype(np.float32)
    t_test = t_test.numpy().astype(np.float32)

    # Save all
    # in a dict on disk
    torch.save(
        {
            "train_loader": train_loader,
            "X_train": X_train,
            "y_train": y_train,
            "t_train": t_train,
            "X_valid": X_valid,
            "y_valid": y_valid,
            "t_valid": t_valid,
            "X_test": X_test,
            "y_test": y_test,
            "t_test": t_test,
            "ps": ps,
        },
        filename_full_data,
    )
    return (
        train_loader,
        X_train,
        y_train,
        t_train,
        X_valid,
        y_valid,
        t_valid,
        X_test,
        y_test,
        t_test,
        ps,
    )


def dgp_iccite_4t_v2(
    hvg,
    subsample_frac,
    use_pca,
    coupling=None,
    batch_size=None,
    n_samples_per_c_in_b=250,
    train_test_split=0.8,
    seed=0,
    preproc=True,
    top_n_effects=None,
    preset=None,
    leave_out_mid=None,
    leave_out_end=None,
    filter_beginning_end=False,
):
    """Create data for the ICCITE experiment."""
    # Print all files in data folder
    filename_full_data = f"/home/rohbeckm/code/mmfm/benchmark/dgp_iccite/data/iccite_4t_vehicle_{hvg}_{use_pca}_{subsample_frac}_{coupling}_{batch_size}_{n_samples_per_c_in_b}_{train_test_split}_{top_n_effects}_{leave_out_mid}_{leave_out_end}_{seed}_{preset}.pt"
    filename_partial_data = f"/home/rohbeckm/code/mmfm/benchmark/dgp_iccite/data/iccite_4t_vehicle_preproc_{hvg}_{use_pca}.h5ad"
    if filter_beginning_end:
        filename_full_data = filename_full_data.replace(".pt", "_filtered.pt")
        filename_partial_data = filename_partial_data.replace(".h5ad", "_filtered.h5ad")
    if preproc & Path(filename_full_data).exists():
        print("Loading FULL preprocessed data")
        data = torch.load(filename_full_data)
        return (
            data["train_loader"],
            data["X_train"],
            data["y_train"],
            data["t_train"],
            data["X_valid"],
            data["y_valid"],
            data["t_valid"],
            data["X_test"],
            data["y_test"],
            data["t_test"],
            data["ps"],
        )

    # # Check if file exists: "data/schiebinger_preproc_{ipsc_timepoint}_{hvg}_{use_pca}.h5ad"
    if preproc & Path(filename_partial_data).exists():
        print("Loading PARTIAL preprocessed data")
        adata = ad.read_h5ad(filename_partial_data)
    else:
        pl.seed_everything(seed)

        if use_pca and hvg:
            raise ValueError("Cannot use PCA and HVG at the same time.")

        adata = sc.read(
            "/home/rohbeckm/code/mmfm/data/icCITE-plex_filtered_top_drugs.h5ad"
        )
        if subsample_frac:
            adata = adata[adata.obs.sample(frac=subsample_frac).index]

        # # Remove all data where dose is != 10 uM
        # # adata.obs["dose"].value_counts()  # 10 uM
        # print(f"Before filtering dosage: {adata.shape}")
        # adata = adata[adata.obs["dose"] == "10 uM"]
        # print(f"After filtering dosage: {adata.shape}")

        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        if use_pca:
            sc.pp.pca(adata, n_comps=use_pca)
            # THIS DOES NOT WORK: adata.X = adata.obsm["X_pca"]
            adata = sc.AnnData(
                X=adata.obsm["X_pca"],
                obs=adata.obs,
                var=pd.DataFrame(
                    index=[f"PC{i+1}" for i in range(adata.obsm["X_pca"].shape[1])]
                ),
                uns=adata.uns,
            )
        elif hvg is not None:
            sc.pp.highly_variable_genes(adata, n_top_genes=hvg)
            adata = adata[:, adata.var["highly_variable"]]

        # Convert age to float timepoints
        adata.obs["timepoint"] = (
            adata.obs["timepoint"].astype(str).str.extract(r"(\d+)").astype(int)
        )
        # Create new timepoint 0 for all non-stimulated data
        adata.obs.loc[
            adata.obs["treatment"].isin(
                ["Vehicle_100nM", "Vehicle_10uM", "Vehicle_1uM"]
            ),
            "timepoint",
        ] = 0
        adata.obs["timepoint"] = adata.obs["timepoint"] / 72
        if filter_beginning_end:
            adata = adata[adata.obs["timepoint"].isin([0, 1])]

        # Round to two digits
        adata.obs["timepoint"] = adata.obs["timepoint"].round(2)

        # Print number of samples per timepoint and perturbation
        # dfx = adata.obs.groupby(["timepoint", "target"]).size().reset_index().pivot(index="timepoint", columns="target")

        adata.write(filename_partial_data)

    if preset is not None:
        if preset == "z":
            # Load df_experiment.to_csv("./data/experiment_50_20_random.csv")
            df_experiment = pd.read_csv(
                "/home/rohbeckm/code/mmfm/benchmark/dgp_iccite/data/experiment_4t_60_30_random.csv"
            )

        elif preset == "y":
            # Load df_experiment.to_csv("./data/experiment_50_20_random.csv")
            df_experiment = pd.read_csv(
                "/home/rohbeckm/code/mmfm/benchmark/dgp_iccite/data/experiment_4t_90_45_random.csv"
            )

        else:
            raise ValueError("Unknown preset")

        treatments = df_experiment.loc[df_experiment["train"], "treatment"].unique()
        leave_out_beg = df_experiment.loc[
            df_experiment["leave_out"] == "beg", "treatment"
        ].unique()
        leave_out_mid = df_experiment.loc[
            df_experiment["leave_out"] == "mid", "treatment"
        ].unique()
        leave_out_end = df_experiment.loc[
            df_experiment["leave_out"] == "end", "treatment"
        ].unique()

        # Remove all leave_out_end from treatments
        if filter_beginning_end:
            treatments = df_experiment.loc[
                (df_experiment["train"]) & (df_experiment["leave_out"] != "end"),
                "treatment",
            ].unique()
        else:
            treatments = df_experiment.loc[df_experiment["train"], "treatment"].unique()

        adata_control = adata[
            adata.obs["treatment"].isin(
                ["No stim_1uM", "No stim_100nM", "No stim_10uM"]
            )
        ].copy()

        adata = adata[
            adata.obs["treatment"].isin(treatments)
            # | adata.obs["treatment"].isin(["No stim_1uM", "No stim_100nM", "No stim_10uM"])
        ]
        print(f"Number of treatments: {len(np.unique(adata.obs['treatment']))}")

        ps = dict(enumerate(adata.obs["treatment"].value_counts().index, 1))
        adata.obs["condition"] = adata.obs["treatment"].map(
            {v: k for k, v in ps.items()}
        )
        adata.obs["condition"] = adata.obs["condition"].astype(int)

    else:
        subset_effects = adata.obs["treatment"].value_counts()[: (top_n_effects + 1)]
        adata = adata[adata.obs["treatment"].isin(subset_effects.index)]

        raise NotImplementedError("No preset implemented for DGP_iccite")

    condition_list = range(1, len(adata.obs["condition"].unique()) + 1)
    all_timepoints = sorted(adata.obs["timepoint"].unique())

    # Sample which conditions to leave out
    X_data = Dict()
    y_data = Dict()
    t_data = Dict()
    X_test = Dict()
    y_test = Dict()
    t_test = Dict()
    for condition in sorted(adata.obs.condition.unique()):
        # We observe condition two only at a shifted point in time
        for timepoint in all_timepoints:
            if timepoint == 0:
                continue
            potential_samples = adata[
                (adata.obs["timepoint"] == timepoint)
                & (adata.obs["condition"] == condition)
            ].obs.index
            potential_test_samples = None
            # Skip if we want to leave out this condition
            # Sample n_samples_per_condition_in_batch samples per condition and timepoint at random
            if len(potential_samples) > n_samples_per_c_in_b:
                idx = np.random.choice(
                    potential_samples, n_samples_per_c_in_b, replace=False
                )
            else:
                # Remove at least one sample for testing
                potential_samples, potential_test_samples = potential_samples[:-1], [
                    potential_samples[-1]
                ]
                idx = np.random.choice(
                    potential_samples, n_samples_per_c_in_b, replace=True
                )

            if potential_test_samples is None:
                potential_test_samples = np.setdiff1d(potential_samples, idx)
            if len(potential_test_samples) > n_samples_per_c_in_b:
                idx_test = np.random.choice(
                    potential_test_samples, n_samples_per_c_in_b, replace=False
                )
            else:
                idx_test = np.random.choice(
                    potential_test_samples, n_samples_per_c_in_b, replace=True
                )

            # if not (
            #     ((ps[condition] in leave_out_end) & (timepoint == 1))
            #     | ((ps[condition] in leave_out_mid) & (timepoint == 0.5))
            # ):
            X_data[timepoint][condition] = np.array(
                adata[adata.obs.index.get_indexer(idx)].X
            )
            y_data[timepoint][condition] = condition * np.ones(
                shape=(X_data[timepoint][condition].shape[0])
            )
            t_data[timepoint][condition] = timepoint * np.ones(
                shape=(X_data[timepoint][condition].shape[0])
            )

            X_test[timepoint][condition] = np.array(
                adata[adata.obs.index.get_indexer(idx_test)].X
            )
            y_test[timepoint][condition] = condition * np.ones(
                shape=(X_test[timepoint][condition].shape[0])
            )
            t_test[timepoint][condition] = timepoint * np.ones(
                shape=(X_test[timepoint][condition].shape[0])
            )

    # Add data from control as timepoint 0
    random_samples_test = np.random.choice(
        adata_control.obs.index, n_samples_per_c_in_b, replace=False
    )
    random_samples_train = np.setdiff1d(adata_control.obs.index, random_samples_test)
    for condition in sorted(adata.obs.condition.unique()):
        subset_sample_train = np.random.choice(
            random_samples_train, n_samples_per_c_in_b, replace=False
        )
        X_data[0][condition] = np.array(
            adata_control[adata_control.obs.index.get_indexer(subset_sample_train)].X
        )
        y_data[0][condition] = condition * np.ones(
            shape=(X_data[0][condition].shape[0])
        )
        t_data[0][condition] = 0 * np.ones(shape=(X_data[0][condition].shape[0]))

        X_test[0][condition] = np.array(
            adata_control[adata_control.obs.index.get_indexer(random_samples_test)].X
        )
        y_test[0][condition] = condition * np.ones(
            shape=(X_test[0][condition].shape[0])
        )
        t_test[0][condition] = 0 * np.ones(shape=(X_test[0][condition].shape[0]))

    X_train, y_train, t_train = Dict(), Dict(), Dict()
    X_valid, y_valid, t_valid = Dict(), Dict(), Dict()

    all_timepoints = sorted(all_timepoints + [0])

    # Split data into train and validation
    for c in condition_list:
        for t in all_timepoints:
            if X_data.get(t, {}).get(c, None) is not None:
                n_samples = X_data[t][c].shape[0]
                n_samples_train_per_c = int(
                    train_test_split * n_samples
                )  # This value always exists
                idx_train = np.random.choice(
                    np.arange(n_samples), size=n_samples_train_per_c, replace=False
                )
                idx_valid = np.setdiff1d(np.arange(n_samples), idx_train)
                X_train[t][c] = torch.tensor(X_data[t][c][idx_train]).float()
                y_train[t][c] = torch.tensor(y_data[t][c][idx_train]).float()
                t_train[t][c] = torch.tensor(t_data[t][c][idx_train]).float()
                X_valid[t][c] = torch.tensor(X_data[t][c][idx_valid]).float()
                y_valid[t][c] = torch.tensor(y_data[t][c][idx_valid]).float()
                t_valid[t][c] = torch.tensor(t_data[t][c][idx_valid]).float()
            if X_test.get(t, {}).get(c, None) is not None:
                X_test[t][c] = torch.tensor(X_test[t][c]).float()
                y_test[t][c] = torch.tensor(y_test[t][c]).float()
                t_test[t][c] = torch.tensor(t_test[t][c]).float()

    # Remove all timepoints and condition from training data which we must not observe
    if not filter_beginning_end:
        timepoint = 1.0
        for c_name in leave_out_end:
            c = [k for k, v in ps.items() if v == c_name][0]
            X_train[timepoint].pop(c)
            y_train[timepoint].pop(c)
            t_train[timepoint].pop(c)
        timepoint = 0.67
        for c_name in leave_out_mid:
            c = [k for k, v in ps.items() if v == c_name][0]
            X_train[timepoint].pop(c)
            y_train[timepoint].pop(c)
            t_train[timepoint].pop(c)
        timepoint = 0.33
        for c_name in leave_out_beg:
            c = [k for k, v in ps.items() if v == c_name][0]
            X_train[timepoint].pop(c)
            y_train[timepoint].pop(c)
            t_train[timepoint].pop(c)

    def swap_levels(nested_dict):
        swapped = Dict()
        for outer_key, inner_dict in nested_dict.items():
            for inner_key, value in inner_dict.items():
                swapped[inner_key][outer_key] = value
        return swapped

    if coupling is None:
        X_train = {
            t: torch.cat([X_train[t][c] for c in X_train[t].keys()], dim=0)
            for t in sorted(X_train.keys())
        }
        y_train = {
            t: torch.cat([y_train[t][c] for c in y_train[t].keys()], dim=0)
            for t in sorted(y_train.keys())
        }
        t_train = {
            t: torch.cat([t_train[t][c] for c in t_train[t].keys()], dim=0)
            for t in sorted(t_train.keys())
        }

        max_dimension = max([X_train[t].shape[0] for t in X_train.keys()])
        padded_train_xs, padded_train_ys, padded_train_ts = [], [], []
        for xs, ys, ts in zip(
            X_train.values(), y_train.values(), t_train.values(), strict=False
        ):
            if xs.shape[0] < max_dimension:
                print("- Upsampling and Padding")
                # Instead of padding with missings, we pad with random points from the same timepoint
                idx_sampling = np.random.choice(
                    xs.shape[0], max_dimension - xs.shape[0], replace=True
                )
                padded_train_xs.append(
                    torch.cat([xs, xs[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_train_ys.append(
                    torch.cat([ys, ys[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_train_ts.append(
                    torch.cat([ts, ts[idx_sampling]], dim=0).unsqueeze(1)
                )
            else:
                padded_train_xs.append(xs.unsqueeze(1))
                padded_train_ys.append(ys.unsqueeze(1))
                padded_train_ts.append(ts.unsqueeze(1))

        X_train = {
            t: padded_train_xs[i].squeeze()
            for i, t in enumerate(sorted(X_train.keys()))
        }
        y_train = {
            t: padded_train_ys[i].squeeze()
            for i, t in enumerate(sorted(y_train.keys()))
        }

        timepoints = sorted(X_train.keys())
        X_train = torch.cat(
            [X_train[t].unsqueeze(1) for t in sorted(X_train.keys())], dim=1
        )
        y_train = torch.cat(
            [y_train[t].unsqueeze(1) for t in sorted(y_train.keys())], dim=1
        )
        t_train = torch.from_numpy(
            np.repeat(np.array(timepoints)[None, :], X_train.shape[0], axis=0)
        )

        X_valid = {
            t: torch.cat([X_valid[t][c] for c in X_valid[t].keys()], dim=0)
            for t in sorted(X_valid.keys())
        }
        y_valid = {
            t: torch.cat([y_valid[t][c] for c in y_valid[t].keys()], dim=0)
            for t in sorted(y_valid.keys())
        }
        t_valid = {
            t: torch.cat([t_valid[t][c] for c in t_valid[t].keys()], dim=0)
            for t in sorted(t_valid.keys())
        }

        max_dimension = max([X_valid[t].shape[0] for t in X_valid.keys()])
        padded_valid_xs, padded_valid_ys, padded_valid_ts = [], [], []
        for xs, ys, ts in zip(
            X_valid.values(), y_valid.values(), t_valid.values(), strict=False
        ):
            if xs.shape[0] < max_dimension:
                print("- Upsampling and Padding")
                # Instead of padding with missings, we pad with random points from the same timepoint
                idx_sampling = np.random.choice(
                    xs.shape[0], max_dimension - xs.shape[0], replace=True
                )
                padded_valid_xs.append(
                    torch.cat([xs, xs[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_valid_ys.append(
                    torch.cat([ys, ys[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_valid_ts.append(
                    torch.cat([ts, ts[idx_sampling]], dim=0).unsqueeze(1)
                )
            else:
                padded_valid_xs.append(xs.unsqueeze(1))
                padded_valid_ys.append(ys.unsqueeze(1))
                padded_valid_ts.append(ts.unsqueeze(1))

        X_valid = {
            t: padded_valid_xs[i].squeeze()
            for i, t in enumerate(sorted(X_valid.keys()))
        }
        y_valid = {
            t: padded_valid_ys[i].squeeze()
            for i, t in enumerate(sorted(y_valid.keys()))
        }

        timepoints = sorted(X_valid.keys())
        X_valid = torch.cat(
            [X_valid[t].unsqueeze(1) for t in sorted(X_valid.keys())], dim=1
        )
        y_valid = torch.cat(
            [y_valid[t].unsqueeze(1) for t in sorted(y_valid.keys())], dim=1
        )
        t_valid = torch.from_numpy(
            np.repeat(np.array(timepoints)[None, :], X_valid.shape[0], axis=0)
        )

    elif coupling == "ot":
        X_train = {
            t: torch.cat([X_train[t][c] for c in X_train[t].keys()], dim=0)
            for t in sorted(X_train.keys())
        }
        y_train = {
            t: torch.cat([y_train[t][c] for c in y_train[t].keys()], dim=0)
            for t in sorted(y_train.keys())
        }
        t_train = {
            t: torch.cat([t_train[t][c] for c in t_train[t].keys()], dim=0)
            for t in sorted(t_train.keys())
        }

        max_dimension = max([X_train[t].shape[0] for t in X_train.keys()])
        padded_train_xs, padded_train_ys, padded_train_ts = [], [], []
        for xs, ys, ts in zip(
            X_train.values(), y_train.values(), t_train.values(), strict=False
        ):
            if xs.shape[0] < max_dimension:
                print("- Upsampling and Padding")
                # Instead of padding with missings, we pad with random points from the same timepoint
                idx_sampling = np.random.choice(
                    xs.shape[0], max_dimension - xs.shape[0], replace=True
                )
                padded_train_xs.append(
                    torch.cat([xs, xs[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_train_ys.append(
                    torch.cat([ys, ys[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_train_ts.append(
                    torch.cat([ts, ts[idx_sampling]], dim=0).unsqueeze(1)
                )
            else:
                padded_train_xs.append(xs.unsqueeze(1))
                padded_train_ys.append(ys.unsqueeze(1))
                padded_train_ts.append(ts.unsqueeze(1))

        X_train = {
            t: padded_train_xs[i].squeeze()
            for i, t in enumerate(sorted(X_train.keys()))
        }
        y_train = {
            t: padded_train_ys[i].squeeze()
            for i, t in enumerate(sorted(y_train.keys()))
        }

        # Couple samples without labels
        X_train, y_train, _ = couple_samples_no_int(X_train, y_train)

        timepoints = sorted(X_train.keys())
        X_train = torch.cat(
            [X_train[t].unsqueeze(1) for t in sorted(X_train.keys())], dim=1
        )
        y_train = torch.cat(
            [y_train[t].unsqueeze(1) for t in sorted(y_train.keys())], dim=1
        )
        t_train = torch.from_numpy(
            np.repeat(np.array(timepoints)[None, :], X_train.shape[0], axis=0)
        )

        X_valid = {
            t: torch.cat([X_valid[t][c] for c in X_valid[t].keys()], dim=0)
            for t in sorted(X_valid.keys())
        }
        y_valid = {
            t: torch.cat([y_valid[t][c] for c in y_valid[t].keys()], dim=0)
            for t in sorted(y_valid.keys())
        }
        t_valid = {
            t: torch.cat([t_valid[t][c] for c in t_valid[t].keys()], dim=0)
            for t in sorted(t_valid.keys())
        }

        max_dimension = max([X_valid[t].shape[0] for t in X_valid.keys()])
        padded_valid_xs, padded_valid_ys, padded_valid_ts = [], [], []
        for xs, ys, ts in zip(
            X_valid.values(), y_valid.values(), t_valid.values(), strict=False
        ):
            if xs.shape[0] < max_dimension:
                print("- Upsampling and Padding")
                # Instead of padding with missings, we pad with random points from the same timepoint
                idx_sampling = np.random.choice(
                    xs.shape[0], max_dimension - xs.shape[0], replace=True
                )
                padded_valid_xs.append(
                    torch.cat([xs, xs[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_valid_ys.append(
                    torch.cat([ys, ys[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_valid_ts.append(
                    torch.cat([ts, ts[idx_sampling]], dim=0).unsqueeze(1)
                )
            else:
                padded_valid_xs.append(xs.unsqueeze(1))
                padded_valid_ys.append(ys.unsqueeze(1))
                padded_valid_ts.append(ts.unsqueeze(1))

        X_valid = {
            t: padded_valid_xs[i].squeeze()
            for i, t in enumerate(sorted(X_valid.keys()))
        }
        y_valid = {
            t: padded_valid_ys[i].squeeze()
            for i, t in enumerate(sorted(y_valid.keys()))
        }

        # Couple samples without labels
        X_valid, y_valid, _ = couple_samples_no_int(X_valid, y_valid)

        timepoints = sorted(X_valid.keys())
        X_valid = torch.cat(
            [X_valid[t].unsqueeze(1) for t in sorted(X_valid.keys())], dim=1
        )
        y_valid = torch.cat(
            [y_valid[t].unsqueeze(1) for t in sorted(y_valid.keys())], dim=1
        )
        t_valid = torch.from_numpy(
            np.repeat(np.array(timepoints)[None, :], X_valid.shape[0], axis=0)
        )

    elif coupling == "cot":
        # Apply OT per class/target/label
        all_timepoints = sorted(X_train.keys())
        # Change nested dict structure from X_train[t][c] to X_train[c][t]
        xs = swap_levels(X_train)
        ys = swap_levels(y_train)

        for c in condition_list:
            X_coupled, y_coupled, _ = couple_samples_no_int(xs[c], ys[c])
            for t in all_timepoints:
                if xs.get(c, {}).get(t, None) is not None:
                    X_train[t][c] = X_coupled[t]
                    y_train[t][c] = y_coupled[t]

        # Sum up all samples for each timepoint
        n_samples = sum(X_train[0][c].shape[0] for c in condition_list)
        n_features = X_train[0][1].shape[1]
        X_train_final = np.nan * torch.ones(n_samples, len(all_timepoints), n_features)
        y_train_final = np.nan * torch.ones(n_samples, len(all_timepoints))
        indices_per_class = {
            c: range(
                sum(X_train[0][cc].shape[0] for cc in condition_list if cc < c),
                sum(X_train[0][cc].shape[0] for cc in condition_list if cc <= c),
            )
            for c in condition_list
        }
        for t in all_timepoints:
            for c in condition_list:
                t_idx = all_timepoints.index(t)
                if X_train.get(t, {}).get(c, None) is not None:
                    X_train_final[indices_per_class[c], t_idx] = X_train[t][c]
                    y_train_final[indices_per_class[c], t_idx] = y_train[t][c]

        X_train = X_train_final
        y_train = y_train_final
        t_train = torch.repeat_interleave(
            torch.tensor(all_timepoints)[None, :], n_samples, dim=0
        )

        # Now for validation data

        # Apply OT per class/target/label
        all_timepoints = sorted(X_valid.keys())
        # Change nested dict structure from X_valid[t][c] to X_valid[c][t]
        xs = swap_levels(X_valid)
        ys = swap_levels(y_valid)

        for c in condition_list:
            X_coupled, y_coupled, _ = couple_samples_no_int(xs[c], ys[c])
            for t in all_timepoints:
                if xs.get(c, {}).get(t, None) is not None:
                    X_valid[t][c] = X_coupled[t]
                    y_valid[t][c] = y_coupled[t]

        # Sum up all samples for each timepoint
        n_samples = sum(X_valid[0][c].shape[0] for c in condition_list)
        n_features = X_valid[0][1].shape[1]
        X_valid_final = np.nan * torch.ones(n_samples, len(all_timepoints), n_features)
        y_valid_final = np.nan * torch.ones(n_samples, len(all_timepoints))
        indices_per_class = {
            c: range(
                sum(X_valid[0][cc].shape[0] for cc in condition_list if cc < c),
                sum(X_valid[0][cc].shape[0] for cc in condition_list if cc <= c),
            )
            for c in condition_list
        }
        for t in all_timepoints:
            for c in condition_list:
                t_idx = all_timepoints.index(t)
                if X_valid.get(t, {}).get(c, None) is not None:
                    X_valid_final[indices_per_class[c], t_idx] = X_valid[t][c]
                    y_valid_final[indices_per_class[c], t_idx] = y_valid[t][c]

        X_valid = X_valid_final
        y_valid = y_valid_final
        t_valid = torch.repeat_interleave(
            torch.tensor(all_timepoints)[None, :], n_samples, dim=0
        )

    # Process test data
    # Note that this is not coupled
    X_test = {
        t: torch.cat([X_test[t][c] for c in X_test[t].keys()], dim=0)
        for t in sorted(X_test.keys())
    }
    y_test = {
        t: torch.cat([y_test[t][c] for c in y_test[t].keys()], dim=0)
        for t in sorted(y_test.keys())
    }
    t_test = {
        t: torch.cat([t_test[t][c] for c in t_test[t].keys()], dim=0)
        for t in sorted(t_test.keys())
    }
    X_test = torch.cat([X_test[t].unsqueeze(1) for t in X_test.keys()], dim=1)
    y_test = torch.cat([y_test[t].unsqueeze(1) for t in y_test.keys()], dim=1)
    t_test = torch.cat([t_test[t].unsqueeze(1) for t in t_test.keys()], dim=1)

    dataset = torch.utils.data.TensorDataset(
        X_train.float(), y_train.float(), t_train.float()
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=X_train.shape[0] if batch_size is None else batch_size,
        shuffle=True,
    )

    # Convert to numpy arrays for analysis
    X_train = X_train.numpy().astype(np.float32)
    y_train = y_train.numpy().astype(np.float32)
    t_train = t_train.numpy().astype(np.float32)
    X_valid = X_valid.numpy().astype(np.float32)
    y_valid = y_valid.numpy().astype(np.float32)
    t_valid = t_valid.numpy().astype(np.float32)
    X_test = X_test.numpy().astype(np.float32)
    y_test = y_test.numpy().astype(np.float32)
    t_test = t_test.numpy().astype(np.float32)

    # Save all
    # in a dict on disk
    torch.save(
        {
            "train_loader": train_loader,
            "X_train": X_train,
            "y_train": y_train,
            "t_train": t_train,
            "X_valid": X_valid,
            "y_valid": y_valid,
            "t_valid": t_valid,
            "X_test": X_test,
            "y_test": y_test,
            "t_test": t_test,
            "ps": ps,
        },
        filename_full_data,
    )
    return (
        train_loader,
        X_train,
        y_train,
        t_train,
        X_valid,
        y_valid,
        t_valid,
        X_test,
        y_test,
        t_test,
        ps,
    )


def dgp_iccite_4t(
    hvg,
    subsample_frac,
    use_pca,
    coupling=None,
    batch_size=None,
    n_samples_per_c_in_b=250,
    train_test_split=0.8,
    seed=0,
    preproc=True,
    top_n_effects=None,
    preset=None,
    leave_out_mid=None,
    leave_out_end=None,
    filter_beginning_end=False,
):
    """Create data for the ICCITE experiment."""
    # Print all files in data folder
    filename_full_data = f"/home/rohbeckm/code/mmfm/benchmark/dgp_iccite/data/iccite_4t_{hvg}_{use_pca}_{subsample_frac}_{coupling}_{batch_size}_{n_samples_per_c_in_b}_{train_test_split}_{top_n_effects}_{leave_out_mid}_{leave_out_end}_{seed}_{preset}.pt"
    filename_partial_data = f"/home/rohbeckm/code/mmfm/benchmark/dgp_iccite/data/iccite_4t_preproc_{hvg}_{use_pca}.h5ad"
    if filter_beginning_end:
        filename_full_data = filename_full_data.replace(".pt", "_filtered.pt")
        filename_partial_data = filename_partial_data.replace(".h5ad", "_filtered.h5ad")
    if preproc & Path(filename_full_data).exists():
        print("Loading FULL preprocessed data")
        data = torch.load(filename_full_data)
        return (
            data["train_loader"],
            data["X_train"],
            data["y_train"],
            data["t_train"],
            data["X_valid"],
            data["y_valid"],
            data["t_valid"],
            data["X_test"],
            data["y_test"],
            data["t_test"],
            data["ps"],
        )

    # # Check if file exists: "data/schiebinger_preproc_{ipsc_timepoint}_{hvg}_{use_pca}.h5ad"
    if preproc & Path(filename_partial_data).exists():
        print("Loading PARTIAL preprocessed data")
        adata = ad.read_h5ad(filename_partial_data)
    else:
        pl.seed_everything(seed)

        if use_pca and hvg:
            raise ValueError("Cannot use PCA and HVG at the same time.")

        adata = sc.read(
            "/home/rohbeckm/code/mmfm/data/icCITE-plex_filtered_top_drugs.h5ad"
        )
        if subsample_frac:
            adata = adata[adata.obs.sample(frac=subsample_frac).index]

        # # Remove all data where dose is != 10 uM
        # # adata.obs["dose"].value_counts()  # 10 uM
        # print(f"Before filtering dosage: {adata.shape}")
        # adata = adata[adata.obs["dose"] == "10 uM"]
        # print(f"After filtering dosage: {adata.shape}")

        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        if use_pca:
            sc.pp.pca(adata, n_comps=use_pca)
            # THIS DOES NOT WORK: adata.X = adata.obsm["X_pca"]
            adata = sc.AnnData(
                X=adata.obsm["X_pca"],
                obs=adata.obs,
                var=pd.DataFrame(
                    index=[f"PC{i+1}" for i in range(adata.obsm["X_pca"].shape[1])]
                ),
                uns=adata.uns,
            )
        elif hvg is not None:
            sc.pp.highly_variable_genes(adata, n_top_genes=hvg)
            adata = adata[:, adata.var["highly_variable"]]

        # Convert age to float timepoints
        adata.obs["timepoint"] = (
            adata.obs["timepoint"].astype(str).str.extract(r"(\d+)").astype(int)
        )
        # Create new timepoint 0 for all non-stimulated data
        adata.obs.loc[
            adata.obs["treatment"].isin(
                ["No stim_1uM", "No stim_100nM", "No stim_10uM"]
            ),
            "timepoint",
        ] = 0
        adata.obs["timepoint"] = adata.obs["timepoint"] / 72
        if filter_beginning_end:
            adata = adata[adata.obs["timepoint"].isin([0, 1])]

        # Round to two digits
        adata.obs["timepoint"] = adata.obs["timepoint"].round(2)

        # Print number of samples per timepoint and perturbation
        # dfx = adata.obs.groupby(["timepoint", "target"]).size().reset_index().pivot(index="timepoint", columns="target")

        adata.write(filename_partial_data)

    if preset is not None:
        if preset == "z":
            # Load df_experiment.to_csv("./data/experiment_50_20_random.csv")
            df_experiment = pd.read_csv(
                "/home/rohbeckm/code/mmfm/benchmark/dgp_iccite/data/experiment_4t_60_30_random.csv"
            )

        elif preset == "y":
            # Load df_experiment.to_csv("./data/experiment_50_20_random.csv")
            df_experiment = pd.read_csv(
                "/home/rohbeckm/code/mmfm/benchmark/dgp_iccite/data/experiment_4t_90_45_random.csv"
            )

        else:
            raise ValueError("Unknown preset")

        treatments = df_experiment.loc[df_experiment["train"], "treatment"].unique()
        leave_out_beg = df_experiment.loc[
            df_experiment["leave_out"] == "beg", "treatment"
        ].unique()
        leave_out_mid = df_experiment.loc[
            df_experiment["leave_out"] == "mid", "treatment"
        ].unique()
        leave_out_end = df_experiment.loc[
            df_experiment["leave_out"] == "end", "treatment"
        ].unique()

        # Remove all leave_out_end from treatments
        if filter_beginning_end:
            treatments = df_experiment.loc[
                (df_experiment["train"]) & (df_experiment["leave_out"] != "end"),
                "treatment",
            ].unique()
        else:
            treatments = df_experiment.loc[df_experiment["train"], "treatment"].unique()

        adata_control = adata[
            adata.obs["treatment"].isin(
                ["No stim_1uM", "No stim_100nM", "No stim_10uM"]
            )
        ].copy()

        adata = adata[
            adata.obs["treatment"].isin(treatments)
            # | adata.obs["treatment"].isin(["No stim_1uM", "No stim_100nM", "No stim_10uM"])
        ]
        print(f"Number of treatments: {len(np.unique(adata.obs['treatment']))}")

        ps = dict(enumerate(adata.obs["treatment"].value_counts().index, 1))
        adata.obs["condition"] = adata.obs["treatment"].map(
            {v: k for k, v in ps.items()}
        )
        adata.obs["condition"] = adata.obs["condition"].astype(int)

    else:
        subset_effects = adata.obs["treatment"].value_counts()[: (top_n_effects + 1)]
        adata = adata[adata.obs["treatment"].isin(subset_effects.index)]

        raise NotImplementedError("No preset implemented for DGP_iccite")

    condition_list = range(1, len(adata.obs["condition"].unique()) + 1)
    all_timepoints = sorted(adata.obs["timepoint"].unique())

    # Sample which conditions to leave out
    X_data = Dict()
    y_data = Dict()
    t_data = Dict()
    X_test = Dict()
    y_test = Dict()
    t_test = Dict()
    for condition in sorted(adata.obs.condition.unique()):
        # We observe condition two only at a shifted point in time
        for timepoint in all_timepoints:
            if timepoint == 0:
                continue
            potential_samples = adata[
                (adata.obs["timepoint"] == timepoint)
                & (adata.obs["condition"] == condition)
            ].obs.index
            potential_test_samples = None
            # Skip if we want to leave out this condition
            # Sample n_samples_per_condition_in_batch samples per condition and timepoint at random
            if len(potential_samples) > n_samples_per_c_in_b:
                idx = np.random.choice(
                    potential_samples, n_samples_per_c_in_b, replace=False
                )
            else:
                # Remove at least one sample for testing
                potential_samples, potential_test_samples = potential_samples[:-1], [
                    potential_samples[-1]
                ]
                idx = np.random.choice(
                    potential_samples, n_samples_per_c_in_b, replace=True
                )

            if potential_test_samples is None:
                potential_test_samples = np.setdiff1d(potential_samples, idx)
            if len(potential_test_samples) > n_samples_per_c_in_b:
                idx_test = np.random.choice(
                    potential_test_samples, n_samples_per_c_in_b, replace=False
                )
            else:
                idx_test = np.random.choice(
                    potential_test_samples, n_samples_per_c_in_b, replace=True
                )

            # if not (
            #     ((ps[condition] in leave_out_end) & (timepoint == 1))
            #     | ((ps[condition] in leave_out_mid) & (timepoint == 0.5))
            # ):
            X_data[timepoint][condition] = np.array(
                adata[adata.obs.index.get_indexer(idx)].X
            )
            y_data[timepoint][condition] = condition * np.ones(
                shape=(X_data[timepoint][condition].shape[0])
            )
            t_data[timepoint][condition] = timepoint * np.ones(
                shape=(X_data[timepoint][condition].shape[0])
            )

            X_test[timepoint][condition] = np.array(
                adata[adata.obs.index.get_indexer(idx_test)].X
            )
            y_test[timepoint][condition] = condition * np.ones(
                shape=(X_test[timepoint][condition].shape[0])
            )
            t_test[timepoint][condition] = timepoint * np.ones(
                shape=(X_test[timepoint][condition].shape[0])
            )

    # Add data from control as timepoint 0
    random_samples_test = np.random.choice(
        adata_control.obs.index, n_samples_per_c_in_b, replace=False
    )
    random_samples_train = np.setdiff1d(adata_control.obs.index, random_samples_test)
    for condition in sorted(adata.obs.condition.unique()):
        subset_sample_train = np.random.choice(
            random_samples_train, n_samples_per_c_in_b, replace=False
        )
        X_data[0][condition] = np.array(
            adata_control[adata_control.obs.index.get_indexer(subset_sample_train)].X
        )
        y_data[0][condition] = condition * np.ones(
            shape=(X_data[0][condition].shape[0])
        )
        t_data[0][condition] = 0 * np.ones(shape=(X_data[0][condition].shape[0]))

        X_test[0][condition] = np.array(
            adata_control[adata_control.obs.index.get_indexer(random_samples_test)].X
        )
        y_test[0][condition] = condition * np.ones(
            shape=(X_test[0][condition].shape[0])
        )
        t_test[0][condition] = 0 * np.ones(shape=(X_test[0][condition].shape[0]))

    X_train, y_train, t_train = Dict(), Dict(), Dict()
    X_valid, y_valid, t_valid = Dict(), Dict(), Dict()

    all_timepoints = sorted(all_timepoints + [0])

    # Split data into train and validation
    for c in condition_list:
        for t in all_timepoints:
            if X_data.get(t, {}).get(c, None) is not None:
                n_samples = X_data[t][c].shape[0]
                n_samples_train_per_c = int(
                    train_test_split * n_samples
                )  # This value always exists
                idx_train = np.random.choice(
                    np.arange(n_samples), size=n_samples_train_per_c, replace=False
                )
                idx_valid = np.setdiff1d(np.arange(n_samples), idx_train)
                X_train[t][c] = torch.tensor(X_data[t][c][idx_train]).float()
                y_train[t][c] = torch.tensor(y_data[t][c][idx_train]).float()
                t_train[t][c] = torch.tensor(t_data[t][c][idx_train]).float()
                X_valid[t][c] = torch.tensor(X_data[t][c][idx_valid]).float()
                y_valid[t][c] = torch.tensor(y_data[t][c][idx_valid]).float()
                t_valid[t][c] = torch.tensor(t_data[t][c][idx_valid]).float()
            if X_test.get(t, {}).get(c, None) is not None:
                X_test[t][c] = torch.tensor(X_test[t][c]).float()
                y_test[t][c] = torch.tensor(y_test[t][c]).float()
                t_test[t][c] = torch.tensor(t_test[t][c]).float()

    # Remove all timepoints and condition from training data which we must not observe
    if not filter_beginning_end:
        timepoint = 1.0
        for c_name in leave_out_end:
            c = [k for k, v in ps.items() if v == c_name][0]
            X_train[timepoint].pop(c)
            y_train[timepoint].pop(c)
            t_train[timepoint].pop(c)
        timepoint = 0.67
        for c_name in leave_out_mid:
            c = [k for k, v in ps.items() if v == c_name][0]
            X_train[timepoint].pop(c)
            y_train[timepoint].pop(c)
            t_train[timepoint].pop(c)
        timepoint = 0.33
        for c_name in leave_out_beg:
            c = [k for k, v in ps.items() if v == c_name][0]
            X_train[timepoint].pop(c)
            y_train[timepoint].pop(c)
            t_train[timepoint].pop(c)

    def swap_levels(nested_dict):
        swapped = Dict()
        for outer_key, inner_dict in nested_dict.items():
            for inner_key, value in inner_dict.items():
                swapped[inner_key][outer_key] = value
        return swapped

    if coupling is None:
        X_train = {
            t: torch.cat([X_train[t][c] for c in X_train[t].keys()], dim=0)
            for t in sorted(X_train.keys())
        }
        y_train = {
            t: torch.cat([y_train[t][c] for c in y_train[t].keys()], dim=0)
            for t in sorted(y_train.keys())
        }
        t_train = {
            t: torch.cat([t_train[t][c] for c in t_train[t].keys()], dim=0)
            for t in sorted(t_train.keys())
        }

        max_dimension = max([X_train[t].shape[0] for t in X_train.keys()])
        padded_train_xs, padded_train_ys, padded_train_ts = [], [], []
        for xs, ys, ts in zip(
            X_train.values(), y_train.values(), t_train.values(), strict=False
        ):
            if xs.shape[0] < max_dimension:
                print("- Upsampling and Padding")
                # Instead of padding with missings, we pad with random points from the same timepoint
                idx_sampling = np.random.choice(
                    xs.shape[0], max_dimension - xs.shape[0], replace=True
                )
                padded_train_xs.append(
                    torch.cat([xs, xs[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_train_ys.append(
                    torch.cat([ys, ys[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_train_ts.append(
                    torch.cat([ts, ts[idx_sampling]], dim=0).unsqueeze(1)
                )
            else:
                padded_train_xs.append(xs.unsqueeze(1))
                padded_train_ys.append(ys.unsqueeze(1))
                padded_train_ts.append(ts.unsqueeze(1))

        X_train = {
            t: padded_train_xs[i].squeeze()
            for i, t in enumerate(sorted(X_train.keys()))
        }
        y_train = {
            t: padded_train_ys[i].squeeze()
            for i, t in enumerate(sorted(y_train.keys()))
        }

        timepoints = sorted(X_train.keys())
        X_train = torch.cat(
            [X_train[t].unsqueeze(1) for t in sorted(X_train.keys())], dim=1
        )
        y_train = torch.cat(
            [y_train[t].unsqueeze(1) for t in sorted(y_train.keys())], dim=1
        )
        t_train = torch.from_numpy(
            np.repeat(np.array(timepoints)[None, :], X_train.shape[0], axis=0)
        )

        X_valid = {
            t: torch.cat([X_valid[t][c] for c in X_valid[t].keys()], dim=0)
            for t in sorted(X_valid.keys())
        }
        y_valid = {
            t: torch.cat([y_valid[t][c] for c in y_valid[t].keys()], dim=0)
            for t in sorted(y_valid.keys())
        }
        t_valid = {
            t: torch.cat([t_valid[t][c] for c in t_valid[t].keys()], dim=0)
            for t in sorted(t_valid.keys())
        }

        max_dimension = max([X_valid[t].shape[0] for t in X_valid.keys()])
        padded_valid_xs, padded_valid_ys, padded_valid_ts = [], [], []
        for xs, ys, ts in zip(
            X_valid.values(), y_valid.values(), t_valid.values(), strict=False
        ):
            if xs.shape[0] < max_dimension:
                print("- Upsampling and Padding")
                # Instead of padding with missings, we pad with random points from the same timepoint
                idx_sampling = np.random.choice(
                    xs.shape[0], max_dimension - xs.shape[0], replace=True
                )
                padded_valid_xs.append(
                    torch.cat([xs, xs[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_valid_ys.append(
                    torch.cat([ys, ys[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_valid_ts.append(
                    torch.cat([ts, ts[idx_sampling]], dim=0).unsqueeze(1)
                )
            else:
                padded_valid_xs.append(xs.unsqueeze(1))
                padded_valid_ys.append(ys.unsqueeze(1))
                padded_valid_ts.append(ts.unsqueeze(1))

        X_valid = {
            t: padded_valid_xs[i].squeeze()
            for i, t in enumerate(sorted(X_valid.keys()))
        }
        y_valid = {
            t: padded_valid_ys[i].squeeze()
            for i, t in enumerate(sorted(y_valid.keys()))
        }

        timepoints = sorted(X_valid.keys())
        X_valid = torch.cat(
            [X_valid[t].unsqueeze(1) for t in sorted(X_valid.keys())], dim=1
        )
        y_valid = torch.cat(
            [y_valid[t].unsqueeze(1) for t in sorted(y_valid.keys())], dim=1
        )
        t_valid = torch.from_numpy(
            np.repeat(np.array(timepoints)[None, :], X_valid.shape[0], axis=0)
        )

    elif coupling == "ot":
        X_train = {
            t: torch.cat([X_train[t][c] for c in X_train[t].keys()], dim=0)
            for t in sorted(X_train.keys())
        }
        y_train = {
            t: torch.cat([y_train[t][c] for c in y_train[t].keys()], dim=0)
            for t in sorted(y_train.keys())
        }
        t_train = {
            t: torch.cat([t_train[t][c] for c in t_train[t].keys()], dim=0)
            for t in sorted(t_train.keys())
        }

        max_dimension = max([X_train[t].shape[0] for t in X_train.keys()])
        padded_train_xs, padded_train_ys, padded_train_ts = [], [], []
        for xs, ys, ts in zip(
            X_train.values(), y_train.values(), t_train.values(), strict=False
        ):
            if xs.shape[0] < max_dimension:
                print("- Upsampling and Padding")
                # Instead of padding with missings, we pad with random points from the same timepoint
                idx_sampling = np.random.choice(
                    xs.shape[0], max_dimension - xs.shape[0], replace=True
                )
                padded_train_xs.append(
                    torch.cat([xs, xs[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_train_ys.append(
                    torch.cat([ys, ys[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_train_ts.append(
                    torch.cat([ts, ts[idx_sampling]], dim=0).unsqueeze(1)
                )
            else:
                padded_train_xs.append(xs.unsqueeze(1))
                padded_train_ys.append(ys.unsqueeze(1))
                padded_train_ts.append(ts.unsqueeze(1))

        X_train = {
            t: padded_train_xs[i].squeeze()
            for i, t in enumerate(sorted(X_train.keys()))
        }
        y_train = {
            t: padded_train_ys[i].squeeze()
            for i, t in enumerate(sorted(y_train.keys()))
        }

        # Couple samples without labels
        X_train, y_train, _ = couple_samples_no_int(X_train, y_train)

        timepoints = sorted(X_train.keys())
        X_train = torch.cat(
            [X_train[t].unsqueeze(1) for t in sorted(X_train.keys())], dim=1
        )
        y_train = torch.cat(
            [y_train[t].unsqueeze(1) for t in sorted(y_train.keys())], dim=1
        )
        t_train = torch.from_numpy(
            np.repeat(np.array(timepoints)[None, :], X_train.shape[0], axis=0)
        )

        X_valid = {
            t: torch.cat([X_valid[t][c] for c in X_valid[t].keys()], dim=0)
            for t in sorted(X_valid.keys())
        }
        y_valid = {
            t: torch.cat([y_valid[t][c] for c in y_valid[t].keys()], dim=0)
            for t in sorted(y_valid.keys())
        }
        t_valid = {
            t: torch.cat([t_valid[t][c] for c in t_valid[t].keys()], dim=0)
            for t in sorted(t_valid.keys())
        }

        max_dimension = max([X_valid[t].shape[0] for t in X_valid.keys()])
        padded_valid_xs, padded_valid_ys, padded_valid_ts = [], [], []
        for xs, ys, ts in zip(
            X_valid.values(), y_valid.values(), t_valid.values(), strict=False
        ):
            if xs.shape[0] < max_dimension:
                print("- Upsampling and Padding")
                # Instead of padding with missings, we pad with random points from the same timepoint
                idx_sampling = np.random.choice(
                    xs.shape[0], max_dimension - xs.shape[0], replace=True
                )
                padded_valid_xs.append(
                    torch.cat([xs, xs[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_valid_ys.append(
                    torch.cat([ys, ys[idx_sampling]], dim=0).unsqueeze(1)
                )
                padded_valid_ts.append(
                    torch.cat([ts, ts[idx_sampling]], dim=0).unsqueeze(1)
                )
            else:
                padded_valid_xs.append(xs.unsqueeze(1))
                padded_valid_ys.append(ys.unsqueeze(1))
                padded_valid_ts.append(ts.unsqueeze(1))

        X_valid = {
            t: padded_valid_xs[i].squeeze()
            for i, t in enumerate(sorted(X_valid.keys()))
        }
        y_valid = {
            t: padded_valid_ys[i].squeeze()
            for i, t in enumerate(sorted(y_valid.keys()))
        }

        # Couple samples without labels
        X_valid, y_valid, _ = couple_samples_no_int(X_valid, y_valid)

        timepoints = sorted(X_valid.keys())
        X_valid = torch.cat(
            [X_valid[t].unsqueeze(1) for t in sorted(X_valid.keys())], dim=1
        )
        y_valid = torch.cat(
            [y_valid[t].unsqueeze(1) for t in sorted(y_valid.keys())], dim=1
        )
        t_valid = torch.from_numpy(
            np.repeat(np.array(timepoints)[None, :], X_valid.shape[0], axis=0)
        )

    elif coupling == "cot":
        # Apply OT per class/target/label
        all_timepoints = sorted(X_train.keys())
        # Change nested dict structure from X_train[t][c] to X_train[c][t]
        xs = swap_levels(X_train)
        ys = swap_levels(y_train)

        for c in condition_list:
            X_coupled, y_coupled, _ = couple_samples_no_int(xs[c], ys[c])
            for t in all_timepoints:
                if xs.get(c, {}).get(t, None) is not None:
                    X_train[t][c] = X_coupled[t]
                    y_train[t][c] = y_coupled[t]

        # Sum up all samples for each timepoint
        n_samples = sum(X_train[0][c].shape[0] for c in condition_list)
        n_features = X_train[0][1].shape[1]
        X_train_final = np.nan * torch.ones(n_samples, len(all_timepoints), n_features)
        y_train_final = np.nan * torch.ones(n_samples, len(all_timepoints))
        indices_per_class = {
            c: range(
                sum(X_train[0][cc].shape[0] for cc in condition_list if cc < c),
                sum(X_train[0][cc].shape[0] for cc in condition_list if cc <= c),
            )
            for c in condition_list
        }
        for t in all_timepoints:
            for c in condition_list:
                t_idx = all_timepoints.index(t)
                if X_train.get(t, {}).get(c, None) is not None:
                    X_train_final[indices_per_class[c], t_idx] = X_train[t][c]
                    y_train_final[indices_per_class[c], t_idx] = y_train[t][c]

        X_train = X_train_final
        y_train = y_train_final
        t_train = torch.repeat_interleave(
            torch.tensor(all_timepoints)[None, :], n_samples, dim=0
        )

        # Now for validation data

        # Apply OT per class/target/label
        all_timepoints = sorted(X_valid.keys())
        # Change nested dict structure from X_valid[t][c] to X_valid[c][t]
        xs = swap_levels(X_valid)
        ys = swap_levels(y_valid)

        for c in condition_list:
            X_coupled, y_coupled, _ = couple_samples_no_int(xs[c], ys[c])
            for t in all_timepoints:
                if xs.get(c, {}).get(t, None) is not None:
                    X_valid[t][c] = X_coupled[t]
                    y_valid[t][c] = y_coupled[t]

        # Sum up all samples for each timepoint
        n_samples = sum(X_valid[0][c].shape[0] for c in condition_list)
        n_features = X_valid[0][1].shape[1]
        X_valid_final = np.nan * torch.ones(n_samples, len(all_timepoints), n_features)
        y_valid_final = np.nan * torch.ones(n_samples, len(all_timepoints))
        indices_per_class = {
            c: range(
                sum(X_valid[0][cc].shape[0] for cc in condition_list if cc < c),
                sum(X_valid[0][cc].shape[0] for cc in condition_list if cc <= c),
            )
            for c in condition_list
        }
        for t in all_timepoints:
            for c in condition_list:
                t_idx = all_timepoints.index(t)
                if X_valid.get(t, {}).get(c, None) is not None:
                    X_valid_final[indices_per_class[c], t_idx] = X_valid[t][c]
                    y_valid_final[indices_per_class[c], t_idx] = y_valid[t][c]

        X_valid = X_valid_final
        y_valid = y_valid_final
        t_valid = torch.repeat_interleave(
            torch.tensor(all_timepoints)[None, :], n_samples, dim=0
        )

    # Process test data
    # Note that this is not coupled
    X_test = {
        t: torch.cat([X_test[t][c] for c in X_test[t].keys()], dim=0)
        for t in sorted(X_test.keys())
    }
    y_test = {
        t: torch.cat([y_test[t][c] for c in y_test[t].keys()], dim=0)
        for t in sorted(y_test.keys())
    }
    t_test = {
        t: torch.cat([t_test[t][c] for c in t_test[t].keys()], dim=0)
        for t in sorted(t_test.keys())
    }
    X_test = torch.cat([X_test[t].unsqueeze(1) for t in X_test.keys()], dim=1)
    y_test = torch.cat([y_test[t].unsqueeze(1) for t in y_test.keys()], dim=1)
    t_test = torch.cat([t_test[t].unsqueeze(1) for t in t_test.keys()], dim=1)

    dataset = torch.utils.data.TensorDataset(
        X_train.float(), y_train.float(), t_train.float()
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=X_train.shape[0] if batch_size is None else batch_size,
        shuffle=True,
    )

    # Convert to numpy arrays for analysis
    X_train = X_train.numpy().astype(np.float32)
    y_train = y_train.numpy().astype(np.float32)
    t_train = t_train.numpy().astype(np.float32)
    X_valid = X_valid.numpy().astype(np.float32)
    y_valid = y_valid.numpy().astype(np.float32)
    t_valid = t_valid.numpy().astype(np.float32)
    X_test = X_test.numpy().astype(np.float32)
    y_test = y_test.numpy().astype(np.float32)
    t_test = t_test.numpy().astype(np.float32)

    # Save all
    # in a dict on disk
    torch.save(
        {
            "train_loader": train_loader,
            "X_train": X_train,
            "y_train": y_train,
            "t_train": t_train,
            "X_valid": X_valid,
            "y_valid": y_valid,
            "t_valid": t_valid,
            "X_test": X_test,
            "y_test": y_test,
            "t_test": t_test,
            "ps": ps,
        },
        filename_full_data,
    )
    return (
        train_loader,
        X_train,
        y_train,
        t_train,
        X_valid,
        y_valid,
        t_valid,
        X_test,
        y_test,
        t_test,
        ps,
    )


#
# Wrapper
#
def dgp_iccite_data(
    hvg,
    subsample_frac,
    use_pca,
    coupling,
    batch_size,
    n_samples_per_c_in_b,
    train_test_split,
    dgp,
    top_n_effects,
    leave_out_mid,
    leave_out_end,
    preset,
    return_data="train-valid",
    seed=0,
):
    """Return dataset created by DGP-ICCITE"""
    if dgp == "a":
        # TOP 50 Drugs
        if return_data == "test":
            if preset == "a":
                _, _, _, _, _, _, _, X_test, y_test, t_test, ps = dgp_iccite(
                    hvg=hvg,
                    use_pca=use_pca,
                    coupling=coupling,
                    batch_size=batch_size,
                    n_samples_per_c_in_b=n_samples_per_c_in_b,
                    train_test_split=train_test_split,
                    subsample_frac=subsample_frac,
                    seed=seed,
                    top_n_effects=top_n_effects,
                    leave_out_mid=leave_out_mid,
                    leave_out_end=leave_out_end,
                    preset=preset,
                )

                timepoints = list(np.unique(t_test))
                all_classes = range(1, 51)
                n_classes = len(all_classes)

            elif preset == "z":
                _, _, _, _, _, _, _, X_test, y_test, t_test, ps = dgp_iccite_4t(
                    hvg=hvg,
                    use_pca=use_pca,
                    coupling=coupling,
                    batch_size=batch_size,
                    n_samples_per_c_in_b=n_samples_per_c_in_b,
                    train_test_split=train_test_split,
                    subsample_frac=subsample_frac,
                    seed=seed,
                    top_n_effects=top_n_effects,
                    leave_out_mid=leave_out_mid,
                    leave_out_end=leave_out_end,
                    preset=preset,
                )

                timepoints = list(np.unique(t_test))
                all_classes = range(1, 61)
                n_classes = len(all_classes)

            elif preset == "y":
                _, _, _, _, _, _, _, X_test, y_test, t_test, ps = dgp_iccite_4t(
                    hvg=hvg,
                    use_pca=use_pca,
                    coupling=coupling,
                    batch_size=batch_size,
                    n_samples_per_c_in_b=n_samples_per_c_in_b,
                    train_test_split=train_test_split,
                    subsample_frac=subsample_frac,
                    seed=seed,
                    top_n_effects=top_n_effects,
                    leave_out_mid=leave_out_mid,
                    leave_out_end=leave_out_end,
                    preset=preset,
                )

                timepoints = list(np.unique(t_test))
                all_classes = range(1, 91)
                n_classes = len(all_classes)

            else:
                raise NotImplementedError("NOT IMPLEMENTED YET, Martin!")

    return X_test, y_test, t_test, n_classes, timepoints, all_classes, ps


def dgp_beijing_data(
    coupling,
    batch_size,
    ns_per_t_and_c,
    dgp,
    return_data="train-valid",
    add_time_cond=None,
    filter_beginning_end=False,
):
    if dgp == "a":
        label_list = [int(x) for x in np.linspace(1, 12, 12)]
        n_classes = len(label_list)

        target = "PM2.5"
        start = 2015
        end = None
        n_quarters = None

        params_global = {
            "coupling": coupling,
            "batch_size": batch_size,
            "target": target,
            "start": start,
            "end": end,
            "n_quarters": n_quarters,
            "ns_per_t_and_c": ns_per_t_and_c
        }

        timepoints_train = [0, 2, 4, 6, 7, 11, 12, 14, 16, 18, 19, 23, 25] if not filter_beginning_end else [0, 25]
        timepoints_train_holdout0 = [0, 5, 9, 10, 17, 20, 25] if not filter_beginning_end else [0, 25]
        timepoints_train_holdout1 = [0, 3, 13, 17, 20, 25] if not filter_beginning_end else [0, 25]
        timepoints_train_holdout2 = [0, 4, 11, 13, 17, 23, 25] if not filter_beginning_end else [0, 25]
        timepoints_valid = list(range(26))
        timepoints_test = list(range(26))

        if return_data == "train-valid":
            # TRAIN
            kwargs = {
                "data_specs": {
                    v: {
                        "timepoints": timepoints_train,
                        "condition": v,
                    }
                    for v in label_list
                },
                **params_global,
            }
            kwargs["data_specs"][4] = {
                "timepoints": timepoints_train_holdout0,
                "condition": 4.0,
            }
            kwargs["data_specs"][7] = {
                "timepoints": timepoints_train_holdout1,
                "condition": 7.0,
            }
            kwargs["data_specs"][10.0] = {
                "timepoints": timepoints_train_holdout2,
                "condition": 10.0,
            }
            if add_time_cond is not None:
                for atc in add_time_cond:
                    current_timepoints = kwargs["data_specs"][atc[0]]["timepoints"]
                    new_timepoints = sorted(
                        current_timepoints
                        + (atc[1] if isinstance(atc[1], list) else [atc[1]])
                    )
                    print(atc[0], new_timepoints)
                    kwargs["data_specs"][atc[0]]["timepoints"] = new_timepoints

            train_loader, X_train, y_train, t_train = dgp_beijing(
                **kwargs
            )

            # VALID
            kwargs = {
                "data_specs": {
                    v: {
                        "timepoints": timepoints_valid,
                        "condition": v,
                    }
                    for v in label_list
                },
                **params_global,
            }

            _, X_valid, y_valid, t_valid = dgp_beijing(**kwargs)

            return (
                train_loader,
                X_train,
                y_train,
                t_train,
                X_valid,
                y_valid,
                t_valid,
                n_classes,
                label_list,
            )

        elif return_data == "test":
            # VALID
            kwargs = {
                "data_specs": {
                    v: {
                        "timepoints": timepoints_test,
                        "condition": v,
                    }
                    for v in label_list
                },
                **params_global,
            }

            timepoints = sorted(
                {
                    item
                    for sublist in [
                        v["timepoints"] for k, v in kwargs["data_specs"].items()
                    ]
                    for item in sublist
                }
            )
            _, X_test, y_test, t_test = dgp_beijing(**kwargs)

            return X_test, y_test, t_test, n_classes, timepoints, label_list
        
    if dgp == "b":
        label_list = [int(x) for x in np.linspace(1, 12, 12)]
        n_classes = len(label_list)

        target = "NO2"
        start = 2015
        end = None
        n_quarters = None

        params_global = {
            "coupling": coupling,
            "batch_size": batch_size,
            "target": target,
            "start": start,
            "end": end,
            "n_quarters": n_quarters,
            "ns_per_t_and_c": ns_per_t_and_c
        }

        timepoints_train = [0, 6, 11, 13, 19, 25] if not filter_beginning_end else [0, 25]
        timepoints_train_holdout0 = [0, 6, 13, 25] if not filter_beginning_end else [0, 25]
        timepoints_train_holdout1 = [0, 6, 13, 25] if not filter_beginning_end else [0, 25]
        timepoints_train_holdout2 = [0, 11, 19, 25] if not filter_beginning_end else [0, 25]
        timepoints_valid = list(range(26))
        timepoints_test = list(range(26))

        if return_data == "train-valid":
            # TRAIN
            kwargs = {
                "data_specs": {
                    v: {
                        "timepoints": timepoints_train,
                        "condition": v,
                    }
                    for v in label_list
                },
                **params_global,
            }
            kwargs["data_specs"][4] = {
                "timepoints": timepoints_train_holdout0,
                "condition": 4.0,
            }
            kwargs["data_specs"][7] = {
                "timepoints": timepoints_train_holdout1,
                "condition": 7.0,
            }
            kwargs["data_specs"][10.0] = {
                "timepoints": timepoints_train_holdout2,
                "condition": 10.0,
            }
            if add_time_cond is not None:
                for atc in add_time_cond:
                    current_timepoints = kwargs["data_specs"][atc[0]]["timepoints"]
                    new_timepoints = sorted(
                        current_timepoints
                        + (atc[1] if isinstance(atc[1], list) else [atc[1]])
                    )
                    kwargs["data_specs"][atc[0]]["timepoints"] = new_timepoints

            train_loader, X_train, y_train, t_train = dgp_beijing(
                **kwargs
            )

            # VALID
            kwargs = {
                "data_specs": {
                    v: {
                        "timepoints": timepoints_valid,
                        "condition": v,
                    }
                    for v in label_list
                },
                **params_global,
            }

            _, X_valid, y_valid, t_valid = dgp_beijing(**kwargs)

            return (
                train_loader,
                X_train,
                y_train,
                t_train,
                X_valid,
                y_valid,
                t_valid,
                n_classes,
                label_list,
            )

        elif return_data == "test":
            # VALID
            kwargs = {
                "data_specs": {
                    v: {
                        "timepoints": timepoints_test,
                        "condition": v,
                    }
                    for v in label_list
                },
                **params_global,
            }

            timepoints = sorted(
                {
                    item
                    for sublist in [
                        v["timepoints"] for k, v in kwargs["data_specs"].items()
                    ]
                    for item in sublist
                }
            )
            _, X_test, y_test, t_test = dgp_beijing(**kwargs)

            return X_test, y_test, t_test, n_classes, timepoints, label_list


def dgp_waves_data(
    coupling,
    batch_size,
    dimension,
    off_diagonal,
    data_std,
    ns_per_t_and_c,
    dgp,
    return_data="train-valid",
    add_time_cond=None,
    filter_beginning_end=False,
):
    """Return dataset created by DGP WAVES"""
    params_global = {
        "coupling": coupling,
        "batch_size": batch_size,
        "dimension": dimension,
        "off_diagonal": off_diagonal,
        "classes_first": True,
    }
    params_local = {
        "std": data_std,
        "n_samples": ns_per_t_and_c,
    }

    if dgp == "a":
        raise NotImplementedError("DGP not implemented")

    elif dgp == "b":
        raise NotImplementedError("DGP not implemented")

    elif dgp == "c":
        label_list = np.linspace(1, 10, 10)
        n_classes = len(label_list)
        params_global["vf"] = "u_sine"
        timepoints_train = (
            [0, 0.25, 0.5, 0.75, 1.0] if not filter_beginning_end else [0, 1.0]
        )
        timepoints_train_holdout = (
            [0, 0.25, 1.0] if not filter_beginning_end else [0, 1.0]
        )
        timepoints_valid = [
            0,
            0.05,
            0.1,
            0.2,
            0.25,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.75,
            0.8,
            0.9,
            0.95,
            1.0,
        ]
        timepoints_test = [
            0,
            0.05,
            0.1,
            0.2,
            0.25,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.75,
            0.8,
            0.9,
            0.95,
            1.0,
        ]
        y0 = (0.0, 0.0)

        if return_data == "train-valid":
            # TRAIN
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": y0,
                        "timepoints": timepoints_train,
                        "condition": v,
                        **params_local,
                    }
                    for _, v in enumerate(label_list)
                },
                **params_global,
            }
            kwargs["data_specs"][5] = {
                "y0": y0,
                "timepoints": timepoints_train_holdout,
                "condition": 5.0,
                **params_local,
            }
            if add_time_cond is not None:
                current_timepoints = kwargs["data_specs"][add_time_cond[0]][
                    "timepoints"
                ]
                new_timepoints = sorted(
                    current_timepoints
                    + (
                        add_time_cond[1]
                        if isinstance(add_time_cond[1], list)
                        else [add_time_cond[1]]
                    )
                )
                kwargs["data_specs"][add_time_cond[0]]["timepoints"] = new_timepoints
            train_loader, X_train, y_train, t_train = dgp_waves(**kwargs)

            # VALID
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": y0,
                        "timepoints": timepoints_valid,
                        "condition": v,
                        **params_local,
                    }
                    for _, v in enumerate(label_list)
                },
                **params_global,
            }
            _, X_valid, y_valid, t_valid = dgp_waves(**kwargs)

            return (
                train_loader,
                X_train,
                y_train,
                t_train,
                X_valid,
                y_valid,
                t_valid,
                n_classes,
                label_list,
            )

        elif return_data == "test":
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": y0,
                        "timepoints": timepoints_test,
                        "condition": v,
                        **params_local,
                    }
                    for _, v in enumerate(label_list)
                },
                **params_global,
            }

            timepoints = sorted(
                {
                    item
                    for sublist in [
                        v["timepoints"] for k, v in kwargs["data_specs"].items()
                    ]
                    for item in sublist
                }
            )
            _, X_test, y_test, t_test = dgp_waves(**kwargs)

            return X_test, y_test, t_test, n_classes, timepoints, label_list

    elif dgp == "d":
        label_list = np.linspace(1, 5.5, 10)
        n_classes = len(label_list)
        params_global["vf"] = "u"
        timepoints_train = (
            [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] if not filter_beginning_end else [0, 1.0]
        )
        timepoints_train_holdout = (
            [0, 0.1, 0.5, 0.9, 1.0] if not filter_beginning_end else [0, 1.0]
        )
        timepoints_valid = [
            0,
            0.05,
            0.075,
            0.08,
            0.09,
            0.1,
            0.11,
            0.12,
            0.125,
            0.15,
            0.2,
            0.25,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.75,
            0.8,
            0.9,
            0.95,
            1.0,
        ]
        timepoints_test = np.linspace(0, 1, endpoint=True, num=21)
        f_y0 = lambda x: (0, (x - 1) / 2)

        if return_data == "train-valid":
            # TRAIN
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": timepoints_train,
                        "condition": v,
                        **params_local,
                    }
                    for v in label_list
                },
                **params_global,
            }
            kwargs["data_specs"][3] = {
                "y0": f_y0(3),
                "timepoints": timepoints_train_holdout,
                "condition": 3.0,
                **params_local,
            }
            if add_time_cond is not None:
                for atc in add_time_cond:
                    current_timepoints = kwargs["data_specs"][atc[0]]["timepoints"]
                    new_timepoints = sorted(
                        current_timepoints
                        + (atc[1] if isinstance(atc[1], list) else [atc[1]])
                    )
                    kwargs["data_specs"][atc[0]]["timepoints"] = new_timepoints
                # current_timepoints = kwargs["data_specs"][add_time_cond[0]]["timepoints"]
                # new_timepoints = sorted(
                #     current_timepoints
                #     + (add_time_cond[1] if isinstance(add_time_cond[1], list) else [add_time_cond[1]])
                # )
                # kwargs["data_specs"][add_time_cond[0]]["timepoints"] = new_timepoints
            train_loader, X_train, y_train, t_train = dgp_waves(**kwargs)

            # VALID
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": timepoints_valid,
                        "condition": v,
                        **params_local,
                    }
                    for _, v in enumerate(label_list)
                },
                **params_global,
            }
            _, X_valid, y_valid, t_valid = dgp_waves(**kwargs)

            return (
                train_loader,
                X_train,
                y_train,
                t_train,
                X_valid,
                y_valid,
                t_valid,
                n_classes,
                label_list,
            )

        elif return_data == "test":
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": timepoints_test,
                        "condition": v,
                        **params_local,
                    }
                    for _, v in enumerate(label_list)
                },
                **params_global,
            }
        timepoints = sorted(
            {
                item
                for sublist in [
                    v["timepoints"] for k, v in kwargs["data_specs"].items()
                ]
                for item in sublist
            }
        )

        _, X_test, y_test, t_test = dgp_waves(**kwargs)

        return X_test, y_test, t_test, n_classes, timepoints, label_list

    elif dgp == "e":
        label_list = np.linspace(1, 10, 10)
        n_classes = len(label_list)
        params_global["vf"] = "u_sine"
        timepoints_train = (
            [0, 0.25, 0.5, 0.75, 1.0] if not filter_beginning_end else [0, 1.0]
        )
        timepoints_train_holdout = (
            [0, 0.25, 0.5] if not filter_beginning_end else [0, 0.5]
        )
        timepoints_valid = [
            0,
            0.05,
            0.1,
            0.2,
            0.25,
            0.3,
            0.4,
            0.5,
            0.51,
            0.52,
            0.525,
            0.55,
            0.6,
            0.7,
            0.75,
            0.8,
            0.9,
            0.95,
            1.0,
        ]
        timepoints_test = [
            0,
            0.05,
            0.1,
            0.2,
            0.25,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.75,
            0.8,
            0.9,
            0.95,
            1.0,
        ]
        y0 = (0.0, 0.0)

        if return_data == "train-valid":
            # TRAIN
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": y0,
                        "timepoints": timepoints_train,
                        "condition": v,
                        **params_local,
                    }
                    for _, v in enumerate(label_list)
                },
                **params_global,
            }
            kwargs["data_specs"][5] = {
                "y0": y0,
                "timepoints": timepoints_train_holdout,
                "condition": 5.0,
                **params_local,
            }
            if add_time_cond is not None:
                for atc in add_time_cond:
                    current_timepoints = kwargs["data_specs"][atc[0]]["timepoints"]
                    new_timepoints = sorted(
                        current_timepoints
                        + (atc[1] if isinstance(atc[1], list) else [atc[1]])
                    )
                    kwargs["data_specs"][atc[0]]["timepoints"] = new_timepoints
            train_loader, X_train, y_train, t_train = dgp_waves(**kwargs)

            # VALID
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": y0,
                        "timepoints": timepoints_valid,
                        "condition": v,
                        **params_local,
                    }
                    for _, v in enumerate(label_list)
                },
                **params_global,
            }
            _, X_valid, y_valid, t_valid = dgp_waves(**kwargs)

            return (
                train_loader,
                X_train,
                y_train,
                t_train,
                X_valid,
                y_valid,
                t_valid,
                n_classes,
                label_list,
            )

        elif return_data == "test":
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": y0,
                        "timepoints": timepoints_test,
                        "condition": v,
                        **params_local,
                    }
                    for _, v in enumerate(label_list)
                },
                **params_global,
            }

            timepoints = sorted(
                {
                    item
                    for sublist in [
                        v["timepoints"] for k, v in kwargs["data_specs"].items()
                    ]
                    for item in sublist
                }
            )
            _, X_test, y_test, t_test = dgp_waves(**kwargs)

            return X_test, y_test, t_test, n_classes, timepoints, label_list

    elif dgp == "f":
        label_list = np.linspace(1, 5.5, 10)
        n_classes = len(label_list)
        params_global["vf"] = "u_strong"
        timepoints_train = (
            [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] if not filter_beginning_end else [0, 1.0]
        )
        timepoints_train_holdout = (
            [0, 0.1, 0.5, 0.9, 1.0] if not filter_beginning_end else [0, 1.0]
        )
        timepoints_valid = [
            0,
            0.05,
            0.075,
            0.08,
            0.09,
            0.1,
            0.11,
            0.12,
            0.125,
            0.15,
            0.2,
            0.25,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.75,
            0.8,
            0.9,
            0.95,
            1.0,
        ]
        timepoints_test = np.linspace(0, 1, endpoint=True, num=21)
        f_y0 = lambda x: (0, (x - 1) / 2)

        if return_data == "train-valid":
            # TRAIN
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": timepoints_train,
                        "condition": v,
                        **params_local,
                    }
                    for v in label_list
                },
                **params_global,
            }
            kwargs["data_specs"][3] = {
                "y0": f_y0(3),
                "timepoints": timepoints_train_holdout,
                "condition": 3.0,
                **params_local,
            }
            if add_time_cond is not None:
                current_timepoints = kwargs["data_specs"][add_time_cond[0]][
                    "timepoints"
                ]
                new_timepoints = sorted(
                    current_timepoints
                    + (
                        add_time_cond[1]
                        if isinstance(add_time_cond[1], list)
                        else [add_time_cond[1]]
                    )
                )
                kwargs["data_specs"][add_time_cond[0]]["timepoints"] = new_timepoints
            train_loader, X_train, y_train, t_train = dgp_waves(**kwargs)

            # VALID
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": timepoints_valid,
                        "condition": v,
                        **params_local,
                    }
                    for _, v in enumerate(label_list)
                },
                **params_global,
            }
            _, X_valid, y_valid, t_valid = dgp_waves(**kwargs)

            return (
                train_loader,
                X_train,
                y_train,
                t_train,
                X_valid,
                y_valid,
                t_valid,
                n_classes,
                label_list,
            )

        elif return_data == "test":
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": timepoints_test,
                        "condition": v,
                        **params_local,
                    }
                    for _, v in enumerate(label_list)
                },
                **params_global,
            }
        timepoints = sorted(
            {
                item
                for sublist in [
                    v["timepoints"] for k, v in kwargs["data_specs"].items()
                ]
                for item in sublist
            }
        )

        _, X_test, y_test, t_test = dgp_waves(**kwargs)

        return X_test, y_test, t_test, n_classes, timepoints, label_list

    elif dgp == "g":
        label_list = np.linspace(1, 5.5, 10)
        n_classes = len(label_list)
        params_global["vf"] = "u"
        timepoints_train = (
            [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] if not filter_beginning_end else [0, 1.0]
        )
        timepoints_train_holdout = (
            [0, 0.1, 0.5, 0.9, 1.0] if not filter_beginning_end else [0, 1.0]
        )
        timepoints_valid = [
            0,
            0.05,
            0.075,
            0.08,
            0.09,
            0.1,
            0.11,
            0.12,
            0.125,
            0.15,
            0.2,
            0.25,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.75,
            0.8,
            0.9,
            0.95,
            1.0,
        ]
        timepoints_test = np.linspace(0, 1, endpoint=True, num=21)
        f_y0 = lambda x: (0, 0)

        if return_data == "train-valid":
            # TRAIN
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": timepoints_train,
                        "condition": v,
                        **params_local,
                    }
                    for v in label_list
                },
                **params_global,
            }
            kwargs["data_specs"][3] = {
                "y0": f_y0(3),
                "timepoints": timepoints_train_holdout,
                "condition": 3.0,
                **params_local,
            }
            if add_time_cond is not None:
                current_timepoints = kwargs["data_specs"][add_time_cond[0]][
                    "timepoints"
                ]
                new_timepoints = sorted(
                    current_timepoints
                    + (
                        add_time_cond[1]
                        if isinstance(add_time_cond[1], list)
                        else [add_time_cond[1]]
                    )
                )
                kwargs["data_specs"][add_time_cond[0]]["timepoints"] = new_timepoints
            train_loader, X_train, y_train, t_train = dgp_waves(**kwargs)

            # VALID
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": timepoints_valid,
                        "condition": v,
                        **params_local,
                    }
                    for _, v in enumerate(label_list)
                },
                **params_global,
            }
            _, X_valid, y_valid, t_valid = dgp_waves(**kwargs)

            return (
                train_loader,
                X_train,
                y_train,
                t_train,
                X_valid,
                y_valid,
                t_valid,
                n_classes,
                label_list,
            )

        elif return_data == "test":
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": timepoints_test,
                        "condition": v,
                        **params_local,
                    }
                    for _, v in enumerate(label_list)
                },
                **params_global,
            }
        timepoints = sorted(
            {
                item
                for sublist in [
                    v["timepoints"] for k, v in kwargs["data_specs"].items()
                ]
                for item in sublist
            }
        )

        _, X_test, y_test, t_test = dgp_waves(**kwargs)

        return X_test, y_test, t_test, n_classes, timepoints, label_list

    # dgp_waves_d_0_0.01_0.01_3_False_0.05_50_64_32_32_None_xavier_normal_LeakyReLU_cosine_cubic_True_False_free_False_False_300_cot_False_True_None_0.5_0.0_0.025_2_adam_False_0.0_False_False_emd_mmfm

    elif dgp == "h":
        label_list = sorted(list(np.linspace(1, 5.5, 10)) + [4.25])
        n_classes = len(label_list)
        params_global["vf"] = "u"
        timepoints_train = (
            [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] if not filter_beginning_end else [0, 1.0]
        )
        timepoints_train_holdout1 = (
            [0, 0.1, 0.5, 0.9, 1.0] if not filter_beginning_end else [0, 1.0]
        )
        timepoints_train_holdout2 = [0, 1.0] if not filter_beginning_end else [0, 1.0]
        timepoints_valid = [
            0,
            0.05,
            0.075,
            0.08,
            0.09,
            0.1,
            0.11,
            0.12,
            0.125,
            0.15,
            0.2,
            0.25,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.75,
            0.8,
            0.9,
            0.95,
            1.0,
        ]
        timepoints_test = np.linspace(0, 1, endpoint=True, num=21)
        f_y0 = lambda x: (0, (x - 1) / 2)

        if return_data == "train-valid":
            # TRAIN
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": timepoints_train,
                        "condition": v,
                        **params_local,
                    }
                    for v in label_list
                },
                **params_global,
            }
            kwargs["data_specs"][3] = {
                "y0": f_y0(3),
                "timepoints": timepoints_train_holdout1,
                "condition": 3.0,
                **params_local,
            }
            kwargs["data_specs"][4.25] = {
                "y0": f_y0(4.25),
                "timepoints": timepoints_train_holdout2,
                "condition": 4.25,
                **params_local,
            }
            if add_time_cond is not None:
                for atc in add_time_cond:
                    current_timepoints = kwargs["data_specs"][atc[0]]["timepoints"]
                    new_timepoints = sorted(
                        current_timepoints
                        + (atc[1] if isinstance(atc[1], list) else [atc[1]])
                    )
                    kwargs["data_specs"][atc[0]]["timepoints"] = new_timepoints
                # current_timepoints = kwargs["data_specs"][add_time_cond[0]]["timepoints"]
                # new_timepoints = sorted(
                #     current_timepoints
                #     + (add_time_cond[1] if isinstance(add_time_cond[1], list) else [add_time_cond[1]])
                # )
                # kwargs["data_specs"][add_time_cond[0]]["timepoints"] = new_timepoints
            train_loader, X_train, y_train, t_train = dgp_waves(**kwargs)

            # VALID
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": timepoints_valid,
                        "condition": v,
                        **params_local,
                    }
                    for _, v in enumerate(label_list)
                },
                **params_global,
            }
            _, X_valid, y_valid, t_valid = dgp_waves(**kwargs)

            return (
                train_loader,
                X_train,
                y_train,
                t_train,
                X_valid,
                y_valid,
                t_valid,
                n_classes,
                label_list,
            )

        elif return_data == "test":
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": timepoints_test,
                        "condition": v,
                        **params_local,
                    }
                    for _, v in enumerate(label_list)
                },
                **params_global,
            }
        timepoints = sorted(
            {
                item
                for sublist in [
                    v["timepoints"] for k, v in kwargs["data_specs"].items()
                ]
                for item in sublist
            }
        )

        _, X_test, y_test, t_test = dgp_waves(**kwargs)

        return X_test, y_test, t_test, n_classes, timepoints, label_list

    elif dgp == "i":
        label_list = sorted(list(np.linspace(1, 6.5, 12)))
        n_classes = len(label_list)
        params_global["vf"] = "u"
        timepoints_train = (
            [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] if not filter_beginning_end else [0, 1.0]
        )
        timepoints_train_holdout1 = (
            [0, 0.1, 0.5, 0.9, 1.0] if not filter_beginning_end else [0, 1.0]
        )
        timepoints_train_holdout2 = (
            [0, 0.3, 0.5, 0.7, 1.0] if not filter_beginning_end else [0, 1.0]
        )
        timepoints_valid = [
            0,
            0.05,
            0.075,
            0.1,
            0.125,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.6,
            0.7,
            0.75,
            0.8,
            0.9,
            0.95,
            1.0,
        ]
        timepoints_test = np.linspace(0, 1, endpoint=True, num=21)
        f_y0 = lambda x: (0, (x - 1) / 2)

        if return_data == "train-valid":
            # TRAIN
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": timepoints_train,
                        "condition": v,
                        **params_local,
                    }
                    for v in label_list
                },
                **params_global,
            }
            kwargs["data_specs"][3] = {
                "y0": f_y0(3),
                "timepoints": timepoints_train_holdout1,
                "condition": 3.0,
                **params_local,
            }
            kwargs["data_specs"][5.0] = {
                "y0": f_y0(5.0),
                "timepoints": timepoints_train_holdout2,
                "condition": 5.0,
                **params_local,
            }
            if add_time_cond is not None:
                for atc in add_time_cond:
                    current_timepoints = kwargs["data_specs"][atc[0]]["timepoints"]
                    new_timepoints = sorted(
                        current_timepoints
                        + (atc[1] if isinstance(atc[1], list) else [atc[1]])
                    )
                    kwargs["data_specs"][atc[0]]["timepoints"] = new_timepoints
                # current_timepoints = kwargs["data_specs"][add_time_cond[0]]["timepoints"]
                # new_timepoints = sorted(
                #     current_timepoints
                #     + (add_time_cond[1] if isinstance(add_time_cond[1], list) else [add_time_cond[1]])
                # )
                # kwargs["data_specs"][add_time_cond[0]]["timepoints"] = new_timepoints
            train_loader, X_train, y_train, t_train = dgp_waves(**kwargs)

            # VALID
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": timepoints_valid,
                        "condition": v,
                        **params_local,
                    }
                    for _, v in enumerate(label_list)
                },
                **params_global,
            }
            _, X_valid, y_valid, t_valid = dgp_waves(**kwargs)

            return (
                train_loader,
                X_train,
                y_train,
                t_train,
                X_valid,
                y_valid,
                t_valid,
                n_classes,
                label_list,
            )

        elif return_data == "test":
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": timepoints_test,
                        "condition": v,
                        **params_local,
                    }
                    for _, v in enumerate(label_list)
                },
                **params_global,
            }
        timepoints = sorted(
            {
                item
                for sublist in [
                    v["timepoints"] for k, v in kwargs["data_specs"].items()
                ]
                for item in sublist
            }
        )

        _, X_test, y_test, t_test = dgp_waves(**kwargs)

        return X_test, y_test, t_test, n_classes, timepoints, label_list

    elif dgp == "z":
        label_list = [np.round(i, 2) for i in np.linspace(1.5, 5, 11)]
        n_classes = len(label_list)
        params_global["vf"] = "u_linear"
        arr = np.random.rand(11, 5)
        arr[:, 0] = 0.0
        arr[:, -1] = 1.0
        arr = np.round(arr, 3)
        timepoints_train = arr

        # Insert random np.nans in training data with x% chance
        timepoints_train[4, 1] = np.nan
        timepoints_train[6, 2] = np.nan
        timepoints_train[8, 3] = np.nan

        t_valid = np.linspace(0, 1, endpoint=True, num=21)
        # Repeat 11 times along axis = 0
        t_valid = np.repeat(t_valid[np.newaxis, :], 11, axis=0)

        arr_valid = np.concatenate([arr, t_valid], axis=1)
        arr_valid = np.round(arr_valid, 3)
        timepoints_valid = arr_valid

        timepoints_test = np.linspace(0, 1, endpoint=True, num=21)
        f_y0 = lambda x: (0, 0)

        if return_data == "train-valid":
            # TRAIN
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": [
                            x for x in timepoints_train[idx] if not np.isnan(x)
                        ],
                        "condition": v,
                        **params_local,
                    }
                    for idx, v in enumerate(label_list)
                },
                **params_global,
            }
            if add_time_cond is not None:
                for atc in add_time_cond:
                    current_timepoints = kwargs["data_specs"][atc[0]]["timepoints"]
                    new_timepoints = sorted(
                        current_timepoints
                        + (atc[1] if isinstance(atc[1], list) else [atc[1]])
                    )
                    kwargs["data_specs"][atc[0]]["timepoints"] = new_timepoints
            train_loader, X_train, y_train, t_train = dgp_waves_independent(**kwargs)

            # VALID
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": [
                            x for x in timepoints_valid[idx] if not np.isnan(x)
                        ],
                        "condition": v,
                        **params_local,
                    }
                    for idx, v in enumerate(label_list)
                },
                **params_global,
            }
            _, X_valid, y_valid, t_valid = dgp_waves_independent(**kwargs)

            return (
                train_loader,
                X_train,
                y_train,
                t_train,
                X_valid,
                y_valid,
                t_valid,
                n_classes,
                label_list,
            )

        elif return_data == "test":
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": list(timepoints_test),
                        "condition": v,
                        **params_local,
                    }
                    for idx, v in enumerate(label_list)
                },
                **params_global,
            }
        timepoints = sorted(
            {
                item
                for sublist in [
                    v["timepoints"] for k, v in kwargs["data_specs"].items()
                ]
                for item in sublist
            }
        )

        _, X_test, y_test, t_test = dgp_waves_independent(**kwargs)

        return X_test, y_test, t_test, n_classes, timepoints, label_list

    elif dgp == "y":
        label_list = [np.round(i, 2) for i in np.linspace(1.5, 5, 21)]
        n_classes = len(label_list)
        params_global["vf"] = "u_linear"
        arr = np.random.rand(21, 5)
        arr[:, 0] = 0.0
        arr[:, -1] = 1.0
        arr = np.round(arr, 3)
        timepoints_train = arr

        # Insert random np.nans in training data with x% chance
        timepoints_train[4, 1] = np.nan
        timepoints_train[6, 2] = np.nan
        timepoints_train[8, 3] = np.nan

        t_valid = np.linspace(0, 1, endpoint=True, num=21)
        # Repeat 21 times along axis = 0
        t_valid = np.repeat(t_valid[np.newaxis, :], 21, axis=0)

        arr_valid = np.concatenate([arr, t_valid], axis=1)
        arr_valid = np.round(arr_valid, 3)
        timepoints_valid = arr_valid

        timepoints_test = np.linspace(0, 1, endpoint=True, num=21)
        f_y0 = lambda x: (0, 0)

        if return_data == "train-valid":
            # TRAIN
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": [
                            x for x in timepoints_train[idx] if not np.isnan(x)
                        ],
                        "condition": v,
                        **params_local,
                    }
                    for idx, v in enumerate(label_list)
                },
                **params_global,
            }
            if add_time_cond is not None:
                for atc in add_time_cond:
                    current_timepoints = kwargs["data_specs"][atc[0]]["timepoints"]
                    new_timepoints = sorted(
                        current_timepoints
                        + (atc[1] if isinstance(atc[1], list) else [atc[1]])
                    )
                    kwargs["data_specs"][atc[0]]["timepoints"] = new_timepoints
            train_loader, X_train, y_train, t_train = dgp_waves_independent(**kwargs)

            # VALID
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": [
                            x for x in timepoints_valid[idx] if not np.isnan(x)
                        ],
                        "condition": v,
                        **params_local,
                    }
                    for idx, v in enumerate(label_list)
                },
                **params_global,
            }
            _, X_valid, y_valid, t_valid = dgp_waves_independent(**kwargs)

            return (
                train_loader,
                X_train,
                y_train,
                t_train,
                X_valid,
                y_valid,
                t_valid,
                n_classes,
                label_list,
            )

        elif return_data == "test":
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": list(timepoints_test),
                        "condition": v,
                        **params_local,
                    }
                    for idx, v in enumerate(label_list)
                },
                **params_global,
            }
        timepoints = sorted(
            {
                item
                for sublist in [
                    v["timepoints"] for k, v in kwargs["data_specs"].items()
                ]
                for item in sublist
            }
        )

        _, X_test, y_test, t_test = dgp_waves_independent(**kwargs)

        return X_test, y_test, t_test, n_classes, timepoints, label_list

    elif dgp == "x":
        label_list = [np.round(i, 2) for i in np.linspace(3, 5, 21)]
        n_classes = len(label_list)
        params_global["vf"] = "u_linear"
        arr = np.random.rand(21, 6)
        arr[:, 0] = 0.0
        arr[:, -1] = 1.0
        arr = np.round(arr, 3)
        timepoints_train = arr

        t_valid = np.linspace(0, 1, endpoint=True, num=21)
        t_valid = np.repeat(t_valid[np.newaxis, :], 21, axis=0)

        arr_valid = np.concatenate([arr, t_valid], axis=1)
        arr_valid = np.round(arr_valid, 3)
        timepoints_valid = arr_valid

        timepoints_test = np.linspace(0, 1, endpoint=True, num=21)
        f_y0 = lambda x: (0, 0)

        if return_data == "train-valid":
            # TRAIN
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": [
                            x for x in timepoints_train[idx] if not np.isnan(x)
                        ],
                        "condition": v,
                        **params_local,
                    }
                    for idx, v in enumerate(label_list)
                },
                **params_global,
            }
            if add_time_cond is not None:
                for atc in add_time_cond:
                    current_timepoints = kwargs["data_specs"][atc[0]]["timepoints"]
                    new_timepoints = sorted(
                        current_timepoints
                        + (atc[1] if isinstance(atc[1], list) else [atc[1]])
                    )
                    kwargs["data_specs"][atc[0]]["timepoints"] = new_timepoints
            train_loader, X_train, y_train, t_train = dgp_waves_independent(**kwargs)

            # VALID
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": [
                            x for x in timepoints_valid[idx] if not np.isnan(x)
                        ],
                        "condition": v,
                        **params_local,
                    }
                    for idx, v in enumerate(label_list)
                },
                **params_global,
            }
            _, X_valid, y_valid, t_valid = dgp_waves_independent(**kwargs)

            return (
                train_loader,
                X_train,
                y_train,
                t_train,
                X_valid,
                y_valid,
                t_valid,
                n_classes,
                label_list,
            )

        elif return_data == "test":
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": list(timepoints_test),
                        "condition": v,
                        **params_local,
                    }
                    for idx, v in enumerate(label_list)
                },
                **params_global,
            }
        timepoints = sorted(
            {
                item
                for sublist in [
                    v["timepoints"] for k, v in kwargs["data_specs"].items()
                ]
                for item in sublist
            }
        )

        _, X_test, y_test, t_test = dgp_waves_independent(**kwargs)

        return X_test, y_test, t_test, n_classes, timepoints, label_list

    elif dgp == "w":
        label_list = [np.round(i, 2) for i in np.linspace(3, 5, 21)]
        n_classes = len(label_list)
        params_global["vf"] = "u_linear"
        arr = np.random.rand(21, 6)
        arr[:, 0] = 0.0
        arr[:, -1] = 1.0
        arr = np.round(arr, 3)
        timepoints_train = arr

        t_valid = np.linspace(0, 1, endpoint=True, num=21)
        t_valid = np.repeat(t_valid[np.newaxis, :], 21, axis=0)

        arr_valid = np.concatenate([arr, t_valid], axis=1)
        arr_valid = np.round(arr_valid, 3)
        timepoints_valid = arr_valid

        timepoints_test = np.linspace(0, 1, endpoint=True, num=21)
        f_y0 = lambda x: (0, (x - 3) / 2)

        if return_data == "train-valid":
            # TRAIN
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": [
                            x for x in timepoints_train[idx] if not np.isnan(x)
                        ],
                        "condition": v,
                        **params_local,
                    }
                    for idx, v in enumerate(label_list)
                },
                **params_global,
            }
            if add_time_cond is not None:
                for atc in add_time_cond:
                    current_timepoints = kwargs["data_specs"][atc[0]]["timepoints"]
                    new_timepoints = sorted(
                        current_timepoints
                        + (atc[1] if isinstance(atc[1], list) else [atc[1]])
                    )
                    kwargs["data_specs"][atc[0]]["timepoints"] = new_timepoints
            train_loader, X_train, y_train, t_train = dgp_waves_independent(**kwargs)

            # VALID
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": [
                            x for x in timepoints_valid[idx] if not np.isnan(x)
                        ],
                        "condition": v,
                        **params_local,
                    }
                    for idx, v in enumerate(label_list)
                },
                **params_global,
            }
            _, X_valid, y_valid, t_valid = dgp_waves_independent(**kwargs)

            return (
                train_loader,
                X_train,
                y_train,
                t_train,
                X_valid,
                y_valid,
                t_valid,
                n_classes,
                label_list,
            )

        elif return_data == "test":
            kwargs = {
                "data_specs": {
                    v: {
                        "y0": f_y0(v),
                        "timepoints": list(timepoints_test),
                        "condition": v,
                        **params_local,
                    }
                    for idx, v in enumerate(label_list)
                },
                **params_global,
            }
        timepoints = sorted(
            {
                item
                for sublist in [
                    v["timepoints"] for k, v in kwargs["data_specs"].items()
                ]
                for item in sublist
            }
        )

        _, X_test, y_test, t_test = dgp_waves_independent(**kwargs)

        return X_test, y_test, t_test, n_classes, timepoints, label_list


#
# Plotting Utility
#
def _plot(ax, X, y, t, odeintegrate=True):
    sns.scatterplot(data=df, x=0, y=1, hue="target", ax=ax, s=10, palette="tab10")
    for idx, _ in enumerate(np.unique(t[~np.isnan(t)])):
        if idx == len(np.unique(t[~np.isnan(t)])) - 1:
            break
        # Collect one index per class
        sample_plot = []
        for c in np.unique(y[~np.isnan(y)]):
            sample_plot.append(np.where(y[:, 0] == c)[0][0].item())
        for n in sample_plot:
            sample_start = X[n, idx]
            offset = 1
            while bool(np.isnan(y[n, idx + offset])):
                if idx + offset < 7:
                    offset += 1
            sample_end = X[n, idx + offset]
            ax.plot(
                [sample_start[0], sample_end[0]],
                [sample_start[1], sample_end[1]],
                color="black",
                linestyle="--",
                # alpha=0.5,
            )
    # Plot ode in background
    if odeintegrate:
        for c in np.unique(y_train[~np.isnan(y_train)]):
            t = np.linspace(0, 10, 101)
            y0 = [0.0, 2.0 * (c - 1)]
            sol = odeint(u, y0, t, args=(c,))
            ax.plot(sol[:, 0], sol[:, 1], color="blue", alpha=0.5)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pytorch_lightning as pl
    import seaborn as sns

    PLOT_WAVES = False  # Waves Experiment
    PLOT_ICCITE = False  # ICCITE Experiment
    PLOT_BEIJING = True  # Beijing Experiment

    if PLOT_WAVES:
        ns_per_t_and_c = 50
        data_std = 0.01
        kwargs = {
            "data_specs": {
                v: {
                    "y0": (0, 0),
                    "timepoints": [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                    "std": data_std,
                    "n_samples": ns_per_t_and_c,
                    "condition": v,
                }
                for v in sorted(list(np.linspace(1, 6.5, 12)))
            },
            "coupling": "ot",
            "batch_size": None,
            "dimension": 2,
            "off_diagonal": 0.0,
            "vf": "u_linear",
            "classes_first": True,
        }

        train_loader, X_train, y_train, t_train = dgp_waves(**kwargs)

        fig, ax = plt.subplots(1, 1, figsize=(8, 5), sharex=True, sharey=True)
        df = pd.DataFrame(X_train.reshape(-1, 2)).assign(
            target=y_train.reshape(-1, 1), time=t_train.reshape(-1, 1)
        )
        # Plot ode in background for all starting points
        for v in kwargs["data_specs"].values():
            y0, c = v["y0"], v["condition"]
            t = np.linspace(0, 1, 501)
            sol = odeint(u_linear, y0, t, args=(c,))
            ax.plot(sol[:, 0], sol[:, 1], color="blue", alpha=0.25)
        sns.scatterplot(data=df, x=0, y=1, hue="target", ax=ax, s=10, palette="tab10")
        ax.set_title("Train")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.suptitle("DGP2 Data")
        plt.legend(loc="upper right", title="Condition")
        # ax.set_aspect("equal")
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=True, sharey=True)
        df = pd.DataFrame(X_train.reshape(-1, 2)).assign(
            target=y_train.reshape(-1, 1), time=t_train.reshape(-1, 1)
        )

        _plot(ax, X_train, y_train, t_train, odeintegrate=False)

        ax.set_title("Train")
        plt.suptitle("DGP2 Data")
        plt.show()

    if PLOT_BEIJING:
        ns_per_t_and_c = 50

        label_list = [int(x) for x in np.linspace(1, 12, 12)]
        n_classes = len(label_list)

        target = "PM2.5"
        start = 2015
        end = None
        n_quarters = None
        n_months = None
        classes_first = True

        params_global = {
            "coupling": "ot",
            "batch_size": None,
            "target": target,
            "start": start,
            "end": end,
            "n_quarters": n_quarters,
            "ns_per_t_and_c": ns_per_t_and_c
        }

        timepoints_train = [0, 3, 7, 11, 13, 17, 20, 23]
        timepoints_train_holdout1 = [0, 3, 7, 13, 17, 20]
        timepoints_train_holdout2 = [0, 3, 11, 13, 17, 23]
        timepoints_valid = list(range(26))
        timepoints_test = list(range(26))

        kwargs = {
            "data_specs": {
                v: {
                    "timepoints": timepoints_train,
                    "condition": v,
                }
                for v in label_list
            },
            **params_global,
        }
        kwargs["data_specs"][7] = {
            "timepoints": timepoints_train_holdout1,
            "condition": 7.0,
        }
        kwargs["data_specs"][10.0] = {
            "timepoints": timepoints_train_holdout2,
            "condition": 10.0,
        }

        train_loader, X_train, y_train, t_train = dgp_beijing(
            **kwargs,
        )

        fig, ax = plt.subplots(1, 1, figsize=(12, 5), sharex=True, sharey=True)
        df = pd.DataFrame({"var": X_train.reshape(-1, 1).squeeze()}).assign(
            target=y_train.reshape(-1, 1), time=t_train.reshape(-1, 1)
        )
        sns.lineplot(data=df, x="time", y="var", hue="target", ax=ax, palette="tab10")
        sns.scatterplot(
            data=df,  x="time", y="var", hue="target", ax=ax, s=50, palette="tab10", legend=False
        )
        ax.set_title("Train")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.grid()
        plt.suptitle("DGP2 Data")
        plt.legend(title="Condition")
        plt.show()

    if PLOT_ICCITE:
        for seed in range(3):
            pl.seed_everything(seed)
            for use_pca in [10, 25]:  #
                for n_samples_per_c_in_b in [100, 250]:  # , 250, 500
                    for coupling in [None, "ot", "cot"]:  #
                        for preset in ["y"]:  # "b", "c", "d",
                            print(
                                f"Seed: {seed}, PCA: {use_pca}, Samples: {n_samples_per_c_in_b} Coupling: {coupling} Preset: {preset}"
                            )
                            if preset in ["y", "z"]:
                                f_call = dgp_iccite_4t
                            else:
                                f_call = dgp_iccite
                            (
                                train_loader,
                                X_train,
                                y_train,
                                t_train,
                                X_valid,
                                y_valid,
                                t_valid,
                                X_test,
                                y_test,
                                t_test,
                                ps,
                            ) = f_call(
                                hvg=None,
                                subsample_frac=None,
                                use_pca=use_pca,
                                coupling=coupling,
                                batch_size=None,
                                n_samples_per_c_in_b=n_samples_per_c_in_b,
                                train_test_split=0.8,
                                top_n_effects=None,
                                leave_out_mid=None,
                                leave_out_end=None,
                                preset=preset,
                                seed=seed,
                                filter_beginning_end=True,
                            )
