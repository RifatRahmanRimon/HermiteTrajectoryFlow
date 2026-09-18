from pathlib import Path

import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from addict import Dict
from tqdm import tqdm

from mmfm.data import dgp_schiebinger, dgp_waves_data
from mmfm.evaluation import compute_metric_set
from mmfm.trajectory import sample_trajectory
from mmfm.plotting import plot_results_mmfm
from mmfm.models import VectorFieldModel


def load_all_fm_models(
    path, dgp=None, production=False, model_cap=100, embedding_type=None, coupling=None, filter_values=None
):
    """Load all model results from the disk."""
    df = pd.DataFrame()
    if dgp is None:
        csv_files = list(Path(path).rglob("*.csv"))
    else:
        if "dgp_schiebinger" in path:
            csv_files = [x for x in Path(path).rglob("*.csv") if f"dgp_schiebinger_{dgp}_" in str(x)]
        elif "dgp_waves" in path:
            csv_files = [x for x in Path(path).rglob("*.csv") if f"dgp_waves_{dgp}_" in str(x)]
            if embedding_type is not None:
                csv_files = [x for x in csv_files if embedding_type in str(x)]
            if coupling is not None:
                csv_files = [x for x in csv_files if coupling in str(x)]
            if filter_values is not None:
                csv_files = [x for x in csv_files if filter_values in str(x)]
            # csv_files = [x for x in csv_files if str(x.parent).endswith("fm")]
        elif "dgp_weather" in path:
            csv_files = [x for x in Path(path).rglob("*.csv") if f"dgp_weather_{dgp}_" in str(x)]
            # if embedding_type is not None:
            #     csv_files = [x for x in csv_files if embedding_type in str(x)]
            # if coupling is not None:
            #     csv_files = [x for x in csv_files if coupling in str(x)]
            # if filter_values is not None:
            #     csv_files = [x for x in csv_files if filter_values in str(x)]
        elif "dgp_vdp" in path:
            csv_files = [x for x in Path(path).rglob("*.csv") if f"dgp_vdp_{dgp}_" in str(x)]
        elif "dgp_iccite" in path:
            csv_files = [x for x in Path(path).rglob("*.csv") if f"dgp_iccite_{dgp}_" in str(x)]
    df_list = []

    for idx, csv_file in tqdm(enumerate(csv_files), total=len(csv_files)):
        if not production and (idx > model_cap):
            break

        # try:
        df_data = pd.read_csv(csv_file)
        df_data["filename"] = str(csv_file)
        for col in [
            "subsample_frac",
            "hvg",
            "normalization",
            "coupling",
            "init_weights",
            "batch_size",
            "lrs",
            "top_n_effects",
            "leave_out_mid",
            "leave_out_end",
            "matching",
        ]:
            if col in df_data.columns:
                df_data[col].replace({np.nan: "None"}, inplace=True)
        df_list.append(df_data)
        # except pd.errors.EmptyDataError:
        #     print("ERROR")
        #     continue

        df_data["filename"] = str(csv_file)

    df = pd.concat(df_list, ignore_index=True)

    df = df.drop(
        columns=[
            "max_grad_norm",
            "sum_time_embed",
            "sum_cond_embed",
            "init_weights",
            "activation",
            "lrs",
            "n_epochs",
            "affine_transform",
            "max_norm_embedding",
            "n_epochs",
            "batch_size",
            "optimizer",
            "dimension",
            "num_out_layers",
            "spectral_norm",
            "dropout",
            "keep_constants",
            "optimizer",
        ]
    )

    # Add time column if not present
    # Divide marginal by max marginal to get time
    # if "time" not in df.columns:
    #     df["time"] = df["marginal"] / df["marginal"].max()

    performance_columns = ["mean_diff_l1", "mean_diff_l2", "kl_div"] + [
        x for x in df.columns if "mmd" in x or "wasserstein" in x
    ]
    grouping_columns = [x for x in df.columns if x not in performance_columns and x != "filename"]

    return df, grouping_columns, performance_columns


def process_all_fm_models(
    df,
    grouping_columns=None,
    performance_columns=None,
    plot=False,
    verbose=False,
    minimum_seeds=3,
    data_cols=None,
):
    """Preprocess the loaded MMFM models."""
    # Drop all rows with nan in train
    df = df.loc[~df["train"].isna()]
    df.loc[:, "train"] = df.loc[:, "train"].astype(bool)

    if data_cols is None:
        data_cols = ["ns_per_t_and_c", "coupling", "data_std", "dgp", "interpolation", "guidance"]

    # Count number of seeds per model and plot histogram
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        x = df.groupby([x for x in grouping_columns if x != "seed"]).size().reset_index(name="counts")["counts"]
        plt.hist(x, bins=20)
        plt.xticks(range(1, 6))
        plt.xlabel("Number of seeds")
        plt.ylabel("Count")
        plt.title("Number of seeds per model")
        plt.show()

    df["seed_counts"] = df.groupby([x for x in grouping_columns if x not in ["seed"]])["seed"].transform("count")
    # Delete all rows with less than minimum_seeds seeds
    if verbose:
        print(f"Number of models before adjusting for seeds: {len(df)}")
    df = df.loc[df["seed_counts"] >= minimum_seeds]
    df = df.drop(columns=["seed_counts"])
    if verbose:
        print(f"Number of models after adjusting for seeds:  {len(df)}\n")

    # Drop all columns with only one unique value
    for column in df.columns:
        if column in data_cols:
            if verbose:
                print(f"- Skipping column {column}")
            continue
        if len(df[column].unique()) == 1:
            val = df[column].unique()[0]
            df = df.drop(columns=[column])
            if verbose:
                print(f"Dropping column {column} with only one unique value: {val}")
            # Remove from grouping_columns
            if column in grouping_columns:
                grouping_columns.remove(column)
            if column in performance_columns:
                performance_columns.remove(column)

    return df, grouping_columns, performance_columns


def get_model_battery_of_best_models(
    df,
    dgp,
    grouping_columns,
    select_by="mmd_mean",
    device="cuda",
    n_top_models=1,
    n_top_seeds=5,
    model_string="dgp_waves",
    label_list=None,
    verbose=True,
    average_out_seed=True,
):
    """Select the best MMFM model and return it with all its seeds."""

    # If eval_points is given, filter df by this
    # Note, df only contains the valid data. We pick the best model averaged over all seeds
    def weighted_avg(group_df, whole_df, values, weights):
        v = whole_df.loc[group_df.index, values]
        w = whole_df.loc[group_df.index, weights]
        return (v * w).sum() / w.sum()

    def weighted_max(group_df, whole_df, values, weights):
        v = whole_df.loc[group_df.index, values]
        w = whole_df.loc[group_df.index, weights]
        return (v * w).max()

    if average_out_seed:
        grouping = [x for x in grouping_columns if x not in ["marginal", "c", "time", "seed"]]
    else:
        grouping = [x for x in grouping_columns if x not in ["marginal", "c", "time"]]

    df_agg = (
        df.drop(columns=["marginal", "c", "time"])
        .groupby(grouping)
        .agg(
            mmd_mean=("mmd", lambda x: weighted_avg(x, df, "mmd", "weight")),
            mmd_max=("mmd", lambda x: weighted_max(x, df, "mmd", "weight")),
            mmd_std=("mmd", "std"),
            mmd_median_mean=("mmd_median", lambda x: weighted_avg(x, df, "mmd_median", "weight")),
            mmd_median_max=("mmd_median", lambda x: weighted_max(x, df, "mmd_median", "weight")),
            mmd_median_std=("mmd_median", "std"),
            wasserstein_mean=("wasserstein", lambda x: weighted_avg(x, df, "wasserstein", "weight")),
            wasserstein_max=("wasserstein", lambda x: weighted_max(x, df, "wasserstein", "weight")),
            wasserstein_std=("wasserstein", "std"),
            mean_diff_l1_mean=("mean_diff_l1", lambda x: weighted_avg(x, df, "mean_diff_l1", "weight")),
            mean_diff_l1_max=("mean_diff_l1", lambda x: weighted_max(x, df, "mean_diff_l1", "weight")),
            mean_diff_l1_std=("mean_diff_l1", "std"),
            mean_diff_l2_mean=("mean_diff_l2", lambda x: weighted_avg(x, df, "mean_diff_l2", "weight")),
            mean_diff_l2_max=("mean_diff_l2", lambda x: weighted_max(x, df, "mean_diff_l2", "weight")),
            mean_diff_l2_std=("mean_diff_l2", "std"),
            # kl_div_mean=("kl_div", lambda x: weighted_avg(x, df, "kl_div", "weight")),
            # kl_div_max=("kl_div", lambda x: weighted_max(x, df, "kl_div", "weight")),
            # kl_div_std=("kl_div", "std"),
            filename_first=("filename", "first"),
        )
    ).reset_index()

    # Find best model according to MMD/Wasserstein mean and std on valid data
    # and compute scores on test data for the best model
    model_battery = Dict()
    model_guidances = Dict()
    model_states = Dict()
    for n in range(1, n_top_models + 1):
        # for c in df["c"].unique():
        # print(f"Optimizing for condition c: {c}")
        df_top_valid = df_agg.sort_values(by=select_by, ascending=True).head(1).reset_index(drop=True)
        if verbose:
            print(f"Best model: {df_top_valid['filename_first'].values[n-1]}")
        model_guidances[n - 1] = df_top_valid["guidance"].values[n - 1]

        # Load the model from its filename and all its seed-variations
        model_path = df_top_valid["filename_first"].values[n - 1]
        model_path = model_path.replace("df_results.csv", "model.pt")

        for seed in range(n_top_seeds):
            # Replace "dgp2_{seed}" with "dgp2_x" in the path
            current_seed = model_path.split("_")[5]
            model_path = model_path.replace(f"{model_string}_{dgp}_{current_seed}", f"{model_string}_{dgp}_{seed}")
            filename = model_path.split("/")[-2]
            try:
                state = torch.load(model_path, weights_only=True)
                if verbose:
                    print(f"✓ {filename}")

            except FileNotFoundError:
                if verbose:
                    print(f"✗ {filename}")
                continue

            mmfm_model = VectorFieldModel(
                data_dim=state["dimension"] if "dimension" in state else state["use_pca"],
                x_latent_dim=state["x_latent_dim"],
                time_embed_dim=state["time_embed_dim"],
                cond_embed_dim=state["cond_embed_dim"],
                conditional_model=state["conditional_model"],
                embedding_type=state["embedding_type"],
                n_classes=state["n_classes"],
                label_list=label_list,
                normalization=state["normalization"],
                activation=state["activation"],
                affine_transform=state["affine_transform"],
                sum_time_embed=state["sum_time_embed"],
                sum_cond_embed=state["sum_cond_embed"],
                max_norm_embedding=state["max_norm_embedding"],
                num_out_layers=state["num_out_layers"],
                spectral_norm=state["spectral_norm"],
                dropout=state["dropout"],
                conditional_bias=state["conditional_bias"],
                keep_constants=state["keep_constants"],
            ).to(device)
            mmfm_model.load_state_dict(state["state_dict"], strict=True)
            model_battery[n - 1][seed] = mmfm_model
            model_states[n - 1][seed] = state

            # Print average values of absolute model weights in state["state_dict"]
            # print(
            #     f"Average absolute model weights: {np.mean([torch.mean(torch.abs(p)).item() for p in mmfm_model.parameters()])}"
            # )

    return df_top_valid, model_battery, model_states, model_guidances


def create_test_data(dgp, coupling, data_std, off_diagonal, batch_size, dimension, n_samples):
    """Create test data for the given DGP."""
    return dgp_waves_data(coupling, batch_size, dimension, off_diagonal, data_std, dgp, return_data="test")


def create_test_data_schiebinger(
    dgp,
    coupling,
    ipsc_timepoint,
    hvg,
    n_samples_per_c_in_b,
    batch_size,
    use_pca,
    skip_timepoints,
    train_test_split,
    seed=0,
    verbose=False,
):
    """Create test data for the given DGP."""
    if dgp == "a":
        _, _, _, _, _, _, _, X_test, y_test, t_test = dgp_schiebinger(
            ipsc_timepoint=ipsc_timepoint,
            hvg=hvg,
            subsample_frac=None,
            use_pca=use_pca,
            coupling=coupling,
            batch_size=batch_size,
            skip_timepoints=skip_timepoints,
            n_samples_per_c_in_b=n_samples_per_c_in_b,
            train_test_split=train_test_split,
            seed=seed,
            verbose=verbose,
        )

        timepoints = list(np.unique(t_test))
        all_classes = [1, 2]
        n_classes = len(all_classes)

    return X_test, y_test, t_test, n_classes, timepoints, all_classes


def predict_on_testset_mmfm(
    model_battery,
    model_states,
    model_guidances,
    X_test,
    y_test,
    t_test,
    device="cuda",
    steps=1001,
    method="dopri5",
):
    """Predict on the test set."""
    df_results = pd.DataFrame()
    # results = []
    traj_test = Dict()
    if len(model_battery) > 1:
        print("Evaluating multiple models is not supported yet.")

    for n, models in model_battery.items():
        for seed, mmfm_model in models.items():
            traj_test[n][seed] = sample_trajectory(
                mmfm_model,
                X=X_test[:, 0],
                y=y_test[:, 0],
                device=device,
                guidance=model_guidances[n],
                conditional_model=model_states[n][seed]["conditional_model"],
                rtol=1e-7,
                atol=1e-9,
                steps=steps,
                method=method,
            )

            from mmfm.evaluation import eval_metrics

            df_results = eval_metrics(traj_test[n][seed], X_test, y_test, t_test, model_guidances[n], train=False)
            df_results["seed"] = seed

    return df_results, traj_test


def plot_results(X_test, y_test, t_test, traj_test, ncols=None, n_classes=None, plot_ode=None, avg_models=False, s=5):
    idx_plot = []
    for c in np.unique(y_test[:, 0]):
        idx = np.where(y_test[:, 0] == c)[0][:2]
        idx_plot.append(idx)
    idx_plot = [x.item() for x in np.array(idx_plot).flatten()]

    for n, traj in traj_test.items():
        if avg_models:
            traj_avg = np.mean([traj[seed] for seed in traj.keys()], axis=0)
            plot_results_mmfm(
                X=X_test,
                y=y_test,
                t=t_test,
                trajectory=traj_avg,
                idx_plot=idx_plot,
                n_classes=n_classes if n_classes is not None else 9,
                title=f"MMFM {n} | Avg",
                save=False,
                filepath="./figure_waves_all.svg",
                s=s,
                ncols=ncols,
                plot_ode=plot_ode,
            )
        else:
            for seed, trajectory in traj.items():
                plot_results_mmfm(
                    X=X_test,
                    y=y_test,
                    t=t_test,
                    trajectory=trajectory,
                    idx_plot=idx_plot,
                    n_classes=n_classes if n_classes is not None else 9,
                    title=f"MMFM {n} | {seed}",
                    save=False,
                    filepath="./figure_waves_all.svg",
                    s=s,
                    ncols=ncols,
                    plot_ode=plot_ode,
                )


def predict_on_testset_fsi(
    results_path,
    X_test,
    y_test,
    t_test,
    seed,
    coupling,
    plot_results,
    ncols,
    verbose=False,
    n_classes=None,
    plot_ode=None,
):
    df_results_fsi = pd.DataFrame()
    results = []

    if coupling == "ot":
        name = "model_ot_fsi.pkl"
    elif coupling == "cot":
        name = "model_cot_fsi.pkl"
    elif coupling == "None":
        name = "model_fsi.pkl"
    else:
        raise ValueError("Coupling not recognized.")

    with open(results_path / name, "rb") as f:
        if verbose:
            print(f"Loading {name}")
        fsi_model = cloudpickle.load(f)

    for marginal, time in enumerate(np.unique(t_test)):
        for c in [x for x in np.unique(y_test) if np.isfinite(x)]:  # FIXME: Is this filtering correct?
            target = X_test[:, marginal][y_test[:, marginal] == c]
            transport = fsi_model.interpolate_from_x0(
                X=X_test[:, 0][y_test[:, 0] == c],
                y=c if coupling == "cot" else None,
                t_query=float(time),  # FIXME: This must always be float or int?
            )

            try:
                mmd, mmd_median, wasserstein, mean_diff_l1, mean_diff_l2, kl_div = compute_metric_set(target, transport)
            except ValueError:
                mmd, mmd_median, wasserstein, mean_diff_l1, mean_diff_l2, kl_div = [np.nan] * 6

            results.append(
                {
                    "seed": seed,
                    "model": "FSI" if coupling == "None" else f"{coupling.upper()}-FSI",  # FIXME: Is this "NONE"?
                    "marginal": marginal,
                    "c": c,
                    "mmd": mmd,
                    "mmd_median": mmd_median,
                    "wasserstein": wasserstein,
                    "mean_diff_l1": mean_diff_l1,
                    "mean_diff_l2": mean_diff_l2,
                    "kl_div": kl_div,
                    "time": time,
                }
            )
    df_results_fsi = pd.DataFrame(results)

    if plot_results:
        idx_plot = []
        for c in np.unique(y_test[:, 0]):
            idx = np.where(y_test[:, 0] == c)[0][:1]
            idx_plot.append(idx)
        idx_plot = [x.item() for x in np.array(idx_plot).flatten()]

        fsi_model.plot_interpolation(
            X_test,
            y_test,
            t_test,
            n_classes=n_classes if n_classes is not None else 9,
            idx_plot=idx_plot,
            title=None,
            save=True,
            filename="fsi",
            filepath="../../figures_paper/",
            coupling=coupling,
            s=5,
            ncols=ncols,
            plot_ode=plot_ode,
        )

    return df_results_fsi


def kl_divergence_gaussian(target, transport_c):
    # Compute means
    target_mean = np.mean(target, axis=0)
    transport_c_mean = np.mean(transport_c, axis=0)

    # Compute covariances
    target_cov = np.cov(target.T)
    transport_c_cov = np.cov(transport_c.T)

    # Compute dimension
    d = target.shape[1]

    # Compute KL divergence
    kl = 0.5 * (
        np.log(np.linalg.det(transport_c_cov) / np.linalg.det(target_cov))
        - d
        + np.trace(np.linalg.inv(transport_c_cov) @ target_cov)
        + (transport_c_mean - target_mean).T @ np.linalg.inv(transport_c_cov) @ (transport_c_mean - target_mean)
    )

    return kl
