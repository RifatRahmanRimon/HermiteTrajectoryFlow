import itertools
import re
from pathlib import Path

import cloudpickle as pickle
import pytorch_lightning as pl

from mmfm.data import dgp_waves_data
from mmfm.fsi import FSI


def main(seed, ns_per_t_and_c, train_test_split, off_diagonal, data_std, dimension, dgp, add_time_cond):
    """Main function to train the (C)(OT) FSI models."""

    pl.seed_everything(seed)

    path_name = "/home/rohbeckm/scratch/results/dgp_waves/results_fsi"
    filename = f"dgp_waves_{dgp}_{seed}_{ns_per_t_and_c}_{train_test_split}_{off_diagonal}_{data_std}_{dimension}"
    if add_time_cond:
        filename = filename + "_" + re.sub(r"[(), ]", "", str(add_time_cond))
    results_path = Path(path_name) / filename
    results_path.mkdir(parents=True, exist_ok=True)
    save = True

    # Check if results_path folder exists and contains a csv file
    if results_path.exists() and (results_path / "df_results.csv").exists():
        print(f"Skipping {filename}")
        return

    batch_size = None

    #
    # VANILLA FSI
    #

    coupling = None
    _, X_train, y_train, t_train, _, _, _, _, _ = dgp_waves_data(
        coupling,
        batch_size,
        dimension,
        off_diagonal,
        data_std,
        ns_per_t_and_c,
        dgp=dgp,
        return_data="train-valid",
        add_time_cond=add_time_cond,
    )

    fsi_model = FSI(conditional=False)
    fsi_model.train(X_train, None, t_train)
    if save:
        fsi_model.save_monge_maps(results_path / "monge_maps_fsi.pkl")
        with open(results_path / "model_fsi.pkl", "wb") as f:
            pickle.dump(fsi_model, f)

    #
    # OT FSI
    #

    coupling = "ot"
    _, X_train, y_train, t_train, _, _, _, _, _ = dgp_waves_data(
        coupling,
        batch_size,
        dimension,
        off_diagonal,
        data_std,
        ns_per_t_and_c,
        dgp=dgp,
        return_data="train-valid",
        add_time_cond=add_time_cond,
    )

    fsi_model = FSI(conditional=False)
    fsi_model.train(X_train, None, t_train)
    if save:
        fsi_model.save_monge_maps(results_path / "monge_maps_ot_fsi.pkl")
        with open(results_path / "model_fsi.pkl", "wb") as f:
            pickle.dump(fsi_model, f)

    #
    # COT FSI
    #
    coupling = "cot"
    _, X_train, y_train, t_train, _, _, _, _, _ = dgp_waves_data(
        coupling,
        batch_size,
        dimension,
        off_diagonal,
        data_std,
        ns_per_t_and_c,
        dgp=dgp,
        return_data="train-valid",
        add_time_cond=add_time_cond,
    )

    fsi_model = FSI(conditional=True)
    fsi_model.train(X_train, y_train, t_train)
    if save:
        fsi_model.save_monge_maps(results_path / "monge_maps_cot_fsi.pkl")
        with open(results_path / "model_cot_fsi.pkl", "wb") as f:
            pickle.dump(fsi_model, f)


if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]
    ns_per_t_and_c = [50]
    train_test_split = [0.5]
    off_diagonal = [0.0]
    data_std = [0.025]
    dimension = [2]
    # CONFIG 1: for DGP i
    # dgp = ["i"]
    # add_time_cond = [
    #     [
    #         (k, 0.15) for k in range(1,11)
    #     ]
    # ]
    # CONFIG 1: for DGP d
    dgp = ["e"]
    add_time_cond = [[(5, 0.55)]]

    param_combinations = itertools.product(
        seeds, ns_per_t_and_c, train_test_split, off_diagonal, data_std, dimension, dgp, add_time_cond
    )

    for params in param_combinations:
        print(f"Running FSI training with parameters: {params}")
        main(*params)
