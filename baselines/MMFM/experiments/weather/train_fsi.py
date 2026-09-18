import itertools
import re
from pathlib import Path

import cloudpickle as pickle
import pytorch_lightning as pl

from mmfm.data import dgp_beijing_data
from mmfm.fsi import FSI


def main(seed, ns_per_t_and_c, train_test_split, dgp, add_time_cond):
    """Main function to train the (C)(OT) FSI models."""

    pl.seed_everything(seed)

    path_name = "/data/m015k/results/dgp_weather/results_fsi"
    filename = f"dgp_weather_{dgp}_{seed}_{ns_per_t_and_c}_{train_test_split}"
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

    # #
    # # VANILLA FSI
    # #

    # coupling = None
    # _, X_train, y_train, t_train, _, _, _, _, _ = dgp_beijing_data(
    #     coupling,
    #     batch_size,
    #     ns_per_t_and_c,
    #     dgp=dgp,
    #     return_data="train-valid",
    #     add_time_cond=add_time_cond,
    # )

    # fsi_model = FSI(conditional=False)
    # fsi_model.train(X_train, None, t_train)
    # if save:
    #     fsi_model.save_monge_maps(results_path / "monge_maps_fsi.pkl")
    #     with open(results_path / "model_fsi.pkl", "wb") as f:
    #         pickle.dump(fsi_model, f)

    # #
    # # OT FSI
    # #

    # coupling = "ot"
    # _, X_train, y_train, t_train, _, _, _, _, _ = dgp_beijing_data(
    #     coupling,
    #     batch_size,
    #     ns_per_t_and_c,
    #     dgp=dgp,
    #     return_data="train-valid",
    #     add_time_cond=add_time_cond,
    # )

    # fsi_model = FSI(conditional=False)
    # fsi_model.train(X_train, None, t_train)
    # if save:
    #     fsi_model.save_monge_maps(results_path / "monge_maps_ot_fsi.pkl")
    #     with open(results_path / "model_fsi.pkl", "wb") as f:
    #         pickle.dump(fsi_model, f)

    #
    # COT FSI
    #
    coupling = "cot"
    _, X_train, y_train, t_train, _, _, _, _, _ = dgp_beijing_data(
        coupling,
        batch_size,
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
    seeds = [0, 1, 2]
    ns_per_t_and_c = [50]
    train_test_split = [0.5]
    dgp = ["a", "b"]
    add_time_cond =  [[(7, 11)]]

    param_combinations = itertools.product(
        seeds, ns_per_t_and_c, train_test_split, dgp, add_time_cond
    )

    for params in param_combinations:
        print(f"Running FSI training with parameters: {params}")
        main(*params)
