import itertools
from pathlib import Path

import cloudpickle as pickle
import pytorch_lightning as pl

from mmfm.data import dgp_iccite_4t
from mmfm.fsi import FSI


def main(
    seed,
    hvg,
    subsample_frac,
    use_pca,
    batch_size,
    n_samples_per_c_in_b,
    train_test_split,
    dgp,
    top_n_effects,
    leave_out_mid,
    leave_out_end,
    preset,
):
    """Main function to train the (C)(OT) FSI models."""
    pl.seed_everything(seed)

    path_name = "/home/rohbeckm/scratch/results/dgp_iccite/results_fsi"
    filename = (
        f"dgp_iccite_4t_{dgp}_{seed}_{hvg}_{subsample_frac}_{use_pca}_{batch_size}_{n_samples_per_c_in_b}"
        + f"_{train_test_split}_{top_n_effects}_{leave_out_mid}_{leave_out_end}_{preset}"
    )
    results_path = Path(path_name) / filename
    results_path.mkdir(parents=True, exist_ok=True)
    save = True

    # Check if results_path folder exists and contains a csv file
    if results_path.exists() and (results_path / "df_results.csv").exists():
        print(f"Skipping {filename}")
        return

    # params = {
    #     "hvg": hvg,
    #     "subsample_frac": subsample_frac,
    #     "use_pca": use_pca,
    #     "batch_size": batch_size,
    #     "n_samples_per_c_in_b": n_samples_per_c_in_b,
    #     "train_test_split": train_test_split,
    #     "top_n_effects": top_n_effects,
    #     "leave_out_mid": leave_out_mid,
    #     "leave_out_end": leave_out_end,
    #     "preset": preset,
    #     "seed": seed,
    # }

    #
    # VANILLA FSI
    #
    # if dgp == "a":
    #     _, X_train, y_train, t_train, _, _, _, _, _, _, _ = dgp_iccite(
    #         hvg=hvg,
    #         use_pca=use_pca,
    #         coupling=None,
    #         batch_size=batch_size,
    #         n_samples_per_c_in_b=n_samples_per_c_in_b,
    #         train_test_split=train_test_split,
    #         subsample_frac=subsample_frac,
    #         seed=seed,
    #         top_n_effects=top_n_effects,
    #         leave_out_mid=leave_out_mid,
    #         leave_out_end=leave_out_end,
    #         preset=preset,
    #     )

    # fsi_model = FSI(conditional=False)
    # fsi_model.train(X_train, None, t_train)
    # if save:
    #     fsi_model.save_monge_maps(results_path / "monge_maps_ot_fsi.pkl")

    #
    # OT FSI
    #
    # if dgp == "a":
    #     _, X_train, y_train, t_train, _, _, _, _, _, _, _ = dgp_iccite(
    #         hvg=hvg,
    #         use_pca=use_pca,
    #         coupling="ot",
    #         batch_size=batch_size,
    #         n_samples_per_c_in_b=n_samples_per_c_in_b,
    #         train_test_split=train_test_split,
    #         subsample_frac=subsample_frac,
    #         seed=seed,
    #         top_n_effects=top_n_effects,
    #         leave_out_mid=leave_out_mid,
    #         leave_out_end=leave_out_end,
    #         preset=preset,
    #     )

    # fsi_model = FSI(conditional=False)
    # fsi_model.train(X_train, None, t_train)
    # if save:
    #     fsi_model.save_monge_maps(results_path / "monge_maps_ot_fsi.pkl")

    #
    # COT FSI
    #
    if dgp == "a":
        _, X_train, y_train, t_train, _, _, _, _, _, _, _ = dgp_iccite_4t(
            hvg=hvg,
            use_pca=use_pca,
            coupling="cot",
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

    fsi_model = FSI(conditional=True)
    fsi_model.train(X_train, y_train, t_train)
    if save:
        fsi_model.save_monge_maps(results_path / "monge_maps_cot_fsi.pkl")
        with open(results_path / "model_cot_fsi.pkl", "wb") as f:
            pickle.dump(fsi_model, f)


if __name__ == "__main__":
    seed = [0, 1, 2]
    hvg = [None]
    subsample_frac = [None]
    use_pca = [10, 25]
    batch_size = [None]
    dgp = ["a"]
    n_samples_per_c_in_b = [100, 250]
    train_test_split = [0.8]
    top_n_effects = [None]
    leave_out_mid = [None]
    leave_out_end = [None]
    preset = ["z"]

    param_combinations = itertools.product(
        seed,
        hvg,
        subsample_frac,
        use_pca,
        batch_size,
        n_samples_per_c_in_b,
        train_test_split,
        dgp,
        top_n_effects,
        leave_out_mid,
        leave_out_end,
        preset,
    )

    for params in param_combinations:
        print(f"Running FSI training with parameters: {params}")
        main(*params)
