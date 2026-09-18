# Surpress all UserWarning
import warnings

# Needed for some anndata warnings
warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.nn.utils import clip_grad_norm_

from mmfm.multi_marginal_fm import MultiMarginalConditionalFlowMatcher
from mmfm.data import dgp_iccite, dgp_iccite_4t
from mmfm.evaluation import eval_metrics
from mmfm.trajectory import sample_trajectory
from mmfm.models import VectorFieldModel


@click.command()
@click.option("--seed", type=int, default=5, help="Random seed")
@click.option("--max_grad_norm", default=False, type=bool, help="Max gradient norm")
@click.option("--p_unconditional", default=0.2, type=float, help="Probability of unconditional")
@click.option("--n_samples_per_c_in_b", default=100, type=int, help="Number of samples per t and c")
@click.option("--x_latent_dim", default=64, type=int, help="Model width")
@click.option("--time_embed_dim", default=64, type=int, help="Model width")
@click.option("--cond_embed_dim", default=64, type=int, help="Model width")
@click.option("--conditional_model", default=True, type=bool, help="Model width")
@click.option("--classifier_free", default=True, type=bool, help="Model width")
@click.option("--embedding_type", default="free", type=str, help="Model width")
@click.option("--sum_time_embed", default=False, type=bool, help="Model width")
@click.option("--sum_cond_embed", default=False, type=bool, help="Model width")
@click.option("--normalization", default=None, type=str, help="Normalization method")
@click.option("--affine_transform", default=False, type=bool, help="Use affine transform")
@click.option("--max_norm_embedding", default=True, type=bool, help="Max norm embedding")
@click.option("--init_weights", default="xavier", type=str, help="Initialization weights")
@click.option("--activation", default="SELU", type=str, help="Activation function")
@click.option("--lrs", default="cosine", type=str, help="Learning rate schedule")
@click.option("--interpolation", default="cubic", type=str, help="MFMF interpolation method")
@click.option("--n_epochs", default=300, type=int, help="Number of epochs")
@click.option("--coupling", default=None, type=str, help="Coupling method")
@click.option("--batch_size", default=None, help="Batch size")
@click.option("--train_test_split", default=0.5, type=float, help="Train test split")
@click.option("--lr", default=1e-2, type=float, help="Learning rate")
@click.option("--flow_variance", help="Flow variance")
@click.option("--num_out_layers", default=3, type=int, help="num_out_layers")
@click.option("--optimizer_name", default="adam", type=str, help="Optimizer")
@click.option("--dgp", default="a", type=str, help="Data Generation Process")
@click.option("--hvg", help="Highly variable genes")
@click.option("--use_pca", default=50, type=int, help="Use PCA")
@click.option("--subsample_frac", help="Subsample fraction")
@click.option("--spectral_norm", default=False, type=bool, help="Subsample fraction")
@click.option("--dropout", default=0.0, type=float, help="Dropout rate")
@click.option("--conditional_bias", default=False, type=bool, help="Conditional bias")
@click.option("--keep_constants", default=False, type=bool, help="Keep constants")
@click.option("--top_n_effects", help="Top n effects")
@click.option("--leave_out_mid", help="Leave out middle")
@click.option("--leave_out_end", help="Leave out end")
@click.option("--preset", default="a", type=str, help="Leave out both")
@click.option("--model_type", default="mmfm", type=str, help="Conditional bias")
@click.option("--matching", default=None, help="Conditional bias")
def main(
    seed,
    max_grad_norm,
    p_unconditional,
    n_samples_per_c_in_b,
    x_latent_dim,
    time_embed_dim,
    cond_embed_dim,
    conditional_model,
    classifier_free,
    embedding_type,
    sum_time_embed,
    sum_cond_embed,
    normalization,
    affine_transform,
    max_norm_embedding,
    init_weights,
    activation,
    lrs,
    interpolation,
    n_epochs,
    coupling,
    batch_size,
    train_test_split,
    lr,
    flow_variance,
    num_out_layers,
    optimizer_name,
    dgp,
    hvg,
    use_pca,
    subsample_frac,
    spectral_norm,
    dropout,
    conditional_bias,
    keep_constants,
    top_n_effects,
    leave_out_mid,
    leave_out_end,
    preset,
    model_type,
    matching,
):
    """Train MMFM model on icCITE data."""
    # Plenty of arguments are passed with incorrect file types when used with LSF batching
    # Fix type issues first
    max_grad_norm = None if max_grad_norm == "None" else max_grad_norm
    normalization = None if normalization == "None" else normalization
    coupling = None if coupling == "None" else coupling
    init_weights = None if init_weights == "None" else init_weights
    lrs = None if lrs == "None" else lrs
    hvg = None if hvg == "None" else hvg
    subsample_frac = None if subsample_frac == "None" else subsample_frac
    top_n_effects = None if top_n_effects == "None" else top_n_effects
    leave_out_mid = None if leave_out_mid == "None" else leave_out_mid
    leave_out_end = None if leave_out_end == "None" else leave_out_end
    preset = None if preset == "None" else preset
    matching = None if matching == "None" else matching
    if isinstance(batch_size, str) & (batch_size == "None"):
        batch_size = None
    else:
        try:
            batch_size = int(batch_size)
        except ValueError:
            raise ValueError("Batch size must be an integer or None") from None
    try:
        flow_variance = float(flow_variance)
    except ValueError:
        pass

    path_name = "/home/rohbeckm/scratch/results/dgp_iccite/results_mmfm"
    Path(path_name).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pl.seed_everything(seed)
    if normalization is None:
        affine_transform = False

    filename = (
        f"dgp_iccite_{dgp}_{seed}_{lr}_{flow_variance}_{num_out_layers}_{max_grad_norm}_{p_unconditional}_{x_latent_dim}_{time_embed_dim}_{cond_embed_dim}"
        + f"_{conditional_model}_{classifier_free}_{embedding_type}_{sum_time_embed}_{sum_cond_embed}_{normalization}_{affine_transform}_{max_norm_embedding}_{init_weights}"
        + f"_{activation}_{lrs}_{interpolation}_{n_epochs}_{coupling}_{batch_size}_{train_test_split}_{hvg}_{use_pca}_{n_samples_per_c_in_b}_{subsample_frac}"
        + f"_{optimizer_name}_{top_n_effects}_{leave_out_mid}_{leave_out_end}_{preset}_{spectral_norm}_{dropout}_{conditional_bias}_{keep_constants}_{matching}_{model_type}"
    )

    results_path = Path(path_name) / filename
    results_path.mkdir(parents=True, exist_ok=True)
    save = True
    guidance_range = np.linspace(0, 1, 11, endpoint=True).tolist() + [1.5, 2.0, 3.0]

    # Check if results_path folder exists and contains a csv file
    if results_path.exists() and (results_path / "df_results.csv").exists():
        print(f"Skipping {filename}")
        return

    #
    # Construct Data for training and validation
    #

    if dgp == "a":
        if preset in ["z", "y"]:
            f_data_loader = dgp_iccite_4t
        else:
            f_data_loader = dgp_iccite
        train_loader, X_train, y_train, t_train, X_valid, y_valid, t_valid, _, _, _, _ = f_data_loader(
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
            filter_beginning_end=True if model_type == "fm" else False,
        )
        if preset == "a":
            n_classes = 50
            label_list = list(range(1, 51))  # We start at 1
        elif preset == "b":
            n_classes = 70
            label_list = list(range(1, 71))  # We start at 1
        elif preset == "c":
            n_classes = 90
            label_list = list(range(1, 91))  # We start at 1
        elif preset == "d":
            n_classes = 123
            label_list = list(range(1, 124))  # We start at 1
        elif preset == "y":
            n_classes = 90
            label_list = list(range(1, 91))  # We start at 1
        elif preset == "z":
            n_classes = 60
            label_list = list(range(1, 61))  # We start at 1


    #
    # MFMF
    #

    mmfm_model = VectorFieldModel(
        data_dim=X_train.shape[2],
        x_latent_dim=x_latent_dim,
        time_embed_dim=time_embed_dim,
        cond_embed_dim=cond_embed_dim,
        conditional_model=conditional_model,
        embedding_type=embedding_type,
        n_classes=n_classes,
        label_list=label_list,
        normalization=normalization,
        activation=activation,
        affine_transform=affine_transform,
        sum_time_embed=sum_time_embed,
        sum_cond_embed=sum_cond_embed,
        max_norm_embedding=max_norm_embedding,
        num_out_layers=num_out_layers,
        spectral_norm=spectral_norm,
        dropout=dropout,
        conditional_bias=conditional_bias,
        keep_constants=keep_constants,
    ).to(device)

    # Initialize weights
    if init_weights.lower() == "xavier":
        for p in mmfm_model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
    elif init_weights.lower() == "xavier_normal":
        for p in mmfm_model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.4 if activation == "LeakyReLU" else 0.75)

    # Initialize optimizer
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(mmfm_model.parameters(), lr=lr)
    elif optimizer_name.lower() == "adam-wd4":
        optimizer = torch.optim.Adam(mmfm_model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(mmfm_model.parameters(), lr=lr, amsgrad=True)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(mmfm_model.parameters(), lr=lr, momentum=0.9, nesterov=True)

    # Initialize learning rate scheduler
    if lrs == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-4)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=10, threshold=0.0001, verbose=True, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    FM = MultiMarginalConditionalFlowMatcher(
        sigma=flow_variance,
        interpolation=interpolation,
    )

    losses_mfmf = []
    for epoch in range(n_epochs):
        for i, batch in enumerate(train_loader):
            x, targets, timepoints = batch
            optimizer.zero_grad()
            x, targets = x.to(device), targets.to(device)
            print(x.shape, timepoints.shape)
            t, xt, ut, _, _ = FM.sample_location_and_conditional_flow(xs=x, timepoints=timepoints)
            if conditional_model:
                # targets are aligned (due to OT) among time steps
                # Sample with probability p_unconditional whether to add conditional information or not
                # Condition 0 means empty set
                mask = (torch.rand(size=(targets.shape[0], 1)) > p_unconditional).to(device).squeeze()
                conditions = torch.where(mask, targets[:, 0], 0)[:, None]
                xt = torch.cat([xt.squeeze(), conditions, t[:, None]], dim=1).to(device)
            else:
                xt = torch.cat([xt.squeeze(), t[:, None]], dim=1).to(device)
            vt = mmfm_model(xt)[:, None]
            loss = torch.mean((vt - ut) ** 2)
            losses_mfmf.append(loss.item())
            loss.backward()
            if max_grad_norm:
                clip_grad_norm_(mmfm_model.parameters(), max_grad_norm)
            optimizer.step()
            print(f"epoch: {epoch}, steps: {i}, loss: {loss.item():.4}", end="\r")
        if lrs == "cosine":
            scheduler.step()

    #
    # Save model, training results and figures
    #

    mmfm_model.eval()
    training_specification = {
        "seed": seed,
        "lr": lr,
        "flow_variance": flow_variance,
        "num_out_layers": num_out_layers,
        "max_grad_norm": max_grad_norm,
        "p_unconditional": p_unconditional,
        "n_samples_per_c_in_b": n_samples_per_c_in_b,
        "x_latent_dim": x_latent_dim,
        "time_embed_dim": time_embed_dim,
        "cond_embed_dim": cond_embed_dim,
        "embedding_type": embedding_type,
        "sum_time_embed": sum_time_embed,
        "sum_cond_embed": sum_cond_embed,
        "normalization": normalization,
        "init_weights": init_weights,
        "activation": activation,
        "lrs": lrs,
        "n_classes": n_classes,
        "interpolation": interpolation,
        "conditional_model": conditional_model,
        "classifier_free": classifier_free,
        "n_epochs": n_epochs,
        "coupling": coupling,
        "affine_transform": affine_transform,
        "max_norm_embedding": max_norm_embedding,
        "batch_size": batch_size,
        "train_test_split": train_test_split,
        "optimizer": optimizer_name,
        "dgp": dgp,
        "hvg": hvg,
        "use_pca": use_pca,
        "subsample_frac": subsample_frac,
        "dimension": X_train.shape[2],
        "spectral_norm": spectral_norm,
        "dropout": dropout,
        "conditional_bias": conditional_bias,
        "keep_constants": keep_constants,
        "top_n_effects": top_n_effects,
        "leave_out_mid": leave_out_mid,
        "leave_out_end": leave_out_end,
        "preset": preset,
        "model_type": model_type,
        "matching": matching,
    }

    # Save model
    if save:
        state = {**training_specification, "state_dict": mmfm_model.state_dict(), "optimizer": optimizer.state_dict()}
        torch.save(state, results_path / "model.pt")

    # Save/Plot loss
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(losses_mfmf)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss {filename}", fontsize=5)
    if save:
        plt.savefig(results_path / "loss.png")
        plt.close()
    else:
        plt.show()

    list_results = []
    for guidance in guidance_range:
        if (p_unconditional == 0.0) & (guidance != 1.0):
            # We do not have an unconditional model, hence guidance should not be different from 1.0
            # since otherwise we mix the conditional with an untrained unconditional model
            continue
        if (p_unconditional == 1.0) & (guidance != 0.0):
            # We do not have a conditional model, hence guidance should not be different from 0.0
            # since otherwise we mix the unconditional with an untrained conditional model
            continue
        print(f"- Guidance: {guidance}")

        try:
            params = {
                "model": mmfm_model,
                "device": device,
                "guidance": guidance,
                "conditional_model": conditional_model,
                "method": "dopri5",
                "steps": 1001,
            }
            traj_train = sample_trajectory(X=X_train[:, 0], y=y_train[:, 0], **params)
            traj_valid = sample_trajectory(X=X_valid[:, 0], y=y_valid[:, 0], **params)

            df_results_train = eval_metrics(traj_train, X_train, y_train, t_train, guidance, train=True)
            df_results_valid = eval_metrics(traj_valid, X_valid, y_valid, t_valid, guidance, train=False)
            list_results.extend([df_results_train, df_results_valid])

        except AssertionError as e:
            print("Error in training trajectory")
            print(e)
            # Write empty file: pd.DataFrame().to_csv(results_path / "df_results.csv", index=False)
            # otherwise the model will be "trained" again by LSF
            continue

    # Set index to all zeros to merge all columns
    df_results = pd.concat(list_results, axis=0)
    df_results.index = np.zeros(df_results.shape[0])
    df_results = pd.DataFrame({**training_specification}, index=[0]).join(df_results).reset_index(drop=True)
    if save:
        # Add parameters in each row
        # Write parameters and mmds to file
        df_results.to_csv(results_path / "df_results.csv", index=False)


if __name__ == "__main__":
    main()
