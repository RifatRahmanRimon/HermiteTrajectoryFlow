from pathlib import Path

# import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.nn.utils import clip_grad_norm_
import timeit

from mmfm.multi_marginal_fm import MultiMarginalFlowMatcher
from mmfm.data import dgp_beijing_data
from mmfm.evaluation import eval_metrics
from mmfm.trajectory import sample_trajectory
from mmfm.models import VectorFieldModel


# @click.command()
# @click.option("--seed", type=int, default=5, help="Random seed")
# @click.option("--max_grad_norm", default=False, type=bool, help="Max gradient norm")
# @click.option("--p_unconditional", default=0.2, type=float, help="Probability of unconditional")
# @click.option("--ns_per_t_and_c", default=100, type=int, help="Number of samples per t and c")
# @click.option("--x_latent_dim", default=64, type=int, help="Model width")
# @click.option("--time_embed_dim", default=64, type=int, help="Model width")
# @click.option("--cond_embed_dim", default=64, type=int, help="Model width")
# @click.option("--conditional_model", default=True, type=bool, help="Model width")
# @click.option("--classifier_free", default=False, type=bool, help="Model width")
# @click.option("--embedding_type", default="free", type=str, help="Model width")
# @click.option("--sum_time_embed", default=False, type=bool, help="Model width")
# @click.option("--sum_cond_embed", default=False, type=bool, help="Model width")
# @click.option("--normalization", default=None, type=str, help="Normalization method")
# @click.option("--affine_transform", default=False, type=bool, help="Use affine transform")
# @click.option("--max_norm_embedding", default=True, type=bool, help="Max norm embedding")
# @click.option("--init_weights", default="xavier", type=str, help="Initialization weights")
# @click.option("--activation", default="SELU", type=str, help="Activation function")
# @click.option("--lrs", default="cosine", type=str, help="Learning rate schedule")
# @click.option("--interpolation", default="cubic", type=str, help="MFMF interpolation method")
# @click.option("--n_epochs", default=300, type=int, help="Number of epochs")
# @click.option("--coupling", default=None, type=str, help="Coupling method")
# @click.option("--batch_size", default=None, help="Batch size")
# @click.option("--train_test_split", default=0.5, type=float, help="Train test split")
# @click.option("--lr", default=1e-2, type=float, help="Learning rate")
# @click.option("--flow_variance", help="Flow variance")
# @click.option("--num_out_layers", default=3, type=int, help="num_out_layers")
# @click.option("--optimizer_name", default="adam", type=str, help="Optimizer")
# @click.option("--dgp", default="a", type=str, help="Data Generation Process")
# @click.option("--dimension", default=2, type=int, help="Dimension")
# @click.option("--spectral_norm", default=False, type=bool, help="Save results")
# @click.option("--dropout", default=0.0, type=float, help="Dropout")
# @click.option("--conditional_bias", default=False, type=bool, help="Conditional bias")
# @click.option("--keep_constants", default=False, type=bool, help="Keep constants")
# @click.option("--spectral_norm", default=False, type=bool, help="Subsample fraction")
# @click.option("--dropout", default=0.0, type=float, help="Dropout rate")
# @click.option("--model_type", default="mmfm", type=str, help="Conditional bias")
# @click.option("--matching", default=None, help="Conditional bias")
def main(
    seed,
    max_grad_norm,
    p_unconditional,
    ns_per_t_and_c,
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
    dimension,
    spectral_norm,
    dropout,
    conditional_bias,
    keep_constants,
    model_type,
    matching,
):
    """Train MMFM models."""
    # Plenty of arguments are passed with incorrect file types when used with LSF batching
    # Fix type issues first
    max_grad_norm = None if max_grad_norm == "None" else max_grad_norm
    normalization = None if normalization == "None" else normalization
    coupling = None if coupling == "None" else coupling
    init_weights = None if init_weights == "None" else init_weights
    lrs = None if lrs == "None" else lrs
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

    path_name = "/data/m015k/results/dgp_weather/results_mmfm"
    Path(path_name).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pl.seed_everything(seed)
    if normalization is None:
        affine_transform = False

    filename = (
        f"dgp_weather_{dgp}_{seed}_{lr}_{flow_variance}_{num_out_layers}_{max_grad_norm}_{p_unconditional}_{ns_per_t_and_c}_{x_latent_dim}_{time_embed_dim}_{cond_embed_dim}_{normalization}_{init_weights}"
        + f"_{activation}_{lrs}_{interpolation}_{conditional_model}_{classifier_free}_{embedding_type}_{sum_time_embed}_{sum_cond_embed}_{n_epochs}_{coupling}_{affine_transform}_{max_norm_embedding}_{batch_size}_{train_test_split}"
        + f"_{dimension}_{optimizer_name}_{spectral_norm}_{dropout}_{conditional_bias}_{keep_constants}_{matching}_{model_type}"
    )

    results_path = Path(path_name) / filename
    results_path.mkdir(parents=True, exist_ok=True)
    save = False
    guidance_range = np.linspace(0, 1, 11, endpoint=True).tolist() + [1.5, 2.0, 3.0]

    # Check if results_path folder exists and contains a csv file
    if results_path.exists() and (results_path / "df_results.csv").exists():
        print(f"Skipping {filename}")
        return

    #
    # Construct Data for training and validation
    #

    train_loader, X_train, y_train, t_train, X_valid, y_valid, t_valid, n_classes, label_list = dgp_beijing_data(
        coupling,
        batch_size,
        ns_per_t_and_c,
        dgp=dgp,
        return_data="train-valid",
        filter_beginning_end=True if model_type == "fm" else False,
    )

    #
    # MFMF
    #

    mmfm_model = VectorFieldModel(
        data_dim=dimension,
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

    FM = MultiMarginalFlowMatcher(
        sigma=flow_variance,
        interpolation=interpolation,
    )

    losses_mfmf = []
    start_time = timeit.timeit()
    for epoch in range(n_epochs):
        for i, batch in enumerate(train_loader):
            x, targets, timepoints = batch
            optimizer.zero_grad()
            x, targets = x.to(device), targets.to(device)
            t, xt, ut, _, _ = FM.sample_location_and_conditional_flow(xs=x, timepoints=timepoints)
            if conditional_model:
                # targets are aligned (due to OT) among time steps
                # Sample with probability p_unconditional whether to add conditional information or not
                # Condition 0 means empty set
                mask = (torch.rand(size=(targets.shape[0], 1)) > p_unconditional).to(device).squeeze()
                conditions = torch.where(mask, targets[:, 0], 0)[:, None]
                xt = torch.cat([xt.squeeze(-1), conditions, t[:, None]], dim=1).to(device)
            else:
                xt = torch.cat([xt.squeeze(-1), t[:, None]], dim=1).to(device)
            vt = mmfm_model(xt)[:, None]
            loss = torch.mean((vt - ut) ** 2)
            # if smoothness_penalty > 0.0:
            #     loss += smoothness_penalty * torch.linalg.norm(func.jacrev(mmfm_model)(xt)) ** 2
            losses_mfmf.append(loss.item())
            loss.backward()
            if max_grad_norm:
                clip_grad_norm_(mmfm_model.parameters(), max_grad_norm)
            optimizer.step()
            print(f"epoch: {epoch}, steps: {i}, loss: {loss.item():.4}", end="\r")
        if lrs == "cosine":
            scheduler.step()
    
    end_time = timeit.timeit()
    print()
    print()
    print(f"Training time: {end_time - start_time:.2f} seconds")

    #
    # Save model, training results and figures
    #

    mmfm_model.eval()
    training_specification = {
        "seed": seed,
        "lr": lr,
        "flow_variance": flow_variance,
        "max_grad_norm": max_grad_norm,
        "p_unconditional": p_unconditional,
        "ns_per_t_and_c": ns_per_t_and_c,
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
        "dimension": dimension,
        "num_out_layers": num_out_layers,
        "spectral_norm": spectral_norm,
        "dropout": dropout,
        "conditional_bias": conditional_bias,
        "keep_constants": keep_constants,
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
            }
            traj_train = sample_trajectory(X=X_train[:, 0], y=y_train[:, 0], **params)
            traj_valid = sample_trajectory(X=X_valid[:, 0], y=y_valid[:, 0], **params)

            # Find samples to plot
            # n_samples_plot_per_class = 2
            # idx_plot_train = []
            # for c in np.unique(y_train[:, 0]):
            #     idx_plot_train.append(np.where(y_train[:, 0] == c)[0][:n_samples_plot_per_class])
            # idx_plot_train = [x.item() for x in np.array(idx_plot_train).flatten()]
            #
            # idx_plot_valid = []
            # for c in np.unique(y_valid[:, 0]):
            #     idx_plot_valid.append(np.where(y_valid[:, 0] == c)[0][:n_samples_plot_per_class])
            # idx_plot_valid = [x.item() for x in np.array(idx_plot_valid).flatten()]
            #
            # plot_results_mmfm(
            #     X_train,
            #     y_train,
            #     t_train,
            #     traj_train,
            #     idx_plot_train,
            #     n_classes=n_classes,
            #     save=save,
            #     title=f"{filename}_{guidance} | Train",
            #     filepath=Path(str(results_path)) / f"figure_train_{guidance}.png",
            #     ncols=5,
            # )
            # plot_results_mmfm(
            #     X_valid,
            #     y_valid,
            #     t_valid,
            #     traj_valid,
            #     idx_plot_valid,
            #     n_classes=n_classes,
            #     save=save,
            #     title=f"{filename}_{guidance} | Valid",
            #     filepath=Path(str(results_path)) / f"figure_valid_{guidance}.png",
            #     ncols=5,
            # )

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
    # main()
    main(
        seed=5,
        max_grad_norm=False,
        p_unconditional=0.2,
        ns_per_t_and_c=50,
        x_latent_dim=16,
        time_embed_dim=16,
        cond_embed_dim=16,
        conditional_model=True,
        classifier_free=False,
        embedding_type="free",
        sum_time_embed=False,
        sum_cond_embed=False,
        normalization=None,
        affine_transform=False,
        max_norm_embedding=True,
        init_weights="xavier",
        activation="SELU",
        lrs="cosine",
        interpolation="cubic",
        n_epochs=300,
        coupling="cot",
        batch_size="None",
        train_test_split=0.5,
        lr=1e-2,
        flow_variance=0.01,
        num_out_layers=3,
        optimizer_name="adam",
        dgp="a",
        dimension=1,
        spectral_norm=False,
        dropout=0.0,
        conditional_bias=False,
        keep_constants=False,
        model_type="mmfm",
        matching=None,
    )
