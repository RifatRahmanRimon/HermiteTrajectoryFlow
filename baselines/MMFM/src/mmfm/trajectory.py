import numpy as np
import torch
import torchdiffeq

from mmfm.models import MMFMModelGuidanceWrapper


def sample_trajectory(
    model, X, y, device, guidance=1.0, conditional_model=True, atol=1e-9, rtol=1e-7, method="dopri5", steps=1001
):
    """Sample trajectory from MMFM model.

    Args:
        model (torch.nn.Module): MMFM model.
        X (np.ndarray or torch.Tensor): Initial conditions. Shape (n_samples, 1, n_features).
        y (np.ndarray or torch.Tensor): Labels. Shape (n_samples, n_timepoints).
        device (torch.device): Device to use.
        guidance (float): Guidance value for sampling see [1].
        conditional_model (bool): Whether the model has an additional condition as input.
        atol (float): Absolute tolerance for ODE solver.
        rtol (float): Relative tolerance for ODE solver.
        method (str): ODE solver method.
        steps (int): Number of time points when sampling.

    [1] Zheng, Qinqing, et al. "Guided flows for generative modeling and decision making."
        arXiv preprint arXiv:2311.13443 (2023).
    """
    settings = {
        "t": torch.linspace(0, 1, steps, device=device),
        "atol": atol,
        "rtol": rtol,
        "method": method,
    }
    with torch.no_grad():
        # Conver numpy to tensor
        if isinstance(X, np.ndarray):
            X = torch.tensor(X).to(device)
            y = torch.tensor(y).to(device)
        else:
            X = X.to(device)
            y = y.to(device)

        if conditional_model:
            trajectory = torchdiffeq.odeint(
                lambda t, x: MMFMModelGuidanceWrapper(model, guidance).forward(x, y, t),
                X.to(device),
                **settings,
            )
        else:
            trajectory = torchdiffeq.odeint(
                lambda t, x: model.forward(
                    torch.cat(
                        [
                            x.squeeze(),
                            t.clone().detach().unsqueeze(0).unsqueeze(0).repeat(x.shape[0], 1),
                        ],
                        dim=1,
                    )
                ),
                X.to(device),
                **settings,
            )

        if device.type == "cuda":
            trajectory = trajectory.cpu()
        trajectory = trajectory.numpy()

        return trajectory
