"""
This module contains the functions to generate synthetic data for:

ODE
1. exponential decay
2. logistic growth
3. harmonic oscillator
4. damped harmonic oscillator
5. lotka volterra
6. lorenz
7. linear nd

8. hopperphysics

tier-2 mechanical systems (positions first, then velocities -- pair map i:(i+n))
1. pendulum            D=2
2. double_pendulum     D=4
3. duffing             D=2   (non-autonomous: harmonically forced)
4. spring_mass_chain   D=10
5. nbody               D=12  (planar, softened)

sde
1. exponential
2. damped harmonic
3. lotka-volterra
4. lorenz
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import torch
import pickle

from pathlib import Path
import sys


# `dm_control` is only needed for the Hopper physics environment. It pulls in
# MuJoCo and is not installed by default, so import it lazily inside the Hopper
# generator instead of at module load (which would break every other system).


def integrate_with_scipy(rhs, times, y0, params, method="RK45", rtol=1e-6, atol=1e-9):
    def wrapped(t, y):
        return rhs(t, y, params).astype(np.float64)

    sol = solve_ivp(
        wrapped,
        t_span=(times[0], times[-1]),
        y0=y0.astype(np.float64),
        t_eval=times,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T.astype(np.float32)


def integrate_sde_em(
    drift_rhs, diffusion, times, y0, params, rng,
    *, dt_internal=None, clamp=None
):
    """
    Euler–Maruyama integration on a user-specified observation grid `times`.

    - drift_rhs(t, x, params) -> shape (d,)
    - diffusion(t, x, params) -> either:
        * shape (d,)   : diagonal diffusion (independent noise per dim)
        * shape (d, m) : full diffusion matrix (m Brownian dims)
    - dt_internal: if provided, subdivide each [t_i, t_{i+1}] into smaller steps
      (recommended for stiff / chaotic / high-noise systems like noisy Lorenz).
    - clamp: optional function clamp(x)->x to enforce constraints (e.g. positivity)
    """
    times = np.asarray(times, dtype=np.float32)
    y0 = np.asarray(y0, dtype=np.float32)
    d = y0.shape[0]
    n = len(times)

    out = np.empty((n, d), dtype=np.float32)
    out[0] = y0

    for i in range(n - 1):
        t0 = float(times[i])
        t1 = float(times[i + 1])
        x = out[i].copy()

        if dt_internal is None:
            nsteps = 1
            h = t1 - t0
        else:
            h = float(dt_internal)
            nsteps = int(np.ceil((t1 - t0) / h))
            h = (t1 - t0) / nsteps  # fit exactly

        t = t0
        for _ in range(nsteps):
            f = np.asarray(drift_rhs(t, x, params), dtype=np.float32)  # (d,)
            G = np.asarray(diffusion(t, x, params), dtype=np.float32)

            if G.ndim == 1:
                # diagonal diffusion
                z = rng.normal(size=d).astype(np.float32)
                x = x + f * h + G * np.sqrt(h) * z
            else:
                # full diffusion matrix (d, m)
                m = G.shape[1]
                z = rng.normal(size=m).astype(np.float32)
                x = x + f * h + (G @ z) * np.sqrt(h)

            if clamp is not None:
                x = clamp(x)

            t += h

        out[i + 1] = x

    return out



def apply_missingness(values, prob, rng, keep_first=True):
    """
    redundant, we only need the mask, the np.nan happens in pre-processing
    """
    mask = np.ones_like(values, dtype=bool)
    if prob <= 0:
        return values, mask
    drop = rng.uniform(size=values.shape) < prob
    if keep_first:
        drop[0] = False
    mask[drop] = False
    values[~mask] = np.nan
    return values, mask

def get_mask(values, prob, rng, keep_first=True, mask_mode='per_dim'):
    """Observation mask for a [T, D] (or [T, ...]) trajectory.

    mask_mode == 'per_dim' (default, original behaviour): each entry is dropped
        independently -> position and its velocity go missing separately.
    mask_mode == 'per_dof': one Bernoulli per time step, applied to the whole
        state at that time -> a dropped observation removes the full degree of
        freedom (both a coordinate and its derivative together), modelling a
        sensor reading that returns the full state or nothing. NOTE: per_dof
        numbers are NOT comparable to the paper's per_dim table.
    """
    mask = np.ones_like(values, dtype=bool)
    if prob <= 0:
        return mask
    if mask_mode == 'per_dim':
        drop = rng.uniform(size=values.shape) < prob
        if keep_first:
            drop[0] = False
        mask[drop] = False
    elif mask_mode == 'per_dof':
        drop = rng.uniform(size=values.shape[0]) < prob   # one draw per time step
        if keep_first:
            drop[0] = False
        mask[drop, ...] = False
    else:
        raise ValueError(f"Unknown mask_mode: {mask_mode!r}")
    return mask

## exponential decay
def rhs_exp_decay(t, state, params):
    return -params["lambda"] * state


def sample_exp_params(rng):
    return {"lambda": rng.uniform(0.2, 1.5)}


def sample_exp_initial(rng, params):
    return rng.uniform(0.5, 2.0, size=1)



# [stochastic] exponential decay :: Orstein-Uhlenbeck Process
def drift_exp_decay_sde(t, state, params):
    # state shape (1,)
    return -params["theta"] * (state)

def diffusion_exp_decay_sde(t, state, params):
    return np.array([params["sigma"]], dtype=np.float32)

def sample_exp_decay_sde_params(rng):
    # return {"theta": rng.uniform(0.2, 1.5), "mu": 0.0, "sigma": rng.uniform(0.05, 0.6)}
    return {"theta": rng.uniform(0.2, 1.5), "mu": 0.0, "sigma": rng.uniform(0.08, 0.08)}

def sample_exp_decay_sde_initial(rng, params):
    return rng.uniform(-2.0, 2.0, size=1)

## logistic growth
def rhs_logistic(t, state, params):
    r, K = params["r"], params["K"]
    return r * state * (1 - state / K)


def sample_logistic_params(rng):
    return {"r": rng.uniform(0.5, 2.0), "K": rng.uniform(1.0, 5.0)}


def sample_logistic_initial(rng, params):
    return rng.uniform(0.1, params["K"], size=1)



## harmonic oscillator
def rhs_harmonic(t, state, params):
    x, v = state
    omega = params["omega"]
    return np.array([v, -omega**2 * x])


def sample_harmonic_params(rng):
    return {"omega": rng.uniform(0.5, 2.5)}


def sample_harmonic_initial(rng, params):
    return rng.uniform(-1.0, 1.0, size=2)


## damped harmonic oscillator
def rhs_damped_harmonic(t, state, params):
    x, v = state
    omega = params["omega"]
    gamma = params["gamma"]
    return np.array([v, -2 * gamma * v - omega**2 * x])


def sample_damped_params(rng):
    return {"omega": rng.uniform(0.8, 2.0), "gamma": rng.uniform(0.1, 0.7)}



## [stochastic] Damped harmonic :: langevin dynamics
def drift_damped_harmonic_sde(t, state, params):
    # [x, v]
    x, v = state
    omega, gamma = params["omega"], params["gamma"]
    return np.array([v, -omega**2 * x - 2 * gamma * v], dtype=np.float32)

def diffusion_damped_harmonic_sde(t, state, params):
    # state = [x, v]
    sigma = params["sigma"]
    return np.array([0.0, sigma], dtype=np.float32)

def sample_damped_harmonic_sde_params(rng):
    # for harmonic: add gamma too (otherwise no stationary behaviour)
    # return {"omega": rng.uniform(0.5, 2.5), "gamma": rng.uniform(0.05, 0.8), "sigma": rng.uniform(0.05, 0.8)}
    return {"omega": rng.uniform(0.5, 2.5), "gamma": rng.uniform(0.05, 0.8), "sigma": rng.uniform(0.08, 0.08)}
    # return {"omega": rng.uniform(0.5, 2.5), "gamma": rng.uniform(0.05, 0.8), "sigma": rng.uniform(5, 8)}

def sample_damped_harmonic_sde_initial(rng, params):
    return rng.uniform(-1.0, 1.0, size=2)



## lotka-volterra
def rhs_lotka_volterra(t, state, params):
    x, y = state
    alpha = params["alpha"]
    beta = params["beta"]
    delta = params["delta"]
    gamma = params["gamma"]
    return np.array([
        alpha * x - beta * x * y,
        delta * x * y - gamma * y,
    ])


def sample_lv_params(rng):
    return {
        "alpha": rng.uniform(0.5, 1.5),
        "beta": rng.uniform(0.02, 0.08),
        "delta": rng.uniform(0.02, 0.08),
        "gamma": rng.uniform(0.5, 1.5),
    }

## lotka-volterra stochastic
# the additive version here can go negative so clamping it

def diffusion_lv_sde(t, state, params):
    return np.array([params["sigma_x"], params["sigma_y"]], dtype=np.float32)

def sample_lv_sde_params(rng):
    p = sample_lv_params(rng)
    # p.update({"sigma_x": rng.uniform(0.01, 0.4), "sigma_y": rng.uniform(0.01, 0.4)})
    p.update({"sigma_x": rng.uniform(0.08, 0.08), "sigma_y": rng.uniform(0.08, 0.08)})
    # p.update({"sigma_x": rng.uniform(4, 10), "sigma_y": rng.uniform(4, 10)})
    return p

def clamp_positive(x, eps=1e-8):
    return np.maximum(x, eps)

## lorenz
def sample_lv_initial(rng, params):
    return rng.uniform(0.5, 2.5, size=2)


def rhs_lorenz(t, state, params):
    x, y, z = state
    sigma = params["sigma"]
    rho = params["rho"]
    beta = params["beta"]
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    ])


def sample_lorenz_params(rng):
    return {
        "sigma": rng.uniform(8.0, 14.0),
        "rho": rng.uniform(20.0, 35.0),
        "beta": rng.uniform(2.0, 3.5),
    }


def sample_lorenz_initial(rng, params):
    return rng.uniform(-15.0, 15.0, size=3)

## lorenz stochastic
def diffusion_lorenz_sde(t, state, params):
    eta = params["eta"]
    return np.array([eta, eta, eta], dtype=np.float32)

def sample_lorenz_sde_params(rng):
    p = sample_lorenz_params(rng)
    # p.update({"eta": rng.uniform(0.1, 2.0)})
    p.update({"eta": rng.uniform(0.08, 0.08)})
    # p.update({"eta": rng.uniform(4, 20)})
    return p



## multivariate linear
def rhs_linear_nd(t, state, params):
    return -params["A"] @ state


def sample_linear_params(rng, dim):
    m = rng.normal(size=(dim, dim))
    a = m.T @ m + dim * np.eye(dim)
    return {"A": a}


def sample_linear_initial(rng, params):
    dim = params["A"].shape[0]
    return rng.uniform(-1.0, 1.0, size=dim)



# ---------------------------------------------------------------------------
# Tier-2 mechanical systems (added for the Hermite work).
#
# Every family below carries an explicit (position, velocity) split so the
# Hermite interpolants have a genuinely observed velocity coordinate to clamp
# on.  The state layout is ALWAYS "all positions first, then all velocities" --
# the convention `generate_hopperphysics_trajectories` already uses for
# qpos/qvel -- so a system with n degrees of freedom pairs as i:(i+n):
#
#     pendulum            D=2   --pairs 0:1
#     double_pendulum     D=4   --pairs 0:2,1:3
#     duffing             D=2   --pairs 0:1
#     spring_mass_chain   D=10  --pairs 0:5,1:6,2:7,3:8,4:9
#     nbody               D=12  --pairs 0:6,1:7,2:8,3:9,4:10,5:11
#
# Params are kept scalar-only, matching the existing families: `params` is
# flattened by IrregularDataset.__getitem__ and collated by the default
# DataLoader collate_fn, and scalars keep that path trivial.
# ---------------------------------------------------------------------------

G_ACCEL = 9.81            # gravitational acceleration [m/s^2]
SPRING_CHAIN_N = 5        # masses in the chain          -> dim = 2 * N
NBODY_N = 3               # bodies in the planar n-body  -> dim = 4 * N


## simple pendulum (nonlinear, optionally damped) -- state = [theta, omega]
def rhs_pendulum(t, state, params):
    theta, omega = state
    return np.array([
        omega,
        -(G_ACCEL / params["length"]) * np.sin(theta) - params["damping"] * omega,
    ])


def sample_pendulum_params(rng):
    return {
        "length": rng.uniform(0.5, 2.0),
        "damping": rng.uniform(0.0, 0.3),
    }


def sample_pendulum_initial(rng, params):
    """Libration-only initial conditions.

    Above the separatrix the bob goes over the top and `theta` grows without
    bound, turning the position coordinate into a ramp and making the
    interpolation comparison meaningless.  Draw theta first, then spend at most
    90% of the separatrix energy E_sep = 2 * g/L on the remaining kinetic term.
    """
    w2 = G_ACCEL / params["length"]
    theta = rng.uniform(-2.2, 2.2)
    e_budget = 0.9 * 2.0 * w2 - w2 * (1.0 - np.cos(theta))
    omega_max = np.sqrt(max(2.0 * e_budget, 0.0))
    return np.array([theta, rng.uniform(-omega_max, omega_max)])


## double pendulum -- state = [theta1, theta2, omega1, omega2]
def rhs_double_pendulum(t, state, params):
    th1, th2, w1, w2 = state
    m1, m2 = params["mass1"], params["mass2"]
    l1, l2 = params["length1"], params["length2"]
    d = th2 - th1
    sin_d, cos_d = np.sin(d), np.cos(d)

    den1 = (m1 + m2) * l1 - m2 * l1 * cos_d * cos_d
    dw1 = (
        m2 * l1 * w1 * w1 * sin_d * cos_d
        + m2 * G_ACCEL * np.sin(th2) * cos_d
        + m2 * l2 * w2 * w2 * sin_d
        - (m1 + m2) * G_ACCEL * np.sin(th1)
    ) / den1

    den2 = (l2 / l1) * den1
    dw2 = (
        -m2 * l2 * w2 * w2 * sin_d * cos_d
        + (m1 + m2) * G_ACCEL * np.sin(th1) * cos_d
        - (m1 + m2) * l1 * w1 * w1 * sin_d
        - (m1 + m2) * G_ACCEL * np.sin(th2)
    ) / den2

    return np.array([w1, w2, dw1, dw2])


def sample_double_pendulum_params(rng):
    return {
        "mass1": rng.uniform(0.8, 1.2),
        "mass2": rng.uniform(0.8, 1.2),
        "length1": rng.uniform(0.8, 1.2),
        "length2": rng.uniform(0.8, 1.2),
    }


def sample_double_pendulum_initial(rng, params):
    """Modest angles and slow starts.

    The double pendulum is chaotic at any non-trivial energy; this keeps the
    trajectories chaotic-but-bounded rather than sending either arm over the
    top, which would make theta unbounded exactly as in the single-pendulum
    case.
    """
    return np.array([
        rng.uniform(-1.0, 1.0),
        rng.uniform(-1.0, 1.0),
        rng.uniform(-0.5, 0.5),
        rng.uniform(-0.5, 0.5),
    ])


## Duffing oscillator (damped, double-well, harmonically forced) -- state = [x, v]
def rhs_duffing(t, state, params):
    """NOTE: the only NON-AUTONOMOUS family in this file -- the forcing term
    depends on `t` explicitly, so the velocity field the MLP learns is a genuine
    function of (x, t) rather than of x alone."""
    x, v = state
    return np.array([
        v,
        -params["delta"] * v
        - params["alpha"] * x
        - params["beta"] * x ** 3
        + params["gamma"] * np.cos(params["omega"] * t),
    ])


def sample_duffing_params(rng):
    # alpha < 0 with beta > 0 is the classic double-well; delta/gamma/omega are
    # drawn around the mildly chaotic regime (delta~0.2, gamma~0.3, omega~1.0).
    return {
        "alpha": rng.uniform(-1.2, -0.8),
        "beta": rng.uniform(0.8, 1.2),
        "delta": rng.uniform(0.1, 0.3),
        "gamma": rng.uniform(0.2, 0.4),
        "omega": rng.uniform(0.9, 1.3),
    }


def sample_duffing_initial(rng, params):
    return np.array([rng.uniform(-1.5, 1.5), rng.uniform(-1.0, 1.0)])


## linear spring-mass chain, both ends pinned to fixed walls
## state = [q_1 .. q_N, v_1 .. v_N]
def rhs_spring_chain(t, state, params):
    n = SPRING_CHAIN_N
    q, v = state[:n], state[n:]
    k, c, m = params["stiffness"], params["damping"], params["mass"]
    # fixed walls: q_0 = q_{N+1} = 0
    padded = np.concatenate(([0.0], q, [0.0]))
    coupling = padded[:-2] - 2.0 * q + padded[2:]
    return np.concatenate([v, (k * coupling - c * v) / m])


def sample_spring_chain_params(rng):
    return {
        "stiffness": rng.uniform(0.5, 2.0),
        "damping": rng.uniform(0.0, 0.2),
        "mass": rng.uniform(0.8, 1.2),
    }


def sample_spring_chain_initial(rng, params):
    n = SPRING_CHAIN_N
    return np.concatenate([
        rng.uniform(-1.0, 1.0, size=n),
        rng.uniform(-0.5, 0.5, size=n),
    ])


## planar gravitational n-body, softened
## state = [x_1, y_1, .., x_N, y_N, vx_1, vy_1, .., vx_N, vy_N]
def rhs_nbody(t, state, params):
    n = NBODY_N
    pos = state[: 2 * n].reshape(n, 2)
    vel = state[2 * n :].reshape(n, 2)
    g, m, eps2 = params["gravity"], params["mass"], params["softening"] ** 2

    diff = pos[None, :, :] - pos[:, None, :]      # diff[i, j] = r_j - r_i
    # Plummer softening: without eps the integrator stalls (and
    # integrate_with_scipy raises) on close encounters.
    inv_r3 = ((diff ** 2).sum(-1) + eps2) ** -1.5
    np.fill_diagonal(inv_r3, 0.0)                 # no self-interaction
    acc = g * m * (diff * inv_r3[:, :, None]).sum(axis=1)

    return np.concatenate([vel.ravel(), acc.ravel()])


def sample_nbody_params(rng):
    # Equal masses: the `vel -= vel.mean(0)` below is only the barycentric
    # momentum correction when every body weighs the same.
    return {
        "gravity": 1.0,
        "mass": rng.uniform(0.8, 1.2),
        "softening": 0.2,
    }


def sample_nbody_initial(rng, params):
    """A jittered rotating ring.

    Uniformly random positions let two bodies start arbitrarily close, which
    produces a near-singular acceleration, ejects a body, and turns its
    coordinate into an unbounded ramp -- the same failure the pendulum energy
    cap avoids.  Instead the bodies start near a regular N-gon, the exact
    relative equilibrium of the equal-mass problem, at its circular speed:

        v^2 = G m / (4 r) * sum_{j=1}^{N-1} 1 / sin(pi j / N)

    (for N=3 this is the Lagrange equilateral triangle).  The jitter below makes
    the orbits quasi-periodic rather than exactly periodic while keeping every
    body bound for the whole time span.
    """
    n = NBODY_N
    r0 = rng.uniform(1.0, 1.4)
    angle = 2.0 * np.pi * np.arange(n) / n + rng.uniform(-0.08, 0.08, size=n)
    radius = r0 * (1.0 + rng.uniform(-0.08, 0.08, size=n))
    pos = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    pos -= pos.mean(axis=0)

    j = np.arange(1, n)
    ring_sum = np.sum(1.0 / np.sin(np.pi * j / n))
    v_circ = np.sqrt(params["gravity"] * params["mass"] * ring_sum / (4.0 * r0))

    r = np.maximum(np.linalg.norm(pos, axis=1, keepdims=True), 1e-3)
    tangent = np.stack([-pos[:, 1], pos[:, 0]], axis=1) / r
    vel = tangent * v_circ * rng.uniform(0.92, 1.08, size=(n, 1))
    vel -= vel.mean(axis=0)

    return np.concatenate([pos.ravel(), vel.ravel()])


### hopper physics
def generate_hopperphysics_trajectories(n_samples, T= 200, D= 14):

		from dm_control import suite  # lazy import; MuJoCo dependency only for Hopper
		env = suite.load('hopper', 'stand')
		physics = env.physics

		# Store the state of the RNG to restore later.
		st0 = np.random.get_state()
		np.random.seed(123)

		data = np.zeros((n_samples, T, D))
		for i in range(n_samples):
			with physics.reset_context():
				# x and z positions of the hopper. We want z > 0 for the hopper to stay above ground.
				physics.data.qpos[:2] = np.random.uniform(0, 0.5, size=2)
				physics.data.qpos[2:] = np.random.uniform(-2, 2, size=physics.data.qpos[2:].shape)
				physics.data.qvel[:] = np.random.uniform(-5, 5, size=physics.data.qvel.shape)
			for t in range(T):
				data[i, t, :D // 2] = physics.data.qpos
				data[i, t, D // 2:] = physics.data.qvel
				physics.step()

		# Restore RNG.
		np.random.set_state(st0)
		return data


## pendulum videos

def generate_pendulum_video_trajectories(
    domain="pendulum",
    task="swingup",
    n_trajectories=5,
    horizon=200,
    render_height=84,
    render_width=84,
    camera_id=0,
):
    """
    can't render on cluster, so uploading it directly after creating it locally
    """
    data_path= Path("/home/c02sara/CISPA-az6/dgdlm-2024/higher_interpolant_matching/data/pendulum_video_precomputed.pkl")

    with data_path.open("rb") as fh:
            trajectories = pickle.load(fh)

    return trajectories



FAMILIES = {
    # ---------------- ODEs ----------------
    "exp_decay": {
        "dim": 1,
        "rhs": rhs_exp_decay,
        "sample_params": sample_exp_params,
        "sample_initial": sample_exp_initial,
    },
    "logistic_growth": {
        "dim": 1,
        "rhs": rhs_logistic,
        "sample_params": sample_logistic_params,
        "sample_initial": sample_logistic_initial,
    },
    "harmonic_oscillator": {
        "dim": 2,
        "rhs": rhs_harmonic,
        "sample_params": sample_harmonic_params,
        "sample_initial": sample_harmonic_initial,
    },
    "damped_harmonic": {
        "dim": 2,
        "rhs": rhs_damped_harmonic,
        "sample_params": sample_damped_params,
        "sample_initial": sample_harmonic_initial,
    },
    "lotka_volterra": {
        "dim": 2,
        "rhs": rhs_lotka_volterra,
        "sample_params": sample_lv_params,
        "sample_initial": sample_lv_initial,
    },
    "lorenz": {
        "dim": 3,
        "rhs": rhs_lorenz,
        "sample_params": sample_lorenz_params,
        "sample_initial": sample_lorenz_initial,
    },

    ## ---------- tier-2 mechanical systems (positions first, then velocities) ----------

    "pendulum": {
        "dim": 2,
        "rhs": rhs_pendulum,
        "sample_params": sample_pendulum_params,
        "sample_initial": sample_pendulum_initial,
    },
    "double_pendulum": {
        "dim": 4,
        "rhs": rhs_double_pendulum,
        "sample_params": sample_double_pendulum_params,
        "sample_initial": sample_double_pendulum_initial,
    },
    "duffing": {
        "dim": 2,
        "rhs": rhs_duffing,
        "sample_params": sample_duffing_params,
        "sample_initial": sample_duffing_initial,
    },
    "spring_mass_chain": {
        "dim": 2 * SPRING_CHAIN_N,
        "rhs": rhs_spring_chain,
        "sample_params": sample_spring_chain_params,
        "sample_initial": sample_spring_chain_initial,
    },
    "nbody": {
        "dim": 4 * NBODY_N,
        "rhs": rhs_nbody,
        "sample_params": sample_nbody_params,
        "sample_initial": sample_nbody_initial,
    },

    ## ---------------- sdes ----------------
    # 1) exponential (OU)
    "exp_decay_sde": {
        "kind": "sde",
        "dim": 1,
        "drift": drift_exp_decay_sde,
        "diffusion": diffusion_exp_decay_sde,
        "sample_params": sample_exp_decay_sde_params,
        "sample_initial": sample_exp_decay_sde_initial,
        "dt_internal": None,
        "clamp": None,
    },

    # 2) damped harmonic (Langevin)
    "damped_harmonic_sde": {
        "kind": "sde",
        "dim": 2,
        "drift": drift_damped_harmonic_sde,
        "diffusion": diffusion_damped_harmonic_sde,
        "sample_params": sample_damped_harmonic_sde_params,
        "sample_initial": sample_damped_harmonic_sde_initial,
        "dt_internal": 0.005,
        "clamp": None,
    },

    # 3) lotka-volterra (additive; clamp to keep positive)
    "lotka_volterra_sde": {
        "kind": "sde",
        "dim": 2,
        "drift": rhs_lotka_volterra,          # reuse deterministic drift
        "diffusion": diffusion_lv_sde,        # additive noise
        "sample_params": sample_lv_sde_params,
        "sample_initial": sample_lv_initial,
        "dt_internal": 1e-2,
        "clamp": clamp_positive,
    },

    # 4) lorenz (additive; use smaller internal dt)
    "lorenz_sde": {
        "kind": "sde",
        "dim": 3,
        "drift": rhs_lorenz,                  # reuse deterministic drift
        "diffusion": diffusion_lorenz_sde,
        "sample_params": sample_lorenz_sde_params,
        "sample_initial": sample_lorenz_initial,
        "dt_internal": 1e-3,                  # important for stability
        "clamp": None,
    },

    ## misc higher D

    "hopperphysics": {
        "dim": 14
    },

    "pendulum_video":{
        "height": 84,
        "width": 84
    }
}

family_via_ivp= ["exp_decay", "logistic_growth", "harmonic_oscillator", "damped_harmonic", "lotka_volterra", "lorenz",
                 "pendulum", "double_pendulum", "duffing", "spring_mass_chain", "nbody"]
family_sde= ["exp_decay_sde", "damped_harmonic_sde", "lotka_volterra_sde", "lorenz_sde"]

def generate_family(family_name, num_param_configs, trajectories_per_config, times=[0.0, 10.0, 0.1], missing_prob=0.0, seed=None, mask_mode='per_dim'):
    if family_name not in FAMILIES:
        raise ValueError(f"Unknown family: {family_name}")

    spec = FAMILIES[family_name]
    rng = np.random.default_rng(seed)
    times = np.arange(times[0], times[1], times[2], dtype=np.float32)

    trajectories = []
    for cfg_idx in range(num_param_configs):

        if family_name in family_via_ivp:
            params = spec["sample_params"](rng)
            for _ in range(trajectories_per_config):
                y0 = spec["sample_initial"](rng, params)
                values = integrate_with_scipy(spec["rhs"], times, y0, params)
                # values, mask = apply_missingness(values.copy(), missing_prob, rng)
                mask = get_mask(values.copy(), missing_prob, rng, mask_mode=mask_mode)
                normalized_times= times/(times[-1]-times[0])
                trajectories.append(
                    {
                        "family": family_name,
                        "times": times.copy(),
                        "values": values,
                        "mask": mask.astype(np.float32),
                        "params": params,
                    }
                )
        elif family_name =='hopperphysics':
            state_dim_hopper= spec["dim"]
            num_samples= trajectories_per_config
            simulated_paths= generate_hopperphysics_trajectories(num_samples, T= len(times), D= state_dim_hopper)
            for i in range(trajectories_per_config):
                values = simulated_paths[i]
                # values, mask = apply_missingness(values.copy(), missing_prob, rng)
                mask = get_mask(values.copy(), missing_prob, rng, mask_mode=mask_mode)
                normalized_times= times/(times[-1]-times[0])
                trajectories.append(
                    {
                        "family": family_name,
                        "times": times.copy(),
                        "values": values,
                        "mask": mask.astype(np.float32),
                        "params": {'placeholder': 0},
                    }
                )
        elif family_name== "pendulum_video":
            height= spec["height"]
            width= spec["width"]
            num_samples= trajectories_per_config
            simulated_frames= generate_pendulum_video_trajectories(domain="pendulum", task="swingup", n_trajectories=num_samples, horizon=len(times), render_height=height, render_width=width, camera_id=0)
            for i in range(trajectories_per_config):
                values= simulated_frames[i]
                mask = get_mask(values.copy(), missing_prob, rng, mask_mode=mask_mode)
                trajectories.append(
                    {
                        "family": family_name,
                        "times": times.copy(),
                        "values": values,
                        "mask": mask.astype(np.float32),
                        "params": {'placeholder': 0},
                    }
                )
        
        elif family_name in family_sde:
            params = spec["sample_params"](rng)
            for _ in range(trajectories_per_config):
                y0 = spec["sample_initial"](rng, params)
                values = integrate_sde_em(spec["drift"], spec["diffusion"], times, y0, params, rng, dt_internal=spec["dt_internal"],clamp=spec["clamp"])
                # values, mask = apply_missingness(values.copy(), missing_prob, rng)
                mask = get_mask(values.copy(), missing_prob, rng, mask_mode=mask_mode)
                normalized_times= times/(times[-1]-times[0])
                trajectories.append(
                    {
                        "family": family_name,
                        "times": times.copy(),
                        "values": values,
                        "mask": mask.astype(np.float32),
                        "params": params,
                    }
                )
        
    return trajectories


def generate_linear_family(dim, num_param_configs, trajectories_per_config, times=None, missing_prob=0.0, seed=None):
    rng = np.random.default_rng(seed)
    if times is None:
        times = np.arange(0.0, 10.0, 0.1, dtype=np.float32)

    trajectories = []
    for cfg_idx in range(num_param_configs):
        params = sample_linear_params(rng, dim)
        for _ in range(trajectories_per_config):
            y0 = sample_linear_initial(rng, params)
            values = integrate_with_scipy(rhs_linear_nd, times, y0, params)
            values, mask = apply_missingness(values.copy(), missing_prob, rng)
            flat_a = {f"A_{i}_{j}": params["A"][i, j] for i in range(dim) for j in range(dim)}
            normalized_times= times/(times[-1]-times[0])
            trajectories.append(
                {
                    "family": f"linear_nd_{dim}",
                    "times": times.copy(),
                    "values": values,
                    "mask": mask.astype(np.float32),
                    "params": params
                }
            )
    return trajectories


def generate_synthetic_trajectories(family_requests, linear_requests=None, times=[0.0, 10.0, 0.1], missing_prob=0.0, seed=None, mask_mode='per_dim'):
    """
    family_requests = {
        "exp_decay": {"num_param_configs": 3, "trajectories_per_config": 5},
        "lorenz": {"num_param_configs": 2, "trajectories_per_config": 6},
        ...
    }
    linear_requests = [
        {"dim": 4, "num_param_configs": 3, "trajectories_per_config": 5},
        {"dim": 8, "num_param_configs": 2, "trajectories_per_config": 4},
    ]
    """
    trajectories = []
    base_seed = seed if seed is not None else np.random.SeedSequence().entropy
    seed_counter = 0

    for family, cfg in family_requests.items():
        local_seed = base_seed + seed_counter
        seed_counter += 1
        num_cfgs = cfg["num_param_configs"]
        per_cfg = cfg["trajectories_per_config"]
        trajectories.extend(
            generate_family(
                family,
                num_cfgs,
                per_cfg,
                times=times,
                missing_prob=missing_prob,
                seed=local_seed,
                mask_mode=mask_mode,
            )
        )

    if linear_requests:
        for item in linear_requests:
            local_seed = base_seed + seed_counter
            seed_counter += 1
            dim = item["dim"]
            num_cfgs = item["num_param_configs"]
            per_cfg = item["trajectories_per_config"]
            trajectories.extend(
                generate_linear_family(
                    dim,
                    num_cfgs,
                    per_cfg,
                    times=times,
                    missing_prob=missing_prob,
                    seed=local_seed,
                )
            )

    return trajectories





# # # ################################################################################ # example usage ################################################################################
# ##### quick plots for sanity check


# from pathlib import Path
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# def quick_plot(trajectories, max_traj=3, out_dir=None, dpi=150):
#     out_path = Path(out_dir) if out_dir is not None else None
#     if out_path is not None:
#         out_path.mkdir(parents=True, exist_ok=True)

#     families = {}
#     for traj in trajectories:
#         families.setdefault(traj["family"], []).append(traj)

#     for fam, fam_trajs in families.items():
#         sample = fam_trajs[0]["values"]
#         dim = sample.shape[1] if sample.ndim > 1 else 1

#         if dim == 3:
#             fig = plt.figure(figsize=(6, 6))
#             ax = fig.add_subplot(111, projection="3d")
#             for idx, traj in enumerate(fam_trajs[:max_traj]):
#                 vals = traj["values"]
#                 ax.plot(vals[:, 0], vals[:, 1], vals[:, 2], linewidth=1.0, label=f"{fam}[{idx}]")
#             ax.set_title(fam)
#             ax.set_xlabel("x")
#             ax.set_ylabel("y")
#             ax.set_zlabel("z")
#             ax.legend()
#         else:
#             fig, ax = plt.subplots(figsize=(8, 4))
#             for idx, traj in enumerate(fam_trajs[:max_traj]):
#                 t = traj["times"]
#                 vals = traj["values"]
#                 if vals.ndim == 1:
#                     vals = vals[:, None]
#                 for j in range(vals.shape[1]):
#                     ax.plot(t, vals[:, j], label=f"{fam}[{idx}] dim{j}")
#             ax.set_title(fam)
#             ax.set_xlabel("time")
#             ax.set_ylabel("value")
#             ax.legend()

#         fig.tight_layout()

#         if out_path is not None:
#             target = out_path / f"{fam}.png"
#             fig.savefig(target, dpi=dpi)
#         plt.show()

# def quick_plot_lorenz(trajectories, max_traj=25):
#     fig = plt.figure(figsize=(6, 6))
#     ax = fig.add_subplot(111, projection="3d")
#     for idx, traj in enumerate(traj for traj in trajectories if traj["family"] == "lorenz"):
#         if idx >= max_traj:
#             break
#         xyz = traj["values"]
#         ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], linewidth=1.0, label=f"lorenz[{idx}]")
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_zlabel("z")
#     ax.legend()
#     plt.tight_layout()
#     plt.show()
#     plt.savefig("synthetic_data_lorenz.png")


# # reqs = {
# #     "exp_decay": {"num_param_configs": 1, "trajectories_per_config": 3},
# # 'harmonic_oscillator': {"num_param_configs": 1, "trajectories_per_config": 3},
# # 'damped_harmonic': {"num_param_configs": 1, "trajectories_per_config": 3},
# # 'logistic_growth': {"num_param_configs": 1, "trajectories_per_config": 3},
# #     "lorenz": {"num_param_configs": 1, "trajectories_per_config": 3},
# # "lotka_volterra": {"num_param_configs": 1, "trajectories_per_config": 3},
# # }

# reqs = {
#     "exp_decay_sde": {"num_param_configs": 1, "trajectories_per_config": 1},
# 'damped_harmonic_sde': {"num_param_configs": 1, "trajectories_per_config": 1},
#     "lorenz_sde": {"num_param_configs": 1, "trajectories_per_config": 1},
# "lotka_volterra_sde": {"num_param_configs": 1, "trajectories_per_config": 1},
# }
# # reqs = {
# #     "exp_decay_sde": {"num_param_configs": 1, "trajectories_per_config": 1},
# # }
# # reqs = {
# #     "harmonic_oscillator": {"num_param_configs": 1, "trajectories_per_config": 1},
# # }

# # linear = [
# #     {"dim": 4, "num_param_configs": 2, "trajectories_per_config": 3},
# # ]


# times = np.arange(-50, 50.0, 0.1, dtype=np.float32)
# # traj_list = generate_synthetic_trajectories(reqs, linear_requests=linear, missing_prob=0.1, seed=42)
# # traj_list = generate_synthetic_trajectories(reqs, times=times, missing_prob=0, seed=42)
# traj_list = generate_synthetic_trajectories(reqs, times=[-10, 10, 0.1], missing_prob=0, seed=42)
# # traj_list = generate_synthetic_trajectories(reqs, times=[-10, 10, 0.1], missing_prob=0.4, seed=42)

# quick_plot(traj_list, out_dir="synthetic_plots")
# # quick_plot(traj_list, out_dir="synthetic_plots_irregular_missing") ## our missingness now happens in the class IrregularDataset

# # quick_plot_lorenz(traj_list)

# import pdb; pdb.set_trace()