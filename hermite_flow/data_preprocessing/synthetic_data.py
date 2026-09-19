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


# ======================================================================= #
# Second-order mechanical systems (position + velocity paired).            #
# State layout is positions-first then velocities, matching the (x, v) and #
# hopper (q, q_dot) convention, so the (p, v) pairing is p -> p + n_dof.    #
#   pendulum        state (theta, theta_dot)                 pairs 0:1      #
#   duffing         state (x, x_dot)                         pairs 0:1      #
#   double_pendulum state (th1, th2, w1, w2)                 pairs 0:2,1:3  #
#   spring_mass     state (x1..x3, v1..v3)                   pairs 0:3,1:4,2:5
#   n_body (2D,3)   state (x1,y1,..,x3,y3, vx1,vy1,..)       pairs 0:6..5:11
# ======================================================================= #

## nonlinear pendulum (state-space): theta'' = -omega0^2 sin(theta) - 2 gamma theta'
def rhs_pendulum(t, state, params):
    theta, omega = state
    omega0 = params["omega0"]
    gamma = params["gamma"]
    return np.array([omega, -omega0**2 * np.sin(theta) - 2.0 * gamma * omega])


def sample_pendulum_params(rng):
    return {"omega0": rng.uniform(0.8, 2.0), "gamma": rng.uniform(0.0, 0.15)}


def sample_pendulum_initial(rng, params):
    """Wide initial angle (to exercise the large-angle regime) but capped BELOW
    the separatrix.

    Drawing theta and omega independently lets the bob go over the top: measured
    over 3000 draws the uncapped version circulated 0.47% of the time, reaching
    |theta| = 9.3 rad. A circulating theta is an unbounded ramp, not an
    oscillation, so its interpolation error says nothing about the method.
    Draw theta first, then spend at most 90% of the separatrix energy
    E_sep = 2 * omega0^2 on the remaining kinetic term.
    """
    w2 = params["omega0"] ** 2
    theta = rng.uniform(-2.2, 2.2)
    e_budget = 0.9 * 2.0 * w2 - w2 * (1.0 - np.cos(theta))
    omega_max = np.sqrt(max(2.0 * e_budget, 0.0))
    return np.array([theta, rng.uniform(-omega_max, omega_max)])


## Duffing oscillator: x'' = -delta x' - alpha x - beta x^3  (alpha<0 -> double well)
def rhs_duffing(t, state, params):
    x, v = state
    delta, alpha, beta = params["delta"], params["alpha"], params["beta"]
    return np.array([v, -delta * v - alpha * x - beta * x**3])


def sample_duffing_params(rng):
    return {"delta": rng.uniform(0.05, 0.30),
            "alpha": rng.uniform(-1.0, 1.0),
            "beta": rng.uniform(0.2, 1.0)}


def sample_duffing_initial(rng, params):
    return np.array([rng.uniform(-1.5, 1.5), rng.uniform(-1.0, 1.0)])


## double pendulum (chaotic). state = [th1, th2, w1, w2]
def rhs_double_pendulum(t, state, params):
    th1, th2, w1, w2 = state
    G = params["g"]; L1 = params["L1"]; L2 = params["L2"]
    M1 = params["m1"]; M2 = params["m2"]
    d = th2 - th1
    cd, sd = np.cos(d), np.sin(d)
    den1 = (M1 + M2) * L1 - M2 * L1 * cd * cd
    a1 = (M2 * L1 * w1 * w1 * sd * cd
          + M2 * G * np.sin(th2) * cd
          + M2 * L2 * w2 * w2 * sd
          - (M1 + M2) * G * np.sin(th1)) / den1
    den2 = (L2 / L1) * den1
    a2 = (-M2 * L2 * w2 * w2 * sd * cd
          + (M1 + M2) * G * np.sin(th1) * cd
          - (M1 + M2) * L1 * w1 * w1 * sd
          - (M1 + M2) * G * np.sin(th2)) / den2
    return np.array([w1, w2, a1, a2])


def sample_double_pendulum_params(rng):
    return {"g": 9.81,
            "L1": rng.uniform(0.8, 1.2), "L2": rng.uniform(0.8, 1.2),
            "m1": rng.uniform(0.8, 1.2), "m2": rng.uniform(0.8, 1.2)}


def sample_double_pendulum_initial(rng, params):
    # moderate angles, zero initial angular velocity
    return np.array([rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), 0.0, 0.0])


## coupled spring-mass chain (linear, multi-DOF), fixed walls at both ends.
## n masses, identical spring constant k and mass m. state = [x_1..x_n, v_1..v_n].
SPRING_MASS_N = 3

def rhs_spring_mass(t, state, params):
    n = params["n"]; k = params["k"]; m = params["m"]
    x = state[:n]; v = state[n:]
    acc = np.zeros(n)
    for i in range(n):
        left = x[i - 1] if i > 0 else 0.0
        right = x[i + 1] if i < n - 1 else 0.0
        acc[i] = (k * (left - x[i]) + k * (right - x[i])) / m
    return np.concatenate([v, acc])


def sample_spring_mass_params(rng):
    return {"n": SPRING_MASS_N, "k": rng.uniform(0.5, 2.0), "m": rng.uniform(0.8, 1.2)}


def sample_spring_mass_initial(rng, params):
    n = params["n"]
    return np.concatenate([rng.uniform(-1.0, 1.0, size=n),
                           rng.uniform(-0.5, 0.5, size=n)])


## gravitational N-body in 2D (softened). state = [pos(2N) ... , vel(2N) ...].
## masses stored as scalar keys m0..m{N-1} so trajectory params stay collatable.
NBODY_N = 3

def rhs_nbody(t, state, params):
    N = params["N"]; G = params["G"]; eps = params["eps"]
    masses = np.array([params[f"m{j}"] for j in range(N)])
    pos = state[:2 * N].reshape(N, 2)
    vel = state[2 * N:].reshape(N, 2)
    acc = np.zeros((N, 2))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            diff = pos[j] - pos[i]
            r2 = diff[0] * diff[0] + diff[1] * diff[1] + eps * eps
            acc[i] += G * masses[j] * diff / (r2 * np.sqrt(r2))
    return np.concatenate([vel.reshape(-1), acc.reshape(-1)])


def sample_nbody_params(rng):
    N = NBODY_N
    p = {"N": N, "G": 1.0, "eps": 0.2}
    for j in range(N):
        p[f"m{j}"] = float(rng.uniform(0.8, 1.2))
    return p


def sample_nbody_initial(rng, params):
    """A jittered rotating ring.

    Uniform-random positions let two bodies start arbitrarily close, producing a
    near-singular acceleration that ejects a body: measured over 600 draws the
    uniform version sent |state| past 10 in 29.3% of cases, peaking at 39.4. An
    escaping body is an unbounded ramp and dominates the z-score statistics for
    the whole dataset.

    Instead start near a regular N-gon -- the relative equilibrium of the
    equal-mass problem (the Lagrange triangle for N=3) -- at its circular speed

        v^2 = G <m> / (4 r) * sum_{j=1}^{N-1} 1 / sin(pi j / N)

    with the barycentre pinned at the origin and net momentum removed (both
    mass-weighted, since the m_j here differ). The jitter keeps the orbits
    quasi-periodic rather than exactly periodic while every body stays bound.
    """
    N = params["N"]
    masses = np.array([params[f"m{j}"] for j in range(N)])
    total_mass = masses.sum()

    r0 = rng.uniform(1.0, 1.4)
    angle = 2.0 * np.pi * np.arange(N) / N + rng.uniform(-0.08, 0.08, size=N)
    radius = r0 * (1.0 + rng.uniform(-0.08, 0.08, size=N))
    pos = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    pos -= (masses[:, None] * pos).sum(axis=0) / total_mass

    j = np.arange(1, N)
    ring_sum = np.sum(1.0 / np.sin(np.pi * j / N))
    v_circ = np.sqrt(params["G"] * (total_mass / N) * ring_sum / (4.0 * r0))

    r = np.maximum(np.linalg.norm(pos, axis=1, keepdims=True), 1e-3)
    tangent = np.stack([-pos[:, 1], pos[:, 0]], axis=1) / r
    vel = tangent * v_circ * rng.uniform(0.92, 1.08, size=(N, 1))
    vel -= (masses[:, None] * vel).sum(axis=0) / total_mass

    return np.concatenate([pos.reshape(-1), vel.reshape(-1)])


### hopper physics
def generate_hopperphysics_trajectories(n_samples, T= 200, D= 14, seed= 123):
		"""Roll out the dm_control Hopper. State layout is positions-first then
		velocities: data[..., :nq] = qpos, data[..., nq:] = qvel.

		Returns (data, times) where `times` is the PHYSICAL time grid in seconds
		(spacing = physics.timestep(), one MuJoCo step per frame). This matters for
		the Hermite interpolants: qvel is d(qpos)/dt in physical seconds, so the
		stored times must be physical for qvel to be the correct tangent of qpos.
		Using a step-index grid (e.g. 0,1,2,...) would put a spurious 1/dt scale on
		the pairing. Hopper joints are all 1-DOF (slide/hinge) so nq == nv and
		qvel[i] is exactly the derivative of qpos[i]."""

		from dm_control import suite  # lazy import; MuJoCo dependency only for Hopper
		env = suite.load('hopper', 'stand')
		physics = env.physics

		nq, nv = int(physics.model.nq), int(physics.model.nv)
		assert nq == D // 2 and nv == D // 2, (
			f"Hopper state layout mismatch: nq={nq}, nv={nv}, but D//2={D // 2}. "
			f"Set D=nq+nv and update the --pairs mapping (p:p+nq for p in range(nq)).")

		dt = float(physics.timestep())              # physical seconds per physics.step()
		times = (np.arange(T, dtype=np.float32) * dt)

		# Store the state of the RNG to restore later; seed initial conditions
		# reproducibly from the caller's seed.
		st0 = np.random.get_state()
		np.random.seed(seed)

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
		return data, times


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

    # ---------- second-order mechanical (position + velocity) ----------
    "pendulum": {
        "dim": 2,
        "rhs": rhs_pendulum,
        "sample_params": sample_pendulum_params,
        "sample_initial": sample_pendulum_initial,
    },
    "duffing": {
        "dim": 2,
        "rhs": rhs_duffing,
        "sample_params": sample_duffing_params,
        "sample_initial": sample_duffing_initial,
    },
    "double_pendulum": {
        "dim": 4,
        "rhs": rhs_double_pendulum,
        "sample_params": sample_double_pendulum_params,
        "sample_initial": sample_double_pendulum_initial,
    },
    "spring_mass": {
        "dim": 2 * SPRING_MASS_N,
        "rhs": rhs_spring_mass,
        "sample_params": sample_spring_mass_params,
        "sample_initial": sample_spring_mass_initial,
    },
    "n_body": {
        "dim": 4 * NBODY_N,
        "rhs": rhs_nbody,
        "sample_params": sample_nbody_params,
        "sample_initial": sample_nbody_initial,
    },

    # ---------------- sdes ----------------
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

family_via_ivp= ["exp_decay", "logistic_growth", "harmonic_oscillator", "damped_harmonic", "lotka_volterra", "lorenz", "pendulum", "duffing", "double_pendulum", "spring_mass", "n_body"]
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
            # hopper_times is the PHYSICAL time grid (seconds); we store that, not the
            # config's step-index grid, so qvel is the correct tangent of qpos.
            simulated_paths, hopper_times = generate_hopperphysics_trajectories(
                num_samples, T= len(times), D= state_dim_hopper,
                seed= seed if seed is not None else 123)
            for i in range(trajectories_per_config):
                values = simulated_paths[i]
                # values, mask = apply_missingness(values.copy(), missing_prob, rng)
                mask = get_mask(values.copy(), missing_prob, rng, mask_mode=mask_mode)
                trajectories.append(
                    {
                        "family": family_name,
                        "times": hopper_times.copy(),
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