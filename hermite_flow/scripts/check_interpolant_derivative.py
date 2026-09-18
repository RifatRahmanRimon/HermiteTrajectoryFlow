#!/usr/bin/env python3
"""
Section 5 validation gate for the Hermite conditional-path interpolants.

Standalone: numpy + scipy only, no network, no training. It measures how well
each interpolant's derivative (mu_dot) matches the *true* time-derivative of the
damped harmonic oscillator, separately for the position slot and the velocity
slot, across missing-data rates.

Ground truth (damped harmonic, state = [x, v]):
    x_dot = v
    v_dot = -omega^2 * x - 2 * gamma * v
(see data_preprocessing/synthetic_data.py::rhs_damped_harmonic).

The interpolant fitting here is the reference implementation that
src/interpolants.py::hermite_interpolant_values reuses (physical time, no
time-rescaling of the observed velocity tangent).

Gate: hedge and pure must beat the B-splines on derivative error, otherwise the
downstream idea does not work.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import make_interp_spline, CubicHermiteSpline

# Make the repo importable so we reuse the *exact* dynamics + mask logic.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from data_preprocessing.synthetic_data import (  # noqa: E402
    rhs_damped_harmonic,
    sample_damped_params,
    sample_harmonic_initial,
    integrate_with_scipy,
    get_mask,
)


# --------------------------------------------------------------------------- #
# Interpolant fitting (reference implementation, physical time)               #
# --------------------------------------------------------------------------- #
def _eval_clamped(spl, deriv_order, t_obs, t_eval, end_values):
    """Evaluate a scipy spline (or its derivative) on t_eval with flat
    extrapolation outside the observed range [t_obs[0], t_obs[-1]], matching
    bspline_interpolant_values. Returns the evaluated array."""
    tmin, tmax = t_obs[0], t_obs[-1]
    t_clip = np.clip(t_eval, tmin, tmax)
    d = spl.derivative(deriv_order) if deriv_order > 0 else spl
    vals = d(t_clip)
    left = t_eval < tmin
    right = t_eval > tmax
    if deriv_order == 0:
        vals[left] = end_values[0]
        vals[right] = end_values[1]
    else:
        # outside the observed range the curve is flat -> derivative 0
        vals[left] = 0.0
        vals[right] = 0.0
    return vals


def _fit_bspline(t_obs, y_obs, degree, t_eval):
    """Plain B-spline fit for one dimension. Returns (mu, mu_dot)."""
    n = t_obs.size
    if n == 0:
        return np.zeros_like(t_eval), np.zeros_like(t_eval)
    if n == 1:
        return np.full_like(t_eval, y_obs[0]), np.zeros_like(t_eval)
    k = min(degree, n - 1)
    spl = make_interp_spline(t_obs, y_obs, k=k)
    mu = _eval_clamped(spl, 0, t_obs, t_eval, (y_obs[0], y_obs[-1]))
    mu_dot = _eval_clamped(spl, 1, t_obs, t_eval, None)
    return mu, mu_dot


def _fit_hermite(t_obs, xp_obs, xv_obs, t_eval, mode):
    """Cubic Hermite fit for a (position, velocity) pair on their common
    observed times. Returns (mu_p, mu_dot_p, mu_v, mu_dot_v)."""
    H = CubicHermiteSpline(t_obs, xp_obs, xv_obs)
    mu_p = _eval_clamped(H, 0, t_obs, t_eval, (xp_obs[0], xp_obs[-1]))
    mu_dot_p = _eval_clamped(H, 1, t_obs, t_eval, None)
    if mode == "hedge":
        # velocity slot is a plain B-spline of the observed velocity
        # (fit here on the intersection times; the repo function fits it on
        #  all v-observed times, but for the gate both p and v use I_pv).
        mu_v, mu_dot_v = _fit_bspline(t_obs, xv_obs, degree=3, t_eval=t_eval)
    elif mode == "pure":
        # velocity slot IS the derivative of the position Hermite curve
        mu_v = _eval_clamped(H, 1, t_obs, t_eval, None)
        mu_dot_v = _eval_clamped(H, 2, t_obs, t_eval, None)
    else:
        raise ValueError(mode)
    return mu_p, mu_dot_p, mu_v, mu_dot_v


def fit_trajectory(values, times, mask, pairs, unpaired, degree, method, t_eval):
    """Fit one masked trajectory and return mu, mu_dot on t_eval, plus the
    number of pairs that fell back to independent B-splines.

    method: "hedge", "pure", or "bspline" (degree given by `degree`).
    """
    D = values.shape[1]
    M = t_eval.size
    mu = np.zeros((M, D))
    mu_dot = np.zeros((M, D))
    fallback = 0

    if method == "bspline":
        for d in range(D):
            obs = mask[:, d].astype(bool)
            mu[:, d], mu_dot[:, d] = _fit_bspline(times[obs], values[obs, d], degree, t_eval)
        return mu, mu_dot, fallback

    # hermite hedge / pure
    for (p, v) in pairs:
        obs_p = mask[:, p].astype(bool)
        obs_v = mask[:, v].astype(bool)
        both = obs_p & obs_v
        if both.sum() >= 2:
            t_pv = times[both]
            mu[:, p], mu_dot[:, p], mu[:, v], mu_dot[:, v] = _fit_hermite(
                t_pv, values[both, p], values[both, v], t_eval, method
            )
        else:
            # fallback: independent B-splines for both dims
            fallback += 1
            mu[:, p], mu_dot[:, p] = _fit_bspline(times[obs_p], values[obs_p, p], degree, t_eval)
            mu[:, v], mu_dot[:, v] = _fit_bspline(times[obs_v], values[obs_v, v], degree, t_eval)

    for u in unpaired:
        obs = mask[:, u].astype(bool)
        mu[:, u], mu_dot[:, u] = _fit_bspline(times[obs], values[obs, u], degree, t_eval)

    return mu, mu_dot, fallback


# --------------------------------------------------------------------------- #
# Ground-truth trajectory generation                                          #
# --------------------------------------------------------------------------- #
def make_trajectory(rng_seed, obs_times, dense_times):
    """Return (obs_values[T,2], params, x_dense, v_dense) for one damped
    harmonic trajectory sampled with the repo's own samplers."""
    rng = np.random.default_rng(rng_seed)
    params = sample_damped_params(rng)
    y0 = sample_harmonic_initial(rng, params)
    obs_values = integrate_with_scipy(rhs_damped_harmonic, obs_times, y0, params)
    dense_values = integrate_with_scipy(rhs_damped_harmonic, dense_times, y0, params)
    return obs_values, params, dense_values[:, 0], dense_values[:, 1]


def true_derivatives(x, v, params):
    """True mu_dot targets on the dense grid: position slot and velocity slot."""
    omega, gamma = params["omega"], params["gamma"]
    x_dot = v                                    # position slot target
    v_dot = -omega**2 * x - 2 * gamma * v        # velocity slot target
    return x_dot, v_dot


def draw_mask(values, prob, rng, mask_mode):
    """per_dim: repo's independent per-(t,d) missingness (get_mask).
    per_dof: one Bernoulli per time step, applied to the whole degree of
    freedom (both dims dropped together), modelling a full-state sensor read."""
    if mask_mode == "per_dim":
        return get_mask(values.copy(), prob, rng)
    T, D = values.shape
    mask = np.ones((T, D), dtype=bool)
    if prob <= 0:
        return mask
    drop = rng.uniform(size=T) < prob
    drop[0] = False  # keep_first
    mask[drop, :] = False
    return mask


# --------------------------------------------------------------------------- #
# Assertions (Section 6)                                                       #
# --------------------------------------------------------------------------- #
def run_assertions():
    """Sec 6 assertions on a fully-observed trajectory."""
    obs_times = np.arange(0.0, 10.0, 0.05, dtype=np.float64)
    values, params, _, _ = make_trajectory(0, obs_times, obs_times)
    T = obs_times.size
    mask = np.ones((T, 2), dtype=bool)
    p, v = 0, 1

    for mode in ("hedge", "pure"):
        mu, mu_dot, fb = fit_trajectory(values, obs_times, mask, [(p, v)], [], 3, mode, obs_times)
        # interpolant passes through observed points (position slot)
        assert np.allclose(mu[:, p], values[:, p], atol=1e-9), f"{mode}: mu[p] mismatch"
        if mode == "pure":
            # velocity slot reproduces observed velocity at knots (no time-scaling)
            assert np.allclose(mu[:, v], values[:, v], atol=1e-9), f"{mode}: mu[v] mismatch"
        else:
            # hedge: position-slot derivative equals observed velocity at knots
            assert np.allclose(mu_dot[:, p], values[:, v], atol=1e-9), f"{mode}: mu_dot[p] mismatch"
    print("[assertions] Section 6 assertions passed (hedge + pure).")


# --------------------------------------------------------------------------- #
# Main sweep                                                                   #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_traj", type=int, default=100, help="trajectories averaged per cell")
    ap.add_argument("--n_dense", type=int, default=2000, help="dense eval grid size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mask_mode", choices=["per_dim", "per_dof", "both"], default="both")
    args = ap.parse_args()

    obs_times = np.arange(0.0, 10.0, 0.05, dtype=np.float64)         # repo grid: 200 pts
    dense_times = np.linspace(0.0, 10.0, args.n_dense, dtype=np.float64)
    missing_probs = [0.0, 0.25, 0.5, 0.75]
    pairs = [(0, 1)]
    unpaired = []

    methods = [("hedge", None), ("pure", None)] + [("bspline", k) for k in range(1, 6)]

    run_assertions()

    mask_modes = ["per_dim", "per_dof"] if args.mask_mode == "both" else [args.mask_mode]
    for mask_mode in mask_modes:
        sweep_one_mask_mode(mask_mode, methods, missing_probs, pairs, unpaired,
                            obs_times, dense_times, args)


def sweep_one_mask_mode(mask_mode, methods, missing_probs, pairs, unpaired,
                        obs_times, dense_times, args):
    print()
    print("#" * 85)
    print(f"# MASK MODE: {mask_mode}"
          + ("   (per-dim independent -- comparable to paper's table)" if mask_mode == "per_dim"
             else "   (full degree-of-freedom drops -- NOT comparable to paper's table)"))
    print("#" * 85)
    header = f"{'p_miss':>7} {'method':>10} {'|err| pos mean':>16} {'pos max':>10} " \
             f"{'|err| vel mean':>16} {'vel max':>10} {'fallback%':>10}"
    print(header)
    print("-" * len(header))

    summary = {}
    for p_miss in missing_probs:
        for (method, k) in methods:
            degree = k if method == "bspline" else 3
            pos_abs_all, vel_abs_all = [], []
            fallback_pairs = 0
            total_pairs = 0
            for i in range(args.n_traj):
                seed_i = args.seed + i
                values, params, x_dense, v_dense = make_trajectory(seed_i, obs_times, dense_times)
                mrng = np.random.default_rng(10_000 + seed_i)
                mask = draw_mask(values, p_miss, mrng, mask_mode)
                mu, mu_dot, fb = fit_trajectory(
                    values, obs_times, mask, pairs, unpaired, degree, method, dense_times
                )
                x_dot_true, v_dot_true = true_derivatives(x_dense, v_dense, params)
                pos_abs_all.append(np.abs(mu_dot[:, 0] - x_dot_true))
                vel_abs_all.append(np.abs(mu_dot[:, 1] - v_dot_true))
                fallback_pairs += fb
                total_pairs += len(pairs)

            pos_abs = np.concatenate(pos_abs_all)
            vel_abs = np.concatenate(vel_abs_all)
            fb_rate = 100.0 * fallback_pairs / max(total_pairs, 1)
            label = method if method != "bspline" else f"bspline{k}"
            print(f"{p_miss:>7.2f} {label:>10} {pos_abs.mean():>16.4e} {pos_abs.max():>10.3e} "
                  f"{vel_abs.mean():>16.4e} {vel_abs.max():>10.3e} {fb_rate:>9.1f}%")
            summary[(p_miss, label)] = (pos_abs.mean(), vel_abs.mean())
        print()

    # --- Gate verdict --------------------------------------------------------
    # R3: cubic Hermite is fixed at cubic, so the honest test is vs the CUBIC
    # B-spline (bspline3). We also show vs the best B-spline of any degree.
    print(f"GATE ({mask_mode}): position-slot mean |error|  (the slot the idea can affect)")
    print("-" * 78)
    print(f"  {'p_miss':>6} | {'hedge':>10} {'pure':>10} | {'bspline3':>10} (cubic) | "
          f"{'best bspl':>10} | matched-degree verdict")
    for p_miss in missing_probs:
        hpos = summary[(p_miss, 'hedge')][0]
        ppos = summary[(p_miss, 'pure')][0]
        b3 = summary[(p_miss, 'bspline3')][0]
        bbest = min(summary[(p_miss, f"bspline{k}")][0] for k in range(1, 6))
        verdict = "hedge/pure BEAT cubic" if min(hpos, ppos) <= b3 + 1e-12 else "cubic bspline wins"
        print(f"  {p_miss:>6.2f} | {hpos:>10.3e} {ppos:>10.3e} | {b3:>10.3e}        | "
              f"{bbest:>10.3e} | {verdict}")
    print()
    print("  Reading: position slot is velocity-aware for hedge/pure. Matched-degree test")
    print("  is hedge/pure (cubic Hermite) vs bspline3 (cubic). Higher-degree B-splines")
    print("  (k=4,5) approximate this very smooth ODE better but are a degree mismatch (R3).")
    print("  hedge's velocity slot == a cubic B-spline of v by construction; pure's velocity")
    print("  slot is H'' (piecewise-linear, discontinuous accel), so pure is worse there.")


if __name__ == "__main__":
    main()
