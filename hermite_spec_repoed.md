# Hermite Conditional Paths — Implementation Brief

Task: add two new interpolants to the SplineFlow codebase. Both use observed
velocity to constrain the position interpolant, instead of fitting every state
dimension independently.

Repo: `https://github.com/santanurathod/SplineFlow`

---

## 0. Setup (do this first, nothing runs without it)

The repo has hardcoded absolute paths from the authors' cluster. Fix both:

- `src/helpers.py:14` — `/home/c02sara/CISPA-az6/dgdlm-2024/higher_interpolant_matching/configs/model_configs/{model_config}.json`
- `main.py:232` — results `base` path
- `evaluation.py:64` — `root` path

Replace with paths relative to the repo root. Verify a baseline run works before
touching anything else:

```bash
python main.py --data_config harmonic_oscillator --interpolant_kind bspline --degree 3 --exp_name smoke --epochs 100
```

---

## 1. Background: what the code currently does

`src/interpolants.py::bspline_interpolant_values(values, times, mask, degree,
subsample_per_interval, dynamics_kind, sigma)`

- `values`: `[B, T, D]`, `times`: `[B, T]`, `mask`: `[B, T, D]`
- For each trajectory `b` and each dimension `d` **independently**: take the
  observed times `t_obs = times[b][mask[b,:,d]]`, fit
  `make_interp_spline(t_obs, values[b, obs, d], k=degree)`, evaluate on a fine
  grid `t_fine`, and evaluate its first derivative.
- Returns `xt` (= `mu_t`), `vt` (= `mu_dot_t`), `tt`, plus SDE terms
  `sigma_t, lambda_t, der_sigma_t`.

Dispatch is in `src/helpers.py::get_flow_matching_inputs`, which already has an
unimplemented `"cubic"` branch. That is where the new kinds go.

### Two facts that matter

**Time is physical, not normalized.** `t_fine = np.linspace(times[b][0],
times[b][-1], ...)` and splines are fit on raw `t_obs`. So the tangent passed to
a Hermite fit is the observed velocity `v` directly — **no time-rescaling
factor**. (Z-score normalization of values/times/targets happens later, in
`data_preprocessing/dataloader.py::ProcessedDataset.__getitem__`, after
interpolation.)

**Masks are per-dimension and independent.** `get_mask` in
`data_preprocessing/synthetic_data.py:131` draws `rng.uniform(size=values.shape)
< prob` over the full `[T, D]` array. So for `missing_prob > 0`, the position
dim and its velocity dim of the same degree of freedom are dropped
independently. Handling this is §4 — it is the main implementation problem.

---

## 2. What to build

State dimensions come in pairs `(p, v)` where dim `v` is the time-derivative of
dim `p`. Dimensions in no pair are `UNPAIRED`.

### Version A — "hedge", `d` fits

| slot | `mu(t)` | `mu_dot(t)` |
|---|---|---|
| position `p` | `H[p](t)` | `H[p].derivative(1)(t)` |
| velocity `v` | `B[v](t)` | `B[v].derivative(1)(t)` |
| unpaired `u` | `B[u](t)` | `B[u].derivative(1)(t)` |

where `H[p] = CubicHermiteSpline(t_obs, values[:,p], values[:,v])` and
`B[·] = make_interp_spline(t_obs, values[:,·], k=degree)`.

Position fit is velocity-aware. Velocity fit is a plain B-spline, so the
acceleration target stays smooth. Trade-off: `mu_dot[p]` and `mu[v]` are no
longer exactly equal away from the knots.

### Version B — "pure", `d/2` fits

| slot | `mu(t)` | `mu_dot(t)` |
|---|---|---|
| position `p` | `H[p](t)` | `H[p].derivative(1)(t)` |
| velocity `v` | `H[p].derivative(1)(t)` | `H[p].derivative(2)(t)` |
| unpaired `u` | `B[u](t)` | `B[u].derivative(1)(t)` |

One curve per degree of freedom, fully self-consistent. Trade-off:
`mu_dot[v]` is piecewise linear and discontinuous at every knot, because cubic
Hermite is only C¹. That discontinuity lands on the acceleration target.

---

## 3. Where the code goes

**`src/interpolants.py`** — add `hermite_interpolant_values(...)` with the same
signature as `bspline_interpolant_values` plus `pairs`, `unpaired`, and
`mode ∈ {"hedge", "pure"}`. Same return tuple.

Copy the structure of `bspline_interpolant_values` and keep its existing edge-case
handling verbatim: empty `dim_idx`, single-observation dims, `k = min(degree,
n_obs - 1)`, clamped flat extrapolation outside `[t_obs_min, t_obs_max]`, and the
`get_velocity_score_and_noise(...)` call for the SDE terms. Only the fitting and
derivative-evaluation lines change.

**`src/helpers.py::get_flow_matching_inputs`** — add two branches:

```python
elif interpolant_kind in ("hermite_hedge", "hermite_pure"):
    xt, vt, t, sigma_t, lambda_t, der_sigma_t = hermite_interpolant_values(
        values, times, mask,
        subsample_per_interval=subsample_per_interval,
        degree=degree,
        dynamics_kind=dynamics_kind,
        sigma=sigma,
        pairs=pairs,
        unpaired=unpaired,
        mode="hedge" if interpolant_kind == "hermite_hedge" else "pure",
    )
```

`pairs`/`unpaired` need to reach this function. Thread them through
`ProcessedDataset.__init__` → `get_flow_matching_inputs`, sourced from a new
`--pairs` CLI arg in `main.py` (format: `"0:1,2:3"`; empty means no pairs).

**`main.py`** — add `--pairs`, and pass through to both `ProcessedDataset`
constructions (lines ~268-269).

### Pair maps

- `harmonic_oscillator`, `damped_harmonic`: state is `(x, v)` → `--pairs 0:1`
- `hopperphysics`: **inspect the actual state layout before assuming.** Read
  `data_preprocessing/synthetic_data.py` / `realworld_data.py` for how the
  Hopper state is assembled, confirm which indices are `q` and which are `q_dot`,
  and report the finding before running anything on it.

---

## 4. The mask problem

Clamping needs position *and* velocity observed at the same time. With
per-dimension independent masks and `missing_prob > 0`, that fails often.

**Default strategy — intersect.** Fit the Hermite curve only at times where both
`mask[b,t,p]` and `mask[b,t,v]` are true. If fewer than 2 such times exist for a
pair, fall back to independent B-splines for that pair and count the fallback.

**Log the fallback rate per config.** If it is high at `missing_prob=0.75`, the
comparison is no longer testing the idea — it is testing the fallback. This
number goes in the results table.

**Add `--mask_mode {per_dim, per_dof}`.** `per_dim` is the existing behaviour.
`per_dof` modifies `get_mask` so that a dropped observation removes the whole
degree of freedom (both `p` and `v` together), which models a sensor reading
returning the full state. Implement it, but note clearly in results that
`per_dof` numbers are **not** comparable to the paper's table.

---

## 5. Validation gate — run before any training

Standalone script, `scripts/check_interpolant_derivative.py`. No network.

For the damped harmonic oscillator, ground truth is
`x_dot = v`, `v_dot = -omega^2 * x - 2*gamma*v` (see
`data_preprocessing/synthetic_data.py` for the parameter sampling).

```
for missing_prob in [0, 0.25, 0.5, 0.75]:
    for method in [hedge, pure, bspline(k=1..5)]:
        fit on the masked trajectory
        evaluate mu_dot on a dense grid
        report mean and max |error| vs truth, SEPARATELY for the
          position slot and the velocity slot
```

Report position and velocity slots separately — the two versions are expected to
differ mainly in the velocity (acceleration) slot, and a combined number hides
that.

**Gate:** if hedge and pure do not beat the B-splines on derivative error here,
stop and report. Nothing downstream will work.

---

## 6. Assertions

```python
# both versions: interpolant passes through observed points
assert np.allclose(mu(t_obs_i), values[i], atol=1e-10)

# pure version: velocity slot reproduces the observed velocity at knots.
# failure here means a time-scaling factor was introduced somewhere.
assert np.allclose(mu(t_obs_i)[v], values[i, v], atol=1e-10)

# hedge version: position slot derivative equals observed velocity at knots
assert np.allclose(mu_dot(t_obs_i)[p], values[i, v], atol=1e-10)
```

---

## 7. Experiments

Only run on systems whose state contains velocity. Exp-Decay, Lotka-Volterra and
Lorenz have no velocity dimension — skip them.

Config name → `missing_prob`: base `= 0`, `_sparse` `= 0.25`, `_v_sparse`
`= 0.5`, `_vv_sparse` `= 0.75`.

```bash
for cfg in damped_harmonic damped_harmonic_sparse damped_harmonic_v_sparse damped_harmonic_vv_sparse; do
  for kind in hermite_hedge hermite_pure; do
    python main.py --data_config $cfg --interpolant_kind $kind --pairs 0:1 --exp_name hermite
  done
  python main.py --data_config $cfg --interpolant_kind bspline --degree 3 --exp_name hermite
  python main.py --data_config $cfg --interpolant_kind linear --exp_name hermite
done
```

Then the same for `harmonic_oscillator*`. Hopper last, and only after the state
layout is confirmed.

### Numbers to beat (MSE, from the paper)

| System | SplineFlow p=0 | p=0.75 | TFM p=0 |
|---|---|---|---|
| Harmonic Oscillator | 3.7e-4 | 1.1e-3 | 0.181 |
| Damped Harmonic | 9.5e-5 | 4.8e-5 | 0.015 |
| HopperPhysics | 1.410 | 3.336 | 1.554 |

Start with damped harmonic at `_vv_sparse`.

---

## 8. Scope

Do §0, §5 and report back before implementing §2. Do not start §7 until the §5
gate passes. Do not modify the training loop, the MLP, the loss, the noise
schedule, or the SDE branch — the only changes are the new interpolant
functions, the dispatch, and the CLI plumbing.
