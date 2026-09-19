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

  > **RESOLVED (commit 98a82ec) — Hopper seed and time grid.**
  > Previously `generate_hopperphysics_trajectories` hardcoded `np.random.seed(123)`
  > and ignored the config `seed`. It now takes `seed=` and `generate_family`
  > threads the config value through, so Hopper behaves like every other family.
  > The same commit fixed a subtler bug worth recording: Hopper's config asks for
  > `times: [0, 200, 1]`, a **step-index** grid, but `qvel` is `d(qpos)/dt` in
  > physical seconds. Storing the index grid would have put a spurious `1/dt`
  > factor on the Hermite tangent — exactly the time-rescaling error §6's third
  > assertion is meant to catch. The generator now returns the physical grid from
  > `physics.timestep()` and that is what gets stored.
  >
  > **STILL OPEN — Hopper values are float64.** The generator allocates
  > `np.zeros((n, T, D))` while every other family produces float32, so the Hopper
  > pickles are ~34 MB per config rather than ~17 MB. Harmless, just wasteful.

### Tier-2 systems (added beyond the paper's three)

Five further mechanical families live in `synthetic_data.py` under
`family_via_ivp`, all with the layout **positions first, then velocities**, so a
system with `n` degrees of freedom pairs as `i:(i+n)`:

| config prefix | `D` | `--pairs` | `times` | note |
|---|---|---|---|---|
| `pendulum` | 2 | `0:1` | `[0,10,0.05]` | energy-capped below the separatrix, so theta librates |
| `double_pendulum` | 4 | `0:2,1:3` | `[0,10,0.05]` | chaotic but bounded |
| `duffing` | 2 | `0:1` | `[0,10,0.05]` | damped double-well, autonomous (no forcing term) |
| `spring_mass` | 6 | `0:3,1:4,2:5` | `[0,10,0.05]` | `SPRING_MASS_N = 3`, both ends pinned |
| `n_body` | 12 | `0:6,1:7,2:8,3:9,4:10,5:11` | `[0,10,0.05]` | `NBODY_N = 3`, Plummer-softened (`eps=0.2`), jittered ring ICs |

Each has the same four sparsity configs, and each config carries a `"pairs"`
field that `main.py::resolve_pairs` reads when `--pairs` is not given on the
command line. `T = 200` for all of them, matching tier 1, so per-run compute is
the same across every system.

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
