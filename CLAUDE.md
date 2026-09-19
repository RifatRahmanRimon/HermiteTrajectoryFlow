# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A research repo implementing **Hermite conditional paths for flow matching**: trajectory
interpolants that use an observed velocity coordinate to constrain the position coordinate,
instead of fitting every state dimension independently.

- `hermite_spec_repoed.md` — the implementation brief (scope, validation gate, experiment plan,
  MSE numbers to beat). Treat it as the source of truth for what to build and what not to touch.
- `hermite_algorithms_pseudo.md` — the math: Algorithms 1 (hedge) / 2 (pure), training, inference,
  and remarks R1–R5 (notably R3: cubic Hermite is *fixed* at cubic, so the matched-degree
  competitor is `bspline --degree 3`).
- `hermite_flow/` — **the working codebase**. A fork of upstream SplineFlow with the two Hermite
  interpolants added. All development happens here.
- `baselines/` — vendored upstream repos for reference/comparison: `SplineFlow` (the unmodified
  original), `MMFM`, `MMSFM`. Read-only; `SplineFlow` still contains the authors' hardcoded
  cluster paths (`src/helpers.py:14`, `main.py:232`) and will not run as-is.

## Commands

All commands run from `hermite_flow/`. There is no test suite, no linter config, and no
requirements file; deps are `torch torchdiffeq torchsde scipy scikit-learn dcor pot matplotlib`
(plus `dm_control` for HopperPhysics only, lazily imported).

```bash
# Validation gate — run this before any training (spec §5/§6). No network, no training.
# Sweeps missing_prob x {hedge, pure, bspline k=1..5}, reports mean/max |mu_dot - truth|
# separately for the position and velocity slots, plus the pair-fallback rate, plus a gate verdict.
python scripts/check_interpolant_derivative.py                  # both mask modes
python scripts/check_interpolant_derivative.py --mask_mode per_dim --n_traj 20

# Smoke train
python main.py --data_config harmonic_oscillator --interpolant_kind bspline --degree 3 \
  --exp_name smoke --epochs 100

# Hermite run (pairs are REQUIRED — without --pairs the hermite kinds reduce to plain B-splines, R5)
python main.py --data_config damped_harmonic_vv_sparse --interpolant_kind hermite_hedge \
  --pairs 0:1 --mask_mode per_dof --exp_name hermite

# Full §7 sweep (damped_harmonic + harmonic_oscillator x 4 sparsities x 4 interpolants)
EPOCHS=10000 MASK_MODE=per_dof bash scripts/run_experiments_hermite.sh

# Pre-generate a dataset pickle without training
python data_preprocessing/dataloader.py --config_name damped_harmonic --create_data True

# Re-evaluate saved checkpoints (see caveat below)
python evaluation.py --data_config damped_harmonic --interpolant_kind bspline --exp_name hermite
```

`results/` and `data/` are gitignored. Each `main.py` run writes to
`results/{exp_name}_{data_config}_{interpolant_kind}_{mask_mode}[_N]/` (`_N` auto-increments so
reruns never overwrite) containing `log.txt`, `train_dynamics.json`, `final_mse_metrics.json`,
`model.pt`, `test_validation_metrics.json`, and trajectory PNGs.

## Architecture

The pipeline is **generate → window → precompute interpolant → train MLP on (x,t)→v → integrate**.

1. `data_preprocessing/synthetic_data.py` integrates ODE/SDE families (exp_decay, harmonic,
   damped_harmonic, lotka_volterra, lorenz, + `generate_hopperphysics_trajectories`,
   `generate_pendulum_video_trajectories`) and draws the observation mask via `get_mask`.
   `realworld_data.py` is empty.
2. `data_preprocessing/dataloader.py::get_dataloaders` reads `configs/data_configs/<name>.json`,
   **caches the generated trajectories to `data/<name>.pkl`** (`<name>__per_dof.pkl` for per-dof
   masking), splits 80/20, and wraps in `IrregularDataset` (fixed windows; `fill_method: "nofill"`
   in all current configs, so gaps stay NaN and the mask is authoritative).
3. `ProcessedDataset.__init__` **precomputes the conditional path once per batch** —
   `get_flow_matching_inputs` → an interpolant in `src/interpolants.py` → `(xt, vt, tt, sigma_t,
   lambda_t, der_sigma_t)`. Z-score normalization happens later, in `__getitem__`, *after*
   interpolation, using stats from the train split (`src/helpers.py::get_data_stats`).
4. `main.py::train_epoch` adds `sigma_t` noise to `xt`, concatenates time, and regresses the MLP
   (`models/NN_models.py`) onto `vt`. Branches on `dynamics_kind`: `ode`, `sde_constant_sigma`,
   `sde_quadratic_sigma` (the latter two also train a score model).
5. `src/validation.py::validate_metrics` integrates the learned field with `torchdiffeq.odeint`
   (or `torchsde.sdeint`) wrapped in `models/odeint_classes.py::NODEFunc` / `StochasticNODE`, and
   reports trajectory MSE (plus Wasserstein/MMD/energy for the stochastic path).

### Invariants that constrain any interpolant change

- **Time is physical, never normalized, inside the interpolants.** `t_fine` spans the raw
  `times[b][0]..times[b][-1]`, so an observed velocity is a valid Hermite tangent with **no
  time-rescaling factor**. `NODEFunc` reapplies the chain rule (`times_std / values_std`) at
  integration time because targets were z-scored afterwards.
- Every interpolant returns the same 6-tuple and must reproduce `bspline_interpolant_values`'
  edge-case handling verbatim: empty `dim_idx`, single observation, `k = min(degree, n_obs - 1)`,
  clamped flat extrapolation outside `[t_obs_min, t_obs_max]` (value pinned to the endpoint,
  derivatives zeroed), and the `get_velocity_score_and_noise` call for the SDE terms. The shared
  helpers `_bspline_slot`, `_eval_spline_clamped`, `_eval_hermite_clamped` exist for exactly this.
- `sigma` defaults to being **estimated from the data** (`estimate_sigma_from_loader`) and the
  train-split value is reused for the test split.

### The Hermite additions

`src/interpolants.py::hermite_interpolant_values(..., pairs, unpaired, mode)`; dispatched from
`src/helpers.py::get_flow_matching_inputs` under `interpolant_kind in ("hermite_hedge",
"hermite_pure")`. Per pair `(p, v)`:

| mode | position slot `p` | velocity slot `v` |
|---|---|---|
| `hedge` (Alg 1, `d` fits) | `H(t)`, `H'(t)` | plain B-spline of observed `v` and its derivative |
| `pure` (Alg 2, `|P|+|U|` fits) | `H(t)`, `H'(t)` | `H'(t)`, `H''(t)` — self-consistent, but accel is piecewise linear and knot-discontinuous |

`H = CubicHermiteSpline` is fit on the **mask intersection** `I_pv = {i : mask[i,p] ∧ mask[i,v]}`.
If `|I_pv| < 2` the pair falls back to independent B-splines; the fallback rate is printed per call
and must be reported alongside any result (R4) — at `missing_prob=0.75` with `per_dim` masking a
high rate means the experiment is measuring the fallback, not the idea.

**`pairs` plumbing** (all four hops are needed; a missing one silently degrades to B-splines):
`main.py --pairs "0:1,2:3"` → `parse_pairs` → `ProcessedDataset(pairs=...)` →
`get_flow_matching_inputs` → `hermite_interpolant_values`. `unpaired` is derived inside
`ProcessedDataset.__init__` as every dim in no pair.

**Pair maps.** `harmonic_oscillator`, `damped_harmonic`: state is `(x, v)` → `--pairs 0:1`.
`hopperphysics`: `generate_hopperphysics_trajectories` writes `qpos` into `[:D//2]` and `qvel`
into `[D//2:]` with `D=14`, so the pairing is `0:7,1:8,2:9,3:10,4:11,5:12,6:13`.
Exp-Decay / Lotka-Volterra / Lorenz have no velocity coordinate — out of scope (§7).

### Masking

`--mask_mode per_dim` (default, `get_mask`) draws one Bernoulli per `[T, D]` entry, so a position
and its velocity drop **independently** — this is upstream behaviour and the only mode comparable
to the paper's table. `--mask_mode per_dof` draws one Bernoulli per time step and drops the whole
state, modelling a sensor that returns everything or nothing; it makes the pairing assumption hold
but its numbers are **not** paper-comparable. Always state which mode a number came from.

### Config naming

`configs/data_configs/<system>[_sde][_sparse|_v_sparse|_vv_sparse].json`. The suffix *is* the
missingness: base `= 0`, `_sparse` `= 0.25`, `_v_sparse` `= 0.5`, `_vv_sparse` `= 0.75`. Configs
also carry `times: [t0, t1, dt]`, `seed`, `fill_method`, and the family request counts.
`configs/model_configs/` holds `MLP.json` (256×4) and `MLP_wide.json`.

## Gotchas

- Editing a data config has no effect until you delete the matching `data/<name>.pkl` — the cache
  is keyed by config name only.
- `evaluation.py` was not updated for the Hermite work: it takes neither `--pairs` nor
  `--mask_mode`, so its run-dir prefix omits the `_{mask_mode}` suffix `main.py` writes, and it
  re-fits the interpolants with no pairs. Its output is only valid for bspline/linear runs.
- `--dynamics_kind sde_time_varying_sigma` is an accepted CLI choice but `train_epoch` has no
  branch for it, so it fails on an unbound `loss`.
- Per the spec's scope (§8): the training loop, the MLP, the loss, the noise schedule, and the SDE
  branch are off-limits. Changes belong in the interpolant functions, the dispatch, and the CLI
  plumbing.
