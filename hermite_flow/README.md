# Hermite Conditional Paths for Flow Matching

> **Start at the [top-level README](../README.md)** for the project overview, the
> repository layout, and how to run every baseline (including SplineFlow and TFM).
> This file is the detailed reference for the `hermite_flow` pipeline itself.

Velocity-aware conditional paths for trajectory flow matching. Instead of fitting
every state dimension independently (the B-spline baseline), the position
interpolant is constrained by the *observed velocity* of its paired dimension, so
the identity `d/dt position = velocity` is enforced at the interpolant level.

This directory is a self-contained working copy of the SplineFlow flow-matching
pipeline plus the Hermite interpolants. The untouched SplineFlow baseline lives in
`../baselines/SplineFlow`.

---

## 1. Setup

Python 3.10+ with:

```bash
pip install torch numpy scipy scikit-learn matplotlib \
            torchdiffeq torchsde torchdyn dcor pot
```

`dm_control` (MuJoCo) is needed **only** for the Hopper system and is imported
lazily; everything else runs without it.

---

## 2. Quick start

Train one model (from this directory):

```bash
python main.py --data_config duffing_vv_sparse --interpolant_kind hermite_pure --mask_mode per_dof
```

You do **not** need to specify the position/velocity pairing by hand — it is read
from the data config's `pairs` field (see §4). The run prints what it resolved:

```
[config] seed=42  mask_mode=per_dof  pairs=[(0, 1)]
```

Outputs land in `results/<exp_name>_<data_config>_<interpolant_kind>_<mask_mode>/`
(`model.pt`, metrics JSONs, trajectory figures, `log.txt`). Generated trajectory
data is cached in `data/` (both are git-ignored).

Full sweep across all systems / missingness levels / interpolants:

```bash
EPOCHS=10000 MASK_MODE=per_dof SEED=42 bash scripts/run_experiments_hermite.sh
# subset: SYSTEMS="pendulum duffing" bash scripts/run_experiments_hermite.sh
```

---

## 3. Interpolants (`--interpolant_kind`)

| value | what it does |
|---|---|
| `hermite_hedge` | position = cubic Hermite `H` (uses observed velocity as tangent); velocity = plain B-spline of `v`. `d` fits. Acceleration target stays smooth. |
| `hermite_pure`  | position = `H`; velocity = `H'`, so `d/dt position = velocity` holds *everywhere*. `d/2` fits. Acceleration target `H''` is piecewise-linear (C¹). |
| `bspline`       | **the SplineFlow baseline** — independent B-spline per dimension (`--degree k`); use `--degree 3` for the matched-degree competitor to cubic Hermite. This *is* the paper's "SplineFlow" (not a separate program). |
| `linear`        | piecewise-linear per dimension — an ablation, **not** a paper baseline; off by default in the sweep. |

---

## 4. Systems and the `pairs` field  (this is the "dim / dof" question)

The Hermite interpolants need to know which state dimension is a **position** and
which is its **velocity**. State is laid out **positions-first, then velocities**,
so a system with `n` degrees of freedom pairs dimension `p` with `p + n`.

Every second-order config carries a `"pairs"` field, so `--pairs` is optional; pass
`--pairs` only to override. Format: `"p1:v1,p2:v2,..."`.

| system | in scope? | state dim | DOF | `pairs` |
|---|---|---|---|---|
| `harmonic_oscillator` | ✅ (linear) | 2 | 1 | `0:1` |
| `damped_harmonic`     | ✅ (linear) | 2 | 1 | `0:1` |
| `pendulum`            | ✅ (nonlinear) | 2 | 1 | `0:1` |
| `duffing`             | ✅ (nonlinear) | 2 | 1 | `0:1` |
| `double_pendulum`     | ✅ (nonlinear, chaotic) | 4 | 2 | `0:2,1:3` |
| `spring_mass`         | ✅ (linear, multi-DOF) | 6 | 3 | `0:3,1:4,2:5` |
| `n_body`              | ✅ (nonlinear, 3 bodies 2D) | 12 | 6 | `0:6,1:7,2:8,3:9,4:10,5:11` |
| `hopperphysics`       | ✅ (needs `dm_control`/MuJoCo installed) | 14 | 7 | `0:7,1:8,2:9,3:10,4:11,5:12,6:13` |
| `exp_decay`, `logistic_growth`, `lotka_volterra`, `lorenz` | ❌ no velocity dim | — | — | — |

Each system has four missingness levels via the config suffix:
`base` = 0, `_sparse` = 0.25, `_v_sparse` = 0.5, `_vv_sparse` = 0.75.

**Hopper note.** State is `qpos` (dims 0–6) then `qvel` (dims 7–13); all joints are
1-DOF (slide/hinge) so `qvel[i]` is exactly `d/dt qpos[i]`. Trajectories are stored
on a **physical time grid** (spacing = `physics.timestep()`), which is required for
`qvel` to be the correct Hermite tangent — do not switch it to a step-index grid.
Generating Hopper needs MuJoCo: `pip install dm_control`.

---

## 5. Masking (`--mask_mode`)

- `per_dof` **(the default; used for all our experiments)** — a dropped observation
  removes the whole state at that time step, so position and its velocity always
  appear together. This matches the paired-sensor scope the method is designed for,
  and the baselines are run under the same mode for a fair comparison.
- `per_dim` (opt-in) — every entry is dropped independently, so a position and its
  velocity can go missing separately. This breaks the pairing assumption; at high
  missingness the Hermite fit is starved of jointly-observed knots. Kept **only** for
  reproducing the original SplineFlow per-coordinate protocol; not our setting.

---

## 6. Reproducibility (`--seed`, default 42)

`--seed` seeds Python `random`, NumPy, and Torch, which fixes model weight init,
the conditional-path noise (`torch.randn_like`), and `DataLoader` shuffling.
Trajectory *generation* is seeded separately by the `seed` field in the data
config. Same `--seed` + same config ⇒ identical run.

---

## 7. Validation gate (no training)

Before training, `scripts/check_interpolant_derivative.py` checks that the Hermite
interpolants' derivative (the flow-matching target) beats the B-splines, and runs
the pass-through / no-time-scaling assertions:

```bash
python scripts/check_interpolant_derivative.py --n_traj 200 --mask_mode both
```

Currently hard-coded to the damped harmonic oscillator (its ground-truth
derivative is known analytically).

---

## 8. Notes for the paper

- Cubic Hermite is fixed at cubic (only value + first derivative are observed), so
  the fair baseline is the **cubic** B-spline (`--degree 3`). Higher-degree
  B-splines can beat it on very smooth systems; that is a degree mismatch, not the
  contribution — the contribution is the exact `d/dt position = velocity` identity.
- The **nonlinear** systems (pendulum, duffing, double_pendulum, n_body) are the
  strongest evidence: the acceleration target is genuinely hard and `hedge` vs
  `pure` diverge there. The linear ones are where a plain spline already does well.
