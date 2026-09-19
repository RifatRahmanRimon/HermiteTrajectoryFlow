# Hermite Trajectory Flow

Velocity-aware conditional paths for trajectory flow matching. Instead of fitting
every state dimension independently, the position interpolant is constrained by the
*observed velocity* of its paired dimension, enforcing `d/dt position = velocity`
at the interpolant level (cubic Hermite). Two variants: **hedge** (`d` fits, smooth
acceleration target) and **pure** (`d/2` fits, exact identity everywhere).

## Repository layout

```
HermiteTrajectoryFlow/
├── hermite_flow/          ← OUR method + the SplineFlow pipeline it extends
│   ├── main.py            ← training entry point
│   ├── src/interpolants.py← hermite_interpolant_values (hedge/pure) + baselines
│   ├── scripts/           ← validation gate + experiment sweep
│   ├── configs/           ← data + model configs
│   └── README.md          ← detailed pipeline reference (flags, systems, tables)
├── baselines/
│   ├── SplineFlow/        ← pristine upstream repo (reference only; see note below)
│   ├── TFM/               ← Trajectory Flow Matching (separate baseline)
│   ├── MMFM/              ← Multi-Marginal Flow Matching (separate baseline)
│   └── MMSFM/             ← Multi-Marginal Stochastic Flow Matching (separate baseline)
├── hermite_algorithms_pseudo.md
└── hermite_spec_repoed.md
```

---

## Which baselines to run (read this first)

The paper's comparison table has three headline columns — **SplineFlow**, **TFM**,
and our method. Here is exactly what each one is and how to produce it:

| Baseline | What it is | How to run |
|---|---|---|
| **Ours** (`hermite_hedge`, `hermite_pure`) | the proposed method | `hermite_flow`, `--interpolant_kind hermite_hedge/hermite_pure` |
| **SplineFlow** | spline-interpolant flow matching — **this IS the `bspline` interpolant in `hermite_flow`**, not a separate program (base method being modified) | `hermite_flow`, `--interpolant_kind bspline --degree 3` |
| **TFM** | Trajectory Flow Matching — primary baseline, **separate codebase** + own conda env | `baselines/TFM` (see §2) |
| **MMFM** | Multi-Marginal Flow Matching (natural cubic spline) — separate codebase | `baselines/MMFM` (see §2) |
| **MMSFM** | Multi-Marginal Stochastic FM — **closest prior art** (monotonic cubic Hermite) — separate codebase | `baselines/MMSFM` (see §2) |
| `linear` | piecewise-linear interpolant — an **ablation, NOT a paper baseline** | optional; `--interpolant_kind linear` |

> **Common mix-up:** `hermite_flow` and SplineFlow share one pipeline (ours is a
> fork of SplineFlow). So running `--interpolant_kind bspline --degree 3` *is*
> running the SplineFlow baseline — you do **not** run `baselines/SplineFlow`
> separately (that copy is kept pristine, with the authors' original cluster paths,
> as a reference only). Running `linear` is not required for the paper; the sweep
> skips it unless you set `INCLUDE_LINEAR=1`.

**TFM, MMFM, and MMSFM are the three external comparison baselines** (see
`dataset_and_baselines_list.md`, Table 1). Each is a standalone repo with its own
environment and README — run whichever comparisons your table needs.

---

## Experiment plan (current)

For now we run **8 synthetic datasets × 4 methods**.

**8 datasets** (all have velocity paired with position; each run at 4 missingness
levels — ρ = 0 / 0.25 / 0.5 / 0.75 via the `base` / `_sparse` / `_v_sparse` /
`_vv_sparse` configs, all under `--mask_mode per_dof`):

| # | dataset | type | state dim |
|---|---|---|---|
| 1 | `harmonic_oscillator` | linear | 2 |
| 2 | `damped_harmonic` | linear | 2 |
| 3 | `pendulum` | nonlinear | 2 |
| 4 | `duffing` | nonlinear | 2 |
| 5 | `double_pendulum` | nonlinear, chaotic | 4 |
| 6 | `spring_mass` | linear, multi-DOF | 6 |
| 7 | `n_body` | nonlinear (3 bodies, 2D) | 12 |
| 8 | `hopperphysics` | MuJoCo (needs `dm_control`) | 14 |

**4 methods:**

| method | how it's run |
|---|---|
| **hedge** | `hermite_flow`, `--interpolant_kind hermite_hedge` |
| **pure** | `hermite_flow`, `--interpolant_kind hermite_pure` |
| **SplineFlow** | `hermite_flow`, `--interpolant_kind bspline --degree 3` (same pipeline) |
| **TFM** | separate repo `baselines/TFM` (own env); ρ=0 comparison only |

hedge / pure / SplineFlow run in the one pipeline under `per_dof` (the sweep does
all three); TFM is run separately. MMFM / MMSFM are available as further baselines
but are not part of this current round.

---

## 1. Our method + SplineFlow (one pipeline)

```bash
cd hermite_flow
pip install torch numpy scipy scikit-learn matplotlib torchdiffeq torchsde torchdyn dcor pot

# single run (SplineFlow baseline)
python main.py --data_config damped_harmonic_vv_sparse --interpolant_kind bspline --degree 3 --mask_mode per_dof

# single run (ours)
python main.py --data_config damped_harmonic_vv_sparse --interpolant_kind hermite_pure --mask_mode per_dof
```

Full sweep — runs **both Hermite variants and the SplineFlow (bspline) baseline**
for every system and missingness level (linear is off by default):

```bash
cd hermite_flow
EPOCHS=10000 MASK_MODE=per_dof SEED=42 bash scripts/run_experiments_hermite.sh
# add the linear ablation:   INCLUDE_LINEAR=1 bash scripts/run_experiments_hermite.sh
# subset of systems:         SYSTEMS="pendulum duffing" bash scripts/run_experiments_hermite.sh
```

See [hermite_flow/README.md](hermite_flow/README.md) for the full flag reference,
the systems ↔ `pairs` table (the "dim/dof" question), `mask_mode`, seeding, and the
validation gate.

---

## 2. TFM baseline (separate codebase)

TFM is a Lightning/Hydra project with its **own environment** — do not mix it with
the `hermite_flow` env:

```bash
cd baselines/TFM
conda create -n tfm python=3.10 && conda activate tfm
conda env create -f environment.yml

# configure a run: add conf/data/<DATA>.yml and conf/model/<MODEL>.yml under src/,
# point conf/config.yaml at them, then:
python src/main.py
```

See [baselines/TFM/README.md](baselines/TFM/README.md) for details and the
`notebook/3Oscillation.ipynb` demo.

**MMFM** and **MMSFM** follow the same pattern — each has its own env
(`environment.yml` / `requirements.txt`) and README:

- [baselines/MMFM/README.md](baselines/MMFM/README.md) — Multi-Marginal Flow Matching
- [baselines/MMSFM/README.md](baselines/MMSFM/README.md) — Multi-Marginal Stochastic FM (has `runner.sh` / `make_venv.sh`)

---

## 3. Reproducibility

Every `hermite_flow` run takes `--seed` (default 42), which fixes model init, the
conditional-path noise, and DataLoader shuffling; data generation is seeded by the
config. Same seed + same config ⇒ identical run. TFM/MMFM/MMSFM manage their own
seeding inside their configs.

---

## Notes for the paper

- The fair baseline for cubic Hermite is the **cubic** B-spline (`--degree 3`);
  higher-degree B-splines can win on very smooth systems (a degree mismatch, not
  the contribution). The contribution is the exact `d/dt position = velocity`
  identity (R1).
- **Missingness mode: we use `per_dof` for all experiments** (the default in
  `main.py` and the sweep). A drop removes the whole paired state at a time step —
  the paired-sensor scope this method targets — so position and velocity are always
  observed together and the Hermite construction is never starved. The **baselines
  (bspline = SplineFlow) run under the same `per_dof` mask**, so the comparison is
  fair. TFM is a ρ=0 comparison and is unaffected by the mask mode.
  Background on why: the ρ (missingness) sweep with *per-coordinate* drops
  (`--mask_mode per_dim`) is SplineFlow's original protocol; under it the Hermite
  advantage collapses at high ρ because the position+velocity intersection shrinks
  like `(1-ρ)²` (see the §5 gate). `per_dim` is kept only for reproducing SplineFlow's
  original per-coordinate numbers; it is not our evaluation setting.
- Nonlinear systems (pendulum, duffing, double_pendulum, n_body) are the strongest
  evidence; the linear ones are where a plain spline already does well.
