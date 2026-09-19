# Reference Tables

## Table 1 — Comparison papers

| Paper | Venue / ID | Coupling | Interpolant | Relevance |
|---|---|---|---|---|
| **SplineFlow** — Rathod, Liò, Zhang | arXiv 2601.23072 | given (ODE); pairwise entropic OT (cells) | B-spline, tunable degree $m$ | Base method being modified |
| **TFM** — Zhang et al. | NeurIPS 2024 | given (paired clinical trajectories) | piecewise linear | Primary baseline — same "given coupling" setting |
| **MMFM** — Rohbeck et al. | ICLR 2025 | MMOT ⇒ chained pairwise $W_2$ | natural cubic spline | First splines-as-conditional-paths in FM |
| **MMSFM** — Lee et al. | arXiv 2508.04351 | overlapping triplets, rolling window | **monotonic cubic Hermite** | Closest prior art; treats $C^1$-only as a feature |
| **3MSBM** — Theodoropoulos et al. | 2026 | global, iteratively refined | phase-space lift, measure-valued splines | Also brings velocity into the dynamics |
| **ALI-CFM** | arXiv 2510.01159 (ICLR 2026) | — | adversarially learnt | Argues spline interpolants give high gradient variance |
| **IMMFM / Longitudinal FM** | arXiv 2510.03569 | — | piecewise quadratic | Benchmarks against TFM |
| **HNN / LNN / Symplectic ODE-Net** | various | n/a | n/a | Same $(q,\dot q)$ structure, imposed in the *architecture* not the interpolant |
| **OT-CFM / MOTFM** — Tong et al. | 2024 | minibatch OT, consecutive pairs | piecewise linear | Background |
| **SF2M** — Tong et al. | 2023 | Schrödinger bridge | linear | Background |
| **Flow Matching** — Lipman et al. | ICLR 2023 | — | linear | Background |

---

## Table 2 — Benchmark datasets

| Dataset | State | Velocity in state? | Source | Notes |
|---|---|---|---|---|
| **Damped Harmonic Osc.** | $(x, \dot x)$ | ✓ | in SplineFlow | Start here. Degree ablation unstable in the paper |
| **Harmonic Oscillator** | $(x, \dot x)$ | ✓ | in SplineFlow | Cleanest case |
| **HopperPhysics** | $(q, \dot q)$ | ✓ | in SplineFlow | Largest; near-linear, degree 1 already wins. Confirm state layout |
| **Pendulum** | $(\theta, \dot\theta)$ | ✓ | simulate | Mild nonlinearity |
| **Double Pendulum** | $(\theta_{1,2}, \dot\theta_{1,2})$ | ✓ | simulate | Chaotic; replaces Lorenz |
| **Duffing oscillator** | $(x, \dot x)$ | ✓ | simulate | High curvature, driven |
| **Spring-mass chain** | $(q_{1:n}, \dot q_{1:n})$ | ✓ | simulate | Scales $d$ for a dimension sweep |
| **$n$-body** | $(q_{1:n}, \dot q_{1:n})$ | ✓ | simulate | Cross-DOF coupling |
| **MuJoCo / Gym rollouts** | $(q, \dot q)$ | ✓ | log yourself | Real-ish, non-synthetic dynamics |
| **CMU MoCap / H3.6M** | joint angles | ✓ (finite diff) | public | Velocity is differenced ⇒ noisy tangents |
| **IMU logs** | pos, vel, accel | ✓ | public | Gives $\ddot q$ directly |
| ~~Exponential Decay~~ | scalar $x$ | ✗ | — | Excluded |
| ~~Lotka-Volterra~~ | $(\text{prey}, \text{pred})$ | ✗ | — | Excluded — first-order 2D |
| ~~Lorenz~~ | $(x,y,z)$ | ✗ | — | Excluded |
| ~~Cellular (PHATE/PCA)~~ | embedding | ✗ | — | Excluded — also unpaired |

### Targets to beat (MSE)

| Dataset | SplineFlow $\rho=0$ | SplineFlow $\rho=0.75$ | TFM $\rho=0$ |
|---|---|---|---|
| Harmonic Oscillator | $3.7\times10^{-4}$ | $1.1\times10^{-3}$ | $0.181$ |
| Damped Harmonic | $9.5\times10^{-5}$ | $4.8\times10^{-5}$ | $0.015$ |
| HopperPhysics | $1.410$ | $3.336$ | $1.554$ |

$\rho$ = per-coordinate missingness rate. Config suffixes: none $=0$, `_sparse` $=0.25$, `_v_sparse` $=0.5$, `_vv_sparse` $=0.75$.
