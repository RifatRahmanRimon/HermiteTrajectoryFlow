#!/usr/bin/env bash
# Section 7 experiment sweep for the Hermite conditional paths.
#
# Scope: only systems whose state contains velocity paired with position.
# Included: harmonic_oscillator, damped_harmonic (linear); pendulum, duffing,
# double_pendulum, n_body (nonlinear); spring_mass (linear, multi-DOF).
# Skipped: exp_decay / logistic / lotka_volterra / lorenz (no velocity dim).
# Hopper is left out until its q / q_dot index split is confirmed.
#
# Masking: --mask_mode per_dof is the DEFAULT here, matching the paper's scope
# (a missing sample drops the paired position+velocity together). The repo's
# original per_dim masking drops them independently and breaks the pairing
# assumption; to reproduce that (NOT comparable to the paper table) set
# MASK_MODE=per_dim below.
#
# Baseline is cubic B-spline (--degree 3), the matched-degree competitor for
# cubic Hermite (R3), plus linear. Run from the repo root.
#
# Numbers to beat (MSE, from the paper):
#   System               | SplineFlow p=0 | p=0.75 | TFM p=0
#   Harmonic Oscillator  |   3.7e-4       | 1.1e-3 | 0.181
#   Damped Harmonic      |   9.5e-5       | 4.8e-5 | 0.015
set -euo pipefail

EXP_NAME="${EXP_NAME:-hermite}"
MASK_MODE="${MASK_MODE:-per_dof}"      # DECIDED DEFAULT: per_dof (paired-sensor scope; drops the
                                       # whole paired state together). Baselines (bspline) run under
                                       # the SAME mask_mode, so the comparison is fair. Use per_dim
                                       # only to reproduce the original SplineFlow per-coordinate numbers.
EPOCHS="${EPOCHS:-10000}"
SEED="${SEED:-42}"
INCLUDE_LINEAR="${INCLUDE_LINEAR:-0}"  # 1 = also run the linear interpolant (ablation, NOT a paper baseline)

# Baselines produced by this sweep:
#   hermite_hedge / hermite_pure  -> the proposed method
#   bspline (--degree 3)          -> the SplineFlow baseline (paper's "SplineFlow" numbers)
#   linear                        -> extra ablation, off by default (INCLUDE_LINEAR=1 to add)
# The TFM baseline is a SEPARATE codebase (../baselines/TFM); see the top-level README.

# The (position:velocity) pairs live in each data_config JSON ("pairs" field), so
# main.py resolves them automatically -- no need to pass --pairs here. State is
# positions-first then velocities, so dof p pairs with p + n_dof:
#   1 dof : harmonic_oscillator, damped_harmonic, pendulum, duffing
#   2 dof : double_pendulum        3 dof : spring_mass        6 dof : n_body (2D,3)
#
# Systems to sweep (override with SYSTEMS="pendulum duffing" bash scripts/... ).
SYSTEMS="${SYSTEMS:-damped_harmonic harmonic_oscillator pendulum duffing double_pendulum spring_mass n_body}"

run_system () {
  local system="$1"
  for cfg in "${system}" "${system}_sparse" "${system}_v_sparse" "${system}_vv_sparse"; do
    for kind in hermite_hedge hermite_pure; do
      python main.py --data_config "$cfg" --interpolant_kind "$kind" \
        --mask_mode "$MASK_MODE" --exp_name "$EXP_NAME" --epochs "$EPOCHS" --seed "$SEED"
    done
    # SplineFlow baseline: matched-degree cubic B-spline (paper's "SplineFlow")
    python main.py --data_config "$cfg" --interpolant_kind bspline --degree 3 \
      --mask_mode "$MASK_MODE" --exp_name "$EXP_NAME" --epochs "$EPOCHS" --seed "$SEED"
    # linear: extra ablation, only if explicitly requested
    if [ "$INCLUDE_LINEAR" = "1" ]; then
      python main.py --data_config "$cfg" --interpolant_kind linear \
        --mask_mode "$MASK_MODE" --exp_name "$EXP_NAME" --epochs "$EPOCHS" --seed "$SEED"
    fi
  done
}

for sys in $SYSTEMS; do
  run_system "$sys"
done

echo "Done. Results under results/${EXP_NAME}_<config>_<kind>_${MASK_MODE}[/_N]"
