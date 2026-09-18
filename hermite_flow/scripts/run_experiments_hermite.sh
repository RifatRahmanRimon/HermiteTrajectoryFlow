#!/usr/bin/env bash
# Section 7 experiment sweep for the Hermite conditional paths.
#
# Scope (see hermite_spec_repoed.md sec 7): only systems whose state contains
# velocity. Exp-Decay / Lotka-Volterra / Lorenz have no velocity dim -> skipped.
# Hopper is left out until its state layout (which indices are q vs q_dot) is
# confirmed from data_preprocessing/synthetic_data.py.
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
MASK_MODE="${MASK_MODE:-per_dof}"      # per_dof (paper scope) | per_dim (repo default)
EPOCHS="${EPOCHS:-10000}"
PAIRS="0:1"                            # (x, v) for both oscillator systems

run_system () {
  local system="$1"
  for cfg in "${system}" "${system}_sparse" "${system}_v_sparse" "${system}_vv_sparse"; do
    for kind in hermite_hedge hermite_pure; do
      python main.py --data_config "$cfg" --interpolant_kind "$kind" \
        --pairs "$PAIRS" --mask_mode "$MASK_MODE" --exp_name "$EXP_NAME" --epochs "$EPOCHS"
    done
    # matched-degree baseline (cubic) + linear
    python main.py --data_config "$cfg" --interpolant_kind bspline --degree 3 \
      --mask_mode "$MASK_MODE" --exp_name "$EXP_NAME" --epochs "$EPOCHS"
    python main.py --data_config "$cfg" --interpolant_kind linear \
      --mask_mode "$MASK_MODE" --exp_name "$EXP_NAME" --epochs "$EPOCHS"
  done
}

# Start with damped harmonic at the sparsest setting, per the brief.
run_system damped_harmonic
run_system harmonic_oscillator

echo "Done. Results under results/${EXP_NAME}_<config>_<kind>_${MASK_MODE}[/_N]"
