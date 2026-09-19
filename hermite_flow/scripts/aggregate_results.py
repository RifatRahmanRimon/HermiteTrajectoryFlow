#!/usr/bin/env python3
"""Turn run_kaggle.py manifests into the paper's tables.

Reads results/_launcher/index_*.jsonl (one JSON record per completed run) and
emits, per mask mode:

  1. the HEADLINE table -- mean +/- std of trajectory MSE over replicate seeds,
     systems x sparsity x {linear, bspline3, hermite_hedge, hermite_pure};
  2. the DEGREE ABLATION -- the same cells against bspline k=1..5, which is what
     answers "why not just raise the spline degree?";
  3. the FALLBACK table -- pair-fallback rate per (system, sparsity, mask mode).
     R4: no Hermite number is reportable without it, and a high rate at p=0.75
     under per_dim means the cell measures the fallback, not the method.

A cell with fewer seeds than requested is printed with its actual n, so a
half-finished sweep can never be mistaken for a complete one.

    python scripts/aggregate_results.py --exp_name hermite
    python scripts/aggregate_results.py --exp_name hermite --csv paper_tables.csv
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = PROJECT_ROOT / "results" / "_launcher"

SPARSITY_ORDER = ["base", "sparse", "v_sparse", "vv_sparse"]
SPARSITY_P = {"base": "0.00", "sparse": "0.25", "v_sparse": "0.50", "vv_sparse": "0.75"}
HEADLINE = ["linear", "bspline3", "hermite_hedge", "hermite_pure"]
DEGREES = ["bspline1", "bspline2", "bspline3", "bspline4", "bspline5"]


def load(exp_name):
    files = sorted(LAUNCHER.glob(f"index_{exp_name}.jsonl")) if exp_name \
        else sorted(LAUNCHER.glob("index_*.jsonl"))
    if not files:
        raise SystemExit(f"no manifests in {LAUNCHER} (looked for index_{exp_name or '*'}.jsonl)")
    # Re-running a cell appends a second record; the last one wins.
    latest = {}
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            latest[(r["system"], r["sparsity"], r["label"], r["mask"], r["seed"])] = r
    return list(latest.values())


def stats(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return None, None, 0
    n = len(vals)
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return mean, sd, n


def fmt(mean, sd, n, n_expected, best=False):
    if mean is None:
        return "--"
    cell = f"{mean:.3e}+-{sd:.0e}" if n > 1 else f"{mean:.3e}"
    if n != n_expected:
        cell += f"[n={n}]"
    return f"**{cell}**" if best else cell


def table(records, mask, labels, title, n_expected, out_rows):
    cells = defaultdict(list)
    for r in records:
        if r["mask"] == mask and r.get("returncode") == 0:
            cells[(r["system"], r["sparsity"], r["label"])].append(r.get("mse"))

    systems = sorted({r["system"] for r in records if r["mask"] == mask})
    if not systems:
        return
    print(f"\n### {title}  (mask_mode = {mask})\n")
    print("| system | p | " + " | ".join(labels) + " |")
    print("|---|---|" + "---|" * len(labels))
    for sysname in systems:
        for sp in SPARSITY_ORDER:
            row = [stats(cells.get((sysname, sp, lab), [])) for lab in labels]
            if all(m is None for m, _, _ in row):
                continue
            finite = [m for m, _, _ in row if m is not None]
            best_val = min(finite) if finite else None
            out = []
            for lab, (m, sd, n) in zip(labels, row):
                out.append(fmt(m, sd, n, n_expected, best=(m is not None and m == best_val)))
                out_rows.append({"mask_mode": mask, "system": sysname,
                                 "p_missing": SPARSITY_P[sp], "interpolant": lab,
                                 "mse_mean": "" if m is None else f"{m:.6e}",
                                 "mse_std": "" if m is None else f"{sd:.6e}",
                                 "n_seeds": n})
            print(f"| {sysname} | {SPARSITY_P[sp]} | " + " | ".join(out) + " |")


def fallback_table(records):
    cells = defaultdict(list)
    for r in records:
        if r.get("fallback_pct") is not None and r.get("returncode") == 0:
            cells[(r["system"], r["sparsity"], r["mask"])].append(r["fallback_pct"])
    if not cells:
        print("\n(no pair-fallback rates recorded -- no hermite runs in these manifests)")
        return
    print("\n### Pair-fallback rate (R4: report alongside every Hermite number)\n")
    print("| system | p | per_dim | per_dof |")
    print("|---|---|---|---|")
    for sysname in sorted({k[0] for k in cells}):
        for sp in SPARSITY_ORDER:
            got = {m: cells.get((sysname, sp, m), []) for m in ("per_dim", "per_dof")}
            if not any(got.values()):
                continue
            out = []
            for m in ("per_dim", "per_dof"):
                v = got[m]
                if not v:
                    out.append("--")
                else:
                    mean = sum(v) / len(v)
                    out.append(f"**{mean:.1f}%**" if mean >= 20 else f"{mean:.1f}%")
            print(f"| {sysname} | {SPARSITY_P[sp]} | " + " | ".join(out) + " |")
    print("\nBold = at or above 20%: that cell is substantially measuring the "
          "independent-B-spline fallback rather than the Hermite path.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", default="hermite")
    ap.add_argument("--seeds_expected", type=int, default=5)
    ap.add_argument("--csv", default=None, help="also write a tidy CSV here")
    args = ap.parse_args()

    records = load(args.exp_name)
    ok = [r for r in records if r.get("returncode") == 0]
    seeds = sorted({r["seed"] for r in ok})
    print(f"# Results: {args.exp_name}\n")
    print(f"- runs recorded : {len(records)}  ({len(records) - len(ok)} failed)")
    print(f"- seeds present : {seeds}")
    print(f"- systems       : {sorted({r['system'] for r in ok})}")
    print(f"\nCells are mean+-std of test trajectory MSE over {args.seeds_expected} "
          f"replicate seeds (each seed re-draws the data AND the training RNG); "
          f"[n=k] marks a cell with only k. Bold = best in row.")

    rows = []
    for mask in ("per_dim", "per_dof"):
        table(ok, mask, HEADLINE, "Headline", args.seeds_expected, rows)
    for mask in ("per_dim", "per_dof"):
        table(ok, mask, ["hermite_hedge", "hermite_pure"] + DEGREES,
              "B-spline degree ablation", args.seeds_expected, rows)
    fallback_table(ok)

    if args.csv:
        seen, uniq = set(), []
        for r in rows:
            k = (r["mask_mode"], r["system"], r["p_missing"], r["interpolant"])
            if k not in seen:
                seen.add(k)
                uniq.append(r)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(uniq[0].keys()))
            w.writeheader()
            w.writerows(uniq)
        print(f"\nCSV written: {args.csv} ({len(uniq)} rows)")


if __name__ == "__main__":
    main()
