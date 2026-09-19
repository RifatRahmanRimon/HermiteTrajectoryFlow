#!/usr/bin/env python3
"""Parallel experiment runner for Kaggle (T4 x2), sized for paper-ready results.

A session runs one COMPLETE cell block: every sparsity x every interpolant x
every mask mode, for the requested systems and seeds, fanned out across both
GPUs with several processes each.

The natural session unit is ONE (system, seed) pair -- 63 runs, ~2.5h -- which
never approaches the 12h session cap and keeps each session's output a
self-contained replicate. Shard across accounts by seed.

What "paper-ready" adds over a single sweep
-------------------------------------------
* --seeds gives REPLICATES, not just repeated training. Seed k is used for the
  data config (so the mask realization and the sampled trajectories differ) AND
  for the training RNG (init, conditional-path noise, shuffle order). Varying
  only the training seed would leave mask-draw variance unmeasured, which is
  indefensible in a paper whose entire subject is missingness. Per-seed configs
  are generated on the fly because data/<name>.pkl is keyed by config NAME.
* The interpolant list takes bspline1..bspline5, so the B-spline DEGREE SWEEP
  is in the trained results and not only in the section-5 gate. The gate shows
  quintic B-splines beating cubic Hermite on derivative error at p=0.75; a
  reviewer will ask whether that survives training, and the answer has to be in
  the table.
* Every completed run appends to results/_launcher/index_<exp>.jsonl with its
  resolved directory, MSE and pair-fallback rate (R4 requires the fallback rate
  to be reported alongside any Hermite number). scripts/aggregate_results.py
  turns those manifests into the paper tables -- no fragile path parsing.

Operational notes
-----------------
* main.py takes device = torch.device("cuda") -> cuda:0 only, no DataParallel,
  so separate processes under CUDA_VISIBLE_DEVICES are the only way to use the
  second T4.
* get_dataset() writes data/<cfg>.pkl with no locking, so concurrent cold-cache
  writers corrupt each other. Caches are always built serially first.
* main.py's run dir is {exp}_{config}_{kind}_{mask} and does NOT include the
  degree, so bspline k=1..5 would all collide. The degree is folded into
  exp_name; the seed rides in the config name.

Examples
--------
    # one replicate of one system -- the standard session unit (~2.5h)
    python scripts/run_kaggle.py --systems pendulum --seeds 0

    # headline interpolants only, all five replicates
    python scripts/run_kaggle.py --systems damped_harmonic --seeds 0,1,2,3,4 \
        --interpolants hermite_hedge,hermite_pure,bspline3,linear

    # plan without running
    python scripts/run_kaggle.py --systems n_body --seeds 0 --dry_run
"""
import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = PROJECT_ROOT / "configs" / "data_configs"

# Layout is "positions first, then velocities" everywhere, so a system with n
# degrees of freedom pairs as i:(i+n). The map itself is NOT duplicated here:
# each config JSON carries a "pairs" field that main.py::resolve_pairs reads.
TIER1 = ["harmonic_oscillator", "damped_harmonic", "hopperphysics"]
TIER2 = ["pendulum", "double_pendulum", "duffing", "spring_mass", "n_body"]
SYSTEMS = TIER1 + TIER2

# The full paper matrix. bspline3 is the matched-degree competitor for cubic
# Hermite (R3) and is what the paper calls SplineFlow; k=1,2,4,5 are the degree
# ablation; linear is the floor.
PAPER_INTERPOLANTS = ["hermite_hedge", "hermite_pure", "linear",
                      "bspline1", "bspline2", "bspline3", "bspline4", "bspline5"]

SPARSITY_SUFFIX = {"base": "", "sparse": "_sparse",
                   "v_sparse": "_v_sparse", "vv_sparse": "_vv_sparse"}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--systems", default="tier2",
                   help="comma-separated names, or 'tier1' / 'tier2' / 'all'")
    p.add_argument("--seeds", default="0",
                   help="comma-separated replicate seeds; each seeds BOTH the data "
                        "config and the training RNG")
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--exp_name", default="hermite")
    p.add_argument("--interpolants", default=",".join(PAPER_INTERPOLANTS),
                   help="hermite_hedge, hermite_pure, linear, bspline1..bspline5")
    p.add_argument("--sparsities", default="base,sparse,v_sparse,vv_sparse")
    p.add_argument("--mask_modes", default="per_dim,per_dof")
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--per_gpu", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--prepare_only", action="store_true")
    return p.parse_args()


def resolve_systems(spec):
    if spec in ("all", "tier1", "tier2"):
        return {"all": SYSTEMS, "tier1": list(TIER1), "tier2": list(TIER2)}[spec]
    out = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [s for s in out if s not in SYSTEMS]
    if unknown:
        sys.exit(f"unknown system(s): {unknown}\nknown: {SYSTEMS}")
    return out


def split_kind(token):
    """'bspline3' -> ('bspline', 3);  'hermite_pure' -> ('hermite_pure', None)."""
    m = re.fullmatch(r"bspline([1-5])", token)
    if m:
        return "bspline", int(m.group(1))
    if token not in ("hermite_hedge", "hermite_pure", "linear", "bspline"):
        sys.exit(f"unknown interpolant '{token}'; expected one of {PAPER_INTERPOLANTS}")
    return token, (3 if token == "bspline" else None)


def seeded_config(base_cfg, seed):
    """Materialise <base_cfg>__s<seed>.json, a copy with "seed" set.

    data/<name>.pkl is keyed by config NAME only, so a per-seed config is what
    gives each replicate its own trajectory draw and its own mask realization.
    """
    name = f"{base_cfg}__s{seed}"
    path = CFG_DIR / f"{name}.json"
    src = CFG_DIR / f"{base_cfg}.json"
    if not src.exists():
        sys.exit(f"missing config: {src}")
    j = json.loads(src.read_text(encoding="utf-8"))
    j["seed"] = int(seed)
    desired = json.dumps(j, indent=4) + "\n"
    if not path.exists() or path.read_text(encoding="utf-8") != desired:
        path.write_text(desired, encoding="utf-8")
    return name


def build_matrix(args):
    systems = resolve_systems(args.systems)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    kinds = [split_kind(k.strip()) for k in args.interpolants.split(",") if k.strip()]
    modes = [m.strip() for m in args.mask_modes.split(",") if m.strip()]
    sparsities = [s.strip() for s in args.sparsities.split(",") if s.strip()]
    for s in sparsities:
        if s not in SPARSITY_SUFFIX:
            sys.exit(f"unknown sparsity '{s}'; expected {list(SPARSITY_SUFFIX)}")

    runs = []
    for sysname, sp, (kind, deg), mode, seed in itertools.product(
            systems, sparsities, kinds, modes, seeds):
        # p=0 makes the mask all-ones, so per_dim and per_dof are the same
        # experiment; run the base config under one mode only.
        if sp == "base" and mode != modes[0]:
            continue
        base_cfg = f"{sysname}{SPARSITY_SUFFIX[sp]}"
        cfg = seeded_config(base_cfg, seed)
        exp = args.exp_name if deg is None else f"{args.exp_name}_k{deg}"
        label = kind if deg is None else f"bspline{deg}"
        runs.append({"system": sysname, "sparsity": sp, "config": cfg,
                     "base_config": base_cfg, "kind": kind, "degree": deg,
                     "mask": mode, "seed": seed, "exp": exp, "label": label,
                     "tag": f"{cfg}__{label}__{mode}"})
    return runs


def prepare_caches(runs):
    """Build every data/*.pkl serially -- get_dataset() has no locking."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from data_preprocessing.dataloader import get_dataset

    for cfg, mode in sorted({(r["config"], r["mask"]) for r in runs}):
        cache = PROJECT_ROOT / "data" / (
            f"{cfg}.pkl" if mode == "per_dim" else f"{cfg}__{mode}.pkl")
        if cache.exists():
            continue
        j = json.loads((CFG_DIR / f"{cfg}.json").read_text(encoding="utf-8"))
        t0 = time.time()
        get_dataset(cfg, j["reqs"], j.get("times", [0.0, 10.0, 0.1]),
                    j.get("missing_prob", 0), j.get("seed", 42), mask_mode=mode)
        print(f"  generated   {cache.name}  ({time.time() - t0:.1f}s)", flush=True)


def result_dirs(run):
    base = f"{run['exp']}_{run['config']}_{run['kind']}_{run['mask']}"
    res = PROJECT_ROOT / "results"
    if not res.exists():
        return []
    return sorted(d for d in res.glob(base + "*")
                  if d.is_dir() and re.fullmatch(re.escape(base) + r"(_\d+)?", d.name))


def finished_dir(run):
    for d in result_dirs(run):
        if (d / "test_validation_metrics.json").exists():
            return d
    return None


def config_pairs(cfg):
    j = json.loads((CFG_DIR / f"{cfg}.json").read_text(encoding="utf-8"))
    return str(j.get("pairs", "")).strip()


def command_for(args, run):
    cmd = [sys.executable, "main.py",
           "--data_config", run["config"],
           "--interpolant_kind", run["kind"],
           "--mask_mode", run["mask"],
           "--exp_name", run["exp"],
           "--epochs", str(args.epochs),
           "--batch_size", str(args.batch_size),
           "--eval_every", str(args.eval_every),
           "--seed", str(run["seed"])]
    if run["kind"] == "bspline":
        cmd += ["--degree", str(run["degree"])]
    if run["kind"].startswith("hermite"):
        pairs = config_pairs(run["config"])
        if not pairs:
            sys.exit(f'{run["config"]}.json has no "pairs" field, and {run["kind"]} '
                     f"without pairs is silently just a B-spline (R5).")
        cmd += ["--pairs", pairs]
    return cmd


def harvest(run, log_path):
    """MSE from the result dir + pair-fallback rate from the run's stdout (R4)."""
    out = {"mse": None, "fallback_pct": None, "result_dir": None}
    d = finished_dir(run)
    if d is not None:
        out["result_dir"] = d.name
        try:
            v = json.loads((d / "test_validation_metrics.json").read_text()).get("mse")
            out["mse"] = float(v) if isinstance(v, (int, float)) else None
        except Exception:
            pass
    if run["kind"].startswith("hermite") and log_path.exists():
        rates = re.findall(r"pair fallback rate:[^(]*\(([\d.]+)%",
                           log_path.read_text(errors="replace"))
        if rates:
            out["fallback_pct"] = max(float(r) for r in rates)
    return out


def main():
    args = parse_args()
    runs = build_matrix(args)
    log_dir = PROJECT_ROOT / "results" / "_launcher"
    log_dir.mkdir(parents=True, exist_ok=True)
    index = log_dir / f"index_{args.exp_name}.jsonl"

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    slots = len(gpus) * args.per_gpu
    # Honest ETA range. 0.08 s/epoch was measured with TWO processes sharing
    # one T4, i.e. that card was already delivering ~25 epochs/s = ~4.9 TFLOPS,
    # about 60% of its fp32 peak. The GPU is therefore close to saturated at 2
    # processes and extra slots mostly divide the same throughput rather than
    # adding to it:
    #   lo -> per-run time unchanged as slots grow (perfect scaling)
    #   hi -> per-GPU throughput pinned at ~25 epochs/s
    work = len(runs) * args.epochs
    n_cache = len({(r['config'], r['mask']) for r in runs})
    fixed_h = (len(runs) * 90 / slots + 12 * n_cache) / 3600
    est_lo = work * 0.08 / slots / 3600 + fixed_h
    est_hi = work / (25.0 * len(gpus)) / 3600 + fixed_h

    print(f"systems     : {sorted({r['system'] for r in runs})}")
    print(f"seeds       : {sorted({r['seed'] for r in runs})}")
    print(f"interpolants: {sorted({r['label'] for r in runs})}")
    print(f"runs        : {len(runs)}  ({args.epochs} epochs each)")
    print(f"parallelism : {len(gpus)} GPU(s) x {args.per_gpu} = {slots} concurrent")
    print(f"ETA         : {est_lo:.1f}-{est_hi:.1f}h  (perfect-scaling .. GPU-saturated)")
    if est_hi > 11.0:
        print("  WARNING: upper estimate exceeds Kaggle's 12h session cap -- "
              "split into fewer seeds/systems per session.")
    print()

    if args.dry_run:
        for r in runs:
            print("  " + " ".join(command_for(args, r)[1:]))
        return

    print("preparing data caches (serial, required before fan-out)")
    prepare_caches(runs)
    if args.prepare_only:
        return

    pending = [r for r in runs if args.force or finished_dir(r) is None]
    if len(pending) < len(runs):
        print(f"\nresuming: {len(runs) - len(pending)} done, {len(pending)} to go")
    print()

    running, done, t0 = [], [], time.time()
    queue = list(pending)
    while queue or running:
        while queue and len(running) < slots:
            run = queue.pop(0)
            gpu = gpus[len(done + running) % len(gpus)]
            log_path = log_dir / f"{run['exp']}__{run['tag']}.log"
            fh = open(log_path, "w")
            proc = subprocess.Popen(command_for(args, run), cwd=str(PROJECT_ROOT),
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
                                    stdout=fh, stderr=subprocess.STDOUT)
            running.append({"run": run, "proc": proc, "fh": fh, "gpu": gpu,
                            "log": log_path, "t": time.time()})
            print(f"[{time.time() - t0:7.1f}s] START  gpu{gpu}  {run['tag']}", flush=True)

        time.sleep(5)
        for it in list(running):
            rc = it["proc"].poll()
            if rc is None:
                continue
            running.remove(it)
            it["fh"].close()
            it["rc"] = rc
            done.append(it)
            info = harvest(it["run"], it["log"])
            rec = dict(it["run"], returncode=rc,
                       minutes=round((time.time() - it["t"]) / 60, 2),
                       epochs=args.epochs, **info)
            with index.open("a", encoding="utf-8") as fo:
                fo.write(json.dumps(rec) + "\n")
            mse = f"mse={info['mse']:.3e}" if info["mse"] is not None else "mse=?"
            fb = "" if info["fallback_pct"] is None else f" fallback={info['fallback_pct']:.1f}%"
            print(f"[{time.time() - t0:7.1f}s] {'OK  ' if rc == 0 else f'FAIL({rc})'} "
                  f"gpu{it['gpu']}  {it['run']['tag']}  "
                  f"{(time.time() - it['t']) / 60:.1f}min  {mse}{fb}", flush=True)

    failed = [d for d in done if d["rc"] != 0]
    print(f"\n{'=' * 78}\nfinished {len(done)} run(s) in {(time.time() - t0) / 3600:.2f}h  "
          f"failures: {len(failed)}")
    for d in failed:
        print(f"\n  {d['run']['tag']}")
        for line in d["log"].read_text(errors="replace").strip().splitlines()[-3:]:
            print(f"      {line}")
    print(f"\nmanifest: {index.relative_to(PROJECT_ROOT)}")
    print(f"aggregate with: python scripts/aggregate_results.py --exp_name {args.exp_name}")


if __name__ == "__main__":
    main()
