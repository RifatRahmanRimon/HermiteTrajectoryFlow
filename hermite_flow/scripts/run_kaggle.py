#!/usr/bin/env python3
"""Parallel experiment runner for Kaggle (T4 x2).

One invocation runs a COMPLETE experiment block for the requested systems:
every sparsity x every interpolant x every mask mode, fanned out across both
GPUs, several processes per GPU.  Designed so each Kaggle session owns one or
two systems and produces a self-contained slice of the results table -- if a
session dies you lose that system, not a random scatter of cells.

Why this exists rather than scripts/run_experiments_hermite.sh:

  * that script is strictly serial and pins everything to cuda:0, so it burns
    ~6.5h of quota to do what a packed session does in well under an hour;
  * main.py takes `device = torch.device("cuda")` -> cuda:0 only, with no
    DataParallel, so the ONLY way to use the second T4 is separate processes
    with CUDA_VISIBLE_DEVICES;
  * get_dataset() writes data/<cfg>.pkl with no locking, so N processes racing
    on a cold cache corrupt each other.  --prepare below generates every pickle
    serially BEFORE any training starts.  Do not skip it.

Kaggle bills wall-clock SESSION time, not GPU time: a T4x2 session costs the
same quota per hour as a T4x1 session.  Both GPUs and every extra process per
GPU are therefore free.  Pack them.

Examples
--------
    # one system, full block, both GPUs, 4 processes each
    python scripts/run_kaggle.py --systems pendulum

    # screening pass over three systems at 2k epochs
    python scripts/run_kaggle.py --systems damped_harmonic,pendulum,duffing --epochs 2000

    # see the plan without running anything
    python scripts/run_kaggle.py --systems n_body --dry_run
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Layout is "positions first, then velocities" everywhere, so a system with n
# degrees of freedom pairs as i:(i+n).  The pair map itself is NOT duplicated
# here: every config JSON carries a "pairs" field and main.py::resolve_pairs
# reads it.  This launcher looks it up from the config and passes it explicitly,
# so the exact mapping is visible in each logged command line -- and it fails
# loudly if a hermite run has no pair map anywhere (R5: a hermite kind without
# pairs silently degrades to a plain B-spline).
TIER1 = ["harmonic_oscillator", "damped_harmonic", "hopperphysics"]
TIER2 = ["pendulum", "double_pendulum", "duffing", "spring_mass", "n_body"]
SYSTEMS = TIER1 + TIER2

# interpolant -> extra CLI args.  bspline k=3 is the matched-degree competitor
# for cubic Hermite (R3); linear is the floor.
INTERPOLANTS = {
    "hermite_hedge": [],
    "hermite_pure":  [],
    "bspline":       ["--degree", "3"],
    "linear":        [],
}

SPARSITIES = {"": 0.0, "_sparse": 0.25, "_v_sparse": 0.5, "_vv_sparse": 0.75}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--systems", default="tier2",
                   help="comma-separated system names, or 'tier1' / 'tier2' / 'all'")
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--exp_name", default="hermite",
                   help="prefix for results/ dirs; give parallel SESSIONS distinct "
                        "names if they might touch the same (config, kind, mask)")
    p.add_argument("--interpolants", default=",".join(INTERPOLANTS))
    p.add_argument("--sparsities", default="base,sparse,v_sparse,vv_sparse")
    p.add_argument("--mask_modes", default="per_dim,per_dof")
    p.add_argument("--gpus", default="0,1", help="GPU ids to spread across")
    p.add_argument("--per_gpu", type=int, default=4,
                   help="concurrent training processes per GPU")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="re-run even if a finished result dir already exists")
    p.add_argument("--prepare_only", action="store_true",
                   help="build the data/*.pkl caches and exit")
    return p.parse_args()


def resolve_systems(spec):
    if spec == "all":
        return TIER1 + TIER2
    if spec == "tier1":
        return list(TIER1)
    if spec == "tier2":
        return list(TIER2)
    out = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [s for s in out if s not in SYSTEMS]
    if unknown:
        sys.exit(f"unknown system(s): {unknown}\nknown: {list(SYSTEMS)}")
    return out


def config_pairs(cfg):
    """The (position, velocity) map for a config, read from its own JSON."""
    j = json.loads((PROJECT_ROOT / "configs" / "data_configs" / f"{cfg}.json").read_text())
    return str(j.get("pairs", "")).strip()


def build_matrix(args):
    """Every (config, interpolant, mask_mode) cell to run.

    At missing_prob == 0 the mask is all-ones, so per_dim and per_dof are
    literally the same experiment -- the base config runs once, under per_dim.
    """
    systems = resolve_systems(args.systems)
    kinds = [k.strip() for k in args.interpolants.split(",") if k.strip()]
    modes = [m.strip() for m in args.mask_modes.split(",") if m.strip()]
    sufs = ["" if s == "base" else "_" + s
            for s in (x.strip() for x in args.sparsities.split(",")) if s]

    runs = []
    for sysname, suf, kind, mode in itertools.product(systems, sufs, kinds, modes):
        if suf == "" and mode != modes[0]:
            continue                                   # p=0: one mask mode only
        cfg = f"{sysname}{suf}"
        if not (PROJECT_ROOT / "configs" / "data_configs" / f"{cfg}.json").exists():
            sys.exit(f"missing config: configs/data_configs/{cfg}.json")
        runs.append({"system": sysname, "config": cfg, "kind": kind, "mask": mode,
                     "tag": f"{cfg}__{kind}__{mode}"})
    return runs


def prepare_caches(runs):
    """Generate every data/*.pkl SERIALLY.

    get_dataset() has no locking: if the training processes hit a cold cache
    together they all generate and all write the same path, and the pickle that
    survives can be a torn interleaving of several writers.
    """
    sys.path.insert(0, str(PROJECT_ROOT))
    from data_preprocessing.dataloader import get_dataset

    wanted = sorted({(r["config"], r["mask"]) for r in runs})
    for cfg, mode in wanted:
        cache = PROJECT_ROOT / "data" / (
            f"{cfg}.pkl" if mode == "per_dim" else f"{cfg}__{mode}.pkl")
        if cache.exists():
            print(f"  cache hit   {cache.name}")
            continue
        j = json.loads((PROJECT_ROOT / "configs" / "data_configs" / f"{cfg}.json").read_text())
        t0 = time.time()
        get_dataset(cfg, j["reqs"], j.get("times", [0.0, 10.0, 0.1]),
                    j.get("missing_prob", 0), j.get("seed", 42), mask_mode=mode)
        print(f"  generated   {cache.name}  ({time.time() - t0:.1f}s)", flush=True)


def result_dirs(exp_name, run):
    base = f"{exp_name}_{run['config']}_{run['kind']}_{run['mask']}"
    res = PROJECT_ROOT / "results"
    return [d for d in res.glob(base + "*") if d.is_dir()] if res.exists() else []


def already_done(exp_name, run):
    # main.py writes test_validation_metrics.json last -> it is the completion mark
    return any((d / "test_validation_metrics.json").exists()
               for d in result_dirs(exp_name, run))


def command_for(args, run):
    cmd = [sys.executable, "main.py",
           "--data_config", run["config"],
           "--interpolant_kind", run["kind"],
           "--mask_mode", run["mask"],
           "--exp_name", args.exp_name,
           "--epochs", str(args.epochs),
           "--batch_size", str(args.batch_size),
           "--eval_every", str(args.eval_every)]
    cmd += INTERPOLANTS[run["kind"]]
    if run["kind"].startswith("hermite"):
        pairs = config_pairs(run["config"])
        if not pairs:
            sys.exit(f'{run["config"]}.json has no "pairs" field, and {run["kind"]} '
                     f'without pairs is silently just a B-spline (R5). Add it.')
        cmd += ["--pairs", pairs]
    return cmd


def read_mse(exp_name, run):
    for d in result_dirs(exp_name, run):
        f = d / "test_validation_metrics.json"
        if f.exists():
            try:
                v = json.loads(f.read_text()).get("mse")
                return float(v) if isinstance(v, (int, float)) else None
            except Exception:
                return None
    return None


def main():
    args = parse_args()
    runs = build_matrix(args)
    log_dir = PROJECT_ROOT / "results" / "_launcher"
    log_dir.mkdir(parents=True, exist_ok=True)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    slots = len(gpus) * args.per_gpu

    print(f"systems     : {sorted({r['system'] for r in runs})}")
    print(f"runs        : {len(runs)}  ({args.epochs} epochs each)")
    print(f"parallelism : {len(gpus)} GPU(s) x {args.per_gpu} = {slots} concurrent")
    print(f"exp_name    : {args.exp_name}\n")

    if args.dry_run:
        for r in runs:
            print("  " + " ".join(command_for(args, r)[1:]))
        return

    print("preparing data caches (serial, required before fan-out)")
    prepare_caches(runs)
    if args.prepare_only:
        return

    pending = [r for r in runs if args.force or not already_done(args.exp_name, r)]
    skipped = len(runs) - len(pending)
    if skipped:
        print(f"\nresuming: {skipped} run(s) already finished, {len(pending)} to go")
    print()

    running, done, t_start = [], [], time.time()
    queue = list(pending)
    while queue or running:
        while queue and len(running) < slots:
            run = queue.pop(0)
            gpu = gpus[len(done + running) % len(gpus)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            log = open(log_dir / f"{args.exp_name}__{run['tag']}.log", "w")
            proc = subprocess.Popen(command_for(args, run), cwd=str(PROJECT_ROOT),
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append({"run": run, "proc": proc, "log": log,
                            "gpu": gpu, "t0": time.time()})
            print(f"[{time.time() - t_start:7.1f}s] START  gpu{gpu}  {run['tag']}", flush=True)

        time.sleep(5)
        for item in list(running):
            rc = item["proc"].poll()
            if rc is None:
                continue
            running.remove(item)
            item["log"].close()
            item["rc"], item["elapsed"] = rc, time.time() - item["t0"]
            done.append(item)
            mse = read_mse(args.exp_name, item["run"])
            status = "OK  " if rc == 0 else f"FAIL({rc})"
            mse_s = f"mse={mse:.3e}" if mse is not None else "mse=?"
            print(f"[{time.time() - t_start:7.1f}s] {status} gpu{item['gpu']}  "
                  f"{item['run']['tag']}  {item['elapsed'] / 60:.1f}min  {mse_s}",
                  flush=True)

    failed = [d for d in done if d["rc"] != 0]
    print(f"\n{'=' * 78}\nfinished {len(done)} run(s) in "
          f"{(time.time() - t_start) / 3600:.2f}h   failures: {len(failed)}")
    if failed:
        print("\nfailed runs (tail of each log):")
        for d in failed:
            log = log_dir / f"{args.exp_name}__{d['run']['tag']}.log"
            tail = log.read_text(errors="replace").strip().splitlines()[-3:]
            print(f"  {d['run']['tag']}")
            for line in tail:
                print(f"      {line}")

    print(f"\n{'config':34s} {'interpolant':15s} {'mask':9s} {'mse':>12s}")
    print("-" * 74)
    for r in sorted(runs, key=lambda r: (r["config"], r["kind"], r["mask"])):
        mse = read_mse(args.exp_name, r)
        print(f"{r['config']:34s} {r['kind']:15s} {r['mask']:9s} "
              f"{(f'{mse:.4e}' if mse is not None else '-'):>12s}")
    print(f"\nlogs: results/_launcher/    results: results/{args.exp_name}_*")


if __name__ == "__main__":
    main()
