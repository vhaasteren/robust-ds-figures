#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global-ish search driver for the FULL problem (high-D).

Usage (typical):
  python run_full_npcc_search.py \
    --optimize-filter ./optimize-filter.py \
    --npsrs 67 --cdf analytic --faprob 2.87e-7 --tau 1.0 \
    --outroot global_search_67 \
    --start-dir inc_spsa_test/n67 \            # optional: dir with 'other' local max
    --jobs 4

What it does:
  • If --start-dir exists and has a compatible x_opt.json (size m=n(n-1)/2), we copy it
    into <outroot>/warm-other/ and use that vector as an alternative seed (via --start-json).
    If it does not exist (or mismatched), we warn and just use NPMV.
  • Runs a small ensemble/grid of SPSA→BOBYQA jobs for BOTH seeds (NPMV and/or OTHER):
      steps:  [0.8, 0.4, 0.2]
      bounds: [8.0, 6.0]
      seeds:  [1..8]   (configurable)
    Each run goes to: <outroot>/ensemble/<seed_label>/step{S}_b{B}_s{seed}/
  • Picks the top-K (default 3) by DP and anneals each in place with a resume schedule:
      stage 1: step*=0.40, bound=min(original_bound, 6.0)
      stage 2: step*=0.30, bound=4.0
      stage 3: step*=0.20, bound=3.5
      final polish in each stage via --polish bobyqa --polish-evals larger
    (You can change schedule with flags.)
  • Prints a summary and the winning outdir. Optionally creates a symlink <outroot>/best → winner.

Notes:
  - This script assumes your optimize-filter.py already supports:
      --optimizer spsa  --spsa-iters  --spsa-step
      --polish bobyqa   --polish-evals
      --resume          --start-json
  - We do NOT mutate the seed vector files; resumes write into each candidate outdir.
  - Use --jobs to parallelize the ensemble stage locally.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import subprocess
import numpy as np


def _safe(s: float) -> str:
    """Turn floats into path-safe tokens (0.8 -> 0p8)."""
    t = f"{s}".replace(".", "p").replace("-", "m")
    return t


def run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    """Run a command, raising on failure."""
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, stdout=sys.stdout, stderr=sys.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")


def read_result(outdir: Path) -> Optional[Dict]:
    j = outdir / "result.json"
    if not j.exists():
        return None
    try:
        with open(j, "r") as fp:
            return json.load(fp)
    except Exception:
        return None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_start_dir(src: Path, dst: Path) -> None:
    """Copy entire tree from src into dst (dst created if missing)."""
    if dst.exists():
        # Merge behavior: copy files that don't exist; keep existing
        for root, _, files in os.walk(src):
            rel = os.path.relpath(root, start=src)
            target_root = dst / rel
            target_root.mkdir(parents=True, exist_ok=True)
            for f in files:
                s = Path(root) / f
                d = target_root / f
                if not d.exists():
                    shutil.copy2(s, d)
    else:
        shutil.copytree(src, dst)


def load_xopt(vec_path: Path) -> Optional[np.ndarray]:
    if not vec_path.exists():
        return None
    try:
        with open(vec_path, "r") as fp:
            arr = np.array(json.load(fp), dtype=float)
        return arr
    except Exception:
        return None


def submit_one(optf: Path, base_args: List[str], outdir: Path,
               start_json: Optional[Path],
               optimizer: str, spsa_iters: int, step: float, bound: float,
               polish: str, polish_evals: int,
               seed: int, resume: bool = False) -> None:
    ensure_dir(outdir)
    cmd = [sys.executable, str(optf)] + base_args + [
        "--optimizer", optimizer,
        "--spsa-iters", str(spsa_iters),
        "--spsa-step", str(step),
        "--bound", str(bound),
        "--polish", polish,
        "--polish-evals", str(polish_evals),
        "--outdir", str(outdir),
        "--seed", str(seed),
    ]
    if start_json is not None:
        cmd += ["--start-json", str(start_json)]
    if resume:
        cmd += ["--resume"]
    run(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="NPCC global-ish search driver for FULL mode.")
    ap.add_argument("--optimize-filter", type=str, default="./optimize-filter.py",
                    help="Path to optimize-filter.py")
    ap.add_argument("--npsrs", type=int, required=True)
    ap.add_argument("--cdf", choices=["analytic", "imhof"], default="analytic")
    ap.add_argument("--faprob", type=float, required=True)
    ap.add_argument("--tau", type=float, default=1.0)

    ap.add_argument("--outroot", type=str, required=True,
                    help="Root output directory for this search.")
    ap.add_argument("--start-dir", type=str, default="",
                    help="Directory containing 'other local maximum' (with x_opt.json).")

    # Ensemble grid
    ap.add_argument("--steps", type=str, default="0.8,0.4,0.2",
                    help="Comma-separated SPSA step sizes for the ensemble.")
    ap.add_argument("--bounds", type=str, default="8.0,6.0",
                    help="Comma-separated bounds for the ensemble.")
    ap.add_argument("--seeds", type=str, default="1,2,3,4,5,6,7,8",
                    help="Comma-separated integer seeds for the ensemble.")
    ap.add_argument("--spsa-iters", type=int, default=3000)
    ap.add_argument("--polish-evals", type=int, default=1200)

    # Anneal schedule (relative step multipliers and absolute bounds)
    ap.add_argument("--topk", type=int, default=3, help="How many top ensemble runs to anneal.")
    ap.add_argument("--anneal-mults", type=str, default="0.40,0.30,0.20",
                    help="Comma-separated multipliers applied to that run's step for resume stages.")
    ap.add_argument("--anneal-bounds", type=str, default="6.0,4.0,3.5",
                    help="Comma-separated bounds used at each resume stage.")
    ap.add_argument("--anneal-iters", type=int, default=3000,
                    help="SPSA iterations per anneal stage.")
    ap.add_argument("--anneal-polish-evals", type=str, default="1600,2000,2400",
                    help="Comma-separated polish-evals per resume stage.")

    # Optional jitter before the FIRST anneal stage (small shake-out)
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="If >0, add N(0, jitter*bound) to x_opt before the first resume (per candidate).")

    ap.add_argument("--jobs", type=int, default=1, help="Parallel jobs for the ensemble grid.")

    args = ap.parse_args()

    optf = Path(args.optimize_filter).resolve()
    if not optf.exists():
        raise FileNotFoundError(f"optimize-filter.py not found at {optf}")

    outroot = Path(args.outroot).resolve()
    ensure_dir(outroot)

    # Parse lists
    steps = [float(s) for s in args.steps.split(",") if s.strip()]
    bounds = [float(b) for b in args.bounds.split(",") if b.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    anneal_mults = [float(s) for s in args.anneal_mults.split(",") if s.strip()]
    anneal_bounds = [float(s) for s in args.anneal_bounds.split(",") if s.strip()]
    anneal_polish = [int(s) for s in args.anneal_polish_evals.split(",") if s.strip()]
    if not (len(anneal_mults) == len(anneal_bounds) == len(anneal_polish)):
        raise ValueError("anneal-mults, anneal-bounds, anneal-polish-evals must have same length.")

    # Basic FULL mode args fixed across runs
    base_args = [
        "--mode", "full",
        "--npsrs", str(args.npsrs),
        "--cdf", args.cdf,
        "--faprob", str(args.faprob),
        "--tau", str(args.tau),
    ]
    optimizer = "spsa"
    polish = "bobyqa"

    # Prepare optional OTHER seed by copying start-dir into <outroot>/warm-other/
    seed_labels: List[Tuple[str, Optional[Path]]] = []
    m = args.npsrs * (args.npsrs - 1) // 2

    if args.start_dir:
        src = Path(args.start_dir).resolve()
        if src.exists():
            dst = outroot / "warm-other"
            copy_start_dir(src, dst)
            vec = load_xopt(dst / "x_opt.json")
            if vec is not None and vec.size == m:
                start_json_other = dst / "x_opt.json"  # use directly
                seed_labels.append(("other", start_json_other))
                print(f"[driver] Using OTHER seed from: {start_json_other}")
            else:
                print(f"[driver][warn] Ignoring OTHER seed: {src} (x_opt.json missing or wrong size). Falling back to NPMV only.")
        else:
            print(f"[driver][warn] start-dir does not exist: {src}. Falling back to NPMV only.")

    # Always include NPMV seed (no start-json)
    seed_labels.append(("npmv", None))

    # --------------- Ensemble grid (parallelizable) ---------------
    print("[driver] Starting ensemble grid...")
    tasks = []
    runs_meta: List[Dict] = []

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for label, start_json in seed_labels:
            for step in steps:
                for bound in bounds:
                    for sd in seeds:
                        sub = outroot / "ensemble" / label / f"step{_safe(step)}_b{_safe(bound)}_s{sd}"
                        meta = {
                            "label": label,
                            "outdir": sub,
                            "step": step,
                            "bound": bound,
                            "seed": sd,
                            "start_json": start_json,
                        }
                        runs_meta.append(meta)
                        cmd = (optf, base_args, sub, start_json,
                               optimizer, args.spsa_iters, step, bound,
                               polish, args.polish_evals, sd, False)
                        tasks.append(ex.submit(submit_one, *cmd))

        # wait for completion
        for fut in as_completed(tasks):
            try:
                fut.result()
            except Exception as e:
                print(f"[driver][warn] An ensemble run failed: {e}")

    # Collect results
    scored: List[Tuple[float, Dict]] = []
    for meta in runs_meta:
        res = read_result(meta["outdir"])
        if res and "DP" in res:
            scored.append((float(res["DP"]), meta))
    if not scored:
        raise RuntimeError("No successful ensemble runs found (no result.json with DP).")

    scored.sort(key=lambda t: t[0], reverse=True)
    print("\n[driver] Top ensemble results:")
    for k, (dp, meta) in enumerate(scored[:max(10, args.topk)]):
        print(f"  {k+1:2d}. DP={dp:.6e}  {meta['outdir']}  (seed={meta['seed']}, step={meta['step']}, bound={meta['bound']}, label={meta['label']})")

    # --------------- Anneal (resume) the top-K in place ---------------
    topK = scored[:args.topk]
    print("\n[driver] Anneal/resume schedule:")
    print(f"  mults={anneal_mults}, bounds={anneal_bounds}, polish={anneal_polish}, iters={args.anneal_iters}")

    for dp0, meta in topK:
        outdir = meta["outdir"]
        step0 = float(meta["step"])
        print(f"\n[driver] Annealing candidate: {outdir} (starting DP={dp0:.6e}, step0={step0}, bound0={meta['bound']})")

        # Optional jitter (shake-out) before first resume
        if args.jitter > 0:
            xpath = outdir / "x_opt.json"
            x = load_xopt(xpath)
            if x is not None:
                rng = np.random.default_rng(meta["seed"] ^ 0xBADC0DE)
                noise = rng.normal(0.0, args.jitter * meta["bound"], size=x.shape)
                x_new = (x + noise).tolist()
                with open(xpath, "w") as fp:
                    json.dump(x_new, fp)
                print(f"[driver] Applied jitter σ={args.jitter*meta['bound']:.3g} to {xpath}")
            else:
                print(f"[driver][warn] No x_opt.json to jitter at {xpath}")

        # Resume stages
        cur_step = step0
        for j, (mult, bnd, pevals) in enumerate(zip(anneal_mults, anneal_bounds, anneal_polish), start=1):
            cur_step = max(1e-4, cur_step * mult)
            print(f"[driver]  Stage {j}: resume with step={cur_step}, bound={bnd}, polish-evals={pevals}")
            submit_one(optf, base_args, outdir, meta["start_json"],
                       optimizer, args.anneal_iters, cur_step, bnd,
                       polish, pevals, meta["seed"], resume=True)

        # Report candidate after anneal
        res = read_result(outdir)
        if res:
            print(f"[driver]  → after anneal: DP={res.get('DP', float('nan')):.6e}  scale={res.get('scale', float('nan')):.6e}")

    # --------------- Pick overall winner ---------------
    final_scored: List[Tuple[float, Path]] = []
    for _, meta in topK:
        res = read_result(meta["outdir"])
        if res and "DP" in res:
            final_scored.append((float(res["DP"]), meta["outdir"]))
    # also consider non-annealed ensemble runs in case they beat the annealed ones
    for dp, meta in scored:
        if (dp, meta["outdir"]) not in final_scored:
            final_scored.append((dp, meta["outdir"]))

    final_scored.sort(key=lambda t: t[0], reverse=True)
    best_dp, best_dir = final_scored[0]
    print("\n[driver] ===================== RESULT =====================")
    print(f"[driver] Best DP = {best_dp:.6e}")
    print(f"[driver] Best outdir = {best_dir}")
    print("[driver] ====================================================")

    # Symlink best
    best_link = outroot / "best"
    try:
        if best_link.exists() or best_link.is_symlink():
            best_link.unlink()
        os.symlink(best_dir, best_link, target_is_directory=True)
        print(f"[driver] Created symlink: {best_link} -> {best_dir}")
    except Exception as e:
        print(f"[driver][warn] Could not create symlink {best_link}: {e}")


if __name__ == "__main__":
    main()
