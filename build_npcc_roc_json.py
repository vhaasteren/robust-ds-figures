#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the ROC curve of the NPCC statistic + baseline curves (DFCC, NPMV, NP) in one pass.

Notation follows the paper *"Optimal robust detection statistics for pulsar
timing arrays"*: Q is a filter, H_N and H_S are the null (noise) and signal
hypotheses with covariances N and S, FAP is the false-alarm probability and
DP is the detection probability.

NPCC is the cross-correlation-only Neyman–Pearson statistic: the zero-diagonal
filter that maximizes the DP at fixed FAP. Its filter depends on the FAP, so
optimize-filter.py is run once per FAP, and the NPCC ROC curve is the envelope
(pointwise maximum) of the ROC curves of those filters.

Outputs (both are always written):
  • --npcc-out (default: ./npcc-figure-data.json)
      Envelope + per-filter diagnostic curves (FAP grid, DP, winners, sources).
  • --fig-out (default: ./genx2-optimized-figure-data.json)
      Baseline CDFs + NPCC CDFs under H_N (hN) and H_S (hS), ready for the figure script:
         - ds_def_cdf_hN,  ds_def_cdf_hS    (deflection filter DF = DFCC for the CURN null)
         - ds_npmv_cdf_hN, ds_npmv_cdf_hS
         - ds_np_cdf_hN,   ds_np_cdf_hS
         - ds_npcc_cdf_hN, ds_npcc_cdf_hS   (CDFs; plot 1 - these)
         - Qdef, Qnpmv, Qnp  (normalized filter matrices for reference)

Key implementation details:
  • Uses the same NG15yr pulsar subset and HD matrix as optimize-filter.py.
  • Complex toy model: eigenvalues λ of L^T Q L enter Imhof as weights λ/2, twice each.
  • For each optimized run (each fap_* folder), we:
       - build CDFs under H_N/H_S over a τ-grid,
       - convert to (FAP, DP) curves,
       - sort by FAP, de-duplicate, enforce monotone DP via cumulative max,
       - linearly interpolate DP onto a shared FAP grid.
    The NPCC envelope is the pointwise max across these interpolated curves.
  • ds_npcc_cdf_* are produced from the envelope via: CDF = 1 - (FAP or DP).

Example:
  python build_npcc_roc_json.py \
     --root  /data/runs/npcc_fap_sweep \
     --npcc-out /data/npcc-figure-data.json \
     --fig-out  /data/genx2-optimized-figure-data.json \
     --npsrs 67 --x-min -40 --x-max 50 --nx 400 \
     --grid-log 1e-8 1e-2 400 --grid-lin 1e-2 1.0 400
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg as sl
import scipy.integrate as sint


# ===================== NG15yr pulsar positions (same as optimize-filter.py) =====================

psrs_pos_15yr = np.array([
    [ 0.235276  , -0.95735312,  0.16769083],
    [ 0.38400231, -0.84770959,  0.36596539],
    [ 0.41542527, -0.76954724,  0.4849937 ],
    [ 0.98151466,  0.10005025,  0.16315308],
    [ 0.98761743,  0.1320268 ,  0.08473928],
    [ 0.42846049,  0.61411581,  0.66278457],
    [ 0.40865968,  0.75692046,  0.5099693 ],
    [ 0.23974131,  0.63501757, -0.73435467],
    [ 0.21645786,  0.96384399,  0.15540512],
    [  0.01039278,  0.96199519,  0.27286854],
    [ -0.01751747,  0.78824583,  0.61511109],
    [ -0.04164258,  0.93260297, -0.3584935 ],
    [ -0.05984842,  0.99758918, -0.03512827],
    [ -0.09763521,  0.61504089,  0.78242704],
    [ -0.12278291,  0.60370332,  0.78769706],
    [ -0.29600044,  0.95123349,  0.08682508],
    [ -0.17079255,  0.36310383,  0.91596153],
    [ -0.75320918,  0.57110276, -0.32637029],
    [ -0.53542437,  0.27117419,  0.7998658 ],
    [ -0.65620075,  0.3335737 , -0.6768524 ],
    [ -0.89776449,  0.40457057,  0.17418833],
    [ -0.90722641,  0.40087068, -0.12744779],
    [ -0.20004911,  0.02989931,  0.97932956],
    [ -0.94989858, -0.31220605,  0.01483487],
    [ -0.68637896, -0.64999598,  0.32617351],
    [ -0.60026435, -0.57865457, -0.55212462],
    [ -0.42623765, -0.74474287, -0.51349734],
    [ -0.41001064, -0.82785089, -0.38282395],
    [ -0.30134139, -0.73299525,  0.60984533],
    [ -0.31514962, -0.8691583 ,  0.38110965],
    [ -0.31941951, -0.92289786, -0.21501328],
    [ -0.22172453, -0.91879364, -0.32658304],
    [ -0.19826516, -0.97072221,  0.1356072 ],
    [ -0.17147358, -0.95224538, -0.25263717],
    [ -0.11864452, -0.91230783, -0.3919412 ],
    [ -0.09176163, -0.99385072,  0.0619721 ],
    [ -0.07820489, -0.96771973,  0.23958825],
    [ -0.0662458 , -0.97739637, -0.2007681 ],
    [ -0.06193343, -0.9819404 ,  0.17876603],
    [ -0.04035017, -0.75802512, -0.65097603],
    [ -0.03227138, -0.87433771, -0.48424387],
    [  0.00848601, -0.93101055, -0.36489361],
    [  0.04511657, -0.91180086, -0.40814664],
    [  0.13956703, -0.97881621, -0.14979946],
    [  0.18584638, -0.96310206,  -0.19466777],
    [  0.22722025, -0.94725437,  0.22600911],
    [  0.27135164, -0.96059139,  0.06027006],
    [  0.23711595, -0.7544394 , -0.61204348],
    [  0.29372463, -0.92928895,  0.22393723],
    [  0.29978395, -0.92373646,  0.23841253],
    [  0.33478861, -0.93502151, -0.11683901],
    [  0.32179346, -0.84518427,  0.42674643],
    [  0.4334279 , -0.88713049,  0.1585552 ],
    [  0.37000908, -0.73873966,  0.56334447],
    [  0.52541139, -0.81868445, -0.23172967],
    [  0.56102516, -0.82105832,  0.10542299],
    [  0.59166697, -0.74744543,  0.3020853 ],
    [  0.62469462, -0.72277152,  0.29556381],
    [  0.64610076, -0.51980032, -0.55889304],
    [  0.8257148 , -0.54735299, -0.13638099],
    [  0.77604174, -0.38418607,  0.50016026],
    [  0.82490391, -0.34232731,  0.44982836],
    [  0.92560095, -0.36281066,  0.10784849],
    [  0.91822693, -0.35808984,  0.16920692],
    [  0.68868989, -0.17559946,  0.70347073],
    [  0.9505931 , -0.17981436,  0.25306037],
    [  0.92132969, -0.15264057,  0.35756462]
])


# ===================== HD matrix & covariances =====================

def hdcorrmat(psrpos: np.ndarray, psrTerm: bool = True) -> np.ndarray:
    """Hellings–Downs correlation matrix (safe logs) with optional pulsar term on diag."""
    cosgamma = np.clip(psrpos @ psrpos.T, -1, 1)
    xp = 0.5 * (1 - cosgamma)
    old = np.seterr(all="ignore")
    logxp = 1.5 * xp * np.log(xp)
    np.fill_diagonal(logxp, 0)
    np.seterr(**old)
    coeff = 0.5 if psrTerm else 0.0
    return logxp - 0.25 * xp + 0.5 + coeff * np.eye(len(cosgamma))


def get_cov_matrices(h: float, hdmat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """N (CURN null hypothesis H_N) and S = I + h^2 * HD (signal hypothesis H_S) for h=1."""
    C_noise = np.identity(len(hdmat))
    C_signal = (h**2) * hdmat
    N = C_noise + np.diag(np.diag(C_signal))  # CURN
    S  = C_noise + C_signal
    return N, S


# ===================== Canonical filters & normalization =====================

def norm_filter_N(Q: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Normalize Q by the N-inner product: Q / sqrt(tr(Q N Q N))."""
    nrm2 = np.trace(Q @ N @ Q @ N)
    if not np.isfinite(nrm2) or nrm2 <= 0:
        raise ValueError("Non-positive norm for filter.")
    return Q / np.sqrt(nrm2)


def get_all_filters(h: float, hdmat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build NP, NPMV, DF filters and Cholesky factors (N = LN LN^T, S = LS LS^T).

    For the diagonal N used here, NPMV is the NP filter with its diagonal set
    to zero, and the deflection filter DF (QDEF) equals DFCC.
    """
    N, S = get_cov_matrices(h, hdmat=hdmat)
    LN = np.diag(np.sqrt(np.diag(N)))  # whitening factor for H_N
    LS = sl.cholesky(S, lower=True)
    N_inv = np.diag(1/np.diag(N))

    QNP  = N_inv - sl.cho_solve((LS, True), np.identity(len(S)))
    QNPMV = QNP.copy(); np.fill_diagonal(QNPMV, 0)
    QDEF = N_inv @ (S - N) @ N_inv

    QNPMV = norm_filter_N(QNPMV, N)
    QNP  = norm_filter_N(QNP , N)
    QDEF = norm_filter_N(QDEF, N)
    return QNP, QNPMV, QDEF, LN, LS


# ===================== GX^2 CDF via Imhof (complex-valued data) =====================

def _imhof_integrand(u: float, x: float, eigs: np.ndarray) -> float:
    """Scalar Imhof integrand at frequency u for threshold x and real-form weights eigs (λ/2, each twice, for complex data)."""
    theta = 0.5 * np.sum(np.arctan(eigs * u)) - 0.5 * x * u
    rho   = np.prod((1.0 + (eigs * u)**2)**0.25)
    if u == 0.0:
        return 0.0
    return float(np.sin(theta) / (u * rho))


def gx2cdf_from_eigs(evals: np.ndarray, xs: np.ndarray,
                     cutoff: float = 1e-6, limit: int = 200, epsabs: float = 1e-9) -> np.ndarray:
    """Imhof CDF for a vector of thresholds xs, with optional small-|λ| cutoff. For complex data, evals must already be the real-form weights (λ/2, each twice)."""
    w = np.asarray(evals, dtype=float)
    if cutoff > 0:
        w = w[np.abs(w) > cutoff]
    if w.size == 0:
        return np.where(xs >= 0.0, 1.0, 0.0)
    out = []
    for x in xs:
        val, _ = sint.quad(lambda u: _imhof_integrand(u, float(x), w),
                           0.0, np.inf, limit=limit, epsabs=epsabs)
        cdf = 0.5 - val / np.pi
        out.append(float(min(1.0, max(0.0, cdf))))
    return np.array(out, dtype=float)


def get_gx2_cdf(L: np.ndarray, Q: np.ndarray, xs: np.ndarray,
                cutoff: float = 1e-6, limit: int = 200, epsabs: float = 1e-9) -> np.ndarray:
    """CDF of z† Q z for z ~ CN(0, I) after the linear map L.

    Eigenvalues λ of L^T Q L are the complex-normal weights. The equivalent
    real generalized-chi-squared uses weight λ/2 with multiplicity 2.
    """
    w = sl.eigvalsh(L.T @ Q @ L)
    if cutoff > 0:
        w = w[np.abs(w) > cutoff]
    w = np.repeat(w, 2) * 0.5
    return gx2cdf_from_eigs(w, xs, cutoff=0.0, limit=limit, epsabs=epsabs)


# ===================== Filesystem helpers for per-FAP runs =====================

def _read_json(p: Path) -> Optional[dict]:
    try:
        with open(p, "r") as fp:
            return json.load(fp)
    except Exception:
        return None


def _parse_fap_from_folder(name: str) -> Optional[float]:
    # expect names like "fap_1e-7", "fap_1e+0"
    if not name.startswith("fap_"):
        return None
    token = name[4:]
    try:
        return float(token)
    except Exception:
        try:
            return float(token.split('_')[0])
        except Exception:
            return None


def _scaled_filter_path(outdir: Path) -> Optional[Path]:
    """Path of the scaled filter of a run: Q_star.npy (D_star.npy for runs
    written by earlier versions of optimize-filter.py), or None if missing."""
    for name in ("Q_star.npy", "D_star.npy"):
        if (outdir / name).exists():
            return outdir / name
    return None


def _pick_best_subrun(fap_dir: Path) -> Optional[Path]:
    """Priority:
       1) If top-level Q_star.npy exists, return fap_dir itself (finished)
       2) search/best if exists
       3) highest-DP result.json under search/ensemble
    """
    if _scaled_filter_path(fap_dir) is not None:
        return fap_dir

    best = fap_dir / "search" / "best"
    if best.exists():
        try:
            return best.resolve()
        except Exception:
            return best

    candidates: List[Tuple[float, Path]] = []
    for rp in (fap_dir / "search").rglob("result.json"):
        res = _read_json(rp)
        if not res:
            continue
        dp = float(res.get("DP", float("-inf")))
        if math.isfinite(dp):
            candidates.append((dp, rp.parent))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def _token_from_fap(fap: float) -> str:
    if fap == 0:
        return "0"
    if fap >= 0.1:
        return f"{fap:.1f}"
    return f"{fap:.0e}"


# ===================== Per-run curve construction (for NPCC envelope) =====================

def build_curve_for_outdir(outdir: Path,
                           xs: np.ndarray,
                           cutoff: float, limit: int, epsabs: float
                           ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Return (FAP_sorted, DP_monotone, meta) for one optimized run outdir."""
    Qpath = _scaled_filter_path(outdir)
    if Qpath is None:
        raise FileNotFoundError(f"Missing Q_star.npy in {outdir}")
    Q = np.load(Qpath)
    n = Q.shape[0]

    # Build LN, LS consistent with optimize-filter
    psrpos = psrs_pos_15yr[:n, :]
    hd = hdcorrmat(psrpos, psrTerm=True)
    N, S = get_cov_matrices(1.0, hd)
    LN = np.diag(np.sqrt(np.diag(N)))
    LS = sl.cholesky(S, lower=True)

    # CDFs along τ-grid
    cdf_N = get_gx2_cdf(LN, Q, xs, cutoff=cutoff, limit=limit, epsabs=epsabs)
    cdf_S = get_gx2_cdf(LS, Q, xs, cutoff=cutoff, limit=limit, epsabs=epsabs)

    fap = 1.0 - cdf_N
    dp  = 1.0 - cdf_S

    # Sort by FAP, de-duplicate FAPs (keep max DP for each FAP), then enforce monotone DP
    order = np.argsort(fap)
    fap_sorted = fap[order]
    dp_sorted  = dp[order]

    funiq, inv = np.unique(fap_sorted, return_inverse=True)
    dp_max = np.zeros_like(funiq)
    for k in range(funiq.size):
        dp_max[k] = np.max(dp_sorted[inv == k])

    dp_iso = np.maximum.accumulate(dp_max)  # monotone non-decreasing DP(FAP)

    meta = _read_json(outdir / "result.json") or _read_json(outdir / "summary.json") or {}
    return funiq, dp_iso, meta


def interpolate_dp_on_grid(fap: np.ndarray, dp: np.ndarray, fap_grid: np.ndarray) -> np.ndarray:
    """Linear interpolation of DP on a shared FAP grid, with edge clamping."""
    tgt = np.clip(fap_grid, float(fap.min()), float(fap.max()))
    return np.interp(tgt, fap, dp)


def build_envelope_on_grid(fap_grid: np.ndarray,
                           curves: Dict[str, Dict[str, np.ndarray]]
                           ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Pointwise max of interpolated curves; also record the winner label."""
    labels = list(curves.keys())
    M = len(labels)
    K = len(fap_grid)
    dp_mat = np.zeros((M, K), dtype=float)
    for i, lab in enumerate(labels):
        dp_mat[i, :] = curves[lab]["dp_interp"]
    winners_idx = np.argmax(dp_mat, axis=0)
    dp_env = dp_mat[winners_idx, np.arange(K)]
    winners = [labels[i] for i in winners_idx]
    return fap_grid, dp_env, winners


# ===================== CLI & main =====================

def main() -> None:
    ap = argparse.ArgumentParser(description="Build NPCC ROC curve (envelope of per-FAP filters) + baseline curves JSON.")
    ap.add_argument("--root", type=str, required=True,
                    help="Root directory containing fap_* folders (e.g., /data/runs/npcc_fap_sweep).")

    # Outputs
    ap.add_argument("--npcc-out", type=str, default="./npcc-figure-data.json",
                    help="Envelope + per-filter curves JSON.")
    ap.add_argument("--fig-out", type=str, default="./genx2-optimized-figure-data.json",
                    help="Figure JSON with baseline curves + NPCC CDFs.")
    # Backward-compat alias for --npcc-out
    ap.add_argument("--out", type=str, default=None,
                    help="(Deprecated) If set and --npcc-out not set, will be used as npcc-out.")

    # Problem size & grids
    ap.add_argument("--npsrs", type=int, default=67, help="# pulsars from the NG15 subset.")
    ap.add_argument("--x-min", type=float, default=-40.0, help="τ grid min for CDF integration.")
    ap.add_argument("--x-max", type=float, default= 50.0, help="τ grid max for CDF integration.")
    ap.add_argument("--nx",    type=int,   default=400,   help="# τ points for CDF integration.")

    # Imhof integration controls
    ap.add_argument("--cutoff", type=float, default=1e-6)
    ap.add_argument("--limit",  type=int,   default=200)
    ap.add_argument("--epsabs", type=float, default=1e-9)

    # Report FAP grid for envelope (merged log + lin)
    ap.add_argument("--grid-log", nargs=3, metavar=("FMIN", "FMAX", "N"),
                    default=["1e-8", "1e-2", "400"])
    ap.add_argument("--grid-lin", nargs=3, metavar=("FMIN", "FMAX", "N"),
                    default=["1e-2", "1.0", "400"])

    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[ERR] Root does not exist: {root}", file=sys.stderr)
        sys.exit(2)

    npcc_out = Path(args.npcc_out if args.out is None else (args.npcc_out or args.out)).resolve()
    fig_out  = Path(args.fig_out).resolve()

    # -------- Baseline problem setup (for DFCC/NP/NPMV curves) --------
    n = int(args.npsrs)
    psrpos = psrs_pos_15yr[:n, :]
    hdmat = hdcorrmat(psrpos, psrTerm=True)
    QNP, QNPMV, QDEF, LN, LS = get_all_filters(1.0, hdmat)
    N = LN @ LN.T

    xs = np.linspace(args.x_min, args.x_max, args.nx)

    # Baseline CDFs under H_N and H_S (complex-valued toy model)
    ds_def_cdf_hN = get_gx2_cdf(LN, QDEF, xs, cutoff=args.cutoff, limit=args.limit, epsabs=args.epsabs)
    ds_npmv_cdf_hN = get_gx2_cdf(LN, QNPMV, xs, cutoff=args.cutoff, limit=args.limit, epsabs=args.epsabs)
    ds_np_cdf_hN  = get_gx2_cdf(LN, QNP , xs, cutoff=args.cutoff, limit=args.limit, epsabs=args.epsabs)

    ds_def_cdf_hS = get_gx2_cdf(LS, QDEF, xs, cutoff=args.cutoff, limit=args.limit, epsabs=args.epsabs)
    ds_npmv_cdf_hS = get_gx2_cdf(LS, QNPMV, xs, cutoff=args.cutoff, limit=args.limit, epsabs=args.epsabs)
    ds_np_cdf_hS  = get_gx2_cdf(LS, QNP , xs, cutoff=args.cutoff, limit=args.limit, epsabs=args.epsabs)

    # -------- Discover per-FAP runs, build per-run curves, interpolate, envelope --------
    fap_dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and p.name.startswith("fap_")]
    if not fap_dirs:
        print(f"[ERR] No fap_* subdirectories under {root}", file=sys.stderr)
        sys.exit(2)

    print(f"[info] Root: {root}")
    print(f"[info] Found {len(fap_dirs)} FAP folders.")

    chosen: Dict[str, Path] = {}
    sources: Dict[str, dict] = {}

    for fd in fap_dirs:
        outdir = _pick_best_subrun(fd)
        if outdir is None:
            print(f"[warn] No successful runs found under {fd}; skipping.")
            continue

        res = _read_json(outdir / "result.json") or _read_json(outdir / "summary.json") or {}
        fap_val = res.get("faprob", None)
        if fap_val is None:
            fap_val = _parse_fap_from_folder(fd.name)
        tok = fd.name if fap_val is None else _token_from_fap(float(fap_val))

        chosen[tok] = outdir
        sources[tok] = {"outdir": str(outdir), "meta": res}
        print(f"[pick] {fd.name} → {outdir} (DP={res.get('DP','?')}, fap={res.get('faprob', fap_val)})")

    if not chosen:
        print("[ERR] No usable runs discovered.", file=sys.stderr)
        sys.exit(2)

    # FAP grid for envelope (merged log + linear)
    fmin_log, fmax_log, n_log = float(args.grid_log[0]), float(args.grid_log[1]), int(args.grid_log[2])
    fmin_lin, fmax_lin, n_lin = float(args.grid_lin[0]), float(args.grid_lin[1]), int(args.grid_lin[2])
    grid_log = np.logspace(np.log10(fmin_log), np.log10(fmax_log), n_log) if n_log > 0 else np.array([], float)
    grid_lin = np.linspace(fmin_lin, fmax_lin, n_lin) if n_lin > 0 else np.array([], float)
    fap_grid = np.unique(np.clip(np.concatenate([grid_log, grid_lin]), 0.0, 1.0))

    # Per-run curves → interpolation on the shared grid
    curves: Dict[str, Dict[str, np.ndarray]] = {}
    for tok, outdir in chosen.items():
        try:
            fap_curve, dp_curve, meta = build_curve_for_outdir(
                outdir, xs=xs,
                cutoff=args.cutoff, limit=args.limit, epsabs=args.epsabs
            )
            dp_interp = interpolate_dp_on_grid(fap_curve, dp_curve, fap_grid)
            curves[tok] = {
                "fap_curve": fap_curve,
                "dp_curve": dp_curve,
                "dp_interp": dp_interp,
                "DP_reported": float(meta.get("DP", np.nan)),
                "scale": float(meta.get("scale", np.nan)),
                "outdir": str(outdir)
            }
            print(f"[curve] {tok}: built & interpolated (n={np.load(_scaled_filter_path(outdir)).shape[0]}), "
                  f"DP_reported={meta.get('DP','?')}, scale={meta.get('scale','?')}")
        except Exception as e:
            print(f"[warn] Failed to build curve for {tok} at {outdir}: {e}")

    if not curves:
        print("[ERR] No curves computed.", file=sys.stderr)
        sys.exit(2)

    # Envelope on the shared grid
    fap_env, dp_env, winners = build_envelope_on_grid(fap_grid, curves)

    # -------- Write NPCC-only diagnostics JSON (envelope + per-filter curves) --------
    npcc_payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "root": str(root),
            "npsrs": n,
            "x_grid": {"min": args.x_min, "max": args.x_max, "nx": args.nx},
            "imhof": {"cutoff": args.cutoff, "limit": args.limit, "epsabs": args.epsabs},
            "envelope_grid": {
                "log": {"min": fmin_log, "max": fmax_log, "n": n_log},
                "lin": {"min": fmin_lin, "max": fmax_lin, "n": n_lin}
            }
        },
        "envelope": {
            "fap": fap_env.tolist(),
            "dp": dp_env.tolist(),
            "winner": winners
        },
        "per_filter_curves": {
            tok: {
                "fap_curve": v["fap_curve"].tolist(),
                "dp_curve": v["dp_curve"].tolist(),
                "dp_interp": v["dp_interp"].tolist(),
                "DP_reported": v["DP_reported"],
                "scale": v["scale"],
                "outdir": v["outdir"]
            }
            for tok, v in curves.items()
        },
        "sources": sources
    }

    npcc_out.parent.mkdir(parents=True, exist_ok=True)
    with open(npcc_out, "w") as fp:
        json.dump(npcc_payload, fp, indent=2)
    print(f"[done] Wrote NPCC diagnostics JSON: {npcc_out}")

    # -------- Write figure JSON (baseline + NPCC CDFs) --------
    # Convert envelope to CDF arrays for plotting (the figure script plots 1 - CDF)
    ds_npcc_cdf_hN = (1.0 - fap_env).tolist()[::-1]  # since FAP = 1 - CDF under H_N
    ds_npcc_cdf_hS = (1.0 - dp_env ).tolist()[::-1]  # since  DP = 1 - CDF under H_S

    fig_payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "root": str(root),
            "npsrs": n,
            "x_grid": {"min": args.x_min, "max": args.x_max, "nx": args.nx},
            "imhof": {"cutoff": args.cutoff, "limit": args.limit, "epsabs": args.epsabs},
        },
        # Baseline CDFs (arrays; plot 1 - these)
        "ds_def_cdf_hN": ds_def_cdf_hN.tolist(),
        "ds_def_cdf_hS": ds_def_cdf_hS.tolist(),
        "ds_npmv_cdf_hN": ds_npmv_cdf_hN.tolist(),
        "ds_npmv_cdf_hS": ds_npmv_cdf_hS.tolist(),
        "ds_np_cdf_hN":  ds_np_cdf_hN.tolist(),
        "ds_np_cdf_hS":  ds_np_cdf_hS.tolist(),
        # NPCC envelope as CDF arrays
        "ds_npcc_cdf_hN": ds_npcc_cdf_hN,
        "ds_npcc_cdf_hS": ds_npcc_cdf_hS,
        # Reference normalized filter matrices (N-inner product)
        "Qdef": QDEF.tolist(),
        "Qnpmv": QNPMV.tolist(),
        "Qnp":  QNP.tolist(),
    }

    fig_out.parent.mkdir(parents=True, exist_ok=True)
    with open(fig_out, "w") as fp:
        json.dump(fig_payload, fp, indent=2)
    print(f"[done] Wrote figure JSON: {fig_out}")


if __name__ == "__main__":
    main()

