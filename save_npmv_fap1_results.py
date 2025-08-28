#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Write NPMV artifacts for FAP=1 into ./fapruns/fap_1/

Outputs:
  D_unscaled.npy  – NPMV normalized under the N-inner product
  D_star.npy      – identical to D_unscaled (scale = 1.0 for FAP=1 placeholder)
  x_opt.json      – strict lower-triangular vector of D_unscaled
  result.json     – lightweight metadata (mode=npmv-fixed, faprob=1.0)

Notes:
- Uses the same NG15yr pulsar subset and HD kernel as optimize-filter.py.
- Default npsrs=67 to match your typical runs; override via --npsrs if needed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.linalg as sl

# ---------------- NG15yr pulsar positions (same as optimize-filter.py) ----------------

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
    [ 0.01039278,  0.96199519,  0.27286854],
    [-0.01751747,  0.78824583,  0.61511109],
    [-0.04164258,  0.93260297, -0.3584935 ],
    [-0.05984842,  0.99758918, -0.03512827],
    [-0.09763521,  0.61504089,  0.78242704],
    [-0.12278291,  0.60370332,  0.78769706],
    [-0.29600044,  0.95123349,  0.08682508],
    [-0.17079255,  0.36310383,  0.91596153],
    [-0.75320918,  0.57110276, -0.32637029],
    [-0.53542437,  0.27117419,  0.7998658 ],
    [-0.65620075,  0.3335737 , -0.6768524 ],
    [-0.89776449,  0.40457057,  0.17418833],
    [-0.90722641,  0.40087068, -0.12744779],
    [-0.20004911,  0.02989931,  0.97932956],
    [-0.94989858, -0.31220605,  0.01483487],
    [-0.68637896, -0.64999598,  0.32617351],
    [-0.60026435, -0.57865457, -0.55212462],
    [-0.42623765, -0.74474287, -0.51349734],
    [-0.41001064, -0.82785089, -0.38282395],
    [-0.30134139, -0.73299525,  0.60984533],
    [-0.31514962, -0.8691583 ,  0.38110965],
    [-0.31941951, -0.92289786, -0.21501328],
    [-0.22172453, -0.91879364, -0.32658304],
    [-0.19826516, -0.97072221,  0.1356072 ],
    [-0.17147358, -0.95224538, -0.25263717],
    [-0.11864452, -0.91230783, -0.3919412 ],
    [-0.09176163, -0.99385072,  0.0619721 ],
    [-0.07820489, -0.96771973,  0.23958825],
    [-0.0662458 , -0.97739637, -0.2007681 ],
    [-0.06193343, -0.9819404 ,  0.17876603],
    [-0.04035017, -0.75802512, -0.65097603],
    [-0.03227138, -0.87433771, -0.48424387],
    [ 0.00848601, -0.93101055, -0.36489361],
    [ 0.04511657, -0.91180086, -0.40814664],
    [ 0.13956703, -0.97881621, -0.14979946],
    [ 0.18584638, -0.96310206,  -0.19466777],
    [ 0.22722025, -0.94725437,  0.22600911],
    [ 0.27135164, -0.96059139,  0.06027006],
    [ 0.23711595, -0.7544394 , -0.61204348],
    [ 0.29372463, -0.92928895,  0.22393723],
    [ 0.29978395, -0.92373646,  0.23841253],
    [ 0.33478861, -0.93502151, -0.11683901],
    [ 0.32179346, -0.84518427,  0.42674643],
    [ 0.4334279 , -0.88713049,  0.1585552 ],
    [ 0.37000908, -0.73873966,  0.56334447],
    [ 0.52541139, -0.81868445, -0.23172967],
    [ 0.56102516, -0.82105832,  0.10542299],
    [ 0.59166697, -0.74744543,  0.3020853 ],
    [ 0.62469462, -0.72277152,  0.29556381],
    [ 0.64610076, -0.51980032, -0.55889304],
    [ 0.8257148 , -0.54735299, -0.13638099],
    [ 0.77604174, -0.38418607,  0.50016026],
    [ 0.82490391, -0.34232731,  0.44982836],
    [ 0.92560095, -0.36281066,  0.10784849],
    [ 0.91822693, -0.35808984,  0.16920692],
    [ 0.68868989, -0.17559946,  0.70347073],
    [ 0.9505931 , -0.17981436,  0.25306037],
    [ 0.92132969, -0.15264057,  0.35756462]
])

# ---------------- Utilities (HD kernel, covariances, normalization) ----------------

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
    """C0 (CURN) and C = I + h^2*HD for h=1."""
    C_noise = np.identity(len(hdmat))
    C_signal = (h**2) * hdmat
    C0 = C_noise + np.diag(np.diag(C_signal))  # CURN
    C  = C_noise + C_signal
    return C0, C

def norm_filter_N(Q: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Normalize Q by the N-inner product: Q / sqrt(tr(Q N Q N))."""
    nrm2 = np.trace(Q @ N @ Q @ N)
    if not np.isfinite(nrm2) or nrm2 <= 0:
        raise ValueError("Non-positive norm for filter.")
    return Q / np.sqrt(nrm2)

def get_lower_triangular_elements(D: np.ndarray) -> np.ndarray:
    i, j = np.tril_indices(D.shape[0], k=-1)
    return D[i, j]

# ---------------- Main: build & save NPMV artifacts ----------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Save NPMV artifacts for FAP=1 into ./fapruns/fap_1")
    ap.add_argument("--npsrs", type=int, default=67, help="Number of pulsars (subset of NG15 list).")
    ap.add_argument("--outdir", type=str, default="./fapruns/fap_1", help="Output directory.")
    ap.add_argument("--tau", type=float, default=1.0, help="Recorded τ (informational only).")
    ap.add_argument("--cdf", choices=["analytic","imhof"], default="analytic",
                    help="Recorded CDF backend (informational only).")
    args = ap.parse_args()

    n = int(args.npsrs)
    if n <= 1 or n > psrs_pos_15yr.shape[0]:
        raise ValueError(f"--npsrs must be in [2, {psrs_pos_15yr.shape[0]}]")

    psrpos = psrs_pos_15yr[:n, :]
    hd = hdcorrmat(psrpos, psrTerm=True)
    C0, C = get_cov_matrices(1.0, hd)
    L0 = np.diag(np.sqrt(np.diag(C0)))          # H0 whitening factor
    L1 = sl.cholesky(C, lower=True)             # H1 Cholesky (for info; not used in outputs)
    N  = L0 @ L0.T

    # NP filter: Q_np = C0^{-1} - C^{-1}; NPMV = off-diagonal of Q_np
    C0_inv = np.diag(1.0 / np.diag(C0))
    Q_np = C0_inv - sl.cho_solve((L1, True), np.eye(n))
    Q_npmv = Q_np.copy()
    np.fill_diagonal(Q_npmv, 0.0)

    # Normalize in N-inner product
    D_unscaled = norm_filter_N(Q_npmv, N)
    D_star = D_unscaled.copy()   # For FAP=1 placeholder we set scale=1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save matrices
    np.save(outdir / "D_unscaled.npy", D_unscaled)
    np.save(outdir / "D_star.npy", D_star)

    # Save vectorized parameters (strict lower triangle)
    x_vec = get_lower_triangular_elements(D_unscaled).tolist()
    with open(outdir / "x_opt.json", "w") as fp:
        json.dump(x_vec, fp, indent=2)

    # Save a minimal result.json compatible with your tooling
    meta = {
        "mode": "npmv-fixed",
        "faprob": 1.0,
        "tau": float(args.tau),
        "DP": 1.0,                 # not evaluated for this placeholder
        "scale": 1.0,               # D_star == D_unscaled
        "npsrs": n,
        "cdf": args.cdf,
        "note": "NPMV (off-diagonal NP) saved without scaling; placeholder for FAP=1."
    }
    with open(outdir / "result.json", "w") as fp:
        json.dump(meta, fp, indent=2)

    print(f"[done] Wrote NPMV artifacts to {outdir} (npsrs={n}, scale=1.0, faprob=1.0)")

if __name__ == "__main__":
    main()

