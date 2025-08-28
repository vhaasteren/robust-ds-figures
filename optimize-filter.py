#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimize quadratic decision filters for PTA detection using a GX^2 CDF.

WHAT'S NEW IN v4.5
------------------
• New optimizers for the FULL high-dimensional problem (keep LN_BOBYQA default):
  --optimizer {bobyqa, subspace_bobyqa, nes, spsa, isres_then_bobyqa}
  - subspace_bobyqa: cycles of random subspaces optimized with BOBYQA
        --subspace-size INT (default 150)
        --subspace-evals INT (default 80 per subspace)
        --subspace-cycles INT (default 2)
  - nes: Natural Evolution Strategies (antithetic sampling) + optional polish
        --nes-pop INT (default 128, uses pairs)
        --nes-iters INT (default 30)
  - spsa: Simultaneous Perturbation Stochastic Approximation + optional polish
        --spsa-iters INT (default 2500)
        --spsa-step FLOAT (default 0.05)
  - isres_then_bobyqa: brief global scatter via NLopt ISRES then BOBYQA polish
        --isres-evals INT (default 3000)
  - Optional polish stage for nes/spsa:
        --polish {none,bobyqa} (default: bobyqa) and --polish-evals INT (default 1000)

• v4.4 features retained:
  - --resume flag (FULL mode only): start from <outdir>/x_opt.json if compatible.
  - Incremental full optimization is strictly progressive (no early stop/fallback).
  - Choice of CDF backend: --cdf analytic | imhof.
  - start_from_npmv toggle (default True) for initializations in multiple modes.
  - Modes: full, full-incremental, legendre, zonal-alpha-aware, zonal-low-rank, bi-spectral
    (legacy aliases: original→full, basis→legendre).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import tempfile
import warnings
from typing import Iterable, List, Optional, Sequence, Tuple

import nlopt
import numpy as np
import scipy.linalg as sl
import scipy.optimize as sopt
import scipy.integrate as sint
from numpy.polynomial import legendre as npleg

# ===================== Pulsar positions (NG15yr subset) =====================

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

# ===================== HD kernel & utilities =====================

def phitheta_to_psrpos(phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Convert spherical (phi,theta) → 3D unit vectors on S^2."""
    return np.array([
        np.cos(phi) * np.sin(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(theta),
    ]).T


def generate_pulsar_positions(npsrs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate n random pulsar sky positions uniformly on S^2."""
    phi = np.random.rand(npsrs) * 2 * np.pi
    theta = np.arccos(np.random.rand(npsrs) * 2 - 1)
    return phi, theta, phitheta_to_psrpos(phi, theta)


def hdfunc(gamma: np.ndarray) -> np.ndarray:
    """Hellings–Downs correlation at angular separation gamma (safe logs)."""
    cosgamma = np.clip(np.cos(gamma), -1.0, 1.0)
    xp = 0.5 * (1 - cosgamma)
    old = np.seterr(all="ignore")
    logxp = 1.5 * xp * np.log(xp)
    np.seterr(**old)
    return logxp - 0.25 * xp + 0.5


def hdcorrmat(psrpos: np.ndarray, psrTerm: bool = True) -> np.ndarray:
    """Build the HD correlation matrix for given pulsar positions."""
    cosgamma = np.clip(psrpos @ psrpos.T, -1, 1)
    npsrs = len(cosgamma)
    xp = 0.5 * (1 - cosgamma)
    old = np.seterr(all="ignore")
    logxp = 1.5 * xp * np.log(xp)
    np.fill_diagonal(logxp, 0)
    np.seterr(**old)
    coeff = 0.5 if psrTerm else 0.0
    return logxp - 0.25 * xp + 0.5 + coeff * np.eye(npsrs)

# ===================== Covariances & canonical filters =====================

def get_cov_matrices(h: float, hdmat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Construct C0 (CURN) and C = I + h^2 * HD."""
    C_noise = np.identity(len(hdmat))
    C_signal = (h**2) * hdmat
    N = C_noise + np.diag(np.diag(C_signal))  # CURN
    C = C_noise + C_signal
    return N, C


def norm_filter(Q: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Normalize Q by the N-inner product: Q / sqrt(tr(Q N Q N))."""
    norm = np.sqrt(np.trace(Q @ N @ Q @ N))
    return Q / norm


def norm_filter_white(Q: np.ndarray) -> np.ndarray:
    """Normalize Q in whitened space (Frobenius): Q / ||Q||_F."""
    return Q / np.linalg.norm(Q)


def get_all_filters(h: float, hdmat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build NP, NPMV (off-diag of NP), DF filters and Cholesky factors."""
    C0, CS = get_cov_matrices(h, hdmat=hdmat)
    L0 = np.diag(np.sqrt(np.diag(C0)))  # whitening factor for H0
    L1 = sl.cholesky(CS, lower=True)
    C0_inv = np.diag(1/np.diag(C0))

    DNP = C0_inv - sl.cho_solve((L1, True), np.identity(len(CS)))
    DNPW = DNP.copy(); np.fill_diagonal(DNPW, 0)
    DDEF = C0_inv @ (CS - C0) @ C0_inv

    DNPW = norm_filter(DNPW, C0)
    DNP  = norm_filter(DNP , C0)
    DDEF = norm_filter(DDEF, C0)
    return DNP, DNPW, DDEF, L0, L1

# ===================== GX^2 CDFs =====================

def _logsumexp_signed(log_pos_list: List[float], log_neg_list: List[float]) -> float:
    """Compute sum(exp(lp)) - sum(exp(ln)) stably in log-domain (analytic CDF)."""
    def lse(arr: List[float]) -> float:
        if not arr:
            return -np.inf
        a = np.max(arr)
        return a + np.log(np.sum(np.exp(np.array(arr) - a)))
    Spos_log = lse(log_pos_list)
    Sneg_log = lse(log_neg_list)
    if Spos_log == -np.inf and Sneg_log == -np.inf:
        return 0.0
    if Spos_log >= Sneg_log:
        return np.exp(Spos_log) - np.exp(Sneg_log)
    else:
        return -(np.exp(Sneg_log) - np.exp(Spos_log))


def gx2cdf_an_from_eigs(evals: np.ndarray, tau: float) -> float:
    """Analytic CDF F(tau;Q) for central complex quadratic form with indefinite Hermitian kernel."""
    lam = np.asarray(evals, dtype=float)
    t = float(tau)
    if lam.size == 0 or not np.isfinite(t):
        return np.nan
    if t <= 0:
        return 1.0 - gx2cdf_an_from_eigs(-lam, -t)

    pos_idx = np.where(lam > 0)[0]
    if pos_idx.size == 0:
        return 1.0

    tiny = 1e-30
    log_pos_terms: List[float] = []
    log_neg_terms: List[float] = []
    for j in pos_idx:
        lj = lam[j]
        den = 1.0 - lam / lj
        den[j] = 1.0
        sgns = np.sign(den)
        absden = np.abs(den) + tiny
        sgn_j = np.prod(sgns)
        log_term = (-0.5 * t / lj) - np.sum(np.log(absden))
        if sgn_j >= 0:
            log_pos_terms.append(log_term)
        else:
            log_neg_terms.append(log_term)
    series = _logsumexp_signed(log_pos_terms, log_neg_terms)
    if not np.isfinite(series):
        return np.nan
    cdf = 1.0 - series
    if not np.isfinite(cdf):
        return np.nan
    return float(min(1.0, max(0.0, cdf)))


# ----- Imhof's method (robust; slower) -----

def _imhof_integrand(u: float, x: float, eigs: np.ndarray, part: str = "cdf") -> float:
    """Scalar Imhof integrand at frequency u for threshold x and eigenvalues eigs."""
    uu = float(u)
    arr = eigs * uu
    theta = 0.5 * np.sum(np.arctan(arr)) - 0.5 * x * uu
    rho = np.prod((1.0 + arr**2)**0.25)
    if uu == 0.0:
        return 0.0
    if part == "cdf":
        return float(np.sin(theta) / (uu * rho))
    else:
        return float(np.cos(theta) / rho)


def gx2cdf_imhof_from_eigs(evals: np.ndarray, tau: float,
                           cutoff: float = 1e-6, limit: int = 200, epsabs: float = 1e-9
                           ) -> float:
    """Imhof CDF F(tau;Q) for central quadratic form with indefinite Hermitian kernel.

    Complex-valued case: eigenvalues are duplicated. Small |λ| can be dropped.
    """
    w = np.asarray(evals, dtype=float)
    ww = np.repeat(w, 2)  # complex-valued duplication
    if cutoff > 0:
        ww = ww[np.abs(ww) > cutoff]
    if ww.size == 0:
        return 1.0 if float(tau) >= 0.0 else 0.0

    val, err = sint.quad(lambda u: _imhof_integrand(u, float(tau), ww, "cdf"),
                         0.0, np.inf, limit=limit, epsabs=epsabs)
    cdf = 0.5 - val / np.pi
    return float(min(1.0, max(0.0, cdf)))


def gx2cdf_from_eigs(evals: np.ndarray, tau: float, method: str = "analytic") -> float:
    """Dispatch CDF evaluation by method."""
    if method == "analytic":
        return gx2cdf_an_from_eigs(evals, tau)
    elif method == "imhof":
        return gx2cdf_imhof_from_eigs(evals, tau)
    else:
        raise ValueError(f"Unknown CDF method: {method}")

# ===================== scaling to target FAP =====================

def scale_to_fap(L: np.ndarray, Q: np.ndarray, tau: float, fap_target: float,
                 max_doubles: int = 60, cdf_method: str = "analytic") -> Optional[float]:
    """Find s>0 with FAP_H0(sQ; tau) = fap_target using the selected CDF."""
    if not (np.isfinite(fap_target) and 0 < fap_target < 1):
        return None
    if tau <= 0:
        return None
    try:
        w0 = sl.eigvalsh(L.T @ Q @ L)
    except Exception:
        return None

    def fap_of_s(s: float) -> float:
        if not np.isfinite(s) or s <= 0:
            return np.nan
        cdf0 = gx2cdf_from_eigs(s * w0, tau, method=cdf_method)
        if not np.isfinite(cdf0):
            return np.nan
        return 1.0 - cdf0

    s_lo = 1e-8; f_lo = fap_of_s(s_lo); nstep = 0
    while (not np.isfinite(f_lo) or f_lo >= fap_target) and nstep < 20:
        s_lo *= 0.5; f_lo = fap_of_s(s_lo); nstep += 1
    s_hi = 1.0; f_hi = fap_of_s(s_hi); nstep = 0
    while (not np.isfinite(f_hi) or f_hi <= fap_target) and nstep < max_doubles:
        s_hi *= 2.0; f_hi = fap_of_s(s_hi); nstep += 1
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
        return None
    if not (f_lo < fap_target < f_hi):
        return None
    try:
        s = sopt.brentq(lambda s_: (fap_of_s(s_) - fap_target), s_lo, s_hi,
                        maxiter=200, xtol=1e-12)
    except Exception:
        return None
    return float(s) if np.isfinite(s) and s > 0 else None


def scale_decision_matrix(L: np.ndarray, Q: np.ndarray, tau: float, faprob: float = 0.1,
                          cdf_method: str = "analytic") -> Optional[np.ndarray]:
    """Scale normalized Q → s*Q so that FAP_H0 = faprob using selected CDF."""
    s = scale_to_fap(L, Q, tau, faprob, cdf_method=cdf_method)
    if s is None or not np.isfinite(s):
        return None
    return s * Q

# ===================== Parameterizations & objectives =====================

# --- Full (free off-diagonal) ---

def construct_decision_matrix(x: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Build a symmetric off-diagonal matrix D from the lower-triangular vector x."""
    elements = x
    m = len(elements)
    n = int(0.5 * (np.sqrt(8*m + 1) + 1))
    i, j = np.tril_indices(n, k=-1)
    D = np.zeros((n, n))
    D[i, j] = elements
    D = D + D.T
    np.fill_diagonal(D, 0.0)
    return D


def get_lower_triangular_elements(D: np.ndarray) -> np.ndarray:
    """Extract strict lower-triangular elements (vectorized)."""
    i, j = np.tril_indices(D.shape[0], k=-1)
    return D[i, j]


def det_prob(x: np.ndarray, L0: np.ndarray, LS: np.ndarray, faprob: float = 0.1,
             normalized_coords: bool = False, tau: float = 1.0,
             cdf_method: str = "analytic") -> float:
    """Compute DP for a filter parameterized by free off-diagonals x."""
    if tau <= 0 or not np.isfinite(tau):
        return -np.inf
    D = construct_decision_matrix(x, normalize=normalized_coords)
    N = L0 @ L0.T
    nrm2 = np.trace(D @ N @ D @ N)
    if not np.isfinite(nrm2) or nrm2 <= 0:
        return -np.inf
    D = D / np.sqrt(nrm2)
    D_scaled = scale_decision_matrix(L0, D, tau=tau, faprob=faprob, cdf_method=cdf_method)
    if D_scaled is None:
        return -np.inf
    try:
        w1 = sl.eigvalsh(LS.T @ D_scaled @ LS)
    except Exception:
        return -np.inf
    cdf1 = gx2cdf_from_eigs(w1, tau, method=cdf_method)
    if not np.isfinite(cdf1):
        return -np.inf
    DP = 1.0 - float(cdf1)
    return float(min(1.0, max(0.0, DP)))


def dp_from_normalized_matrix(D: np.ndarray, L0: np.ndarray, L1: np.ndarray,
                              faprob: float, tau: float, cdf_method: str) -> float:
    """Detection probability for a given matrix D already normalized in the N-inner product."""
    s = scale_to_fap(L0, D, tau, faprob, cdf_method=cdf_method)
    if s is None or not np.isfinite(s):
        return -np.inf
    try:
        w1 = sl.eigvalsh(L1.T @ (s*D) @ L1)
    except Exception:
        return -np.inf
    cdf1 = gx2cdf_from_eigs(w1, tau, method=cdf_method)
    if not np.isfinite(cdf1):
        return -np.inf
    return float(min(1.0, max(0.0, 1.0 - cdf1)))

# --- Legendre (zonal matrix basis) ---

def build_legendre_basis(cosgamma: np.ndarray, Lmax: int) -> List[np.ndarray]:
    """Build [B_ℓ] with B_ℓ = P_ℓ(cosγ), zero diagonal, for ℓ=1..Lmax (ℓ=0 skipped)."""
    B: List[np.ndarray] = []
    for ell in range(1, Lmax + 1):
        M = npleg.legval(cosgamma, [0]*ell + [1.0])
        np.fill_diagonal(M, 0.0)
        B.append(M)
    return B


def gram_matrix(B: Sequence[np.ndarray], N: np.ndarray) -> np.ndarray:
    """Gram matrix G_ij = <B_i, B_j>_N for normalization and projections."""
    L = len(B)
    G = np.empty((L, L))
    for i in range(L):
        BiN = B[i] @ N
        for j in range(L):
            G[i, j] = np.trace(BiN @ B[j] @ N)
    return G


def D_from_alpha(alpha: np.ndarray, B: Sequence[np.ndarray], G: np.ndarray, N: np.ndarray
                 ) -> Optional[np.ndarray]:
    """Construct normalized D = sum_i alpha_i B_i under the N-inner product."""
    q = float(alpha @ G @ alpha)
    if not np.isfinite(q) or q <= 0:
        return None
    a = alpha / np.sqrt(q)
    D = np.zeros_like(B[0])
    for ai, Bi in zip(a, B):
        D += ai * Bi
    np.fill_diagonal(D, 0.0)
    nrm2 = np.trace(D @ N @ D @ N)
    if not np.isfinite(nrm2) or nrm2 <= 0:
        return None
    D /= np.sqrt(nrm2)
    return D


def project_D_to_alpha(D: np.ndarray, B: Sequence[np.ndarray], G: np.ndarray, N: np.ndarray
                       ) -> np.ndarray:
    """Least-squares projection of a matrix D onto span{B_i} under the N-inner product."""
    b = np.array([np.trace((Bi @ N) @ (D @ N)) for Bi in B], dtype=float)
    try:
        alpha = sl.solve(G + 1e-12*np.eye(len(G)), b, assume_a='pos')
    except Exception:
        alpha = np.linalg.lstsq(G, b, rcond=1e-10)[0]
    return alpha


def det_prob_alpha(alpha: np.ndarray, L0: np.ndarray, L1: np.ndarray, B: Sequence[np.ndarray],
                   G: np.ndarray, N: np.ndarray, faprob: float = 0.1, tau: float = 1.0,
                   cdf_method: str = "analytic") -> float:
    """Detection probability for a Legendre-basis parameterization."""
    if tau <= 0 or not np.isfinite(tau):
        return -np.inf
    D = D_from_alpha(alpha, B, G, N)
    if D is None:
        return -np.inf
    s = scale_to_fap(L0, D, tau, faprob, cdf_method=cdf_method)
    if s is None or not np.isfinite(s):
        return -np.inf
    try:
        w1 = sl.eigvalsh(L1.T @ (s*D) @ L1)
    except Exception:
        return -np.inf
    cdf1 = gx2cdf_from_eigs(w1, tau, method=cdf_method)
    if not np.isfinite(cdf1):
        return -np.inf
    DP = 1.0 - float(cdf1)
    return float(min(1.0, max(0.0, DP)))

# ===================== Safe JSON save =====================

def atomic_save_json(path: str, obj: dict) -> None:
    """Atomic JSON save (write temp then rename)."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=d, delete=False) as tmp:
        json.dump(obj, tmp, indent=2)
        tmp.flush(); os.fsync(tmp.fileno()); tmp_name = tmp.name
    os.replace(tmp_name, path)

# ===================== New basis helpers =====================

def zonal_weights(Lmax: int) -> np.ndarray:
    """Weights w_ℓ=(2ℓ+1)/(4π) for ℓ=1..Lmax (ℓ=0 skipped as we enforce cross-only)."""
    ells = np.arange(1, Lmax+1)
    return (2*ells + 1) / (4*np.pi)


def nullspace_w(w: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis U for the nullspace of w^T (i.e., {x: w·x=0})."""
    w = w.reshape(1, -1)
    U, S, Vt = np.linalg.svd(w, full_matrices=True)
    return Vt[1:, :].T  # shape L×(L-1)


def build_lowrank_basis_from_residual(D_target: np.ndarray, B: Sequence[np.ndarray], G: np.ndarray,
                                      N: np.ndarray, r: int) -> List[np.ndarray]:
    """Build r low-rank symmetric basis matrices from the off-diagonal residual (NP − zonal fit)."""
    alpha_fit = project_D_to_alpha(D_target, B, G, N)
    Dz = D_from_alpha(alpha_fit, B, G, N)
    if Dz is None:
        Dz = np.zeros_like(D_target)
    R = D_target - Dz
    R = (R + R.T) / 2.0
    np.fill_diagonal(R, 0.0)
    try:
        evals, evecs = sl.eigh(R)
    except Exception:
        evals, evecs = np.linalg.eigh(R)
    idx = np.argsort(np.abs(evals))[::-1][:max(0, r)]
    mats: List[np.ndarray] = []
    for k in idx:
        v = evecs[:, k].reshape(-1, 1)
        Mk = v @ v.T
        np.fill_diagonal(Mk, 0.0)
        mats.append(Mk)
    return mats

# ===================== Optimization drivers (standard modes) =====================

def optimize_with_legendre(psrpos: np.ndarray, faprob: float, tau: float, Lmax: int,
                           n_starts: int, seed: int, outdir: str, cdf_method: str,
                           start_from_npmv: bool
                           ) -> Tuple[np.ndarray, dict]:
    """Legacy 'basis' → 'legendre': BOBYQA in Legendre basis (zonal), DP objective."""
    cosgamma = np.clip(psrpos @ psrpos.T, -1.0, 1.0)
    np.fill_diagonal(cosgamma, 1.0)
    hdmat = hdcorrmat(psrpos, psrTerm=True)
    h_opt = 1.0
    DNP, DNPW, DDEF, L_H0, L_H1 = get_all_filters(h_opt, hdmat)
    N = L_H0 @ L_H0.T
    B = build_legendre_basis(cosgamma, Lmax)
    G = gram_matrix(B, N)

    rng = np.random.default_rng(seed)
    init_D = DNPW if start_from_npmv else DDEF
    alpha0 = project_D_to_alpha(init_D, B, G, N)

    best = {"DP": -np.inf, "alpha": None, "scale": None}
    for k in range(n_starts):
        if k == 0 and np.all(np.isfinite(alpha0)):
            x0 = alpha0.copy()
        else:
            x0 = rng.normal(0, 1, size=Lmax)
            x0 /= (np.linalg.norm(x0) + 1e-12)

        opt = nlopt.opt(nlopt.LN_BOBYQA, Lmax)
        opt.set_lower_bounds(-10*np.ones(Lmax))
        opt.set_upper_bounds(+10*np.ones(Lmax))
        opt.set_xtol_rel(1e-6)
        opt.set_maxeval(2000)

        def obj(a: np.ndarray, grad: np.ndarray) -> float:
            DP = det_prob_alpha(a, L_H0, L_H1, B, G, N,
                                faprob=faprob, tau=tau, cdf_method=cdf_method)
            return -float(DP if np.isfinite(DP) else -np.inf)

        opt.set_min_objective(obj)
        try:
            a_opt = opt.optimize(x0)
            DP = det_prob_alpha(a_opt, L_H0, L_H1, B, G, N,
                                faprob=faprob, tau=tau, cdf_method=cdf_method)
            if DP > best["DP"]:
                D_opt = D_from_alpha(a_opt, B, G, N)
                s_opt = scale_to_fap(L_H0, D_opt, tau, faprob, cdf_method=cdf_method)
                best.update(DP=float(DP), alpha=a_opt.tolist(), scale=float(s_opt))
        except nlopt.RoundoffLimited:
            continue
        except Exception:
            continue

    if best["alpha"] is None:
        raise RuntimeError("No successful optimization. Try increasing starts/Lmax.")

    alpha_best = np.array(best["alpha"])  # unscaled normalized D
    D_unscaled = D_from_alpha(alpha_best, B, G, N)
    s_best = float(best["scale"])
    D_star = s_best * D_unscaled

    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "D_star.npy"), D_star)
    np.save(os.path.join(outdir, "D_unscaled.npy"), D_unscaled)
    atomic_save_json(os.path.join(outdir, "result.json"),
        {
            "mode": "legendre",
            "Lmax": int(Lmax),
            "n_starts": int(n_starts),
            "seed": int(seed),
            "faprob": float(faprob),
            "tau": float(tau),
            "DP": float(best["DP"]),
            "scale": float(s_best),
            "alpha": best["alpha"],
            "cdf": cdf_method,
            "start_from_npmv": bool(start_from_npmv),
        }
    )
    return D_star, best

# ========= FULL helpers: objective wrapper, finalize, polish, subspace ==========

def _dp_obj_factory(L_H0, L_H1, faprob, tau, cdf_method):
    def eval_dp(x: np.ndarray) -> float:
        DP = det_prob(x, L_H0, L_H1, faprob=faprob, normalized_coords=False,
                      tau=tau, cdf_method=cdf_method)
        return float(DP if np.isfinite(DP) else -np.inf)
    return eval_dp

def _finalize_and_save_full(x_best: np.ndarray, L_H0, L_H1, N, faprob, tau, cdf_method,
                            outdir: str, seed: int, m: int, maxeval: int, bound: float,
                            start_json: Optional[str], start_from_npmv: bool, resume: bool,
                            optimizer_name: str, extra_meta: dict) -> Tuple[np.ndarray, dict]:
    """Common post-processing: normalize, scale to FAP, write outputs."""
    D = construct_decision_matrix(x_best, normalize=False)
    nrm2 = np.trace(D @ N @ D @ N)
    if not np.isfinite(nrm2) or nrm2 <= 0:
        raise RuntimeError("Final D normalization failed.")
    D /= np.sqrt(nrm2)
    s_opt = scale_to_fap(L_H0, D, tau, faprob, cdf_method=cdf_method)
    if s_opt is None:
        raise RuntimeError("Final scaling to FAP failed.")
    D_unscaled = D.copy()
    D_star = s_opt * D_unscaled
    DP = det_prob(x_best, L_H0, L_H1, faprob=faprob, normalized_coords=False,
                  tau=tau, cdf_method=cdf_method)

    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "D_star.npy"), D_star)
    np.save(os.path.join(outdir, "D_unscaled.npy"), D_unscaled)
    atomic_save_json(os.path.join(outdir, "result.json"),
        dict({
            "mode": "full",
            "optimizer": optimizer_name,
            "seed": int(seed),
            "faprob": float(faprob),
            "tau": float(tau),
            "DP": float(DP),
            "scale": float(s_opt),
            "x": x_best.tolist(),
            "m_params": int(m),
            "maxeval": int(maxeval),
            "bound": float(bound),
            "start_json": start_json if start_json is not None else "",
            "cdf": cdf_method,
            "start_from_npmv": bool(start_from_npmv),
            "resume": bool(resume),
        }, **extra_meta)
    )
    atomic_save_json(os.path.join(outdir, "x_opt.json"), x_best.tolist())
    return D_star, {"DP": float(DP), "scale": float(s_opt)}

def _polish_bobyqa(x0: np.ndarray, eval_dp, bound: float, eval_budget: int) -> np.ndarray:
    """Run a short full-dimensional BOBYQA polish."""
    m = x0.size
    x0c = np.clip(x0, -bound, +bound)
    opt = nlopt.opt(nlopt.LN_BOBYQA, m)
    opt.set_lower_bounds(-bound*np.ones(m))
    opt.set_upper_bounds(+bound*np.ones(m))
    opt.set_xtol_rel(1e-6)
    opt.set_maxeval(max(1, int(eval_budget)))

    def obj(x: np.ndarray, grad: np.ndarray) -> float:
        return -eval_dp(x)

    opt.set_min_objective(obj)
    try:
        x_opt = opt.optimize(x0c)
        return x_opt
    except Exception:
        return x0c

def _run_subspace_bobyqa(x_init: np.ndarray, eval_dp, bound: float, seed: int,
                         size: int, evals_per_subspace: int, n_cycles: int) -> np.ndarray:
    """Cycle random disjoint subspaces and optimize each via BOBYQA."""
    rng = np.random.default_rng(seed)
    x = np.clip(x_init.copy(), -bound, +bound)
    m = x.size
    size = max(1, min(size, m))
    blocks_per_cycle = int(np.ceil(m / size))

    for cyc in range(n_cycles):
        perm = rng.permutation(m)
        for b in range(blocks_per_cycle):
            block = perm[b*size : min((b+1)*size, m)]
            if block.size == 0:
                continue

            # Local subspace optimizer
            opt = nlopt.opt(nlopt.LN_BOBYQA, block.size)
            opt.set_lower_bounds(-bound*np.ones(block.size))
            opt.set_upper_bounds(+bound*np.ones(block.size))
            opt.set_xtol_rel(1e-6)
            opt.set_maxeval(max(1, int(evals_per_subspace)))

            def obj_sub(y: np.ndarray, grad: np.ndarray) -> float:
                x_full = x.copy()
                x_full[block] = y
                return -eval_dp(x_full)

            opt.set_min_objective(obj_sub)
            try:
                y0 = x[block].copy()
                y_opt = opt.optimize(y0)
                x[block] = np.clip(y_opt, -bound, +bound)
            except Exception:
                # keep previous x[block]
                pass
    return x

# --- New Mode 2: Zonal α-aware with continuum-style constraint ---

def optimize_zonal_alpha_aware(psrpos: np.ndarray, faprob: float, tau: float, Lmax: int,
                               seed: int, outdir: str, maxeval: int, cdf_method: str,
                               start_from_npmv: bool
                               ) -> Tuple[np.ndarray, dict]:
    """Zonal optimization with continuum constraint ∑ w_ℓ q_ℓ = 0 via nullspace."""
    cosgamma = np.clip(psrpos @ psrpos.T, -1.0, 1.0)
    np.fill_diagonal(cosgamma, 1.0)
    hdmat = hdcorrmat(psrpos, psrTerm=True)
    DNP, DNPW, DDEF, L_H0, L_H1 = get_all_filters(1.0, hdmat)
    N = L_H0 @ L_H0.T

    B = build_legendre_basis(cosgamma, Lmax)
    G = gram_matrix(B, N)

    w = zonal_weights(Lmax)
    U = nullspace_w(w)  # L × (L-1)

    init_D = DNPW if start_from_npmv else DNP  # cross-only-friendly init
    alpha_init = project_D_to_alpha(init_D, B, G, N)

    y0 = U.T @ alpha_init
    y0 /= (np.linalg.norm(y0) + 1e-12)

    best = {"DP": -np.inf, "y": None, "scale": None}

    opt = nlopt.opt(nlopt.LN_BOBYQA, U.shape[1])
    opt.set_lower_bounds(-10*np.ones(U.shape[1]))
    opt.set_upper_bounds(+10*np.ones(U.shape[1]))
    opt.set_xtol_rel(1e-6)
    opt.set_maxeval(maxeval)

    def obj(y: np.ndarray, grad: np.ndarray) -> float:
        alpha = U @ y
        DP = det_prob_alpha(alpha, L_H0, L_H1, B, G, N,
                            faprob=faprob, tau=tau, cdf_method=cdf_method)
        return -float(DP if np.isfinite(DP) else -np.inf)

    opt.set_min_objective(obj)
    try:
        y_opt = opt.optimize(y0)
        alpha = U @ y_opt
        DP = det_prob_alpha(alpha, L_H0, L_H1, B, G, N,
                            faprob=faprob, tau=tau, cdf_method=cdf_method)
        if DP > best["DP"]:
            D = D_from_alpha(alpha, B, G, N)
            s_opt = scale_to_fap(L_H0, D, tau, faprob, cdf_method=cdf_method)
            best.update(DP=float(DP), y=y_opt.tolist(), scale=float(s_opt))
            D_unscaled = D
            D_star = s_opt * D_unscaled
    except nlopt.RoundoffLimited:
        raise

    if best["y"] is None:
        raise RuntimeError("α-aware zonal optimization failed.")

    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "D_star.npy"), D_star)
    np.save(os.path.join(outdir, "D_unscaled.npy"), D_unscaled)
    atomic_save_json(os.path.join(outdir, "result.json"),
        {
            "mode": "zonal-alpha-aware",
            "Lmax": int(Lmax),
            "seed": int(seed),
            "faprob": float(faprob),
            "tau": float(tau),
            "DP": float(best["DP"]),
            "scale": float(best["scale"]),
            "y": best["y"],
            "cdf": cdf_method,
            "start_from_npmv": bool(start_from_npmv),
        }
    )
    return D_star, best

# --- New Mode 2: Zonal + low-rank anisotropy ---

def optimize_zonal_lowrank(psrpos: np.ndarray, faprob: float, tau: float, Lmax: int,
                           r_lowrank: int, seed: int, outdir: str, maxeval: int,
                           cdf_method: str, start_from_npmv: bool) -> Tuple[np.ndarray, dict]:
    """Augment the Legendre zonal basis with r_lowrank anisotropic directions."""
    cosgamma = np.clip(psrpos @ psrpos.T, -1.0, 1.0)
    np.fill_diagonal(cosgamma, 1.0)
    hdmat = hdcorrmat(psrpos, psrTerm=True)
    DNP, DNPW, DDEF, L_H0, L_H1 = get_all_filters(1.0, hdmat)
    N = L_H0 @ L_H0.T

    B = build_legendre_basis(cosgamma, Lmax)
    G = gram_matrix(B, N)

    lowrank = build_lowrank_basis_from_residual(DNP, B, G, N, r=r_lowrank)
    B_all = list(B) + lowrank
    G_all = gram_matrix(B_all, N)

    rng = np.random.default_rng(seed)
    alpha0_z = project_D_to_alpha(DNPW if start_from_npmv else DNP, B, G, N)
    x0 = np.concatenate([alpha0_z, np.zeros(len(lowrank))])
    x0 /= (np.linalg.norm(x0) + 1e-12)

    best = {"DP": -np.inf, "alpha": None, "scale": None}

    opt = nlopt.opt(nlopt.LN_BOBYQA, len(B_all))
    opt.set_lower_bounds(-10*np.ones(len(B_all)))
    opt.set_upper_bounds(+10*np.ones(len(B_all)))
    opt.set_xtol_rel(1e-6)
    opt.set_maxeval(maxeval)

    def obj(a: np.ndarray, grad: np.ndarray) -> float:
        DP = det_prob_alpha(a, L_H0, L_H1, B_all, G_all, N,
                            faprob=faprob, tau=tau, cdf_method=cdf_method)
        return -float(DP if np.isfinite(DP) else -np.inf)

    opt.set_min_objective(obj)

    try:
        a_opt = opt.optimize(x0)
        DP = det_prob_alpha(a_opt, L_H0, L_H1, B_all, G_all, N,
                            faprob=faprob, tau=tau, cdf_method=cdf_method)
        if DP > best["DP"]:
            D_opt = D_from_alpha(a_opt, B_all, G_all, N)
            s_opt = scale_to_fap(L_H0, D_opt, tau, faprob, cdf_method=cdf_method)
            best.update(DP=float(DP), alpha=a_opt.tolist(), scale=float(s_opt))
            D_unscaled = D_opt
            D_star = s_opt * D_unscaled
    except nlopt.RoundoffLimited:
        raise

    if best["alpha"] is None:
        raise RuntimeError("zonal-low-rank optimization failed.")

    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "D_star.npy"), D_star)
    np.save(os.path.join(outdir, "D_unscaled.npy"), D_unscaled)
    atomic_save_json(os.path.join(outdir, "result.json"),
        {
            "mode": "zonal-low-rank",
            "Lmax": int(Lmax),
            "r_lowrank": int(r_lowrank),
            "seed": int(seed),
            "faprob": float(faprob),
            "tau": float(tau),
            "DP": float(best["DP"]),
            "scale": float(best["scale"]),
            "alpha": best["alpha"],
            "cdf": cdf_method,
            "start_from_npmv": bool(start_from_npmv),
        }
    )
    return D_star, best

# --- New Mode 3: Bi-spectral zonal spectrum ---

def optimize_bispectral(psrpos: np.ndarray, faprob: float, tau: float, Lmax: int,
                        kmax: int, seed: int, outdir: str, cdf_method: str,
                        start_from_npmv: bool
                        ) -> Tuple[np.ndarray, dict]:
    """Two-level zonal spectrum: choose subset S of ℓ with q_ℓ=+1, set q_ℓ=q_-<0 on complement."""
    cosgamma = np.clip(psrpos @ psrpos.T, -1.0, 1.0)
    np.fill_diagonal(cosgamma, 1.0)
    hdmat = hdcorrmat(psrpos, psrTerm=True)
    DNP, DNPW, DDEF, L_H0, L_H1 = get_all_filters(1.0, hdmat)
    N = L_H0 @ L_H0.T

    B = build_legendre_basis(cosgamma, Lmax)
    G = gram_matrix(B, N)

    # Spectral score from NP or NPMV projection (init choice)
    alpha_proj = project_D_to_alpha(DNPW if start_from_npmv else DNP, B, G, N)
    score = np.abs(alpha_proj)
    ell_indices = np.argsort(score)[::-1]

    w = zonal_weights(Lmax)
    best = {"DP": -np.inf, "alpha": None, "scale": None, "k": None}

    for k in range(1, min(kmax, Lmax-1) + 1):
        S_mask = np.zeros(Lmax, dtype=bool)
        S_mask[ell_indices[:k]] = True
        wS = float(np.sum(w[S_mask])); wSc = float(np.sum(w[~S_mask]))
        if wSc == 0:
            continue
        q_plus = 1.0
        q_minus = - q_plus * (wS / wSc)
        alpha = np.where(S_mask, q_plus, q_minus)
        D = D_from_alpha(alpha, B, G, N)
        if D is None:
            continue
        s_opt = scale_to_fap(L_H0, D, tau, faprob, cdf_method=cdf_method)
        if s_opt is None:
            continue
        try:
            w1 = sl.eigvalsh(L_H1.T @ (s_opt*D) @ L_H1)
        except Exception:
            continue
        cdf1 = gx2cdf_from_eigs(w1, tau, method=cdf_method)
        if not np.isfinite(cdf1):
            continue
        DP = 1.0 - float(cdf1)
        if DP > best["DP"]:
            best.update(DP=float(DP), alpha=alpha.tolist(), scale=float(s_opt), k=int(k))
            D_unscaled = D
            D_star = s_opt * D_unscaled

    if best["alpha"] is None:
        raise RuntimeError("bi-spectral search found no valid candidate (try larger kmax/Lmax).")

    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "D_star.npy"), D_star)
    np.save(os.path.join(outdir, "D_unscaled.npy"), D_unscaled)
    atomic_save_json(os.path.join(outdir, "result.json"),
        {
            "mode": "bi-spectral",
            "Lmax": int(Lmax),
            "k": int(best["k"]),
            "seed": int(seed),
            "faprob": float(faprob),
            "tau": float(tau),
            "DP": float(best["DP"]),
            "scale": float(best["scale"]),
            "alpha": best["alpha"],
            "cdf": cdf_method,
            "start_from_npmv": bool(start_from_npmv),
        }
    )
    return D_star, best

# ===================== New: Incremental FULL optimization =====================

def _embed_old_into_new(D_old: np.ndarray, n_new: int) -> np.ndarray:
    """Embed a smaller symmetric matrix into the top-left block of a larger one."""
    n_old = D_old.shape[0]
    D_new = np.zeros((n_new, n_new))
    D_new[:n_old, :n_old] = D_old
    np.fill_diagonal(D_new, 0.0)
    return D_new


def optimize_with_full_incremental(psrpos: np.ndarray, faprob: float, tau: float, seed: int,
                                   outdir: str, maxeval: int, bound: float, cdf_method: str,
                                   inc_start: int, inc_step: int, start_from_npmv: bool
                                   ) -> Tuple[np.ndarray, dict]:
    """
    Incremental optimization for the full problem (free off-diagonals).

    Procedure:
      1) Solve the full problem for n = inc_start (standard 'full' optimization).
      2) For n ← n + inc_step up to target:
         (a) Initialize by embedding previous optimal solution into n×n and
             filling *new* entries using NPMV (off-diagonal NP) as a guess.
         (b) Optimize only the newly added parameters (old ones frozen).
         (c) Optimize all parameters jointly.
      At each n, also compute DP(NPMV). If our optimized DP < DP(NPMV), we WARN
      and CONTINUE (no early stop, no fallback).
    """
    rng = np.random.default_rng(seed)
    n_target = psrpos.shape[0]
    n0 = max(2, min(inc_start, n_target))
    step = max(1, inc_step)

    stages: List[dict] = []

    # ---- Stage 0: solve for n0 using standard full optimization ----
    sub_psr = psrpos[:n0, :]
    hdmat = hdcorrmat(sub_psr, psrTerm=True)
    DNP, DNPW, DDEF, L0, L1 = get_all_filters(1.0, hdmat)
    N = L0 @ L0.T

    # DP for NPMV at n0 (same backend)
    DP_npmv_n0 = dp_from_normalized_matrix(DNPW, L0, L1, faprob=faprob, tau=tau, cdf_method=cdf_method)

    # Stage-0 init: NPMV or classic
    D_start = DNPW.copy() if start_from_npmv else norm_filter_white(DDEF.copy())
    x0 = get_lower_triangular_elements(D_start)

    m0 = n0*(n0-1)//2
    opt0 = nlopt.opt(nlopt.LN_BOBYQA, m0)
    opt0.set_lower_bounds(-bound*np.ones(m0))
    opt0.set_upper_bounds(+bound*np.ones(m0))
    opt0.set_xtol_rel(1e-6)
    opt0.set_maxeval(maxeval)

    def obj0(x: np.ndarray, grad: np.ndarray) -> float:
        return -det_prob(x, L0, L1, faprob=faprob, tau=tau, cdf_method=cdf_method)

    opt0.set_min_objective(obj0)
    x_prev = opt0.optimize(x0)
    DP0 = det_prob(x_prev, L0, L1, faprob=faprob, tau=tau, cdf_method=cdf_method)

    # Keep a copy of previous D (matrix form) for embedding
    D_prev = construct_decision_matrix(x_prev, normalize=False)

    # Report and warn if below NPMV, but DO NOT stop.
    print(f"[increment] n={n0}: base full optimization  DP={DP0:.6e}  vs NPMV={DP_npmv_n0:.6e}")
    if DP0 < DP_npmv_n0:
        print(f"[increment][warn] n={n0}: DP below NPMV; continuing (no fallback).")

    stages.append({"n": int(n0), "DP_base": float(DP0), "DP_NPMV": float(DP_npmv_n0)})

    # ---- Incremental loop ----
    n_prev = n0

    while n_prev < n_target:
        n_cur = min(n_target, n_prev + step)
        sub_psr = psrpos[:n_cur, :]
        hdmat = hdcorrmat(sub_psr, psrTerm=True)
        DNP, DNPW, DDEF, L0, L1 = get_all_filters(1.0, hdmat)
        N = L0 @ L0.T

        # DP for NPMV at n_cur (same backend)
        DP_npmv_cur = dp_from_normalized_matrix(DNPW, L0, L1, faprob=faprob, tau=tau, cdf_method=cdf_method)

        # Seed: embed previous solution; fill new entries using DNPW (NPMV)
        D_seed = _embed_old_into_new(D_prev, n_cur)
        i_tril, j_tril = np.tril_indices(n_cur, k=-1)
        old_mask = (i_tril < n_prev) & (j_tril < n_prev)
        new_mask = ~old_mask

        x_seed = get_lower_triangular_elements(D_seed)
        x_guess_np = get_lower_triangular_elements(DNPW)
        x_seed[new_mask] = x_guess_np[new_mask]  # keep old, fill new from NPMV

        # --- Step 1: optimize only newly added parameters ---
        d_free = int(np.sum(new_mask))
        opt1 = nlopt.opt(nlopt.LN_BOBYQA, d_free)
        opt1.set_lower_bounds(-bound*np.ones(d_free))
        opt1.set_upper_bounds(+bound*np.ones(d_free))
        opt1.set_xtol_rel(1e-6)
        opt1.set_maxeval(maxeval)

        def obj1(x_new: np.ndarray, grad: np.ndarray) -> float:
            x_full = x_seed.copy()
            x_full[new_mask] = x_new
            return -det_prob(x_full, L0, L1, faprob=faprob, tau=tau, cdf_method=cdf_method)

        opt1.set_min_objective(obj1)
        x_new0 = x_seed[new_mask].copy()
        x_new_opt = opt1.optimize(x_new0)

        # Construct full vector after step 1 and evaluate
        x_step1 = x_seed.copy()
        x_step1[new_mask] = x_new_opt
        DP1 = det_prob(x_step1, L0, L1, faprob=faprob, tau=tau, cdf_method=cdf_method)

        # For reporting, compute scale s1 of the normalized matrix
        D1_mat = construct_decision_matrix(x_step1, normalize=False)
        nrm2 = np.trace(D1_mat @ N @ D1_mat @ N)
        s1 = None
        if np.isfinite(nrm2) and nrm2 > 0:
            D1n = D1_mat / np.sqrt(nrm2)
            s_tmp = scale_to_fap(L0, D1n, tau, faprob, cdf_method=cdf_method)
            s1 = float(s_tmp) if (s_tmp is not None and np.isfinite(s_tmp)) else None

        print(f"[increment] n={n_cur}: step1 (new-only, d_free={d_free})  "
              f"DP={DP1:.6e}  vs NPMV={DP_npmv_cur:.6e}"
              + (f"  scale={s1:.6e}" if s1 is not None else ""))

        # --- Step 2: optimize all parameters jointly (start from step1 solution) ---
        m_cur = n_cur*(n_cur-1)//2
        opt2 = nlopt.opt(nlopt.LN_BOBYQA, m_cur)
        opt2.set_lower_bounds(-bound*np.ones(m_cur))
        opt2.set_upper_bounds(+bound*np.ones(m_cur))
        opt2.set_xtol_rel(1e-6)
        opt2.set_maxeval(maxeval)

        def obj2(x: np.ndarray, grad: np.ndarray) -> float:
            return -det_prob(x, L0, L1, faprob=faprob, tau=tau, cdf_method=cdf_method)

        opt2.set_min_objective(obj2)
        x2_opt = opt2.optimize(x_step1)
        DP2 = det_prob(x2_opt, L0, L1, faprob=faprob, tau=tau, cdf_method=cdf_method)

        # Update previous solution for next increment
        x_prev = x2_opt
        D_prev = construct_decision_matrix(x_prev, normalize=False)

        # Final scale at this stage (for reporting)
        nrm2 = np.trace(D_prev @ N @ D_prev @ N)
        s2 = None
        if np.isfinite(nrm2) and nrm2 > 0:
            Dn = D_prev / np.sqrt(nrm2)
            s_tmp = scale_to_fap(L0, Dn, tau, faprob, cdf_method=cdf_method)
            s2 = float(s_tmp) if (s_tmp is not None and np.isfinite(s_tmp)) else None

        print(f"[increment] n={n_cur}: step2 (all-free)  "
              f"DP={DP2:.6e}  vs NPMV={DP_npmv_cur:.6e}"
              + (f"  scale={s2:.6e}" if s2 is not None else ""))

        if DP2 < DP_npmv_cur:
            print(f"[increment][warn] n={n_cur}: DP below NPMV; continuing (no fallback).")

        stages.append({
            "n": int(n_cur),
            "DP_step1": float(DP1),
            "scale_step1": (float(s1) if s1 is not None else None),
            "DP_step2": float(DP2),
            "scale_step2": (float(s2) if s2 is not None else None),
            "DP_NPMV": float(DP_npmv_cur),
        })

        n_prev = n_cur

    # ---- Final output at last processed size ----
    n_final = D_prev.shape[0]  # size of the last optimized matrix
    hdmat = hdcorrmat(psrpos[:n_final, :], psrTerm=True)
    DNP, DNPW, DDEF, L0, L1 = get_all_filters(1.0, hdmat)
    N = L0 @ L0.T

    # Compute DP(NPMV) at final size (same backend)
    DP_npmv_final = dp_from_normalized_matrix(DNPW, L0, L1, faprob=faprob, tau=tau, cdf_method=cdf_method)

    nrm2 = np.trace(D_prev @ N @ D_prev @ N)
    if not np.isfinite(nrm2) or nrm2 <= 0:
        raise RuntimeError("Final D normalization failed in incremental path.")
    D_unscaled = D_prev / np.sqrt(nrm2)
    s_final = scale_to_fap(L0, D_unscaled, tau, faprob, cdf_method=cdf_method)
    if s_final is None or not np.isfinite(s_final):
        raise RuntimeError("Final scaling to FAP failed in incremental path.")
    D_star = float(s_final) * D_unscaled

    DP_final = det_prob(get_lower_triangular_elements(D_prev), L0, L1,
                        faprob=faprob, tau=tau, cdf_method=cdf_method)

    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "D_star.npy"), D_star)
    np.save(os.path.join(outdir, "D_unscaled.npy"), D_unscaled)
    atomic_save_json(os.path.join(outdir, "result.json"),
        {
            "mode": "full-incremental",
            "seed": int(seed),
            "faprob": float(faprob),
            "tau": float(tau),
            "npsrs": int(n_final),
            "inc_start": int(n0),
            "inc_step": int(step),
            "DP": float(DP_final),
            "DP_NPMV_final": float(DP_npmv_final),
            "scale": float(s_final),
            "stages": stages,
            "cdf": cdf_method,
            "start_from_npmv": bool(start_from_npmv),
        }
    )
    return D_star, {"DP": float(DP_final), "scale": float(s_final)}

# ===================== FULL (with multiple optimizers) =====================

def optimize_with_full(psrpos: np.ndarray, faprob: float, tau: float, seed: int,
                       outdir: str, start_json: Optional[str], maxeval: int, bound: float,
                       cdf_method: str, start_from_npmv: bool, resume: bool = False,
                       optimizer: str = "bobyqa",
                       subspace_size: int = 150, subspace_evals: int = 80, subspace_cycles: int = 2,
                       nes_pop: int = 128, nes_iters: int = 30,
                       spsa_iters: int = 2500, spsa_step: float = 0.05,
                       isres_evals: int = 3000,
                       polish: str = "bobyqa", polish_evals: int = 1000
                       ) -> Tuple[np.ndarray, dict]:
    """FULL optimization with selectable optimizers."""
    hdmat = hdcorrmat(psrpos, psrTerm=True)
    h_opt = 1.0
    DNP, DNPW, DDEF, L_H0, L_H1 = get_all_filters(h_opt, hdmat)
    N = L_H0 @ L_H0.T

    n = psrpos.shape[0]
    m = n*(n-1)//2

    # ---------- pick starting vector x0 ----------
    x0 = None
    resume_path = os.path.join(outdir, "x_opt.json") if resume else None
    if resume_path:
        if os.path.exists(resume_path):
            try:
                with open(resume_path, 'r') as fp:
                    x_resume = np.array(json.load(fp), dtype=float)
                if x_resume.size == m:
                    x0 = x_resume
                    print(f"[full][resume] Starting from previous solution: {resume_path}")
                else:
                    warnings.warn(f"[full][resume] Ignoring {resume_path}: size {x_resume.size} != {m}.")
            except Exception as e:
                warnings.warn(f"[full][resume] Failed to read {resume_path}: {e}")
        else:
            warnings.warn(f"[full][resume] No {resume_path} found; falling back to standard init.")

    if x0 is None and start_json is not None:
        if os.path.exists(start_json):
            with open(start_json, 'r') as fp:
                x_try = np.array(json.load(fp), dtype=float)
            if x_try.size == m:
                x0 = x_try
                print(f"[full] Using start-json: {start_json}")
            else:
                warnings.warn(f"start-json size {x_try.size} != {m}; ignoring.")
        else:
            warnings.warn(f"start-json path does not exist: {start_json}")

    if x0 is None:
        opt_files = glob.glob('genx2-xopt-it*.json')
        if opt_files:
            indices = [int(of.split('genx2-xopt-it')[1].split('.')[0])
                       for of in opt_files if 'genx2-xopt-it' in of]
            if indices:
                legacy_json = f'genx2-xopt-it{np.max(indices)}.json'
                try:
                    with open(legacy_json, 'r') as fp:
                        x_try = np.array(json.load(fp), dtype=float)
                    if x_try.size == m:
                        x0 = x_try
                        print(f"[full] Using legacy start-json: {legacy_json}")
                    else:
                        warnings.warn(f"Legacy start-json size {x_try.size} != {m}; ignoring.")
                except Exception as e:
                    warnings.warn(f"Failed to read legacy start-json {legacy_json}: {e}")

    if x0 is None:
        D0 = DNPW.copy() if start_from_npmv else norm_filter_white(DDEF.copy())
        x0 = get_lower_triangular_elements(D0)
        print(f"[full] Using {'NPMV' if start_from_npmv else 'classic'} initialization.")

    x0 = np.clip(x0, -bound, +bound)
    eval_dp = _dp_obj_factory(L_H0, L_H1, faprob, tau, cdf_method)
    rng = np.random.default_rng(seed)

    # ---------- optimizer dispatch ----------
    optimizer = optimizer.lower()
    extra_meta: dict = {}
    x_best = x0.copy()

    if optimizer == "bobyqa":
        # Classic full-dimensional BOBYQA
        opt = nlopt.opt(nlopt.LN_BOBYQA, m)
        opt.set_lower_bounds(-bound*np.ones(m))
        opt.set_upper_bounds(+bound*np.ones(m))
        opt.set_xtol_rel(1e-6)
        opt.set_maxeval(maxeval)

        def obj(x: np.ndarray, grad: np.ndarray) -> float:
            return -eval_dp(x)

        opt.set_min_objective(obj)
        x_best = opt.optimize(x0)
        extra_meta.update({"optimizer_settings": {
            "maxeval": int(maxeval)
        }})

    elif optimizer == "subspace_bobyqa":
        x_best = _run_subspace_bobyqa(
            x_init=x0, eval_dp=eval_dp, bound=bound, seed=seed,
            size=subspace_size, evals_per_subspace=subspace_evals, n_cycles=subspace_cycles
        )
        extra_meta.update({"optimizer_settings": {
            "subspace_size": int(subspace_size),
            "subspace_evals": int(subspace_evals),
            "subspace_cycles": int(subspace_cycles)
        }})

    elif optimizer == "nes":
        # NES hyperparams
        pop = max(2, nes_pop)
        if pop % 2 == 1:
            pop += 1
        pairs = pop // 2
        sigma = 0.05 * bound  # perturbation scale
        alpha = 0.1          # base learning rate
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m_adam = np.zeros_like(x0)
        v_adam = np.zeros_like(x0)

        x = x0.copy()
        DP_best = eval_dp(x)
        x_best = x.copy()
        for t in range(1, nes_iters + 1):
            E = rng.normal(size=(pairs, m))
            g = np.zeros_like(x)
            for i in range(pairs):
                eps_i = E[i]
                xp = np.clip(x + sigma*eps_i, -bound, +bound)
                xm = np.clip(x - sigma*eps_i, -bound, +bound)
                fp = eval_dp(xp)
                fm = eval_dp(xm)
                if not np.isfinite(fp): fp = -1.0
                if not np.isfinite(fm): fm = -1.0
                g += (fp - fm) * eps_i
            g /= (2.0 * sigma * pairs)

            # Adam step on x
            m_adam = beta1*m_adam + (1-beta1)*g
            v_adam = beta2*v_adam + (1-beta2)*(g*g)
            m_hat = m_adam / (1 - beta1**t)
            v_hat = v_adam / (1 - beta2**t)
            x = np.clip(x + alpha * m_hat / (np.sqrt(v_hat) + eps), -bound, +bound)

            DP_x = eval_dp(x)
            if DP_x > DP_best:
                DP_best = DP_x
                x_best = x.copy()

        # optional polish
        if polish == "bobyqa" and polish_evals > 0:
            x_best = _polish_bobyqa(x_best, eval_dp, bound, polish_evals)
        extra_meta.update({"optimizer_settings": {
            "nes_pop": int(nes_pop),
            "nes_iters": int(nes_iters),
            "sigma": float(sigma),
            "alpha": float(alpha),
            "polish": polish,
            "polish_evals": int(polish_evals),
        }})

    elif optimizer == "spsa":
        # SPSA schedule (standard a_k, c_k)
        A = 0.1 * max(1, spsa_iters)
        alpha, gamma = 0.602, 0.101
        a0 = float(spsa_step)
        c0 = 0.05 * bound

        x = x0.copy()
        DP_best = eval_dp(x)
        x_best = x.copy()
        for k in range(1, spsa_iters + 1):
            ak = a0 / ((k + A) ** alpha)
            ck = c0 / (k ** gamma)
            delta = rng.choice([-1.0, +1.0], size=m)
            xp = np.clip(x + ck*delta, -bound, +bound)
            xm = np.clip(x - ck*delta, -bound, +bound)
            fp = eval_dp(xp)
            fm = eval_dp(xm)
            if not np.isfinite(fp): fp = -1.0
            if not np.isfinite(fm): fm = -1.0
            ghat = (fp - fm) / (2.0 * ck) * delta
            x = np.clip(x + ak * ghat, -bound, +bound)

            DP_x = eval_dp(x)
            if DP_x > DP_best:
                DP_best = DP_x
                x_best = x.copy()

        # optional polish
        if polish == "bobyqa" and polish_evals > 0:
            x_best = _polish_bobyqa(x_best, eval_dp, bound, polish_evals)
        extra_meta.update({"optimizer_settings": {
            "spsa_iters": int(spsa_iters),
            "spsa_step": float(spsa_step),
            "A": float(A),
            "alpha": float(alpha),
            "gamma": float(gamma),
            "c0": float(c0),
            "polish": polish,
            "polish_evals": int(polish_evals),
        }})

    elif optimizer == "isres_then_bobyqa":
        # Short ISRES global scatter, then full BOBYQA polish
        opt = nlopt.opt(nlopt.GN_ISRES, m)
        opt.set_lower_bounds(-bound*np.ones(m))
        opt.set_upper_bounds(+bound*np.ones(m))
        opt.set_maxeval(max(1, int(isres_evals)))

        def obj(x: np.ndarray, grad: np.ndarray) -> float:
            return -eval_dp(x)

        opt.set_min_objective(obj)
        try:
            x_isres = opt.optimize(x0)
        except Exception:
            x_isres = x0.copy()

        # polish with BOBYQA using polish_evals (or fallback to maxeval if not set)
        pe = polish_evals if polish_evals > 0 else maxeval
        x_best = _polish_bobyqa(x_isres, eval_dp, bound, pe)
        extra_meta.update({"optimizer_settings": {
            "isres_evals": int(isres_evals),
            "polish": "bobyqa",
            "polish_evals": int(pe),
        }})

    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    # ---------- finalize ----------
    return _finalize_and_save_full(
        x_best=x_best, L_H0=L_H0, L_H1=L_H1, N=N, faprob=faprob, tau=tau,
        cdf_method=cdf_method, outdir=outdir, seed=seed, m=m, maxeval=maxeval,
        bound=bound, start_json=start_json, start_from_npmv=start_from_npmv,
        resume=resume, optimizer_name=optimizer, extra_meta=extra_meta
    )

# ===================== CLI =====================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize quadratic filter with a chosen GX^2 CDF.\n"
            "Modes: full, full-incremental, legendre, zonal-alpha-aware, zonal-low-rank, bi-spectral.\n"
            "CDF:   analytic (fast) or imhof (robust, slower).\n"
            "NOTE: legacy aliases --mode original→full, basis→legendre are accepted."
        )
    )
    parser.add_argument("--mode", choices=[
        "full", "full-incremental", "legendre", "zonal-alpha-aware", "zonal-low-rank", "bi-spectral",
        # legacy aliases
        "original", "basis"
    ], default="full",
        help="Optimization mode (see description; 'original' and 'basis' map to new names).")
    parser.add_argument("--cdf", choices=["analytic", "imhof"], default="analytic",
                        help="CDF evaluator to use for FAP scaling and DP (default: analytic).")

    # init from NPMV toggle (default True)
    parser.add_argument("--start_from_npmv", dest="start_from_npmv", action="store_true",
                        help="Initialize from NPMV (off-diagonal NP) where applicable (default).")
    parser.add_argument("--no_start_from_npmv", dest="start_from_npmv", action="store_false",
                        help="Disable NPMV-based initialization.")
    parser.set_defaults(start_from_npmv=True)

    # NEW: resume for FULL mode
    parser.add_argument("--resume", action="store_true",
                        help="(FULL mode) Start from <outdir>/x_opt.json if present and compatible.")

    # ----- Optimizer options (FULL mode only) -----
    parser.add_argument("--optimizer", choices=["bobyqa", "subspace_bobyqa", "nes", "spsa", "isres_then_bobyqa"],
                        default="bobyqa",
                        help="(FULL mode) Choose optimization algorithm (default: bobyqa).")

    # Subspace-BOBYQA controls
    parser.add_argument("--subspace-size", type=int, default=150,
                        help="(FULL/subspace_bobyqa) Subspace size (default: 150).")
    parser.add_argument("--subspace-evals", type=int, default=80,
                        help="(FULL/subspace_bobyqa) Max evals per subspace (default: 80).")
    parser.add_argument("--subspace-cycles", type=int, default=2,
                        help="(FULL/subspace_bobyqa) Number of full cycles over disjoint subspaces (default: 2).")

    # NES controls
    parser.add_argument("--nes-pop", type=int, default=128,
                        help="(FULL/nes) NES population size (even; default: 128).")
    parser.add_argument("--nes-iters", type=int, default=30,
                        help="(FULL/nes) NES iterations (default: 30).")

    # SPSA controls
    parser.add_argument("--spsa-iters", type=int, default=2500,
                        help="(FULL/spsa) SPSA iterations (default: 2500).")
    parser.add_argument("--spsa-step", type=float, default=0.05,
                        help="(FULL/spsa) Base step size a0 (default: 0.05).")

    # ISRES pre-phase
    parser.add_argument("--isres-evals", type=int, default=3000,
                        help="(FULL/isres_then_bobyqa) Max evals for ISRES pre-phase (default: 3000).")

    # Optional polish stage (for nes/spsa; isres_then_bobyqa always polishes with BOBYQA)
    parser.add_argument("--polish", choices=["none", "bobyqa"], default="bobyqa",
                        help="(FULL/nes,spsa) Optional polish algorithm (default: bobyqa).")
    parser.add_argument("--polish-evals", type=int, default=1000,
                        help="(FULL/nes,spsa) Eval budget for polish (default: 1000).")

    parser.add_argument("--Lmax", type=int, default=20, help="Max Legendre order (zonal modes).")
    parser.add_argument("--starts", type=int, default=20, help="Multistarts (legendre mode).")
    parser.add_argument("--r_lowrank", type=int, default=8, help="# low-rank anisotropic directions (zonal-low-rank).")
    parser.add_argument("--kmax", type=int, default=8, help="Max |S| to sweep in bi-spectral mode.")
    parser.add_argument("--faprob", type=float, default=2.87e-7, help="Target false-alarm probability.")
    parser.add_argument("--tau", type=float, default=1.0, help="Threshold τ for the statistic (use τ>0).")
    parser.add_argument("--npsrs", type=int, default=67, help="# pulsars to use from the NG15 set.")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed (used for starts/nullspace init).")
    parser.add_argument("--outdir", type=str, default="opt_output", help="Output directory.")
    # full-mode extras
    parser.add_argument("--start-json", type=str, default=None,
                        help="Path to JSON with starting vector x0 (full mode).")
    parser.add_argument("--maxeval", type=int, default=4000, help="NLopt maxeval (full/zonal modes).")
    parser.add_argument("--bound", type=float, default=5.0,
                        help="Abs bound per parameter for full-mode BOBYQA.")
    # incremental extras (underscore versions)
    parser.add_argument("--inc_start", type=int, default=10,
                        help="Starting # pulsars for full-incremental mode (default 10).")
    parser.add_argument("--inc_step", type=int, default=1,
                        help="Increment size in # pulsars for full-incremental mode (default 1).")

    args = parser.parse_args()

    # Select pulsars
    npsrs = args.npsrs
    psrpos = psrs_pos_15yr[:npsrs, :]

    # Legacy alias mapping
    if args.mode == "original":
        warnings.warn("--mode original is deprecated; using --mode full.")
        args.mode = "full"
    if args.mode == "basis":
        warnings.warn("--mode basis is deprecated; using --mode legendre.")
        args.mode = "legendre"

    if args.mode == "legendre":
        D_star, best = optimize_with_legendre(psrpos, faprob=args.faprob, tau=args.tau,
                                              Lmax=args.Lmax, n_starts=args.starts,
                                              seed=args.seed, outdir=args.outdir,
                                              cdf_method=args.cdf, start_from_npmv=args.start_from_npmv)
    elif args.mode == "full":
        D_star, best = optimize_with_full(psrpos, faprob=args.faprob, tau=args.tau,
                                          seed=args.seed, outdir=args.outdir,
                                          start_json=args.start_json,
                                          maxeval=args.maxeval, bound=args.bound,
                                          cdf_method=args.cdf, start_from_npmv=args.start_from_npmv,
                                          resume=args.resume,
                                          optimizer=args.optimizer,
                                          subspace_size=args.subspace_size,
                                          subspace_evals=args.subspace_evals,
                                          subspace_cycles=args.subspace_cycles,
                                          nes_pop=args.nes_pop, nes_iters=args.nes_iters,
                                          spsa_iters=args.spsa_iters, spsa_step=args.spsa_step,
                                          isres_evals=args.isres_evals,
                                          polish=args.polish, polish_evals=args.polish_evals)
    elif args.mode == "full-incremental":
        D_star, best = optimize_with_full_incremental(psrpos, faprob=args.faprob, tau=args.tau,
                                                      seed=args.seed, outdir=args.outdir,
                                                      maxeval=args.maxeval, bound=args.bound,
                                                      cdf_method=args.cdf,
                                                      inc_start=args.inc_start,
                                                      inc_step=args.inc_step,
                                                      start_from_npmv=args.start_from_npmv)
    elif args.mode == "zonal-alpha-aware":
        D_star, best = optimize_zonal_alpha_aware(psrpos, faprob=args.faprob, tau=args.tau,
                                                  Lmax=args.Lmax, seed=args.seed,
                                                  outdir=args.outdir, maxeval=args.maxeval,
                                                  cdf_method=args.cdf, start_from_npmv=args.start_from_npmv)
    elif args.mode == "zonal-low-rank":
        D_star, best = optimize_zonal_lowrank(psrpos, faprob=args.faprob, tau=args.tau,
                                              Lmax=args.Lmax, r_lowrank=args.r_lowrank,
                                              seed=args.seed, outdir=args.outdir,
                                              maxeval=args.maxeval, cdf_method=args.cdf,
                                              start_from_npmv=args.start_from_npmv)
    elif args.mode == "bi-spectral":
        D_star, best = optimize_bispectral(psrpos, faprob=args.faprob, tau=args.tau,
                                           Lmax=args.Lmax, kmax=args.kmax,
                                           seed=args.seed, outdir=args.outdir,
                                           cdf_method=args.cdf, start_from_npmv=args.start_from_npmv)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print(f"[done] mode={args.mode}  DP={best['DP']:.6e}  scale={best['scale']:.6e}  outdir={args.outdir}")


if __name__ == "__main__":
    main()

