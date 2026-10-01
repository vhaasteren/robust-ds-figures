#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arbitrary-precision evaluation of the CDF F(tau, C, Q) of a quadratic
detection statistic (single-frequency-bin complex toy model).

This is the implementation described in the subsection "Numerical evaluation
of the CDF F(tau, C, Q)" of the paper *"Optimal robust detection statistics
for pulsar timing arrays."*

Given the n (nonzero, nondegenerate) eigenvalues lambda_j of
E = C^{1/2} Q C^{1/2}, the CDF is

    F(tau) =     sum_{j | lambda_j < 0} c_j exp(-tau/lambda_j)    for tau <= 0
    F(tau) = 1 - sum_{j | lambda_j > 0} c_j exp(-tau/lambda_j)    for tau >= 0

    c_j = prod_{k != j} lambda_j / (lambda_j - lambda_k)

Notation follows the paper: Q is the filter of the quadratic statistic
D = z^dagger Q z, H_N and H_S are the null (noise) and signal hypotheses with
covariances N and S, FAP is the false-alarm probability (1 - F under H_N) and
DP is the detection probability (1 - F under H_S).

The coefficients c_j have alternating signs and can be enormous (|c_j| ~ 1e32
for the 67 NANOGrav pulsars, ~ 1e226 for 300 pulsars), so the sum suffers from
catastrophic cancellation. `cdf` first estimates, in double precision, the
number of mantissa bits needed to reach the requested accuracy. If fewer than
53 bits suffice it completes the sum in double precision; otherwise it
evaluates the sum with `mpmath` at the required precision.

Contents:
  • Pulsar positions -> Hellings-Downs matrix -> NP / NPMV / DF filters ->
    normalized eigenvalues of E under H_N (noise only) and H_S (signal).
  • cdf / ccdf: the CDF and complementary CDF, with the number of mantissa
    bits used.
  • Diagnostics: CDF plots, ROC curves, and the detection probability at a
    fixed false-alarm probability.

Usage:
  As a module (see ComputeCDF.ipynb):

      import compute_cdf as cc
      evals = cc.make_eigenvalue_dict(cc.PULSAR_FILES['67'])
      F, bits = cc.cdf(-10.0, evals['EVNnpmv'], accuracy=1e-12)

  As a script, to print the detection probability at the 5-sigma FAP:

      python3 compute_cdf.py                  # all pulsar sets
      python3 compute_cdf.py --cases 30 67    # selected pulsar sets

Requirements: numpy, matplotlib, mpmath (substantially faster with the
optional gmpy2 backend).
"""

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

# Pulsar-position files shipped with this repository, keyed by number of
# pulsars. Line format: pulsar_name   RA_degrees   DEC_degrees
_DATA_DIR = Path(__file__).resolve().parent
PULSAR_FILES = {
    '30':  _DATA_DIR / 'ppta-ra-dec.txt',
    '67':  _DATA_DIR / 'ng-ra-dec.txt',
    '121': _DATA_DIR / 'five_PTA.txt',
    '150': _DATA_DIR / 'Random150.txt',
    '200': _DATA_DIR / 'Random200.txt',
    '250': _DATA_DIR / 'Random250.txt',
    '300': _DATA_DIR / 'Random300.txt',
}

# The three detection statistics: Neyman-Pearson (NP), Neyman-Pearson minimum
# variance (NPMV), and deflection (DF). For the CURN null hypothesis used here,
# DF equals DFCC, the traditional "optimal" cross-correlation statistic.
STATISTICS = ['np', 'npmv', 'def']
LABELS = {'np': 'NP', 'npmv': 'NPMV', 'def': 'DF'}

# False-alarm probability corresponding to 5 sigma
FAP_5SIGMA = 2.87e-7


def check_mpmath_backend():
    """Warn if mpmath is not using the (much faster) gmpy2 backend."""
    backend = mp.libmp.BACKEND
    if backend != 'gmpy':
        print(f"mpmath backend: {backend}")
        print("mpmath is NOT using gmpy2.")
        print("Installing gmpy2 and restarting Python will speed up the code.")


# ===================== Pulsars, filters and eigenvalues =====================

# Read datafile. Line format: pulsar_name   RA_degrees   DEC_degrees
def readPulsars(datafile):
    name, ra, dec = np.loadtxt(datafile,
                               dtype={'names': ('name', 'ra', 'dec'),
                                      'formats': ('U16', float, float)},
                               unpack=True)
    # convert to spherical polar coordinates theta, phi (in radians)
    theta = np.pi*(90.0 - dec)/180.0
    phi = np.pi*ra/180.0
    return name, theta, phi

# compute HD matrix from a pulsar data file
def HD_from_data(dataFile):
    # theta and phi for each pulsar
    names, theta, phi = readPulsars(dataFile)
    print(f'Read data for {len(theta)} pulsars from file {Path(dataFile).name}')
    # Use broadcasting to compute HD matrix
    ti = theta[:, None]
    pi = phi[:, None]
    tj = theta[None, :]
    pj = phi[None, :]
    cosg = np.cos(ti)*np.cos(tj) + np.sin(ti)*np.sin(tj)*np.cos(pi-pj)
    cosg = np.clip(cosg, -1.0, 1.0)
    # compute Hellings and Downs function
    z = 0.5*(1.0-cosg)
    logz = 0.0*z
    np.log(z, where = z > 0, out = logz)
    tot = 1.0/3.0 - z/6.0 + z*logz
    # double along the diagonal for (1 + delta_ab) factor
    diag = (1.0/3.0)*np.identity(tot.shape[0])
    return tot+diag

# take the square root of a non-negative matrix
def sqrtmat(A):
    eigvals, eigvecs = np.linalg.eigh(A)
    sqrt_eigvals = np.sqrt(np.clip(eigvals, 0, None))
    return eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.conj().T

# return sorted eigenvalues of matrix, either computing or using scale factor
def RenormedEigenvals(mat, scalefac = None):
    arr = np.sort(np.linalg.eigvals(mat))
    if scalefac is None:
        # find factor that makes product of eigenvalues one
        d = arr.size
        logdet = np.sum(np.log(np.abs(arr)))
        scalefac = np.exp(-logdet/d)
    arr *= scalefac
    return arr, scalefac

# Compute different filters Q and create dictionary of the corresponding eigenvalues
def make_eigenvalue_dict(filename):
    """
    Eigenvalues of E = C^{1/2} Q C^{1/2} for the pulsars in `filename`.

    Returns a dictionary with keys 'EVN<stat>' (C = N, noise-only hypothesis;
    gives false-alarm probabilities) and 'EVS<stat>' (C = S, signal
    hypothesis; gives false-dismissal probabilities), for <stat> in
    'np', 'npmv', 'def'.

    For each statistic the filter Q is rescaled so that the product of the
    |eigenvalues| under H_N is one; the same scale factor is applied under
    H_S, so that both sets of eigenvalues refer to the same threshold tau.
    """

    # Get HD curve and identity matrix of same shape
    hd = HD_from_data(filename)
    I = np.identity(hd.shape[0])

    # Magnitude of pulsar and GWB variance
    # Toy model of the paper: sigma^2 = 1 and h^2 = 1, with an HD matrix that
    # is 1 on the diagonal. Factor 3/2 because the HD matrix returned by
    # HD_from_data is 2/3 on the diagonal (not 1)
    s2 = 1.0
    h2 = 1.5

    # covariance matrices and inverses for H_S ...
    S = h2*hd + s2*I
    Sinv = np.linalg.inv(S)
    Ssqrt = sqrtmat(S)

    # ... and for H_N
    N = np.diag(np.diag(S))
    Ninv = np.linalg.inv(N)
    Nsqrt = sqrtmat(N)

    # Three filters: deflection, Neyman-Pearson, and NP min variance
    Qnp =  Ninv - Sinv
    Qnpmv = Qnp.copy()
    np.fill_diagonal(Qnpmv, 0.0)
    Qdef = Ninv @ (S - N) @ Ninv

    # E matrices for CDF (false alarm/p-values)
    ENnp =   Nsqrt @ Qnp   @ Nsqrt
    ENnpmv = Nsqrt @ Qnpmv @ Nsqrt
    ENdef =  Nsqrt @ Qdef  @ Nsqrt

    # E matrices for CDF (false dismissal)
    ESnp   = Ssqrt @ Qnp   @ Ssqrt
    ESnpmv = Ssqrt @ Qnpmv @ Ssqrt
    ESdef  = Ssqrt @ Qdef  @ Ssqrt

    # make dictionary of normalized Eigenvalues
    output = {}
    output['EVNnp'],npNorm   = RenormedEigenvals(ENnp)
    output['EVSnp'],_        = RenormedEigenvals(ESnp,  npNorm)
    output['EVNnpmv'],mvNorm = RenormedEigenvals(ENnpmv)
    output['EVSnpmv'],_      = RenormedEigenvals(ESnpmv, mvNorm)
    output['EVNdef'],defNorm = RenormedEigenvals(ENdef)
    output['EVSdef'],_       = RenormedEigenvals(ESdef, defNorm)

    return output

# Print the number of eigenvalues of each sign
def print_eigenvalue_signs(ev):
    print('Negative eigenvalues: ', np.sum(ev < 0))
    print('Zero eigenvalues:     ', np.sum(ev == 0))
    print('Positive eigenvalues: ', np.sum(ev > 0))


# ===================== CDF with automatic precision =====================

# Check that eigenvalues are nonzero and nondegenerate, and sort
def validate_and_sort_eigenvalues(ev):
    if np.any(ev == 0):
        raise ValueError("eigenvalues must be nonzero!")
    evout = np.sort(ev)
    ratios = evout[1:]/evout[:-1]
    if np.any(np.abs(ratios - 1.0) < 1.e-12):
        raise ValueError("eigenvalues must be nondegenerate!")
    return evout

def cdf(tau, eigenvalues, accuracy = 1.e-8):
    """
    Compute CDF F(tau, E), which depends upon eigenvalues of E = C^1/2 Q C^1/2.

    Parameters
    ----------
    tau : float, threshold.
    eigenvalues : numpy.ndarray, nonzero and nondegenerate eigenvalues of E.
    accuracy : float, desired absolute accuracy of the result.

    Returns
    -------
    (F, bits) : the CDF value (float in [0, 1]) and the number of mantissa
        bits needed for the computation. If bits < 53 the sum was evaluated
        in double precision, otherwise with mpmath using `bits` bits.
    """
    # use reflection formula to guarantee tau <= 0
    if tau > 0:
        val, bits = cdf(-tau,-eigenvalues, accuracy)
        return 1.0 - val, bits
    eva = validate_and_sort_eigenvalues(eigenvalues)
    # eigenvalues are sorted, set m to the number of negative ones
    m = np.searchsorted(eva, 0.0)
    if m == 0:
        return 0.0, 0
    # evaluate log of the terms in double precision, using broadcasting:
    # r[j,k] = (lambda_j - lambda_k)/lambda_j, so that 1/c_j = prod_k r[j,k]
    evn = eva[:m]
    lj = evn[:, None]
    lk = eva[None, :]
    r = (lj - lk)/lj
    np.fill_diagonal(r, 1.0)
    # log_abs_terms[j] = -log|c_j exp(-tau/lambda_j)|
    log_abs_terms = np.sum(np.log(np.abs(r)), axis=1) + tau/evn
    # bits left and right of binary point, assume errors add in quadrature
    bitsR = max(0, -math.log2(accuracy*(m**-0.5)))
    bitsL = max(0, -np.min(log_abs_terms)/math.log(2))
    bits_needed = math.ceil(bitsR + bitsL)
    if bits_needed < 53:
        # double precision good enough
        signs = np.prod(np.sign(r), axis=1)
        val = np.sum(signs * np.exp(-log_abs_terms))
    else:
        # use arbitrary precision library
        with mp.workprec(bits_needed):
            val = cdf_mp(tau, eva, m)
    return np.clip(val, 0.0, 1.0),bits_needed

# Compute CDF with high precision, assuming tau <= 0, eigenvalues sorted,
# and the first m eigenvalues negative. Uses the current mpmath precision.
def cdf_mp(tau, eigenvalues, m):
    n = len(eigenvalues)
    tau_mp = mp.mpf(tau)
    eva = [mp.mpf(x) for x in eigenvalues]
    val = mp.mpf(0)
    for j in range(m):
        lj = eva[j]
        prod = mp.mpf(1)
        for k in range(n):
            if k == j:
                continue
            lk = eva[k]
            rjk = (lj - lk)/lj
            prod *= rjk
        val += mp.exp(-tau_mp/lj)/prod
    return float(val)

# Compute complementary CDF (survival function) bar-F(tau, E) = 1 - F(tau, E)
# also returns number of mantissa bits needed for computation
def ccdf(tau, eigenvalues, accuracy = 1.e-8):
    return cdf(-tau, -eigenvalues, accuracy)


# ===================== Diagnostics and plots =====================

# Illustrate precision: compare F computed with decreasing target error
# against the value computed with target error 1e-16
def print_precision_test(tau, ev):
    Fprecise,bits = cdf(tau, ev , 10**-16)
    error=10**-2
    print(f'F(tau={tau},Eigenvalues) = {Fprecise} mantissa bits = {bits}')
    while error > 10**-16:
        F,bits = cdf(tau, ev , error)
        print(f'F-Faccurate = {F-Fprecise}  target error = {error}')
        error /= 10

# Plot the cumulative distribution function for the eigenvalues of a matrix E,
# together with the number of mantissa bits needed to compute it
# Pick the range of tau to cover 0.001 < F < 0.999
def plot_cdf(eigens, name):
    dtau = 1.0
    tol = 1e-8
    Fmin = 0.001
    Fmax = 0.999

    # start at tau = 0
    F0, b0 = cdf(0.0, eigens, tol)

    # walk to the right until CDF >= Fmax
    tauplus = [0.0]
    Fplus = [F0]
    bitsplus = [b0]

    tau = dtau
    while True:
        F, b = cdf(tau, eigens, tol)
        tauplus.append(tau)
        Fplus.append(F)
        bitsplus.append(b)
        if F >= Fmax:
            break
        tau += dtau

    # walk to the left until CDF <= Fmin
    tauminus = []
    Fminus = []
    bitsminus = []

    tau = -dtau
    while True:
        F, b = cdf(tau, eigens, tol)
        tauminus.append(tau)
        Fminus.append(F)
        bitsminus.append(b)
        if F <= Fmin:
            break
        tau -= dtau

    # combine left and right sides into increasing tau order
    F = np.array(Fminus[::-1] + Fplus)
    mask = (F >= Fmin) & (F <= Fmax)
    F = F[mask]
    tau = np.array(tauminus[::-1] + tauplus)[mask]
    bits = np.array(bitsminus[::-1] + bitsplus)[mask]

    fig, ax1 = plt.subplots()

    # Left y-axis (CDF)
    ax1.plot(tau, F, 'b-', label='F')
    ax1.set_xlabel('tau')
    ax1.set_ylabel('CDF F', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Right y-axis (bits)
    ax2 = ax1.twinx()
    ax2.plot(tau, bits, 'r--', label='bits')
    ax2.set_ylabel('mantissa bits', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(bottom=0)

    plt.title(f'CDF for {name} pulsars')
    plt.show()

# For each statistic, plot false-alarm and false-dismissal probability vs
# threshold, then plot the ROC curves of all statistics together.
# evecs is a dictionary from make_eigenvalue_dict; case labels the plots.
def plot_all_ROC(evecs, case):
    stats = STATISTICS

    dtau = 1.0
    pmin = 1e-8
    drop = 1e-10
    tol = 1e-12

    # store ROC data for combined plot
    roc_data = {}

    for statistic in stats:
        E_N = evecs[f'EVN{statistic}']
        E_S = evecs[f'EVS{statistic}']

        # build positive thresholds and probabilities
        tauplus, faplus, fdplus = [], [], []
        t = 0.0
        while True:
            fa, _   = ccdf(t, E_N, tol)
            omfd, _ = ccdf(t, E_S, tol) # one minus false dismissal
            tauplus.append(t)
            faplus.append(fa)
            fdplus.append(1.0-omfd)
            if fa < pmin or omfd < pmin:
                break
            t += dtau

        # build negative thresholds and probabilities
        tauminus, faminus, fdminus = [], [], []
        t = -dtau
        while True:
            omfa, _ = cdf(t, E_N, tol) # one minus false alarm
            fd, _   = cdf(t, E_S, tol)
            tauminus.append(t)
            faminus.append(1.0-omfa)
            fdminus.append(fd)
            if omfa < pmin or fd < pmin:
                break
            t -= dtau

        # merge negative and positive sides
        tau = np.array(tauminus[::-1] + tauplus)
        FA  = np.array(faminus[::-1] + faplus)
        FD  = np.array(fdminus[::-1] + fdplus)

        # drop points where FA or FD is too small
        mask = (FA >= drop) & (FD >= drop)
        tau = tau[mask]
        FA  = FA[mask]
        FD  = FD[mask]

        # --- Plot FA and FD vs tau ---
        plt.figure()
        plt.title(f'{LABELS[statistic]} statistic performance for {case} pulsars')
        plt.plot(tau, FA, label='False Alarm')
        plt.plot(tau, FD, label='False Dismissal')
        plt.xlabel(r'Threshold $\tau$')
        plt.ylabel('Probability')
        plt.yscale('log')
        plt.ylim(top=1.05)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # --- Prepare ROC data ---
        idx = np.argsort(FA)
        FA_sorted = FA[idx]
        DP_sorted = (1.0 - FD)[idx]

        keep = FA_sorted >= 10**-8
        roc_data[statistic] = (FA_sorted[keep], DP_sorted[keep])

    # --- Combined ROC plot ---
    plt.figure()
    for statistic in stats:
        FA_plot, DP_plot = roc_data[statistic]
        plt.plot(FA_plot, DP_plot, lw=2, label=LABELS[statistic])

    plt.xlabel('False Alarm Probability')
    plt.ylabel('Detection Probability')
    plt.title(f'ROC Curves for {case} pulsars')
    plt.xscale('log')
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ===================== Detection probability at fixed FAP =====================

def _make_probability_functions(E_N, E_S, tol):
    def fap_at_tau(tau):
        fa, _ = ccdf(tau, E_N, tol)
        return fa

    def fd_at_tau(tau):
        omfd, _ = ccdf(tau, E_S, tol)   # detection probability
        return 1.0 - omfd               # false dismissal probability

    return fap_at_tau, fd_at_tau


def _relative_error(value, target):
    return abs(value - target) / target


def _print_result(statistic, tau, fap, fd):
    dp = 1.0 - fd
    print(
        f"{statistic:4s}  "
        f"threshold = {tau:.5g}   "
        f"FAP = {fap:.5g}   "
        f"FDP = {fd:.5g}   "
        f"DP = {dp:.5g}"
    )


def _expand_bracket(fap_at_tau, target_fap, tau0, step0, max_expand):
    """
    Find lo, hi such that
        fap_at_tau(lo) - target_fap
    and
        fap_at_tau(hi) - target_fap
    have opposite signs.
    """
    fa0 = fap_at_tau(tau0)
    f0 = fa0 - target_fap

    if f0 == 0.0:
        return tau0, tau0

    # Decide which direction to search first
    direction = -1.0 if fa0 < target_fap else 1.0

    lo = hi = tau0
    flo = fhi = f0
    step = step0

    for _ in range(max_expand):
        tau_new = tau0 + direction * step
        f_new = fap_at_tau(tau_new) - target_fap

        if direction > 0:
            lo, flo = hi, fhi
            hi, fhi = tau_new, f_new
        else:
            hi, fhi = lo, flo
            lo, flo = tau_new, f_new

        if flo * fhi <= 0:
            return lo, hi

        step *= 2.0

    raise RuntimeError("Could not bracket solution")


def _bisect_for_fap(fap_at_tau, target_fap, lo, hi, reltol, max_bisect):
    """
    Solve fap_at_tau(tau) ~= target_fap by bisection on [lo, hi].
    """
    if lo == hi:
        tau = lo
        return tau, fap_at_tau(tau)

    for _ in range(max_bisect):
        mid = 0.5 * (lo + hi)
        fa_mid = fap_at_tau(mid)

        if _relative_error(fa_mid, target_fap) <= reltol:
            return mid, fa_mid

        if fa_mid > target_fap:
            lo = mid
        else:
            hi = mid

    tau = 0.5 * (lo + hi)
    return tau, fap_at_tau(tau)


def _solve_statistic(E_N, E_S, target_fap, reltol, tol, tau0, step0,
                     max_expand, max_bisect):
    fap_at_tau, fd_at_tau = _make_probability_functions(E_N, E_S, tol)

    fa0 = fap_at_tau(tau0)
    if _relative_error(fa0, target_fap) <= reltol:
        tau_star = tau0
        fa_star = fa0
    else:
        lo, hi = _expand_bracket(
            fap_at_tau=fap_at_tau,
            target_fap=target_fap,
            tau0=tau0,
            step0=step0,
            max_expand=max_expand,
        )
        tau_star, fa_star = _bisect_for_fap(
            fap_at_tau=fap_at_tau,
            target_fap=target_fap,
            lo=lo,
            hi=hi,
            reltol=reltol,
            max_bisect=max_bisect,
        )

    fd_star = fd_at_tau(tau_star)
    return tau_star, fa_star, fd_star


def find_tau_for_fap(evecs, case, target_fap=FAP_5SIGMA, reltol=1e-3, tol=1e-12,
                     tau0=0.0, step0=1.0, max_expand=200, max_bisect=200):
    """
    For each statistic in ['np', 'npmv', 'def'], find the threshold tau
    such that the false-alarm probability (FAP) ccdf(tau, E_N) ~= target_fap.

    evecs is a dictionary from make_eigenvalue_dict; case labels the output.

    Prints:
      (a) threshold
      (b) FAP
      (c) false-dismissal probability (FDP)
      (d) detection probability (DP = 1 - FDP)
    """
    stats = STATISTICS

    print(f"case = {case} pulsars")
    print(f"target FAP = {target_fap:.12g}")
    print()

    for statistic in stats:
        E_N = evecs[f'EVN{statistic}']
        E_S = evecs[f'EVS{statistic}']

        tau_star, fa_star, fd_star = _solve_statistic(
            E_N=E_N,
            E_S=E_S,
            target_fap=target_fap,
            reltol=reltol,
            tol=tol,
            tau0=tau0,
            step0=step0,
            max_expand=max_expand,
            max_bisect=max_bisect,
        )

        _print_result(LABELS[statistic], tau_star, fa_star, fd_star)


# ===================== Command-line interface =====================

def main():
    ap = argparse.ArgumentParser(
        description="Detection probability of the NP, NPMV and DF statistics "
                    "at a fixed false-alarm probability, using the "
                    "arbitrary-precision CDF.")
    ap.add_argument("--cases", nargs="+", default=list(PULSAR_FILES),
                    choices=list(PULSAR_FILES),
                    help="pulsar sets to use, by number of pulsars (default: all)")
    ap.add_argument("--fap", type=float, default=FAP_5SIGMA,
                    help="target false-alarm probability (default: 5 sigma)")
    ap.add_argument("--reltol", type=float, default=1e-5,
                    help="relative tolerance on the FAP when solving for the threshold")
    args = ap.parse_args()

    check_mpmath_backend()
    for case in args.cases:
        print("==================================")
        evecs = make_eigenvalue_dict(PULSAR_FILES[case])
        find_tau_for_fap(evecs, case, target_fap=args.fap, reltol=args.reltol)


if __name__ == "__main__":
    main()
