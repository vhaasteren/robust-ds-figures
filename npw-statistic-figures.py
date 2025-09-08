import numpy as np
import scipy.linalg as sl
import scipy.integrate as sint
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import json
import pickle
import gzip

# Tweak matplotlib settings
npw_rcparams = {
    "backend": "module://matplotlib_inline.backend_inline",
    #"backend": "pdf",
    "axes.labelsize": 20,
    "lines.markersize": 4,
    "font.size": 16, # 10,
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.major.size": 6,
    "ytick.minor.size": 3,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "lines.markeredgewidth": 1,
    "axes.linewidth": 1.2,
    "legend.fontsize": 16,  #7,
    "xtick.labelsize": 12,
    "xtick.direction": "in",
    "xtick.minor.visible": True,
    "xtick.major.top": True,
    "xtick.minor.top": True,
    "ytick.labelsize": 12,
    "ytick.direction": "in",
    "ytick.minor.visible": True,
    "ytick.major.right": True,
    "ytick.minor.right": True,
    "savefig.dpi": 400,
    "path.simplify": True,
    "font.family": "serif",
    "font.serif": "Times",
    "text.usetex": True,
    #"text.latex.preamble": r'\usepackage{cmbright}',
    "figure.figsize": [10.0, 7.0]}
    #"figure.figsize": [7.0, 5.0]}

mpl.rcParams.update(npw_rcparams)

# Functions for Generalized Chi-squared distributions
def imhof(u, x, eigen_values, output='cdf'):
    theta = 0.5 * np.sum(np.arctan(eigen_values[:,np.newaxis] * u), axis=0) - 0.5 * x * u
    rho = np.prod((1.0 + (eigen_values[:,np.newaxis] * u)**2)**0.25, axis=0)

    rv = np.sin(theta) / (u * rho) if output=='cdf' else np.cos(theta) / rho

    return rv

def gx2pdf(eigen_values, xs, cutoff=1e-6, limit=100, epsabs=1e-6):
    """Calculate the GX2 PDF as a function of sx, based off of eigenvalues 'eigen_values'"""

    eigen_values = eigen_values[:cutoff] if cutoff > 1 else eigen_values[np.abs(eigen_values) > cutoff]

    return np.array([sint.quad(lambda u: float(imhof(u, x, eigen_values, output='pdf')),
                                                0, np.inf, limit=limit, epsabs=epsabs)[0] / (2*np.pi) for x in xs])

def gx2cdf(eigr, xs, cutoff=1e-6, limit=100, epsabs=1e-6):
    """Calculate the GX2 CDF as a function of sx, based off of eigenvalues 'eigr'"""

    eigen_values = eigr[:cutoff] if cutoff > 1 else eigr[np.abs(eigr) > cutoff]

    return np.array([0.5 - sint.quad(lambda u: float(imhof(u, x, eigen_values)),
                                                0, np.inf, limit=limit, epsabs=epsabs)[0] / np.pi for x in xs])

def get_gx2_cdf(L, Q, complex_valued=False):
    LQL = np.dot(L.T, np.dot(Q, L))
    w = sl.eigvalsh(LQL)

    x = np.linspace(-10, 20, 200)

    if complex_valued:
        ww = w.repeat(2)
    else:
        ww = w

    ds_an = gx2pdf(ww, x, cutoff=1e-6, limit=200, epsabs=1e-9)
    ds_an_cdf = gx2cdf(ww, x, cutoff=1e-6, limit=200, epsabs=1e-9)

    return ds_an, ds_an_cdf, x, w


def get_auc(cdf_h0, cdf_hs, skip_vals=1):
    """
    Calculate the Area Under the Curve (AUC) for the given CDFs.

    :param cdf_h0:  H0 CDF
    :param cdf_hs:  HS CDF
    :skip_vals:     To reduce numerical problems, we can skip edge values
    """
    # Reverse the CDFs and skip the first and last values
    cdf_h0 = cdf_h0[::-1][skip_vals:-skip_vals]
    cdf_hs = cdf_hs[::-1][skip_vals:-skip_vals]

    # Calculate AUC using trapezoidal rule
    auc = np.trapz(1 - cdf_hs, 1 - cdf_h0)
    return auc


# Functions to convert filters
def norm_filter(Q):
    """Normalize per Equation (6), but assume whitened H0: N=I"""
    QQ = np.dot(Q, Q)
    return Q / np.sqrt(2 * np.trace(QQ))

def B_to_filters(Bmat, idx):
    """
    Starting with 'B' from Equation (23) of van Haasteren, Allen, Romano,
    derive all the quadratic filters
    """
    C = Bmat + np.identity(len(Bmat))
    L = sl.cholesky(C, lower=True)
    Q_def = norm_filter(Bmat)

    # Also calculate Q_np and Q_npw
    BBi = sl.cho_solve((L, True), Bmat)
    Q_np = norm_filter(BBi)

    for ii in range(len(idx)-1):
        # Zero out block-diagonals of auto terms of pulsars
        slc = slice(idx[ii], idx[ii+1])
        BBi[slc, slc] = 0

    Q_npw = norm_filter(BBi)

    return C, L, Q_def, Q_np, Q_npw

def dp_at_fap(cdf_h0, cdf_h1, fap0, use_loglog=True):
    """
    Interpolate detection probability (DP) at a target FAP (fap0) using ROC-space interpolation.
    By default does log–log PCHIP for stability at tiny probabilities.
    """
    fap = 1.0 - np.asarray(cdf_h0)
    dp  = 1.0 - np.asarray(cdf_h1)

    # Keep finite, positive entries (needed for log-space)
    m = np.isfinite(fap) & np.isfinite(dp)
    fap, dp = fap[m], dp[m]

    # Sort by FAP ascending and deduplicate (strictly increasing for PCHIP)
    order = np.argsort(fap)
    fap, dp = fap[order], dp[order]
    fap, uniq_idx = np.unique(fap, return_index=True)
    dp = dp[uniq_idx]

    # Clip query inside available range
    fap0 = float(np.clip(fap0, fap.min(), fap.max()))

    # Prefer log–log PCHIP for tiny probabilities
    if use_loglog:
        # Filter strictly positive pairs for log-space interpolation
        pos = (fap > 0) & (dp > 0)
        fap_pos, dp_pos = fap[pos], dp[pos]
        try:
            from scipy.interpolate import PchipInterpolator
            interp = PchipInterpolator(np.log(fap_pos), np.log(dp_pos), extrapolate=True)
            return float(np.exp(interp(np.log(fap0))))
        except Exception:
            # Fallback to NumPy linear interp in log–log space
            return float(np.exp(np.interp(np.log(fap0), np.log(fap_pos), np.log(dp_pos))))
    else:
        # Linear–linear interpolation
        try:
            from scipy.interpolate import PchipInterpolator
            interp = PchipInterpolator(fap, dp, extrapolate=True)
            return float(interp(fap0))
        except Exception:
            return float(np.interp(fap0, fap, dp))


# B of equation (23)
with gzip.GzipFile('./Bmatrix.npy.gz', 'rb') as fp:
    Bmat = np.load(fp).astype(float)

# Pulsar indices in the B matrix
with open('./Bmatrix-indices.json', 'r') as fp:
    Bidx = json.load(fp)

# Detection statistic values for the real data
# Here for reference, in case you want to check
# Note that ds_np is not zero-centered, because 
# Trace(Q C) < 0 when including auto terms
ds_def = 5.46239
ds_npw = 3.43332
ds_np = 0.51654

# Form all the filters
C, L, Q_def, Q_np, Q_npw = B_to_filters(Bmat, Bidx)
II = np.identity(len(C))

# Create the CDF curves
ds_def_h0, ds_def_cdf_h0, x, _ = get_gx2_cdf(II, Q_def)
ds_npw_h0, ds_npw_cdf_h0, x, _ = get_gx2_cdf(II, Q_npw)
ds_np_h0,  ds_np_cdf_h0,  x, _ = get_gx2_cdf(II, Q_np)
ds_def_h1, ds_def_cdf_h1, x, _ = get_gx2_cdf(L, Q_def)
ds_npw_h1, ds_npw_cdf_h1, x, _ = get_gx2_cdf(L, Q_npw)
ds_np_h1,  ds_np_cdf_h1,  x, _ = get_gx2_cdf(L, Q_np)

# Get the AUC values
auc_def = get_auc(ds_def_cdf_h0, ds_def_cdf_h1)
auc_npw = get_auc(ds_npw_cdf_h0, ds_npw_cdf_h1)
auc_np  = get_auc(ds_np_cdf_h0, ds_np_cdf_h1)

auc_def, auc_npw, auc_np

# Create Figure 1
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4.5))

# Plot an invisible dummy line to get colors as below
ax.plot([], [], ' ', label="Area under curve:")
ax.plot(1-ds_np_cdf_h0, 1-ds_np_cdf_h1, lw=2, color='0.4',  label=f'NP AUC = {auc_np:.3f}')
ax.plot(1-ds_npw_cdf_h0, 1-ds_npw_cdf_h1, lw=2, color='0',  label=f'NPMV AUC = {auc_npw:.3f}')
ax.plot(1-ds_def_cdf_h0, 1-ds_def_cdf_h1, lw=2, color='0.75', label=f'DFCC AUC = {auc_def:.3f}')
# Diagonal “chance” line
ax.plot([0,1], [0,1],lw=1, color='0', ls='--', label= 'Chance AUC = 0.5')

ax.set_xlabel("False-alarm probability (FAP)")
ax.set_ylabel("Detection probability")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(loc='lower right')
ax.grid(True)

def smart_formatter(x, pos):
    if abs(x) < 1e-10:  # Handle floating-point zero safely
        return '0'
    return f'{x:g}'  # Default format: removes trailing zeros and decimals

ax.xaxis.set_major_formatter(mticker.FuncFormatter(smart_formatter))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(smart_formatter))

fig.savefig("fig-ng15-roc-linearscale.pdf", dpi=300, bbox_inches='tight')

###################################################

# Create Figure 2
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4.5))

dp0 = 2.87e-7
dp_def = dp_at_fap(ds_def_cdf_h0, ds_def_cdf_h1, dp0, use_loglog=True)
dp_npw = dp_at_fap(ds_npw_cdf_h0, ds_npw_cdf_h1, dp0, use_loglog=True)
dp_np  = dp_at_fap(ds_np_cdf_h0,  ds_np_cdf_h1,  dp0, use_loglog=True)

print('DP (npw) =', dp_npw, 'DP (def) =', dp_def, 'DP improvement =', 100*(dp_npw-dp_def)/dp_def, 'percent')

# Plot an invisible dummy line to add the custom header in the legend
ax.plot([], [], ' ', label='At ``$5$-$\sigma$" FAP of $2.9 \\times 10^{-7}$:')
ax.plot(1-ds_np_cdf_h0, 1-ds_np_cdf_h1, lw=2, color='0.4', label=f'NP detection probability {100*dp_np:.1f}\%')
ax.plot(1-ds_npw_cdf_h0, 1-ds_npw_cdf_h1, lw=2, color='0', label=f'NPMV detection probability {100*dp_npw:.1f}\%')
ax.plot(1-ds_def_cdf_h0, 1-ds_def_cdf_h1, lw=2, color='0.75', label=f'DFCC detection probability {100*dp_def:.1f}\%')

ax.set_xlabel("False-alarm probability (FAP)")
ax.set_ylabel("Detection probability")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(2.87e-7, 1)
ax.set_ylim(2e-2, 1)
ax.legend(loc='lower right')
ax.grid(True)


fig.savefig("fig-ng15-roc-logscale.pdf", dpi=300, bbox_inches='tight')

###################################################
# Create Figure 4
with open('genx2-figure-data.json', 'r') as fp:
    fig4_data = json.load(fp)

# Could also re-generate the h0 and h1 CDFs, but this is faster
# The filter matrices are in the dictionary, and are under keys:
# 'Ddef', 'Dnpw', 'Dnp', 'Dnpcc'. Just call the get_gx2_cdf function
ds_def_cdf_h0 = np.array(fig4_data['ds_def_cdf_h0']) 
ds_def_cdf_h1 = np.array(fig4_data['ds_def_cdf_h1'])
ds_npw_cdf_h0 = np.array(fig4_data['ds_npw_cdf_h0'])
ds_npw_cdf_h1 = np.array(fig4_data['ds_npw_cdf_h1'])
ds_np_cdf_h0 =  np.array(fig4_data['ds_np_cdf_h0'])
ds_np_cdf_h1 =  np.array(fig4_data['ds_np_cdf_h1'])
ds_npcc_cdf_h0 = np.array(fig4_data['ds_npcc_cdf_h0'])
ds_npcc_cdf_h1 = np.array(fig4_data['ds_npcc_cdf_h1'])

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4.5))

dp0 = 2.87e-7
dp_def  = dp_at_fap(ds_def_cdf_h0, ds_def_cdf_h1, dp0, use_loglog=True)
dp_npw  = dp_at_fap(ds_npw_cdf_h0, ds_npw_cdf_h1, dp0, use_loglog=True)
dp_np   = dp_at_fap(ds_np_cdf_h0,  ds_np_cdf_h1,  dp0, use_loglog=True)
dp_npcc = dp_at_fap(ds_npcc_cdf_h0,  ds_npcc_cdf_h1,  dp0, use_loglog=True)

#print('DP (npw) =', dp_npw, 'DP (def) =', dp_def, 'DP improvement =', 100*(dp_npw-dp_def)/dp_def, 'percent')

# Plot an invisible dummy line to add the custom header in the legend
ax.plot([], [], ' ', label='At ``$5$-$\sigma$" FAP of $2.9 \\times 10^{-7}$:')
ax.plot(1-ds_np_cdf_h0, 1-ds_np_cdf_h1, lw=2, color='0.4', label=f'NP detection probability {100*dp_np:.1f}\%')
ax.plot(1-ds_npcc_cdf_h0, 1-ds_npcc_cdf_h1, lw=2, color='0.8', label=f'NPCC detection probability {100*dp_npcc:.1f}\%')
ax.plot(1-ds_npw_cdf_h0, 1-ds_npw_cdf_h1, lw=2, color='0', ls=':', label=f'NPMV detection probability {100*dp_npw:.1f}\%')
ax.plot(1-ds_def_cdf_h0, 1-ds_def_cdf_h1, lw=2, color='0.75', label=f'DFCC detection probability {100*dp_def:.1f}\%')

ax.set_xlabel("False-alarm probability (FAP)")
ax.set_ylabel("Detection probability")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(2.87e-7, 1)
ax.set_ylim(9e-2, 1)
ax.legend(loc='lower right')
ax.grid(True)

fig.savefig("fig-toy-roc-logscale.pdf", dpi=300, bbox_inches='tight')


###############################################################
# Create Figure 5
# Get the AUC values
auc_def = get_auc(ds_def_cdf_h0, ds_def_cdf_h1)
auc_npw = get_auc(ds_npw_cdf_h0, ds_npw_cdf_h1)
auc_npcc = get_auc(ds_npcc_cdf_h0, ds_npcc_cdf_h1)
auc_np  = get_auc(ds_np_cdf_h0, ds_np_cdf_h1)

auc_def, auc_npw, auc_npcc, auc_np

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4.5))

# Plot an invisible dummy line to get colors as below
ax.plot([], [], ' ', label="Area under curve:")
ax.plot(1-ds_np_cdf_h0, 1-ds_np_cdf_h1, lw=2, color='0.4',  label=f'NP AUC = {auc_np:.3f}')
ax.plot(1-ds_npcc_cdf_h0, 1-ds_npcc_cdf_h1, lw=2, color='0.8',  label=f'NPCC AUC = {auc_npcc:.3f}')
ax.plot(1-ds_npw_cdf_h0, 1-ds_npw_cdf_h1, lw=2, ls=':', color='0',  label=f'NPMV AUC = {auc_npw:.3f}')
ax.plot(1-ds_def_cdf_h0, 1-ds_def_cdf_h1, lw=2, color='0.75', label=f'DFCC AUC = {auc_def:.3f}')
# Diagonal “chance” line
ax.plot([0,1], [0,1],lw=1, color='0', ls='--', label= 'Chance AUC = 0.5')

ax.set_xlabel("False-alarm probability (FAP)")
ax.set_ylabel("Detection probability")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(loc='lower right')
ax.grid(True)

def smart_formatter(x, pos):
    if abs(x) < 1e-10:  # Handle floating-point zero safely
        return '0'
    return f'{x:g}'  # Default format: removes trailing zeros and decimals

ax.xaxis.set_major_formatter(mticker.FuncFormatter(smart_formatter))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(smart_formatter))

fig.savefig("fig-toy-roc-linearscale.pdf", dpi=300, bbox_inches='tight')


