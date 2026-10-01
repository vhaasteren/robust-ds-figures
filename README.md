# robust-ds-figures

Code and data to reproduce the **ROC figures** (Figs. 2 and 3) of the paper *“Optimal robust detection statistics for pulsar timing arrays,”* and the arbitrary-precision evaluation of the CDF described in its Section VI.
The repository produces ROC curves and related diagnostics for four quadratic detection statistics: **DFCC** (the literature-standard “optimal” cross-correlation statistic), **NP** (Neyman–Pearson), **NPMV** (Neyman–Pearson-Minimum-Variance; our recommended statistic), and **NPCC** (cross-correlation-only Neyman–Pearson, found numerically by per-FAP optimizations). We use the NG15yr pulsar subset and a Hellings–Downs correlation model.

**Notation** follows the paper: `D = z† Q z` is the detection statistic and `Q` its filter; `H_N` and `H_S` are the null (noise) and signal hypotheses, with covariance matrices `N` and `S`; **FAP** is the false-alarm probability and **DP** the detection probability. See [Notes on the statistics](#notes-on-the-statistics) for the definitions of the four statistics.

> TL;DR pipeline (copy–paste, from the repository root):
>
> ```bash
> # 1) Run the NPCC optimization sweep over many FAP targets
> bash ./run_fap_sweep.sh
> #    (optional) add the FAP=1 NPMV placeholder
> python3 ./save_npmv_fap1_results.py
>
> # 2) Merge all runs and build the NPCC ROC curve + figure JSON
> bash ./build_roc.sh
>
> # 3) Generate the ROC figures (PDFs written to the repo root)
> python3 ./npmv-statistic-figures.py
> ```
>
> The repository ships `genx2-figure-data.json`, so step 3 alone regenerates the figures. Steps 1 and 2 recompute the NPCC data from scratch.
>
> The arbitrary-precision CDF of Section VI is independent of this pipeline: `python3 ./compute_cdf.py` (see [below](#arbitrary-precision-cdf-section-vi)).

---

## Contents

* **`optimize-filter.py`** – Main optimizer (FULL mode and others) for the filter `Q` of a quadratic detection statistic. Maximizing the DP at fixed FAP over zero-diagonal filters gives the NPCC filter. Supports multiple optimizers (BOBYQA, subspace-BOBYQA, NES, SPSA, ISRES→BOBYQA) and CDF backends (analytic / Imhof). The FAP target is set per run; results are written under an `--outdir`: the filter matrices `Q_star.npy` and `Q_unscaled.npy`, and metadata in `result.json` (plus `x_opt.json` in FULL mode).
* **`run_fap_sweep.sh`** – Drives a **sweep over many FAP values**, running `optimize-filter.py` once per FAP (17 values from 1e-8 to 3e-1) and writing results to `./fapruns/fap_*`. These runs are the inputs for the NPCC ROC curve.
* **`build_npcc_roc_json.py`** – Reads all `./fapruns/fap_*` outputs, computes CDFs via Imhof, converts to ROC space, **interpolates and envelopes** the curves to build the NPCC ROC curve, and writes:

  * `npcc-figure-data.json` – NPCC envelope + per-run diagnostics (FAP grid, DP, winner, sources). Not tracked in the repository.
  * `genx2-figure-data.json` – Full set of **figure-ready** arrays: baseline (DFCC/NPMV/NP) CDFs under H_N/H_S and **NPCC CDFs**. Also includes the normalized filter matrices for reference.
* **`build_roc.sh`** – Convenience wrapper that calls `build_npcc_roc_json.py` with sensible grids and paths (`--root ./fapruns`).
* **`npmv-statistic-figures.py`** – Creates and saves the ROC figures of the paper (Figs. 2 and 3). The other figures of the paper are not produced by this repository.

  * The curves for the NANOGrav 15-year model (DFCC/NPMV/NP) are computed directly from `Bmatrix.npy.gz`.
  * The toy-model curves (DFCC/NPMV/NP and **NPCC**) are read from `genx2-figure-data.json`.
  * Outputs:

    * `fig-toy-roc-linearscale.pdf` – toy model, Fig. 2 (left panel)
    * `fig-toy-roc-logscale.pdf` – toy model, Fig. 2 (right panel)
    * `fig-ng15-roc-logscale.pdf` – NANOGrav 15-year model, Fig. 3
    * `fig-ng15-roc-linearscale.pdf` – NANOGrav 15-year model on linear axes (not shown in the paper)
* **`save_npmv_fap1_results.py`** – Optional helper that saves **NPMV artifacts at FAP=1** into `./fapruns/fap_1/` (handy to anchor the envelope at the right edge). Not required for the main pipeline.
* **`run_full_npcc_search.py`** – Advanced driver for global-ish searches with alternative seeds and annealing schedules. **Not required** for the reproducibility pipeline below, but can be used to confirm convergence when starting from different points in parameter space.
* **`compute_cdf.py`** – Arbitrary-precision evaluation of the CDF \(F(\tau,\mathrm{C},\mathrm{Q})\) for complex-valued data in Section VI. Given the eigenvalues of \(\mathrm{C}^{1/2}\mathrm{Q}\,\mathrm{C}^{1/2}\), it estimates the mantissa bits required and evaluates the sum in double precision or with `mpmath`. This is the implementation described in the numerical-evaluation subsection. Importable as a module (`cdf`, `ccdf`, plus helpers for the filters, ROC curves, and detection probabilities), or run stand-alone to print the detection probabilities of NP, NPMV, and DF (equal to DFCC for the CURN null hypothesis used here) at the 5σ FAP: `python3 compute_cdf.py [--cases 30 67 ...]`.
* **`ComputeCDF.ipynb`** – Thin notebook that calls `compute_cdf.py`: precision test, CDF plots with the mantissa bits needed, ROC curves, and detection probabilities at the 5σ FAP for each set of pulsars.
* **`ComputeF.ipynb`** – Earlier fixed-decimal-precision version of the same CDF, also using `mpmath`. Self-contained (it does not use `compute_cdf.py`); it reads `ng-ra-dec.txt`, so run it from the repository root.
* **`ng-ra-dec.txt`**, **`ppta-ra-dec.txt`**, **`five_PTA.txt`**, **`Random150.txt`**, **`Random200.txt`**, **`Random250.txt`**, **`Random300.txt`** – Pulsar positions used by `compute_cdf.py` (30, 67, 121, 150, 200, 250, and 300 pulsars).
* **`Bmatrix.npy.gz`**, **`Bmatrix-indices.json`** – Filter matrix for the ROC curves of the more realistic NANOGrav 15-year model, and the per-pulsar block indices. `B` is the whitened deflection filter \(\tilde{\mathrm{Q}}_{\rm DF} = \mathrm{A} - \mathrm{I}\) of the section “Relating DF and NP filters for realistic PTA data”; used by `npmv-statistic-figures.py`.
* **`genx2-figure-data.json`** – (Generated; a copy is included) Figure data produced by `build_roc.sh` / `build_npcc_roc_json.py`, read by `npmv-statistic-figures.py`. Overwritten when you run step 2.
* **`fapruns/`** – (Generated, not tracked) Output of the FAP sweep, one folder per FAP.
* **`LICENSE`** – MIT License.
* **`README.md`** – This file.

---

## Requirements

* **Python**: 3.9 or newer (tested with 3.12)
* **Python packages**:

  * `numpy`, `scipy`, `matplotlib` – all scripts. Use `numpy < 2.4`: `npmv-statistic-figures.py` calls `np.trapz`, which was removed in NumPy 2.4.
  * `nlopt` – only for `optimize-filter.py` (step 1).
  * `matplotlib-inline` – only for `npmv-statistic-figures.py`, which selects the Jupyter inline backend. It is installed together with Jupyter/IPython.
  * `mpmath` – only for `compute_cdf.py`, `ComputeCDF.ipynb`, and `ComputeF.ipynb` (much faster with the optional `gmpy2` backend).
  * `jupyter` – only to run the two notebooks.
* **TeX (optional but recommended)**:

  * Figures use `text.usetex=True`, which needs a LaTeX installation that includes the `type1ec.sty` package (part of `cm-super`). If LaTeX is missing, either install TeX Live/MacTeX or set `text.usetex=False` in `npmv-statistic-figures.py`.

Install with pip (example):

```bash
python3 -m pip install "numpy<2.4" scipy matplotlib matplotlib-inline nlopt mpmath
```

> Note: On some systems `nlopt` may require system headers. If `pip install nlopt` fails, install a system package (e.g., `apt-get install libnlopt-dev`) and retry.

---

## Recreate the ROC figures (step-by-step)

The workflow is intentionally linear and fully scripted. Run all commands from the repository root: the scripts use relative paths.

### 1) Run the FAP sweep (NPCC optimizations)

This step runs the FULL problem at a set of target FAPs and writes one folder per FAP under `./fapruns/`.

```bash
bash ./run_fap_sweep.sh
```

* Uses SPSA→BOBYQA with a conservative schedule and a fixed RNG seed for stable, repeatable outcomes.
* Outputs like:

  ```
  fapruns/
    fap_1e-8/        Q_star.npy  result.json  ...
    fap_3e-8/        ...
    ...
    fap_3e-1/        ...
  ```

  The folder name is the FAP with `.` replaced by `p` (for example `fap_2p87e-7`). Runs written by earlier versions of the code contain `D_star.npy` instead of `Q_star.npy`; `build_npcc_roc_json.py` reads both.
* The script calls `python`. To use another interpreter: `PYTHON_BIN=python3 bash ./run_fap_sweep.sh`.

*Optional:* If you also want a placeholder at **FAP=1** (NPMV normalized under the N-inner product), run:

```bash
python3 ./save_npmv_fap1_results.py
```

This creates `fapruns/fap_1/` and can slightly improve the right-edge envelope, but it is **not required**.

### 2) Build the NPCC ROC curve and figure JSON

Merge all runs, compute CDFs, convert to ROC, **envelope** across curves, and write JSONs needed by the figure code.

```bash
bash ./build_roc.sh
```

This writes:

* `npcc-figure-data.json` – diagnostics and per-FAP curves
* `genx2-figure-data.json` – arrays consumed by the figure script (baseline + NPCC)

`build_roc.sh` calls `python` and needs neither `nlopt` nor TeX.

### 3) Generate the figures

Create the ROC figures exactly as used in the paper (Figs. 2 and 3).

```bash
python3 ./npmv-statistic-figures.py
```

You should see these PDFs in the repo root:

* `fig-ng15-roc-linearscale.pdf`
* `fig-ng15-roc-logscale.pdf`
* `fig-toy-roc-logscale.pdf`
* `fig-toy-roc-linearscale.pdf`

The script also prints the detection probabilities of NPMV and DFCC at the 5σ FAP for the NANOGrav 15-year model, and the relative improvement.

---

## Arbitrary-precision CDF (Section VI)

`compute_cdf.py` is independent of the ROC pipeline above. It needs `numpy`, `matplotlib`, and `mpmath`, and reads the pulsar-position files next to it, so it can be run from any directory.

```bash
# detection probabilities of NP, NPMV and DF at the 5σ FAP, for all sets of pulsars
python3 ./compute_cdf.py

# selected sets of pulsars (by number of pulsars), another FAP
python3 ./compute_cdf.py --cases 30 67 --fap 1e-4
```

As a module:

```python
import compute_cdf as cc
evals = cc.make_eigenvalue_dict(cc.PULSAR_FILES['67'])    # eigenvalues for NP, NPMV, DF under H_N and H_S
F, bits = cc.cdf(-10.0, evals['EVNnpmv'], accuracy=1e-12)  # CDF value and mantissa bits used
```

`ComputeCDF.ipynb` calls the same functions and adds the precision test and the plots (CDF with mantissa bits, ROC curves). Without `gmpy2` the notebook takes tens of minutes, because the plots evaluate the CDF at many thresholds for up to 300 pulsars.

---

## Notes on the statistics

* **DFCC**: Cross-correlation deflection statistic: the traditional “optimal” cross-correlation statistic of the PTA literature. It maximizes the deflection (signal-to-noise ratio), not the detection probability. For the CURN null hypothesis used here it coincides with the general deflection statistic **DF**; the code abbreviates it as `def`.
* **NP**: Neyman–Pearson statistic. It maximizes the detection probability at fixed FAP, but uses autocorrelations and is therefore not robust; shown for reference.
* **NPMV**: Neyman–Pearson-Minimum-Variance statistic (our recommended robust statistic). The quadratic statistic that is as close as possible to NP (minimum variance of the difference under `H_N`) while using only cross-correlations. For a block-diagonal `N`, as in all models here, its filter is the NP filter with the autocorrelation (diagonal) blocks set to zero.
* **NPCC**: Cross-correlation-only Neyman–Pearson statistic: the quadratic statistic that maximizes the detection probability at fixed FAP while using only cross-correlations. Its filter depends on the FAP and is only known numerically, so we optimize it separately **at each FAP** and take the envelope of the resulting curves in ROC space. It performs only modestly better than NPMV and is cumbersome to use; it is shown for comparison.

In performance the ranking is NP, NPCC, NPMV, DFCC.

---

## Tips & troubleshooting

* **LaTeX**: If you don’t have a TeX installation, open `npmv-statistic-figures.py` and set:

  ```python
  "text.usetex": False
  ```

  The PDFs will still be produced (fonts differ slightly).
* **Non-determinism**: The sweep uses fixed seeds and a stable schedule, but small numerical differences across platforms/BLAS/NLopt builds can slightly change the detection probability at the ≥1e-3 level. The envelope construction (sorting, de-duplication, monotonicity) makes the final curves robust.
* **Starting from other seeds**: If you want to stress-test convergence, use:

  ```bash
  python3 ./run_full_npcc_search.py --npsrs 67 --cdf analytic --faprob 2.87e-7 \
    --outroot ./global_search_67 --jobs 4
  ```

  This is **optional** and not part of the main reproducibility pipeline.

---

## Clean re-run

To rebuild from scratch:

```bash
rm -rf ./fapruns npcc-figure-data.json genx2-figure-data.json \
       fig-ng15-roc-*.pdf fig-toy-roc-*.pdf
bash ./run_fap_sweep.sh
python3 ./save_npmv_fap1_results.py
bash ./build_roc.sh
python3 ./npmv-statistic-figures.py
```

---

## License

MIT — see `LICENSE`.
