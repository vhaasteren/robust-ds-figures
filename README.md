# robust-ds-figures

Code and data to **ROC figures** in the paper *“Optimal robust detection statistics for pulsar timing arrays.”*
The repository produces ROC curves and related diagnostics for the baseline filters **DFCC** (“optimal” cross-correlation), **NP** (full Neyman–Pearson), **NPMV** (off-diagonal NP; our recommended statistic), and the **NPCC** envelope built from per-FAP optimizations. We use the NG15yr pulsar subset and a Hellings–Downs correlation model.

> TL;DR pipeline (copy–paste):
>
> ```bash
> # 1) Run the NPCC optimization sweep over many FAP targets
> bash ./run_fap_sweep.sh
>
> # 2) Save the FAP=1 NPMV results
> python3 ./save_npmv_fap1_results.py
>
> # 3) Merge all runs and build the NPCC envelope + figure JSON
> bash ./build_roc.sh
>
> # 4) Generate all figures (PDFs written to the repo root)
> python3 ./npw-statistic-figures.py
> ```

---

## Contents

* **`optimize-filter.py`** – Main optimizer (FULL mode and others) for quadratic decision filters. Supports multiple optimizers (BOBYQA, subspace-BOBYQA, NES, SPSA, ISRES→BOBYQA) and CDF backends (analytic / Imhof). The FAP target is set per run; results are written under an `--outdir` with matrices and metadata.
* **`run_fap_sweep.sh`** – Drives a **sweep over many FAP values**, running `optimize-filter.py` once per FAP and writing results to `./fapruns/fap_*`. These runs are the inputs for the NPCC envelope.
* **`build_npcc_roc_json.py`** – Reads all `./fapruns/fap_*` outputs, computes CDFs via Imhof, converts to ROC space, **interpolates and envelopes** the curves to build NPCC, and writes:

  * `npcc-figure-data.json` – NPCC envelope + per-run diagnostics (FAP grid, DP, winner, sources).
  * `genx2-figure-data.json` – Full set of **figure-ready** arrays: baseline (DFCC/NPMV/NP) CDFs under H0/H1 and **NPCC CDFs**. Also includes the normalized filter matrices for reference.
* **`build_roc.sh`** – Convenience wrapper that calls `build_npcc_roc_json.py` with sensible grids and paths (`--root ./fapruns`).
* **`npw-statistic-figures.py`** – Creates and saves all PDF figures used in the paper.

  * Figures based on the **baseline filters** (DFCC/NPMV/NP) are computed directly.
  * **NPCC** curves are read from `genx2-figure-data.json`.
  * Outputs:

    * `fig-ng15-roc-linearscale.pdf`
    * `fig-ng15-roc-logscale.pdf`
    * `fig-toy-roc-logscale.pdf`
    * `fig-toy-roc-linearscale.pdf`
* **`save_npmv_fap1_results.py`** – Optional helper that saves **NPMV artifacts at FAP=1** into `./fapruns/fap_1/` (handy to anchor the envelope at the right edge). Not required for the main pipeline.
* **`run_full_npcc_search.py`** – Advanced driver for global-ish searches with alternative seeds and annealing schedules. **Not required** for the reproducibility pipeline below, but can be used to confirm convergence when starting from different points in parameter space.
* **`Bmatrix.npy.gz`**, **`Bmatrix-indices.json`** – Filter matrices for creating the Real NG 15-year ROC curves (`B` is `QDF` of Eq. 54); used by `npw-statistic-figures.py`.
* **`genx2-figure-data.json`** – (Generated) Figure data produced by `build_roc.sh` / `build_npcc_roc_json.py`. Overwritten when you run step 2.
* **`LICENSE`** – MIT License.
* **`README.md`** – This file.

---

## Requirements

* **Python**: 3.9–3.12
* **Python packages**:

  * `numpy`, `scipy`, `matplotlib`, `nlopt`, `json`, `gzip`
* **TeX (optional but recommended)**:

  * Figures use `text.usetex=True`. If LaTeX is missing, either install TeX Live/MacTeX or set `text.usetex=False` in `npw-statistic-figures.py`.

Install with pip (example):

```bash
python3 -m pip install numpy scipy matplotlib nlopt
```

> Note: On some systems `nlopt` may require system headers. If `pip install nlopt` fails, install a system package (e.g., `apt-get install libnlopt-dev`) and retry.

---

## Recreate all figures (step-by-step)

The workflow is intentionally linear and fully scripted.

### 1) Run the FAP sweep (NPCC optimizations)

This step runs the FULL problem at a set of target FAPs and writes one folder per FAP under `./fapruns/`.

```bash
bash ./run_fap_sweep.sh
```

* Uses SPSA→BOBYQA with a conservative schedule and a fixed RNG seed for stable, repeatable outcomes.
* Outputs like:

  ```
  fapruns/
    fap_1e-8/        D_star.npy  result.json  ...
    fap_3e-8/        ...
    ...
    fap_3e-1/        ...
  ```

*Optional:* If you also want a placeholder at **FAP=1** (NPMV normalized under the N-inner product), run:

```bash
python3 ./save_npmv_fap1_results.py
```

This creates `fapruns/fap_1/` and can slightly improve the right-edge envelope, but it is **not required**.

### 2) Build the NPCC envelope and figure JSON

Merge all runs, compute CDFs, convert to ROC, **envelope** across curves, and write JSONs needed by the figure code.

```bash
bash ./build_roc.sh
```

This writes:

* `npcc-figure-data.json` – diagnostics and per-FAP curves
* `genx2-figure-data.json` – arrays consumed by the figure script (baseline + NPCC)

### 3) Generate the figures

Create all figures exactly as used in the paper.

```bash
python3 ./npw-statistic-figures.py
```

You should see these PDFs in the repo root:

* `fig-ng15-roc-linearscale.pdf`
* `fig-ng15-roc-logscale.pdf`
* `fig-toy-roc-logscale.pdf`
* `fig-toy-roc-linearscale.pdf`

---

## Notes on the statistics

* **DFCC**: Traditional “optimal” cross-correlation statistic (baseline in PTA literature).
* **NPMV**: Off-diagonal NP filter (our recommended robust statistic).
* **NP**: Full Neyman–Pearson filter (includes auto terms; shown for reference).
* **NPCC**: Envelope built from filters optimized separately **at each FAP** and then enveloped in ROC space. Shown for completeness; **we do not recommend NPCC** for standard analysis, but include it to compare achievable ROC envelopes under per-FAP optimization.

---

## Tips & troubleshooting

* **LaTeX**: If you don’t have a TeX installation, open `npw-statistic-figures.py` and set:

  ```python
  "text.usetex": False
  ```

  The PDFs will still be produced (fonts differ slightly).
* **Non-determinism**: The sweep uses fixed seeds and a stable schedule, but small numerical differences across platforms/BLAS/NLopt builds can slightly change DP at the ≥1e-3 level. The envelope construction (sorting, de-duplication, monotonicity) makes the final curves robust.
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
python3 ./npw-statistic-figures.py
```

---

## License

MIT — see `LICENSE`.
