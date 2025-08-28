#!/usr/bin/env bash
set -euo pipefail

# Optional: pass a custom Python path or script path
PYTHON_BIN="${PYTHON_BIN:-python}"
OPT_SCRIPT="${1:-./optimize-filter.py}"

# Output root
OUTROOT="fapruns"
mkdir -p "${OUTROOT}"

# FAPs to sweep (as strings, so we can preserve scientific notation)
FAPS=(
  "1e-8"
  "3e-8"
  "1e-7"
  "2.87e-7"
  "3e-7"
  "1e-6"
  "3e-6"
  "1e-5"
  "3e-5"
  "1e-4"
  "3e-4"
  "1e-3"
  "3e-3"
  "1e-2"
  "3e-2"
  "1e-1"
  "3e-1"
)

# Common flags (these are exactly the ones you said “simply work”)
COMMON=(
  --mode full
  --cdf analytic
  --bound 8.0
  --optimizer spsa
  --spsa-iters 3000
  --spsa-step 0.08
  --polish bobyqa
  --polish-evals 800
  --seed 123
)

echo "[info] Using script: ${OPT_SCRIPT}"
echo "[info] Output root:  ${OUTROOT}"
echo

for FAP in "${FAPS[@]}"; do
  # Tokenize FAP for folder name: 2.87e-7 -> 2p87e-7
  FTOK="${FAP//./p}"
  OUTDIR="${OUTROOT}/fap_${FTOK}"

  echo "[run] FAP=${FAP}  ->  ${OUTDIR}"
  mkdir -p "${OUTDIR}"

  # Your known-good invocation uses --fap (not --faprob), so we keep that.
  ${PYTHON_BIN} "${OPT_SCRIPT}" \
    "${COMMON[@]}" \
    --fap "${FAP}" \
    --outdir "${OUTDIR}"

  echo "[done] ${OUTDIR}"
  echo
done

echo "[all done] results in ${OUTROOT}/"
