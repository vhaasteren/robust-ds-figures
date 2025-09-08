#! /bin/bash
python ./build_npcc_roc_json.py \
  --root ./fapruns \
  --npcc-out  ./npcc-figure-data.json \
  --fig-out genx2-figure-data.json \
  --grid-log 1e-8 0.9999999 1200 \
  --grid-lin 1e-9 1.0 1200 \
  --x-min -24 --x-max 24 --nx 800
