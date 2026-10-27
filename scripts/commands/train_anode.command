#!/bin/zsh
# Double-click in Finder: simulated anode data -> LR + SR training -> checks.
# Each run gets a new folder: run/<YYYYMMDD-HHMM>-anode-lr and -sr.
# For a short trial from a terminal: STEPS=200 SR_STEPS=200 scripts/commands/train_anode.command

cd "${0:A:h:h:h}" || exit 1
source scripts/commands/lib.zsh
RUN=run/$(date +%Y%m%d-%H%M)-anode

if [[ ! -d data/generated/simul_anode/volumes ]]; then
  step "Generate simulated anode data" $PY scripts/simul/run.py --config scripts/simul/config_anode.yaml
fi
step "Check training crops" $PY scripts/check/01_dataset.py --sample simul_anode --no-view
step "Train $RUN-lr, then $RUN-sr" caffeinate -i $PY train.py simul_anode \
  --run-dir $RUN ${STEPS:+--steps=$STEPS} ${SR_STEPS:+--sr-steps=$SR_STEPS}
step "Check LR generation (top half)" $PY scripts/check/02_generated.py --weight $RUN-lr --no-view
step "Check LR generation (bottom half)" $PY scripts/check/02_generated.py --weight $RUN-lr \
  --height-origin 128 --no-view
step "Check anchors" $PY scripts/check/03_anchor.py --weight $RUN-lr --no-view
step "Check particle tilt vs real" $PY scripts/check/07_tilt.py --weight $RUN-lr \
  --real data/generated/simul_anode/volumes/*.tiff --real-stride 2
step "Check SR generation" $PY scripts/check/06_hr.py --weight $RUN-sr --no-view

finish 0 Done "LR model" $RUN-lr "SR model" $RUN-sr Checks checks/
