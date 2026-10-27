#!/bin/zsh
# Double-click in Finder: default sample -> LR + SR training -> checks.
# Each run gets a new folder: run/<YYYYMMDD-HHMM>-default-lr and -sr.
# For a short trial from a terminal: STEPS=200 SR_STEPS=200 commands/train_default.command

cd "${0:A:h:h}" || exit 1
source commands/lib.zsh
RUN=run/$(date +%Y%m%d-%H%M)-default

step "Check training crops" $PY scripts/check/01_dataset.py --sample default --no-view
step "Train $RUN-lr, then $RUN-sr" caffeinate -i $PY run_train.py default \
  --run-dir $RUN ${STEPS:+--steps=$STEPS} ${SR_STEPS:+--sr-steps=$SR_STEPS}
step "Check LR generation" $PY scripts/check/02_generated.py --weight $RUN-lr --no-view
step "Check anchors" $PY scripts/check/03_anchor.py --weight $RUN-lr --no-view
step "Check SR generation" $PY scripts/check/06_hr.py --weight $RUN-sr --no-view

finish 0 Done "LR model" $RUN-lr "SR model" $RUN-sr Checks checks/
