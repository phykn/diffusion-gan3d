#!/bin/zsh
# Double-click in Finder: default sample -> LR training -> LR and anchor checks.
# Each run gets a new folder: run/<YYYYMMDD-HHMM>-default-lr; train_default_2nd.command
# then trains its SR model. For a short trial: STEPS=200 scripts/commands/train_default_1st.command

cd "${0:A:h:h:h}" || exit 1
source scripts/commands/lib.zsh
RUN=run/$(date +%Y%m%d-%H%M)-default

step "Check training crops" $PY scripts/check/01_dataset.py --sample default --no-view
step "Train $RUN-lr" caffeinate -i $PY train.py default --lr-only \
  --run-dir $RUN ${STEPS:+--steps=$STEPS}
step "Check LR generation" $PY scripts/check/02_generated.py --weight $RUN-lr --no-view
step "Check anchors" $PY scripts/check/03_anchor.py --weight $RUN-lr --no-view

finish 0 Done "LR model" $RUN-lr Checks checks/ Next train_default_2nd.command
