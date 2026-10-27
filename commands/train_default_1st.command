#!/bin/zsh
# Double-click in Finder: default sample -> LR training -> LR and anchor checks.
# Each run gets a new folder: run/<YYYYMMDD-HHMM>-default-lr; train_default_2nd.command
# then trains its SR model. For a short trial: STEPS=200 commands/train_default_1st.command

cd "${0:A:h:h}" || exit 1
source commands/lib.zsh
RUN=run/$(date +%Y%m%d-%H%M)-default

step "Check training crops" $PY scripts/01_check_dataset.py --sample default --no-view
step "Train $RUN-lr" caffeinate -i $PY run_train.py default --lr-only \
  --run-dir $RUN ${STEPS:+--steps=$STEPS}
step "Check LR generation" $PY scripts/02_check_generated.py --weight $RUN-lr --no-view
step "Check anchors" $PY scripts/03_check_anchor.py --weight $RUN-lr --no-view

finish 0 Done "LR model" $RUN-lr Checks checks/ Next train_default_2nd.command
