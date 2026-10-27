#!/bin/zsh
# Double-click in Finder: simulated anode data -> LR training -> LR and anchor checks.
# Each run gets a new folder: run/<YYYYMMDD-HHMM>-anode-lr; train_anode_2nd.command
# then trains its SR model. For a short trial: STEPS=200 scripts/commands/train_anode_1st.command

cd "${0:A:h:h:h}" || exit 1
source scripts/commands/lib.zsh
RUN=run/$(date +%Y%m%d-%H%M)-anode

if [[ ! -d data/generated/simul_anode/volumes ]]; then
  step "Generate simulated anode data" $PY scripts/simul/run.py --config scripts/simul/config_anode.yaml
fi
step "Check training crops" $PY scripts/check/01_dataset.py --sample simul_anode --no-view
step "Train $RUN-lr" caffeinate -i $PY train.py simul_anode --lr-only \
  --run-dir $RUN ${STEPS:+--steps=$STEPS}
step "Check LR generation (top half)" $PY scripts/check/02_generated.py --weight $RUN-lr --no-view
step "Check LR generation (bottom half)" $PY scripts/check/02_generated.py --weight $RUN-lr \
  --height-origin 128 --no-view
step "Check anchors" $PY scripts/check/03_anchor.py --weight $RUN-lr --no-view
step "Check particle tilt vs real" $PY scripts/check/07_tilt.py --weight $RUN-lr \
  --real data/generated/simul_anode/volumes/*.tiff --real-stride 2

finish 0 Done "LR model" $RUN-lr Checks checks/ Next train_anode_2nd.command
