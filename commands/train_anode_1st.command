#!/bin/zsh
# Double-click in Finder: simulated anode data -> LR training -> LR and anchor checks.
# Each run gets a new folder: run/<YYYYMMDD-HHMM>-anode-lr; train_anode_2nd.command
# then trains its SR model. For a short trial: STEPS=200 commands/train_anode_1st.command

cd "${0:A:h:h}" || exit 1
PY=.venv/bin/python
RUN=run/$(date +%Y%m%d-%H%M)-anode

# Colors only in a terminal; NO_COLOR turns them off.
if [[ -t 1 && -z $NO_COLOR ]]; then
  B=$'\e[1m' D=$'\e[2m' R=$'\e[31m' G=$'\e[32m' C=$'\e[36m' N=$'\e[0m'
fi
STEP=0

# finish CODE HEADLINE [LABEL VALUE]...: a green or red summary, then wait.
finish() {
  local code=$1 color=$G mark="✓"
  (( code )) && color=$R mark="✗"
  echo
  echo "${B}${color}${mark} $2${N}"
  shift 2
  while (( $# >= 2 )); do
    printf "  ${D}%-12s${N}%s\n" "$1" "$2"
    shift 2
  done
  echo
  read "?${D}Press Enter to close.${N}"
  exit "$code"
}

step() {
  local title=$1
  (( ++STEP ))
  echo
  echo "${B}${C}━━ Step $STEP · $title${N}"
  shift
  "$@" || finish 1 "Failed at step $STEP: $title" Command "$*"
}

[[ -x $PY ]] || finish 1 "Missing $PY. Create the project .venv first."
if [[ ! -d data/generated/simul_anode/volumes ]]; then
  step "Generate simulated anode data" $PY simul/run.py --config simul/config_anode.yaml
fi
step "Check training crops" $PY scripts/01_check_dataset.py --sample simul_anode --no-view
step "Train $RUN-lr" caffeinate -i $PY run_train.py simul_anode --lr-only \
  --run-dir $RUN ${STEPS:+--steps=$STEPS}
step "Check LR generation (top half)" $PY scripts/02_check_generated.py --weight $RUN-lr --no-view
step "Check LR generation (bottom half)" $PY scripts/02_check_generated.py --weight $RUN-lr \
  --height-origin 128 --no-view
step "Check anchors" $PY scripts/03_check_anchor.py --weight $RUN-lr --no-view
step "Check particle tilt vs real" $PY scripts/07_check_tilt.py --weight $RUN-lr \
  --real data/generated/simul_anode/volumes/*.tiff --real-stride 2

finish 0 Done "LR model" $RUN-lr Checks checks/ Next train_anode_2nd.command
