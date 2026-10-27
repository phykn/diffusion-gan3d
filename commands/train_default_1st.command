#!/bin/zsh
# Double-click in Finder: default sample -> LR training -> LR and anchor checks.
# Each run gets a new folder: run/<YYYYMMDD-HHMM>-default-lr; train_default_2nd.command
# then trains its SR model. For a short trial: STEPS=200 commands/train_default_1st.command

cd "${0:A:h:h}" || exit 1
PY=.venv/bin/python
RUN=run/$(date +%Y%m%d-%H%M)-default

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
step "Check training crops" $PY scripts/01_check_dataset.py --sample default --no-view
step "Train $RUN-lr" caffeinate -i $PY run_train.py default --lr-only \
  --run-dir $RUN ${STEPS:+--steps=$STEPS}
step "Check LR generation" $PY scripts/02_check_generated.py --weight $RUN-lr --no-view
step "Check anchors" $PY scripts/03_check_anchor.py --weight $RUN-lr --no-view

finish 0 Done "LR model" $RUN-lr Checks checks/ Next train_default_2nd.command
