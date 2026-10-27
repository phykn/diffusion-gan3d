#!/bin/zsh
# Double-click in Finder: SR training from the newest run/*-anode-lr -> HR check.
# Run train_anode_1st.command first; the SR model goes to the same run's -sr folder.
# For a short trial from a terminal: SR_STEPS=200 commands/train_anode_2nd.command

cd "${0:A:h:h}" || exit 1
PY=.venv/bin/python

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
# Run names start with a timestamp, so reverse name order is newest first.
lr=(run/*-anode-lr(N/On[1]))
[[ -n $lr ]] || finish 1 "No run/*-anode-lr. Run train_anode_1st.command first."
[[ -f $lr/.done ]] || finish 1 "$lr is still training or was interrupted."
RUN=${lr%-lr}
[[ -f $RUN-sr/.done ]] && finish 1 "$RUN-sr is already trained. Run train_anode_1st.command for a new run."
step "Train $RUN-sr from $RUN-lr" caffeinate -i $PY run_train.py simul_anode \
  --run-dir $RUN ${SR_STEPS:+--sr-steps=$SR_STEPS}
step "Check SR generation" $PY scripts/06_check_hr.py --weight $RUN-sr --no-view

finish 0 Done "SR model" $RUN-sr Checks checks/
