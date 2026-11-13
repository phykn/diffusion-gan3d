# Shared by the .command files: source after cd to the project root.

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

new_run() {
  print -r -- "run/$(date +%Y%m%d-%H%M)-$1"
}

[[ -x $PY ]] || finish 1 "Missing $PY. Create the project .venv first."

check_lr() {
  local name=$1 run=$2
  if [[ $name == anode ]]; then
    step "Check LR generation (top half)" $PY scripts/check/02_generate.py --weight "$run-lr" --no-view
    step "Check LR generation (bottom half)" $PY scripts/check/02_generate.py --weight "$run-lr" \
      --height-origin 128 --no-view
  else
    step "Check LR generation" $PY scripts/check/02_generate.py --weight "$run-lr" --no-view
  fi
  step "Check anchors" $PY scripts/check/03_anchor.py --weight "$run-lr" --no-view
  if [[ $name == anode ]]; then
    local -a real
    real=(data/generated/simul_anode/volumes/*.tiff)
    step "Check particle tilt vs real (top half)" $PY scripts/check/05_tilt.py \
      --weight "$run-lr" --real "${real[@]}" --real-stride 2
    step "Check particle tilt vs real (bottom half)" $PY scripts/check/05_tilt.py \
      --weight "$run-lr" --real "${real[@]}" --real-stride 2 --height-origin 128
  fi
}

launch_training() {
  local name=$1 mode=$2 sample=$1 run lr
  local -a stage_flag extra_steps
  [[ $name == anode ]] && sample=simul_anode
  if [[ $mode == sr ]]; then
    local -a candidates
    candidates=(run/*-${name}-lr(N/On[1]))
    lr=${candidates[1]}
    [[ -n $lr ]] || finish 1 "No run/*-${name}-lr. Run train_${name}_1st.command first."
    [[ -f $lr/.done ]] || finish 1 "$lr is still training or was interrupted."
    run=${lr%-lr}
    [[ ! -f $run-sr/.done ]] || finish 1 "$run-sr is already trained. Run train_${name}_1st.command for a new run."
    step "Train $run-sr from $run-lr" caffeinate -i $PY train.py "$sample" \
      --run-dir "$run" ${SR_STEPS:+--sr-steps=$SR_STEPS}
  else
    run=$(new_run "$name")
    if [[ $name == anode && ! -d data/generated/simul_anode/volumes ]]; then
      step "Generate simulated anode data" $PY scripts/simul/run.py --config scripts/simul/config_anode.yaml
    fi
    step "Check training crops" $PY scripts/check/01_dataset.py --sample "$sample" --no-view
    local title="Train $run-lr"
    if [[ $mode == lr ]]; then
      stage_flag=(--lr-only)
    else
      title+=", then $run-sr"
      extra_steps=(${SR_STEPS:+--sr-steps=$SR_STEPS})
    fi
    step "$title" caffeinate -i $PY train.py "$sample" "${stage_flag[@]}" \
      --run-dir "$run" ${STEPS:+--steps=$STEPS} "${extra_steps[@]}"
    check_lr "$name" "$run"
    if [[ $mode == lr ]]; then
      finish 0 Done "LR model" "$run-lr" Checks checks/ Next "train_${name}_2nd.command"
    fi
  fi
  step "Check SR generation" $PY scripts/check/04_hr.py --weight "$run-sr" --no-view
  if [[ $mode == sr ]]; then
    finish 0 Done "SR model" "$run-sr" Checks checks/
  fi
  finish 0 Done "LR model" "$run-lr" "SR model" "$run-sr" Checks checks/
}
