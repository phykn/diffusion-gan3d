#!/bin/zsh
# Double-click in Finder: SR training from the newest run/*-anode-lr -> HR check.
# Run train_anode_1st.command first; the SR model goes to the same run's -sr folder.
# For a short trial from a terminal: SR_STEPS=200 commands/train_anode_2nd.command

cd "${0:A:h:h}" || exit 1
source commands/lib.zsh

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
