#!/bin/zsh
cd "${0:A:h:h:h}" || exit 1
source scripts/commands/lib.zsh
launch_training default sr
