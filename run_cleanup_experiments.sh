#!/bin/bash
set -euo pipefail

# Simple launcher for CleanUp experiments (C0–C4).
PROJECT="socialjax-cleanup"
ENTITY="${ENTITY:-""}"
TOTAL_STEPS="${TOTAL_STEPS:-200000}"

run_one() {
  local COND="$1"
  local SEED="$2"

  case "$COND" in
    C0)
      python algorithms/IPPO/ippo_cnn_cleanup.py \
        PROJECT=$PROJECT ENTITY="$ENTITY" TOTAL_TIMESTEPS=$TOTAL_STEPS SEED=$SEED \
        ENV_KWARGS.shared_rewards=False REWARD=individual \
        CONDITION=C0 MECHANISM_CLASS=Baseline RUN_NAME=C0_seed${SEED}
      ;;
    C1)
      python algorithms/IPPO/ippo_cnn_cleanup.py \
        PROJECT=$PROJECT ENTITY="$ENTITY" TOTAL_TIMESTEPS=$TOTAL_STEPS SEED=$SEED \
        ENV_KWARGS.shared_rewards=True REWARD=common \
        CONDITION=C1 MECHANISM_CLASS=ClassI RUN_NAME=C1_seed${SEED}
      ;;
    C2)
      python algorithms/SVO/svo_cnn_cleanup.py \
        PROJECT=$PROJECT ENTITY="$ENTITY" TOTAL_TIMESTEPS=$TOTAL_STEPS SEED=$SEED \
        CONDITION=C2 MECHANISM_CLASS=ClassI RUN_NAME=C2_seed${SEED} \
        ENV_KWARGS.svo_ideal_angle_degrees=90 ENV_KWARGS.svo_w=0.5
      ;;
    C3)
      python algorithms/MAPPO/mappo_cnn_cleanup.py \
        PROJECT=$PROJECT ENTITY="$ENTITY" TOTAL_TIMESTEPS=$TOTAL_STEPS SEED=$SEED \
        CONDITION=C3 MECHANISM_CLASS=ClassII+III RUN_NAME=C3_seed${SEED}
      ;;
    C4)
      python algorithms/IPPO/ippo_cnn_cleanup_c4.py \
        PROJECT=$PROJECT ENTITY="$ENTITY" TOTAL_TIMESTEPS=$TOTAL_STEPS SEED=$SEED \
        CONDITION=C4 MECHANISM_CLASS=ClassIV RUN_NAME=C4_seed${SEED}
      ;;
    *)
      echo "Unknown condition: $COND" >&2
      return 1
      ;;
  esac
}

# Examples:
#   ./run_cleanup_experiments.sh C0 0
#   ./run_cleanup_experiments.sh C1 1
if [[ "${BASH_SOURCE[0]}" == "$0" && "${1-}" != "" ]]; then
  run_one "$@"
fi
