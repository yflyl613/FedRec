#!/bin.sh

MODEL_TYPE="MF"
DATA='ml'
LR=2e-3
WEIGHT_DECAY=1e-5
AGG_TYPE='FedAdam'

# No Attack
EXP_NAME=${DATA}-${MODEL_TYPE}-${AGG_TYPE}
if [ $1 == "train" ]
then
    python -u train.py --EXP_NAME ${EXP_NAME} --DATA ${DATA} --MODEL_TYPE ${MODEL_TYPE} \
    --AGG_TYPE ${AGG_TYPE} --LR ${LR} --WEIGHT_DECAY ${WEIGHT_DECAY}
elif [ $1 == "test" ]
then
    python -u test.py --EXP_NAME ${EXP_NAME} --DATA ${DATA} --MODEL_TYPE ${MODEL_TYPE}
fi

# # Attack
# ATTACKER_RATIO=0.01
# ATTACKER_STRAT='ClusterAttack'
# EXP_NAME=${DATA}-${MODEL_TYPE}-${AGG_TYPE}-${ATTACKER_STRAT}-${ATTACKER_RATIO}

# if [ $1 == "train" ]
# then
#     python -u train.py --EXP_NAME ${EXP_NAME} --DATA ${DATA} --MODEL_TYPE ${MODEL_TYPE} \
#     --ATTACKER_RATIO ${ATTACKER_RATIO} --ATTACKER_STRAT ${ATTACKER_STRAT} --AGG_TYPE ${AGG_TYPE} \
#     --LR ${LR} --WEIGHT_DECAY ${WEIGHT_DECAY}
# elif [ $1 == "test" ]
# then
#     python -u test.py --EXP_NAME ${EXP_NAME} --DATA ${DATA} --MODEL_TYPE ${MODEL_TYPE}
# fi