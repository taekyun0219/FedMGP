#!/bin/bash

GPU=$1                                    # GPU ID
DATASET=$2                                # Dataset name
SHOTS=${3:-8}                             # Number of shots, default 8
CONFIG_FILE=${4:-"base2novel_vit_b16"}    # Config file name
SEED=${5:-0}                              # Random seed, default 0

DATA="${COOP_DATASET}"                    # Dataset root path
cfg_file=${CONFIG_FILE}
trainer=FedMoPG
model=FedMoPG
SUBSAMPLE_CLASSES=base
USEALL=False

DIR=output/${DATASET}/${trainer}/${cfg_file}/${SHOTS}shots/seed${SEED}/${SUBSAMPLE_CLASSES}

if [ -d "$DIR" ]; then
  echo "Warning: Directory ${DIR} exists, will overwrite"
fi

mkdir -p ${DIR}

CUDA_VISIBLE_DEVICES=${GPU} python federated_main.py \
--root ${DATA} \
--output-dir ${DIR} \
--seed ${SEED} \
--model ${model} \
--trainer ${trainer} \
--config-file configs/trainers/${trainer}/${cfg_file}.yaml \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--num_shots ${SHOTS} \
--debug-mode true \
DATASET.SUBSAMPLE_CLASSES ${SUBSAMPLE_CLASSES} \
DATASET.USEALL ${USEALL}
