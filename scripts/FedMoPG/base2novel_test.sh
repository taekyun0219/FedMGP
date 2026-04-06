#!/bin/bash

GPU=$1                                    # GPU ID
DATASET=$2                                # Dataset name
SUBSAMPLE_CLASSES=${3:-"base"}            # Class subset to test (base/new)
SHOTS=${4:-8}                             # Number of shots, default 8
CONFIG_FILE=${5:-"base2novel_vit_b16"}    # Config file name
SEED=${6:-0}                              # Random seed, default 0

DATA="${COOP_DATASET}"                    # Dataset root path
cfg_file=${CONFIG_FILE}
trainer=FedMoPG
model=FedMoPG
USEALL=False

# Base model path for loading weights
BASE_DIR=output/${DATASET}/${trainer}/${cfg_file}/${SHOTS}shots/seed${SEED}/base

# Test output path
OUTPUT_DIR=output/${DATASET}/${trainer}/${cfg_file}/${SHOTS}shots/seed${SEED}/${SUBSAMPLE_CLASSES}_test

if [ ! -d "$BASE_DIR" ]; then
  echo "Error: Base model not found at ${BASE_DIR}"
  exit 1
fi

# Find model weights (best_model.pt or last_model.pt)
MODEL_PATH=""
if [ -f "${BASE_DIR}/best_model.pt" ]; then
  MODEL_PATH="${BASE_DIR}/best_model.pt"
  echo "Using best model: ${MODEL_PATH}"
elif [ -f "${BASE_DIR}/last_model.pt" ]; then
  MODEL_PATH="${BASE_DIR}/last_model.pt"
  echo "Using last model: ${MODEL_PATH}"
else
  echo "Error: Model weights not found"
  exit 1
fi

mkdir -p ${OUTPUT_DIR}

echo "Testing on ${SUBSAMPLE_CLASSES} classes..."
CUDA_VISIBLE_DEVICES=${GPU} python federated_main.py \
--root ${DATA} \
--output-dir ${OUTPUT_DIR} \
--seed ${SEED} \
--model ${model} \
--trainer ${trainer} \
--config-file configs/trainers/${trainer}/${cfg_file}.yaml \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--num_shots ${SHOTS} \
--eval-only \
--model-dir ${MODEL_PATH} \
DATASET.SUBSAMPLE_CLASSES ${SUBSAMPLE_CLASSES} \
DATASET.USEALL ${USEALL}

echo "Testing completed, results saved to ${OUTPUT_DIR}"
