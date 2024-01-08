#!/bin/sh

OPTIMIZER="adam"
LOSS="binary_crossentropy"
EPOCHS=100
BATCH_SIZE=64

for model in densenet inceptionnet xception resnetv2 inceptionresnet convnext;
  do
    python -m scripts.azure.machine_learning.run_job \
      --model "$model" \
      --optimizer "$OPTIMIZER" \
      --loss "$LOSS" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE"
  done

