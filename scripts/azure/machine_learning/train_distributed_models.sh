#!/bin/sh

OPTIMIZER="adam"
LOSS="binary_crossentropy"
EPOCHS=100
BATCH_SIZE=128

for model in densenet inceptionnet xception resnetv2 convnext inceptionresnet;
  do
    python -m scripts.azure.machine_learning.run_training_job \
      --model "$model" \
      --optimizer "$OPTIMIZER" \
      --loss "$LOSS" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --distributed
  done

