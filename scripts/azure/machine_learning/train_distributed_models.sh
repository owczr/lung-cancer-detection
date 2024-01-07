#!/bin/sh

OPTIMIZER="adam"
LOSS="binary_crossentropy"
EPOCHS=100
BATCH_SIZE=128

for model in mobilenet nasnet efficientnet efficientnetv2 densenet inceptionnet xception resnet resnetv2 convnext inceptionresnet vgg;
  do
    python -m scripts.azure.machine_learning.run_job \
      --model "$model" \
      --optimizer "$OPTIMIZER" \
      --loss "$LOSS" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --distributed
  done

