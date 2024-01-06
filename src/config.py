import os
import logging
from datetime import datetime

import tensorflow as tf
import tensorflow_addons as tfa

from src.model.builders import (
    ConvNeXtBuilder,
    DenseNetBuilder,
    EfficientNetBuilder,
    EfficientNetV2Builder,
    InceptionNetBuilder,
    InceptionResNetBuilder,
    MobileNetBuilder,
    ResNetBuilder,
    ResNetV2Builder,
    VGGBuilder,
    XceptionBuilder,
)

# Get the current date to create a dynamic log filename
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"lung_cancer_detection_{current_date}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

EARLY_STOPPING_CONFIG = {
    "monitor": "val_loss",
    "min_delta": 0.001,
    "patience": 3,
}

REDUCE_LR_CONFIG = {
    "monitor": "val_loss",
    "factor": 0.1,
    "patience": 2,
    "min_delta": 0.001,
}

MODELS = [
    "convnext",
    "densenet",
    "efficientnet",
    "efficientnetv2",
    "inceptionnet",
    "inceptionresnet",
    "mobilenet",
    "resnet",
    "resnetv2",
    "vgg",
    "xception",
]

BUILDERS = {
    "convnext": ConvNeXtBuilder,
    "densenet": DenseNetBuilder,
    "efficientnet": EfficientNetBuilder,
    "efficientnetv2": EfficientNetV2Builder,
    "inceptionnet": InceptionNetBuilder,
    "inceptionresnet": InceptionResNetBuilder,
    "mobilenet": MobileNetBuilder,
    "resnet": ResNetBuilder,
    "resnetv2": ResNetV2Builder,
    "vgg": VGGBuilder,
    "xception": XceptionBuilder,
}


METRICS = [
    tf.keras.metrics.Accuracy(),
    tfa.metrics.F1Score(num_classes=2),
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    tf.keras.metrics.AUC(),
]


CALLBACKS = [
    tf.keras.callbacks.EarlyStopping(**EARLY_STOPPING_CONFIG),
    tf.keras.callbacks.ReduceLROnPlateau(**REDUCE_LR_CONFIG),
]

