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
    NASNetBuilder,
    ResNetBuilder,
    ResNetV2Builder,
    VGGBuilder,
    XceptionBuilder,
)

def config_logging():
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"lung_cancer_detection_{current_date}.log"
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

config_logging()

RANDOM_SEED = 42

MODELS = [
    "convnext",
    "densenet",
    "efficientnet",
    "efficientnetv2",
    "inceptionnet",
    "inceptionresnet",
    "mobilenet",
    "nasnet",
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
    "nasnet": NASNetBuilder,
    "resnet": ResNetBuilder,
    "resnetv2": ResNetV2Builder,
    "vgg": VGGBuilder,
    "xception": XceptionBuilder,
}

_threshold = 0.5

# lambda is needed, because metrics need to be created within strategy scope
METRICS = [
    lambda : tf.keras.metrics.BinaryAccuracy(threshold=_threshold, name="accuracy"),
    lambda : tfa.metrics.F1Score(num_classes=1, threshold=_threshold, name="f1"),
    lambda : tf.keras.metrics.Precision(thresholds=_threshold, name="precision"),
    lambda : tf.keras.metrics.Recall(thresholds=_threshold, name="recall"),
    lambda : tf.keras.metrics.AUC(thresholds=[_threshold], curve="ROC", name="roc_auc"),
]

EARLY_STOPPING_CONFIG = {
    "monitor": "loss",
    "min_delta": 0.001,
    "patience": 6,
}

REDUCE_LR_CONFIG = {
    "monitor": "loss",
    "factor": 0.1,
    "patience": 4,
    "min_delta": 0.001,
}

CALLBACKS = [
    tf.keras.callbacks.EarlyStopping(**EARLY_STOPPING_CONFIG),
    tf.keras.callbacks.ReduceLROnPlateau(**REDUCE_LR_CONFIG),
]

