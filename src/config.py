import logging
from datetime import datetime

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
