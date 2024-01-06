import os
import logging
from datetime import datetime

import click
import tensorflow as tf

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
from src.model.director import ModelDirector
from src.dataset.dataset_loader import DatasetLoader
from src.config import EARLY_STOPPING_CONFIG, REDUCE_LR_CONFIG, MODELS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model", type=click.Choice(MODELS), default="mobilenet", help="Model to train"
)
@click.option(
    "--train", type=click.Path(exists=True), help="Path to the training dataset"
)
@click.option("--test", type=click.Path(exists=True), help="Path to the test dataset")
@click.option(
    "--optimizer",
    type=click.Choice(["adam", "sgd"]),
    default="adam",
    help="Optimizer to use",
)
@click.option(
    "--loss",
    type=click.Choice(["binary_crossentropy", "categorical_crossentropy"]),
    default="binary_crossentropy",
    help="Loss function to use",
)
@click.option(
    "--metric",
    type=click.Choice(["accuracy", "f1"]),
    default="accuracy",
    help="Metrics to use",
)
@click.option("--epochs", type=click.INT, default=10, help="Number of epochs to train for")
@click.option("--batch_size", type=click.INT, default=64, help="Batch size for dataset loaders")
def run(model, train, test, optimizer, loss, metric, epochs, batch_size):
    logger.info(f"Started training run at {datetime.now()}")
    logger.info(
        f"Run parameters - optimizer: {optimizer}, loss: {loss}, metrics: {metric}"
    )

    builder = {
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
    }[model]()
    director = ModelDirector(builder)
    model = director.make()
    logger.info(f"Built model with {str(builder)}")

    train_loader = DatasetLoader(train)
    test_loader = DatasetLoader(test)

    train_dataset = train_loader.get_dataset()
    test_dataset = test_loader.get_dataset()
    logger.info("Loaded train and test datasets")

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    logger.info("Compiled model")

    ec = tf.keras.callbacks.EarlyStopping(**EARLY_STOPPING_CONFIG)
    lr = tf.keras.callbacks.ReduceLROnPlateau(**REDUCE_LR_CONFIG)

    model.fit(train_dataset, epochs=epochs, callbacks=[ec, lr])
    logger.info("Trained model")

    model.evaluate(test_dataset)
    logger.info("Evaluated model")

    logger.info(f"Finished training at {datetime.now()}")


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
