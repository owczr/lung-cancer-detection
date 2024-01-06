import os
import logging
from datetime import datetime

import click
import tensorflow as tf
from azureml.core import Run

from src.model.director import ModelDirector
from src.dataset.dataset_loader import DatasetLoader
from src.config import EARLY_STOPPING_CONFIG, REDUCE_LR_CONFIG, MODELS, BUILDERS, CALLBACKS, METRICS


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
@click.option("--epochs", type=click.INT, default=10, help="Number of epochs to train for")
@click.option("--batch_size", type=click.INT, default=64, help="Batch size for dataset loaders")
def run(model, train, test, optimizer, loss, epochs, batch_size):
    run = Run.get_context()

    logger.info(f"Started training run at {datetime.now()}")
    logger.info(
        f"Run parameters - optimizer: {optimizer}, loss: {loss}, metrics: {metric}"
    )

    builder = BUILDERS[model]()

    director = ModelDirector(builder)
    model = director.make()
    logger.info(f"Built model with {str(builder)}")

    train_loader = DatasetLoader(train)
    test_loader = DatasetLoader(test)

    train_dataset = train_loader.get_dataset()
    test_dataset = test_loader.get_dataset()
    logger.info("Loaded train and test datasets")

    model.compile(optimizer=optimizer, loss=loss, metrics=METRICS)
    logger.info("Compiled model")

    history = model.fit(train_dataset, epochs=epochs, callbacks=CALLBACKS)
    logger.info("Trained model")

    for metric, values in history.history.items():
        for epoch, value in enumerate(values)):
            run.log(str(epoch), value)

    results = model.evaluate(test_dataset, return_dict=True)
    logger.info("Evaluated model")

    for metric, value in results.items():
        run.log(metrics, value)

    logger.info(f"Finished training at {datetime.now()}")


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
