import os
import logging
from datetime import datetime

import click
import mlflow
import numpy as np
import tensorflow as tf
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

from src.model.director import ModelDirector
from src.dataset.dataset_loader import DatasetLoader
from src.config import (
    RANDOM_SEED, 
    EARLY_STOPPING_CONFIG, 
    REDUCE_LR_CONFIG, 
    MODELS, 
    BUILDERS, 
    CALLBACKS, 
    METRICS, 
    config_logging
)

config_logging()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("azure")


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
@click.option("--job_name", type=click.STRING, help="Azure Machine Learning job name")
def run(model, train, test, optimizer, loss, epochs, batch_size, job_name):
    mlflow.set_experiment("lung-cancer-detection")
    mlflow_run = mlflow.start_run(run_name=f"train_{model}_{datetime.now().strftime('%Y%m%d%H%M%S')}")

    mlflow.log_param("optimizer", optimizer)
    mlflow.log_param("loss", loss)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("random_seed", RANDOM_SEED)

    logger.info(f"Started training run at {datetime.now()}")
    logger.info(
        f"Run parameters - optimizer: {optimizer}, loss: {loss}"
    )

    builder = BUILDERS[model]()

    director = ModelDirector(builder)
    model_nn = director.make()
    logger.info(f"Built model_nn with {str(builder)}")

    train_loader = DatasetLoader(train)
    test_loader = DatasetLoader(test)

    train_loader.set_seed(RANDOM_SEED)
    test_loader.set_seed(RANDOM_SEED)

    train_dataset = train_loader.get_dataset()
    test_dataset = test_loader.get_dataset()
    logger.info("Loaded train and test datasets")

    model_nn.compile(optimizer=optimizer, loss=loss, metrics=METRICS)
    logger.info("Compiled model")

    history = model_nn.fit(train_dataset, epochs=epochs, callbacks=CALLBACKS)
    logger.info("Trained model")

    for metric, values in history.history.items():
        for step, value in enumerate(values):
            mlflow.log_metric(f"{metric}", value, step=step)

    results = model_nn.evaluate(test_dataset, return_dict=True)
    logger.info("Evaluated model")

    for metric, value in results.items():
        mlflow.log_metric(f"Final {metric}", value)

    logger.info(f"Finished training at {datetime.now()}")

    mlflow.tensorflow.save_model(
        model=model_nn,
        path=os.path.join(job_name, model),
    )

    mlflow.tensorflow.log_model(
        model=model_nn,
        registered_model_name=model,
        artifact_path=model,
    )

    mlflow.end_run()


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
