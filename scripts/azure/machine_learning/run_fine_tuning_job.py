import os
import logging
from datetime import datetime

import click

from src.config import MODELS
from src.azure.utils import connect_to_workspace, get_compute, register_model
from src.azure.jobs import submit_fine_tuning_job


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--model", type=click.Choice(MODELS), help="Model to train")
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
@click.option("--epochs", type=int, default=10, help="Number of epochs to train for")
@click.option("--batch_size", type=int, default=32, help="Batch size to use")
@click.option("--distributed", is_flag=True, help="Use distributed startegy")
def run(model, optimizer, loss, epochs, batch_size, distributed):
    if model not in MODELS:
        raise ValueError(f"Model {model} not supported")

    ml_client = connect_to_workspace()

    get_compute(ml_client=ml_client)

    returned_job = submit_fine_tuning_job(
        ml_client=ml_client,
        model=model,
        optimizer=optimizer,
        loss=loss,
        epochs=epochs,
        batch_size=batch_size,
        distributed=distributed,
    )

    click.echo(
        f"Job {returned_job.name} created.\n"
        f"  - id: {returned_job.id}\n"
        f"  - url: {returned_job.studio_url}\n"
    )

    logger.info(
        f"Created a {model} training job at {datetime.now()}\n"
        f"  - id: {returned_job.id}\n"
        f"  - name: {returned_job.name}\n"
        f"  - url: {returned_job.studio_url}\n\n"
        "Training parameters:\n"
        f"  - optimizer: {optimizer}\n"
        f"  - loss: {loss}\n"
        f"  - epochs: {epochs}\n"
        f"  - batch size: {batch_size}\n"
    )


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
