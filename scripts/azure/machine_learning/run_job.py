import os
import logging
from datetime import datetime

import click
from dotenv import load_dotenv
from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute, Model
from azure.ai.ml.constants import AssetTypes

from src.config import MODELS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def connect_to_workspace():
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    workspace = os.getenv("AZURE_WORKSPACE")

    ml_client = MLClient(
        DefaultAzureCredential(), subscription_id, resource_group, workspace
    )

    return ml_client


def get_compute(ml_client):
    cpu_compute_target = os.getenv("AZURE_COMPUTE_TARGET")

    try:
        ml_client.compute.get(cpu_compute_target)
    except Exception:
        click.echo(f"Compute {cpu_compute_target} not found.")

def submit_job(ml_client, model, optimizer, loss, epochs, batch_size, distributed):
    code = os.getenv("AZURE_CODE_PATH")
    environment = os.getenv("AZURE_ENVIRONMENT")
    type_ = os.getenv("AZURE_STORAGE_TYPE")
    path = os.getenv("AZURE_STORAGE_PATH")
    compute = os.getenv("AZURE_COMPUTE_TARGET")

    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "test")

    job_name = f"train_{model}_{datetime.now().strftime('%Y%m%d%H%M%S')}" 

    command_var = (
        "python -m scripts.azure.machine_learning.train"
        " --train ${{inputs.train}} --test ${{inputs.test}}"
        " --epochs ${{inputs.epochs}} --optimizer ${{inputs.optimizer}}"
        " --loss ${{inputs.loss}}  --batch_size ${{inputs.batch_size}}"
        " --model ${{inputs.model}} --job_name ${{inputs.job_name}}"
    )

    command_var = command_var + " --distributed" if distributed else command_var

    command_job = command(
        code=code,
        command=command_var,
        environment=environment,
        inputs={
            "train": Input(
                type=type_,
                path=train_path,
            ),
            "test": Input(
                type=type_,
                path=test_path,
            ),
            "optimizer": optimizer,
            "loss": loss,
            "epochs": epochs,
            "batch_size": batch_size,
            "model": model,
            "job_name": job_name,
        },
        compute=compute,
        name=job_name,
    )

    returned_job = ml_client.jobs.create_or_update(command_job)

    return returned_job


def register_model(ml_client, returned_job, run_name, run_description):
    run_model = Model(
        path=f"azureml://jobs/{returned_job.name}/outputs/artifacts/paths/model/",
        name=run_name,
        description=run_description,
        type=AssetTypes.MLFLOW_MODEL,
    )

    ml_client.models.create_or_update(run_model)


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

    returned_job = submit_job(
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
