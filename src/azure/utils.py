import os

from dotenv import load_dotenv
from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute, Model
from azure.ai.ml.constants import AssetTypes


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


def register_model(ml_client, returned_job, run_name, run_description):
    run_model = Model(
        path=f"azureml://jobs/{returned_job.name}/outputs/artifacts/paths/model/",
        name=run_name,
        description=run_description,
        type=AssetTypes.MLFLOW_MODEL,
    )

    ml_client.models.create_or_update(run_model)

