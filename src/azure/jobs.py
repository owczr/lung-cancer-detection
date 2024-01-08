import os
from datetime import datetime

from dotenv import load_dotenv
from azure.ai.ml import MLClient, command, Input, Output
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute, Model
from azure.ai.ml.constants import AssetTypes

load_dotenv()


def submit_dataset_splitting_job(ml_client):
    code = os.getenv("AZURE_CODE_PATH")
    environment = os.getenv("AZURE_ENVIRONMENT")
    type_ = os.getenv("AZURE_STORAGE_TYPE")
    input_path = os.getenv("AZURE_STORAGE_PATH")
    output_path = os.getenv("AZURE_STORAGE_OUTPUT_PATH")
    compute = os.getenv("AZURE_COMPUTE_TARGET")

    job_name = f"split_dataset{datetime.now().strftime('%Y%m%d%H%M%S')}" 

    command_var = (
        "python -m scripts.azure.machine_learning.split_dataset"
        " --input_path ${{inputs.input_path}} --output_path ${{outputs.output_path}}"
    )

    command_job = command(
        code=code,
        command=command_var,
        environment=environment,
        inputs={
            "input_path": Input(
                type=type_,
                path=input_path,
            ),
        },
        outputs={
            "output_path": Output(
                type=type_,
                path=output_path,
            )
        },
        compute=compute,
        name=job_name,
    )

    returned_job = ml_client.jobs.create_or_update(command_job)

    return returned_job


def submit_fine_tuning_job(ml_client, model, optimizer, loss, epochs, batch_size, distributed):
    pass


def submit_training_job(ml_client, model, optimizer, loss, epochs, batch_size, distributed):
    code = os.getenv("AZURE_CODE_PATH")
    environment = os.getenv("AZURE_ENVIRONMENT")
    type_ = os.getenv("AZURE_STORAGE_TYPE")
    path = os.getenv("AZURE_STORAGE_PATH")
    compute = os.getenv("AZURE_COMPUTE_TARGET")

    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "test")
    val_path = os.path.join(path, "validation")

    job_name = f"train_{model}_{datetime.now().strftime('%Y%m%d%H%M%S')}" 

    command_var = (
        "python -m scripts.azure.machine_learning.train"
        " --train ${{inputs.train}} --test ${{inputs.test}} --val ${{inputs.val}}"
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
            "val": Input(
                type=type_,
                path=val_path,
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

