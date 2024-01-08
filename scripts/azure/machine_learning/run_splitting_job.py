import os
import logging
from datetime import datetime

import click
from dotenv import load_dotenv


from src.config import MODELS
from src.azure.utils import connect_to_workspace, get_compute
from src.azure.jobs import submit_dataset_splitting_job


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def run():
    ml_client = connect_to_workspace()
    get_compute(ml_client)
    submit_dataset_splitting_job(ml_client)


if __name__ == "__main__":
    run()

