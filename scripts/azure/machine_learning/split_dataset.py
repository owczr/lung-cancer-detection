import os
import shutil
import logging
import random
from datetime import datetime

import click

from src.preprocessing.utils import NODULE, NON_NODULE


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--input_path", help="Path to the input dataset on Azure ML")
@click.option("--output_path", help="Path to the output dataset on Azure ML")
def run(input_path, output_path):
   # for label in [NODULE, NON_NODULE]:
   #     path = os.path.join(input_path, 'test', label)

   #     files = [f for f in os.listdir(path)]
   #     random.shuffle(files)  

   #     split_idx = len(files) // 2
   #     test_files = files[:split_idx]
   #     validation_files = files[split_idx:]

   #     test_output_dir = os.path.join(output_path, 'test', label)
   #     validation_output_dir = os.path.join(output_path, 'validation', label)

   #     os.makedirs(test_output_dir, exist_ok=True)
   #     os.makedirs(validation_output_dir, exist_ok=True)

   #     for f in test_files:
   #         shutil.copy(os.path.join(path, f), os.path.join(test_output_dir, f))

   #     for f in validation_files:
   #         shutil.copy(os.path.join(path, f), os.path.join(validation_output_dir, f))

   #     logger.info(f"Completed shuffling and splitting for label '{label}'.")

    for label in [NODULE, NON_NODULE]:
        path = os.path.join(input_path, 'train', label)

        train_files = [f for f in os.listdir(path)]

        train_output_dir = os.path.join(output_path, 'train', label)

        os.makedirs(train_output_dir, exist_ok=True)

        for f in train_files:
            shutil.copy(os.path.join(path, f), os.path.join(train_output_dir, f))

        logger.info(f"Completed moving for label '{label}'.")


if __name__ == "__main__":
    run()

