import os

import click

from src.preprocessing.dataset_processor import DatasetProcessor


@click.command()
@click.option("-d", "--dataset_path", type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to directory containing processed dataset")
@click.option("-t", "--train_size", type=float, default=0.8,
    help="Train size for train/test split")
def run(dataset_path, train_size):
    try:
        dp = DatasetProcessor(dataset_path)
        click.echo(f"Splitting processed dataset at {dataset_path}\nTrain split: {train_size}")
        dp.train_test_split(dataset_path, train_size=train_size)
        click.echo(f"Splitting completed.")
    except Exception as e:
        click.echo(f"An error occurred: {e}", err=True)

if __name__ == "__main__":
    run()
