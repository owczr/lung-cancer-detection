import os

import click

from src.preprocessing.dataset_processor import DatasetProcessor


@click.command()
@click.option("-i", "--input_path", type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to directory containing patient data with Dicom images")
@click.option("-o", "--output_path", type=click.Path(file_okay=False, dir_okay=True, writable=True),
    help="Path to output directory where processed dicoms will be saved")
@click.option("-t", "--train_size", type=float, default=0.8,
    help="Train size for train/test split")
def run(input_path, output_path, train_size):
    try:
        dp = DatasetProcessor(input_path)
        dp.process_and_save(output_path)
        dp.train_test_split(output_path, train_size=train_size)
        dp.remove_processed_data(output_path)

        click.echo(f"Processing completed. Data saved to {output_path}")
    except Exception as e:
        click.echo(f"An error occurred: {e}", err=True)

if __name__ == "__main__":
    run()
