import os

import click

from src.preprocessing.dataset_processor import DatasetProcessor


@click.command()
@click.option("-i", "--input_path", type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to directory containing patient data with Dicom images")
@click.option("-o", "--output_path", type=click.Path(file_okay=False, dir_okay=True, writable=True),
    help="Path to output directory where processed dicoms will be saved")
def run(input_path, output_path):
    try:
        dp = DatasetProcessor(input_path)
        dp.process_and_save(output_path)
        click.echo(f"Processing completed. Data saved to {output_path}")
    except Exception as e:
        click.echo(f"An error occurred: {e}", err=True)

if __name__ == "__main__":
    run()
