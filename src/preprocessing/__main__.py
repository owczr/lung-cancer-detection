import os

import click

from processing import process_directory


@click.command()
@click.option("-i", "--input_path", type=click.STRING, help="Path to directory containing patient data with Dicom images")
@click.option("-o", "--output_path", type=click.STRING, help="Path to output directory where processed dicoms will be saved")
def run(input_path, output_path):
    # print(os.path.realpath(os.curdir))
    process_directory(input_path, output_path)


if __name__ == "__main__":
    run()
