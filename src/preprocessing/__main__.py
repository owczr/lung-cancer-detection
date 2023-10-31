import os

import click


@click.command()
@click.option("-i", "--input_path", type=click.STRING, help="Path to directory containing patient data with Dicom images")
@click.option("-o", "--output_path", type=click.STRING, help="Path to output directory where processed dicoms will be saved")
def run(input_path, output_path):
    pass

if __name__ == "__main__":
    run()
