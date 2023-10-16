import os

import pydicom

from segmentation import chest_segmentation, lung_segmentation


def process_directory(input_path: str, output_path: str):
    pass


def _generate_annotation_and_dicom_paths(path: str) -> tuple:
    for root, _, files in os.walk(input_path, topdown=False):
        paths_to_dicoms = []
        paths_to_annotations = []
        if len(files) == 0:
            continue

        for file in files:
            if ".dcm" in file:
                paths_to_dicoms.append(os.path.realpath(os.path.join(root, file)))
            elif ".xml" in file:
                paths_to_annotations.append(os.path.realpath(os.path.join(root, file)))
            else:
                continue
        yield paths_to_annotations, paths_to_dicoms


def process_dicom():
    pass


def _read_dicom():
    pass


def _save_processed():
    pass
