import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Mapping

import numpy as np

from base import BaseProcessor
from preprocessing import AnnotationProcessor, DicomProcessor
from utils import *


class DatasetProcessor(BaseProcessor):
    """
    Processor for handling entire datasets.

    This class processes directories containing DICOM and XML files. It's capable of parallel processing 
    and can save the processed data in an organized fashion.
    
    Attributes:
        path (str): The path to the directory containing DICOM and XML files.
        _data (dict): Dictionary containing processed DICOMs and labels.

    Methods inherited from BaseProcessor:
        process, save, process_and_save.

    Private Methods:
        _process_parallel: Processes the dataset in parallel using multiple CPU cores.
        _process: Processes a batch of DICOM files.
        _process_and_save: Processes and saves a batch of DICOM files.
        _generate_annotation_and_dicom_paths: Generates paths for DICOM and XML files.
        _get_annotation_xml_path: Returns the path to the XML annotation within a directory.
    """
    def __init__(self, path: str):
        self.path = path
        self._data = {
            DICOM_KEY: [],
            ANNOTATION_KEY: [],
        }

    def process(self) -> Mapping[list[ProcessedDicom], list[str]]:
        """Processes whole directory and returns dictionary with processed dicoms and labels"""
        self._process_parallel(self._process)
        return self._data

    def save(self, path: str) -> None:
        for processed_dicom, label in self._data.items():
            # Save image in correct label folder
            filename = f"{processed_dicom.uid}{NUMPY_EXTENSION}"
            output_path = os.path.join(path, label, filename)
            np.save(output_path, processed_dicom.image)


    def process_and_save(self, path: str) -> None:
        """Processes whole directory and saves it to given output directory"""
        self._process_parallel(self._process_and_save, path)

    @property
    def data(self):
        return self._data

    @data.getter
    def data(self):
        if self._data is None:
            self.process()
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def _process_parallel(self, worker_func, **kwargs):
        path = kwargs.get("path")

        # Create output directory if it doesn't exist
        os.makedirs(path, exists_ok=True)

        # Create label directories, here processed dicoms will be saved
        for label in [NODULE, NON_NODULE]:
            label_path = os.path.join(self.output_directory, label)
            os.makedirs(label_path, exists_ok=True)

        for paths_dictionary in self._generate_annotation_and_dicom_paths(self.path):
            # Unpack the dictionary
            dicom_paths = paths_dictionary[DICOM_KEY]
            annotation_path = paths_dictionary[ANNOTATION_KEY]

            # Get all z position of nodules
            annotation_processor = AnnotationProcessor(path=annotation_path)
            nodule_positions = annotation_processor.process()

            # List of all future tasks
            futures = []     

            # Create batches from dicom_paths, this will reduco I/O frequency
            dicom_batches = [
                dicom_paths[i:i + BATCH_SIZE] for i in range(0, len(dicom_paths), BATCH_SIZE)
            ]

            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for dicom_batch in dicom_batches:
                    # Submit a new task to the executor.
                    # The _process_and_save function will process the DICOM and save it.
                    future = executor.submit(
                        worker_func, dicom_batch, nodule_positions, path
                    )
                    futures.append(future)

            # Wait for all tasks to complete and handle exceptions if necessary
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Task generated an exception: {e}")

    def _process(
        self, dicom_paths:
        list[str],
        nodule_positions: set[float],
        **kwargs,
    ) -> None:
        for dicom_path in dicom_paths:
            # Get processed image, uid for filename and slice z position
            dicom_processor = DicomProcessor(path=dicom_path)
            processed_dicom = dicom_processor.process()

            # Check whether slice contains a nodule
            label = (
                NODULE
                if processed_dicom.z_position in nodule_positions
                else NON_NODULE
            )

            self._data[DICOM_KEY].append(processed_dicom)
            self._data[ANNOTATION_KEY].appemd(label)

    def _process_and_save(
        self, dicom_paths: list[str],
        nodule_positions: set[float],
        path: str,
    ) -> None:
    """Reads, processes and saves dicom images"""
        for dicom_path in dicom_paths:
            # Get processed image, uid for filename and slice z position
            dicom_processor = DicomProcessor(path=dicom_path)
            processed_dicom = dicom_processor.process()

            # Check whether slice contains a nodule
            label = (
                NODULE
                if processed_dicom.z_position in nodule_positions
                else NON_NODULE
            )

            # Save image in correct label folder
            filename = f"{processed_dicom.uid}{NUMPY_EXTENSION}"
            output_path = os.path.join(path, label, filename)
            np.save(output_path, processed_dicom.image)

    def _generate_annotation_and_dicom_paths(self, path: str) -> tuple:
        """Yields a dictionary with path to annotation and paths to dicoms"""
        for root, _, files in os.walk(path, topdown=False):
            if len(files) == 0:
                continue

            dicom_paths = [
                os.path.realpath(os.path.join(root, f))
                for f in files
                if f.endswith(DICOM_EXTENSION)
            ]

            annotation_path = self._get_annotation_xml_path(root)

            paths_dictionary = {
                DICOM_KEY: dicom_paths,
                ANNOTATION_KEY: annotation_path,
            }

            yield paths_dictionary

    @staticmethod
    def _get_annotation_xml_path(directory: str) -> str:
        """Returns path to annotation xml"""
        # List all files in the directory
        files = [
            f
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]

        # Filter the list to get .xml files
        xml_files = [f for f in files if f.endswith(ANNOTATION_EXTENSION)]

        # Check if there's exactly one .xml file
        if len(xml_files) == 1:
            return os.path.join(directory, xml_files[0])
        elif len(xml_files) == 0:
            raise ValueError("No XML files found in the directory.")
        else:
            raise ValueError("Multiple XML files found. Expected only one.")
