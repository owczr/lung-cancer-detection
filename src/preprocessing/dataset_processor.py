import os
import shutil
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Mapping
from datetime import datetime

import numpy as np
from tqdm import tqdm

import src.config
from src.preprocessing.base import BaseProcessor
from src.preprocessing.annotation_processor import AnnotationProcessor
from src.preprocessing.dicom_processor import DicomProcessor
from src.preprocessing.utils import *


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        logger.info(f"Initialized DatasetProcessor for {path}")

    def process(
        self, as_output: bool = False
    ) -> Mapping[list[ProcessedDicom], list[str]]:
        """Processes whole directory and optionaly returns dictionary with processed dicoms and labels"""
        self._process_parallel(self._process)
        return self._data if as_output else None

    def save(self, path: str) -> None:
        for processed_dicom, label in zip(
            self._data[DICOM_KEY], self._data[ANNOTATION_KEY]
        ):
            # Save image in correct label folder
            filename = f"{processed_dicom.uid}{NUMPY_EXTENSION}"
            output_path = os.path.join(path, label, filename)
            np.save(output_path, processed_dicom.image)

    def process_and_save(self, path: str) -> None:
        """Processes whole directory and saves it to given output directory"""
        logger.info(f"Processings started at {datetime.now()}")
        self._process_parallel(self._process_and_save, path)
        logger.info(f"Processing ended at {datetime.now()}")

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

    def train_test_split(self, path: str, train_size: float = 0.8):
        """Splits the dataset into training and testing sets"""
        # Check if nodue and non nodule folders exist
        if not all(
            os.path.exists(os.path.join(path, category))
            for category in [NODULE, NON_NODULE]
        ):
            logger.info(
                f"Directories {NODULE} and {NON_NODULE} do not exist."
                "Process the dataset first."
            )
            return

        train_dir = os.path.join(path, TRAIN_FOLDER)
        test_dir = os.path.join(path, TEST_FOLDER)

        for category in [NODULE, NON_NODULE]:
            os.makedirs(os.path.join(train_dir, category), exist_ok=True)
            os.makedirs(os.path.join(test_dir, category), exist_ok=True)

            self._split_data(
                os.path.join(path, category),
                os.path.join(train_dir, category),
                os.path.join(test_dir, category),
                train_size,
            )

    def _split_data(self, source, train_dir, test_dir, split_size):
        files = os.listdir(source)

        # Shuffle the files randomly
        shuffled_files = np.random.permutation(files)

        # Calculate the split index
        split_index = int(len(shuffled_files) * split_size)

        train_files = shuffled_files[:split_index]
        test_files = shuffled_files[split_index:]

        # Copy files to the respective directories
        with tqdm(total=len(train_files)) as pbar:
            for file in train_files:
                shutil.copy(os.path.join(source, file), train_dir)
                pbar.update(1)

        with tqdm(total=len(test_files)) as pbar:
            for file in test_files:
                shutil.copy(os.path.join(source, file), test_dir)
                pbar.update(1)

    def remove_processed_data(self, path: str):
        """Removes all processed data from the directory"""
        categories = []
        # Check if nodule folder exists
        if not os.path.exists(os.path.join(path, NODULE)):
            logger.info(f"Directory {NODULE} does not exist.")
        else:
            categories.append(NODULE)

        # Check if non_nodule folder exists
        if not os.path.exists(os.path.join(path, NON_NODULE)):
            logger.info(f"Directory {NON_NODULE} does not exist.")
        else:
            categories.append(NON_NODULE)
        
        if len(categories) == 0:
            logger.info(f"No nodule and non nodule directories to remove.")
            return

        for category in categories:
            shutil.rmtree(os.path.join(path, category))
        logger.info(f"Removed processed data.") 

    def remove_train_test_data(self, path: str):
        """Removes train and test directories"""
        categories = []
        # Check if train folder exists
        if not os.path.exists(os.path.join(path, TRAIN_FOLDER)):
            logger.info(f"Directory {TRAIN_FOLDER} does not exist.")
        else:
            categories.append(TRAIN_FOLDER)
        
        # Check if test folder exists
        if not os.path.exists(os.path.join(path, TEST_FOLDER)):
            logger.info(f"Directory {TEST_FOLDER} does not exist.")
        else:
            categories.append(TEST_FOLDER)
        
        if len(categories) == 0:
            logger.info(f"No train and test directories to remove.")
            return

        for category in categories:
            shutil.rmtree(os.path.join(path, category))
        logger.info(f"Removed train and test directories.")

    def _process_parallel(self, worker_func, path=None):
        if path:
            # Create output directory if it doesn't exist
            os.makedirs(path, exist_ok=True)

            # Create label directories, here processed dicoms will be saved
            for label in [NODULE, NON_NODULE]:
                label_path = os.path.join(path, label)
                os.makedirs(label_path, exist_ok=True)

        dataset_size = self._check_dataset_size()
        logger.info(f"Processing dataset with {dataset_size} scans.")

        with tqdm(total=dataset_size) as pbar:
            for paths_dictionary in self._generate_annotation_and_dicom_paths():
                # Unpack the dictionary
                dicom_paths = paths_dictionary[DICOM_KEY]
                annotation_path = paths_dictionary[ANNOTATION_KEY]

                # Get all z position of nodules
                ap = AnnotationProcessor(path=annotation_path)
                ap.process()
                nodule_positions = ap.z_positions

                # List of all future tasks
                futures = []

                # Create batches from dicom_paths, this will reduco I/O frequency
                dicom_batches = [
                    dicom_paths[i : i + BATCH_SIZE]
                    for i in range(0, len(dicom_paths), BATCH_SIZE)
                ]

                logger.info(f"Starting parallel processing.")

                with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    for dicom_batch in dicom_batches:
                        # Submit a new task to the executor.
                        # The _process_and_save function will process the DICOM and save it.
                        future = executor.submit(
                            worker_func,
                            dicom_batch,
                            nodule_positions,
                            ap.data,
                            path,
                        )
                        futures.append(future)

                    # Wait for all tasks to complete and handle exceptions if necessary
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            logger.exception(f"Error processing batch:\n{e}")

                logger.info(f"Finished parallel processing succesfully.")
                pbar.update(1)

    def _process(
        self,
        dicom_paths: list[str],
        nodule_positions: set[float],
        annotations: list[ProcessedAnnotation],
        path=None,
    ) -> None:
        for dicom_path in dicom_paths:
            # Get processed image, uid for filename and slice z position
            dp = DicomProcessor(path=dicom_path, annotations=annotations)
            processed_dicom = dp.process(as_output=True)

            if processed_dicom is None:
                logger.error(f"Processing dicom {dp.path} returned None.")
                return

            # Check whether slice contains a nodule
            label = (
                NODULE if processed_dicom.z_position in nodule_positions else NON_NODULE
            )

            self._data[DICOM_KEY].append(processed_dicom)
            self._data[ANNOTATION_KEY].append(label)

    def _process_and_save(
        self,
        dicom_paths: list[str],
        nodule_positions: set[float],
        annotations: list[ProcessedAnnotation],
        path: str,
    ) -> None:
        for dicom_path in dicom_paths:
            # Get processed image, uid for filename and slice z position
            dp = DicomProcessor(path=dicom_path, annotations=annotations)
            processed_dicom = dp.process(as_output=True)

            if processed_dicom is None:
                logger.error(f"Processing dicom {dp.path} returned None")
                return

            # Check whether slice contains a nodule
            label = (
                NODULE if processed_dicom.z_position in nodule_positions else NON_NODULE
            )

            # Save image in correct label folder
            filename = f"{processed_dicom.uid}{NUMPY_EXTENSION}"
            output_path = os.path.join(path, label, filename)
            np.save(output_path, processed_dicom.image)
            logger.info(f"Saved DICOM Image to {output_path}")

    def _generate_annotation_and_dicom_paths(self) -> tuple:
        """Yields a dictionary with path to annotation and paths to dicoms"""
        for root, _, files in os.walk(self.path, topdown=False):
            if len(files) == 0:
                continue

            dicom_paths = [
                os.path.realpath(os.path.join(root, f))
                for f in files
                if f.endswith(DICOM_EXTENSION)
            ]

            annotation_path = self._get_annotation_xml_path(root)
            if annotation_path is None:
                continue

            paths_dictionary = {
                DICOM_KEY: dicom_paths,
                ANNOTATION_KEY: annotation_path,
            }

            yield paths_dictionary

    def _get_annotation_xml_path(self, directory: str) -> str:
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
            logger.exception("No XML files found in the directory.")
            return None
        else:
            logger.exception("Multiple XML files found. Expected only one.")
            return None

    def _check_dataset_size(self) -> int:
        return sum(1 for _ in self._generate_annotation_and_dicom_paths())
