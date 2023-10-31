import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

import pydicom
import numpy as np
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_closing,
    binary_opening,
    disk,
)

class DicomDatasetProcessor:
    def __init__(self, input_directory: str, output_directory: str):
        self.input_directory = input_directory
        self.output_directory = output_directory

    def process_directory(self) -> None:
        """Processes whole directory and saves it to given output directory"""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_directory, exists_ok=True)

        # Create label directories, here processed dicoms will be saved
        for label in [NODULE, NON_NODULE]:
            path = os.path.join(self.output_directory, label)
            os.makedirs(path, exists_ok=True)

        for paths_dictionary in self.__generate_annotation_and_dicom_paths(
            self.input_directory
        ):
            # Unpack the dictionary
            dicom_paths = paths_dictionary[DICOM_KEY]
            annotation_path = paths_dictionary[ANNOTATION_KEY]

            # Get all z position of nodules
            nodule_positions = self.__process_annotation(annotation_path)

            # List of all future tasks
            futures = []     

            # Create batches from dicom_paths, this will reduco I/O frequency
            dicom_batches = [
                dicom_paths[i:i + BATCH_SIZE] for i in range(0, len(dicom_paths), BATCH_SIZE)
            ]

            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for dicom_batch in dicom_batches:
                    # Submit a new task to the executor.
                    # The __process_and_save function will process the DICOM and save it.
                    future = executor.submit(
                        self.__process_and_save, dicom_batch, nodule_positions
                    )
                    futures.append(future)

            # Wait for all tasks to complete and handle exceptions if necessary
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Task generated an exception: {e}")

    def __process_and_save(
        self, dicom_paths: list[str],
        nodule_positions: set[float],
    ) -> None:
    """Reads, processes and saves dicom images"""
        for dicom_path in dicom_paths:
            # Get processed image, uid for filename and slice z position
            processed_dicom = self.process_dicom(dicom_path)

            # Check whether slice contains a nodule
            label = (
                NODULE
                if processed_dicom.z_position in nodule_positions
                else NON_NODULE
            )

            # Save image in correct label folder
            filename = f"{processed_dicom.uid}{NUMPY_EXTENSION}"
            output_path = os.path.join(self.output_directory, label, filename)
            np.save(output_path, processed_dicom.image)

    def __generate_annotation_and_dicom_paths(self, path: str) -> tuple:
        """Yields a dictionary with path to annotation and paths to dicoms"""
        for root, _, files in os.walk(path, topdown=False):
            if len(files) == 0:
                continue

            dicom_paths = [
                os.path.realpath(os.path.join(root, f))
                for f in files
                if f.endswith(DICOM_EXTENSION)
            ]

            annotation_path = self.__get_annotation_xml_path(root)

            paths_dictionary = {
                DICOM_KEY: dicom_paths,
                ANNOTATION_KEY: annotation_path,
            }

            yield paths_dictionary

    @staticmethod
    def __get_annotation_xml_path(directory: str) -> str:
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

    @staticmethod
    def __process_annotation(xml_path: str) -> set[float]:
        """Returns a set with z positions of nodules"""
        # Get root from the xml file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        _ = lambda s: f"{ANNOTATION_NAMESPACE}{str(s)}"  # adds namespace name to string

        # Create a list of z positions that contain a nodule
        nodule_z_positions = [
            float(z_position.text)
            for reading_session in root.findall(_(READING_SESSION))
            for unblinded_read_nodule in reading_session.findall(
                _(UNBLINDED_READ_NODULE)
            )
            for roi in unblinded_read_nodule.findall(_(ROI))
            for z_position in roi.findall(_(IMAGE_Z_POSITION))
        ]

        return set(nodule_z_positions)

    def process_dicom(self, dicom_path: str) -> ProcessedDicom:
        """Returns a normalized and segmented image with uid and z position"""
        # Read the dicom file
        dicom = pydicom.dcmread(dicom_path)

        # Create a lung mask
        mask = self.create_lung_mask(dicom)

        # Get image from dicom object
        image = dicom.pixel_array

        # Segment lungs by multiplying image with mask
        image_segmented = image * mask

        # Normalize image
        image_processed = image_segmented / np.max(image_segmented)

        # Get UID from dicom for new filename
        uid = dicom.SOPInstanceUID

        # Get z_position to check whether it contains a nodule
        z_position = dicom.SliceLocation

        processed_dicom = ProcessedDicom(
            image=image_processed,
            uid=uid,
            z_position=z_position,
        )

        return processed_dicom
    
    @staticmethod
    def __fix_outliers(
        dcm: pydicom.dataset.FileDataset
    ) -> pydicom.dataset.FileDataset:
        """Returns dicom object without negative values"""
        # Get image from dicom object
        image = dcm.pixel_array

        # Set negative values to 0
        image[image < 0] = 0

        # Save image data back to dicom object
        dcm.PixelData = image.tobytes()

        return dcm

    def create_lung_mask(self, dcm: pydicom.dataset.FileDataset) -> np.ndarray:
        """Returns a binary mask from dicom object"""
        # Fix outliers and load dicom pixel array
        dcm = self.__fix_outliers(dcm)
        image = dcm.pixel_array

        # Select threshold using the Otsu method
        thresh = threshold_otsu(image)

        # Reverse binarization of the image
        mask_binary = image < thresh

        # Remove border, now only lungs and some noise is visible
        mask_borderless = clear_border(mask_binary)

        # Remove small artifacts around lungs
        mask_without_objects = remove_small_objects(mask_borderless)

        # Fill holes within lungs
        mask_without_holes = remove_small_holes(mask_without_objects)

        # Binary closing remove larger holes
        closing_disk = disk(CLOSING_DISK_DIAMETER)
        mask_closed = binary_closing(mask_without_holes)

        # Binary opening to disconnect lungs
        opening_disk = disk(OPENING_DISK_DIAMETER)
        mask_opened = binary_opening(mask_closed)

        return mask_opened
