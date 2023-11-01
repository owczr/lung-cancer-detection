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

from base import BaseProcessor
from utils import *


class DicomProcessor(BaseProcessor):
    """
    Processor for handling DICOM files.

    This class is responsible for reading DICOM files, processing images,
    creating masks, and normalization.
    
    Attributes:
        path (str): The path to the DICOM file.
        _data (ProcessedDicom): Processed DICOM data after processing.

    Methods inherited from BaseProcessor:
        process, save, process_and_save.

    Private Methods:
        _create_lung_mask: Creates a binary mask from the DICOM data.
        _fix_outliers: Fixes outlier values in the DICOM data.
    """
    def __init__(self, path: str):
        self.path = path
        self._data = None

    def process(self) -> ProcessedDicom:
        """Returns a normalized and segmented image with uid and z position"""
        # Read the dicom file
        dicom = pydicom.dcmread(self.path)

        # Create a lung mask
        mask = self._create_lung_mask(dicom)

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

        self._data = processed_dicom

        return self._data

    def save(self, path):
        if self._data is None:
            message = "Data is empty, call process method first!"
            raise Exception(message)
        
        # Create filename and save image
        filename = f"{self.data.uid}{NUMPY_EXTENSION}"
        output_path = os.path.join(path, filename)
        np.save(output_path, self.data.image)


    def process_and_save(self, path: str) -> None:
    """Reads, processes and saves dicom images"""
        # Get processed image, uid for filename and slice z position
        self.process()
        self.save(path)

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

    
    def _create_lung_mask(self, dcm: pydicom.dataset.FileDataset) -> np.ndarray:
        """Returns a binary mask from dicom object"""
        # Fix outliers and load dicom pixel array
        dcm = self._fix_outliers(dcm)
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

    @staticmethod
    def _fix_outliers(
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
