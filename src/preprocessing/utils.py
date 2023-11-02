import logging
from dataclasses import dataclass

import numpy as np

# Labels
NODULE = "nodule"
NON_NODULE = "non_nodule"
# Extensions
DICOM_EXTENSION = ".dcm"
NUMPY_EXTENSION = ".npy"
ANNOTATION_EXTENSION = ".xml"
# Path dictionary
DICOM_KEY = "dicom"
ANNOTATION_KEY = "annotation"
# Annotation xml
READING_SESSION = "readingSession"
UNBLINDED_READ_NODULE = "unblindedReadNodule"
ROI = "roi"
IMAGE_Z_POSITION = "imageZposition"
EDGE_MAP = "edgeMap"
Y_COORD = "yCoord"
X_COORD = "xCoord"
ANNOTATION_NAMESPACE = "{http://www.nih.gov}"
# Dicom tags
SLICE_LOCATION = "SliceLocation"
SOP_INSTANCE_UID = "SOPInstanceUID"
PIXEL_DATA = "PixelData"
# Parallelization
BATCH_SIZE = 10
MAX_WORKERS = None  # this will use all cores
# Dicom image processing
CLOSING_DISK_DIAMETER = 15
OPENING_DISK_DIAMETER = 5
# Logging
PREPROCESSING_LOG = "preprocessing.log"


@dataclass
class ProcessedDicom:
    """Processed dicom image with uid and slice location"""
    image: np.ndarray
    uid: str
    z_position: float

@dataclass
class ProcessedAnnotation:
    """Annotation coordinates on a single dicom"""
    z_position: float
    x_positions: list[float]
    y_positions: list[float]