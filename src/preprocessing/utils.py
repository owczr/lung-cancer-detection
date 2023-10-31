from dataclasses import dataclass

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
ANNOTATION_NAMESPACE = "{http://www.nih.gov}"
# Parallelization
BATCH_SIZE = 10
MAX_WORKERS = None  # this will use all cores
# Dicom image processing
CLOSING_DISK_DIAMETER = 15
OPENING_DISK_DIAMETER = 5


@dataclass
class ProcessedDicom:
    image: np.ndarray
    uid: str
    z_position: float
