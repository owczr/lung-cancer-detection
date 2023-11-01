import xml.etree.ElementTree as ET

from base import BaseProcessor
from utils import *


class AnnotationProcessor(BaseProcessor):
    """
    Processor for handling annotations.

    This class is responsible for processing annotations and extracting relevant information.
    
    Attributes:
        path (str): The path to the XML file containing annotations.
        _data (set[float]): Set of z positions of nodules after processing.

    Methods inherited from BaseProcessor:
        process, save, process_and_save.
    """
    def __init__(self, path):
        self.path = path
        self._data = None

    def process(self) -> set[float]:
        """Returns a set with z positions of nodules"""
        # Get root from the xml file
        tree = ET.parse(self.path)
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

        self._data = set(nodule_z_positions)

        return self._data

    def save(self, path: str) -> None:
        raise NotImplementedError(
            "Save method was not needed and it wasn't implemented"
        )

    def process_and_save(self, path: str) -> None:
        self.process()
        self.save()

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
