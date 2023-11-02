import xml.etree.ElementTree as ET

from src.preprocessing.base import BaseProcessor
from src.preprocessing.utils import *


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
        self._data = []

    def process(self, as_output: bool = False) -> list[ProcessedAnnotation]:
        """Returns a list of processed annotations""" 
        # Get root from the xml file
        tree = ET.parse(self.path)
        root = tree.getroot()
        self._process(root)

        return self._data if as_output else None

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

    @property
    def z_positions(self):
        z_positions = [pa.z_position for pa in self._data]
        return set(z_positions)

    def _process(self, root):
        _ = lambda s: f"{ANNOTATION_NAMESPACE}{str(s)}"  # adds namespace name to string

        for reading_session in root.findall(_(READING_SESSION)):
            for unblinded_read_nodule in reading_session.findall(
                _(UNBLINDED_READ_NODULE)
            ):
                for roi in unblinded_read_nodule.findall(_(ROI)):
                    nodule_z_positions = []
                    nodule_x_positions = []
                    nodule_y_positions = []
                    for z_position in roi.findall(_(IMAGE_Z_POSITION)):
                        nodule_z_positions.append(float(z_position.text))
                    for edge_map in roi.findall(_(EDGE_MAP)):
                        for x_position in edge_map.findall(_(X_COORD)):
                            nodule_x_positions.append(float(x_position.text))
                        for y_position in edge_map.findall(_(Y_COORD)):
                            nodule_y_positions.append(float(y_position.text))

                    if len(nodule_z_positions) > 1:
                        logger.error("Found more z positions for on roi in AnnotationProcessor")
                    if any([
                        len(nodule_x_positions) == 0,
                        len(nodule_y_positions) == 0,
                        len(nodule_z_positions) == 0,
                    ]):
                        logger.error(f"Did not found nodule coordinates for one roi in {self.path}")
                        continue

                    processed_annotation = ProcessedAnnotation(
                        z_position=nodule_z_positions[0],
                        x_positions=nodule_x_positions,
                        y_positions=nodule_y_positions
                    )
                    self._data.append(processed_annotation)
