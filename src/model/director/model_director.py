"""
Module for the ModelDirector class
"""
import logging

from src.model.builders import ModelBuilder


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelDirector:
    """Director class for model builders."""
    def __init__(self, builder: ModelBuilder):
        self.__builder = builder
        logger.info(f"Initialized ModelDirector with {builder}")

    def make(self):
        """Builds and returns the built model."""
        self.__builder.reset()
        self.__builder.set_preprocessing_layers()
        self.__builder.set_model_layers()
        self.__builder.set_output_layers()
        return self.__builder.build()

    @property
    def builder(self):
        """ModelBuilder object property"""
        return self.__builder

    @builder.setter
    def builder(self, builder: ModelBuilder):
        """Sets the ModelBuilder object property"""
        self.__builder = builder
        logger.info(f"ModelDirector builder set to {builder}")
