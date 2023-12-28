"""
Module for the ModelDirector class
"""
import logging

from src.model.director.base import BaseModelDirector
from src.model.builders import ModelBuilder


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelDirector(BaseModelDirector):
    """Director class for model builders."""
    def make(self):
        """Builds and returns the built model."""
        self._builder.reset()
        self._builder.set_preprocessing_layers()
        self._builder.set_model_layers()
        self._builder.set_output_layers()
        return self._builder.build()

    @property
    def builder(self):
        """ModelBuilder object property"""
        return self._builder

    @builder.setter
    def builder(self, builder: ModelBuilder):
        """Sets the ModelBuilder object property"""
        self._builder = builder
        logger.info(f"ModelDirector builder set to {str(builder)}")
