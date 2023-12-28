"""
Module for the BaseModelDirector interface
"""
import logging
from abc import ABC, abstractmethod

from src.model.builders import ModelBuilder


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseModelDirector(ABC):
    """Interface for model director for model builders."""
    def __init__(self, builder: ModelBuilder):
        self._builder: ModelBuilder = builder
        logger.info(f"Initialized ModelDirector with {str(builder)}")

    @abstractmethod
    def make(self):
        """Builds and returns the built model."""
        pass

    @property
    @abstractmethod
    def builder(self):
        """ModelBuilder object property"""
        pass

    @builder.setter
    @abstractmethod
    def builder(self, builder: ModelBuilder):
        """Sets the ModelBuilder object property"""
        pass
    