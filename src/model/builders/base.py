"""
Module for the base class of model builders
"""
import logging
from abc import ABC, abstractmethod

import tensorflow as tf


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelBuilderError(Exception):
    """Base class for exceptions in this module"""
    ...


class ModelBuilder(ABC):
    """Base class for model builders"""

    def __init__(self):
        self.preprocessing_layers = None
        self.model_layers = None
        self.output_layers = None
        self.__model = None

    @abstractmethod
    def reset(self):
        """Resets the model builder"""
        self.model = None
        self.preprocessing_layers = None
        self.model_layers = None
        self.output_layers = None

    @property
    def model(self):
        """Built model"""
        return self.__model

    @model.setter
    def model(self, model) -> None:
        """Sets the built model"""
        self.__model = model

    @abstractmethod
    def set_preprocessing_layers(self) -> None:
        """Sets the preprocessing layers of the model"""
        pass

    @abstractmethod
    def set_model_layers(self) -> None:
        """Sets the model layers of the model"""
        pass

    @abstractmethod
    def set_output_layers(self) -> None:
        """Sets the output layers of the model"""
        pass

    @abstractmethod
    def build(self):
        """Builds the model"""
        layers = [self.preprocessing_layers, self.model_layers, self.output_layers]

        if not all(layers):
            raise ModelBuilderError("Not all layers are set")

        self.model = tf.keras.Sequential(layers)

        return self.model
