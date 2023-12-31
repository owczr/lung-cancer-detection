"""
Module for the MobileNet concrete builder
"""
import logging

import tensorflow as tf

from src.model.builders import ModelBuilder
from src.preprocessing.utils import HEIGHT, WIDTH


MOBILENET_INPUT_SHAPE = (224, 224, 3)
MOBILENET_DENSE_UNITS = 128
MOBILENET_DENSE_ACTIVATION = "relu"
MOBILENET_DROPOUT_RATE = 0.2
MOBILENET_OUTPUT_ACTIVATION = "sigmoid"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MobileNetBuilder(ModelBuilder):
    """Concrete builder for MobileNet."""
    def __str__(self):
        return "MobileNetBuilder"

    def set_preprocessing_layers(self):
        """Sets the preprocessing layers for the MobileNet model"""
        reshape_layer = tf.keras.layers.Reshape(
            target_shape=(HEIGHT, WIDTH, 1)
        )

        greyscale_layer = tf.keras.layers.Lambda(
           tf.image.grayscale_to_rgb
        )

        resize_layer = tf.keras.layers.Resizing(
            height=MOBILENET_INPUT_SHAPE[0], width=MOBILENET_INPUT_SHAPE[1]
        )

        preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input

        preprocess_input_layer = tf.keras.layers.Lambda(
            preprocess_input, input_shape=MOBILENET_INPUT_SHAPE
        )

        self.preprocessing_layers = tf.keras.Sequential(
            [
                reshape_layer,
                greyscale_layer,
                resize_layer,
                preprocess_input_layer,
            ]
        )
        logger.info("MobileNet preprocessing layers set")

    def set_model_layers(self):
        """Sets the model layers for the MobileNet model"""
        base_model = tf.keras.applications.MobileNetV3Small(
            input_shape=MOBILENET_INPUT_SHAPE, include_top=False, weights="imagenet"
        )

        base_model.trainable = False

        self.model_layers = base_model
        logger.info("MobileNet model layers set")

    def set_output_layers(self):
        """Sets the output layers for the MobileNet model"""
        pooling_layer = tf.keras.layers.GlobalAveragePooling2D()

        dense_layer = tf.keras.layers.Dense(
            units=MOBILENET_DENSE_UNITS, activation=MOBILENET_DENSE_ACTIVATION
        )

        dropout_layer = tf.keras.layers.Dropout(rate=MOBILENET_DROPOUT_RATE)

        output_layer = tf.keras.layers.Dense(1, activation=MOBILENET_OUTPUT_ACTIVATION)

        self.output_layers = tf.keras.Sequential(
            [pooling_layer, dense_layer, dropout_layer, output_layer]
        )
        logger.info("MobileNet output layers set")

    def build(self):
        """Builds the MobileNet model"""
        model = super().build()
        logger.info("MobileNet built")
        return model

    def reset(self):
        """Resets the MobileNet model"""
        super().reset()
        logger.info("MobileNet reset")
