"""
Module for the EfficientNet concrete builder
"""
import logging

import tensorflow as tf

from src.model.builders import ModelBuilder
from src.preprocessing.utils import HEIGHT, WIDTH


EFFICIENTNET_INPUT_SHAPE = (224, 224, 3)
EFFICIENTNET_DENSE_UNITS = 128
EFFICIENTNET_DENSE_ACTIVATION = "relu"
EFFICIENTNET_DROPOUT_RATE = 0.2
EFFICIENTNET_OUTPUT_ACTIVATION = "sigmoid"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EfficientNetBuilder(ModelBuilder):
    """EfficientNet concrete builder"""

    def __str__(self):
        return "EfficientNet"

    def set_preprocessing_layers(self):
        """Sets the preprocessing layers for the EfficientNet model"""
        reshape_layer = tf.keras.layers.Reshape(target_shape=(HEIGHT, WIDTH, 1))

        greyscale_layer = tf.keras.layers.Lambda(tf.image.grayscale_to_rgb)

        resize_layer = tf.keras.layers.Resizing(
            height=EFFICIENTNET_INPUT_SHAPE[0], width=EFFICIENTNET_INPUT_SHAPE[1]
        )

        preprocess_input = tf.keras.applications.efficientnet.preprocess_input

        preprocess_input_layer = tf.keras.layers.Lambda(
            preprocess_input, input_shape=EFFICIENTNET_INPUT_SHAPE
        )

        self.preprocessing_layers = tf.keras.Sequential(
            [
                reshape_layer,
                greyscale_layer,
                resize_layer,
                preprocess_input_layer,
            ]
        )
        logger.info("EfficientNet preprocessing layers set")

    def set_model_layers(self):
        """Sets the model layers for the EfficientNet model"""
        base_model = tf.keras.applications.EfficientNetB7(
            input_shape=EFFICIENTNET_INPUT_SHAPE, include_top=False, weights="imagenet"
        )

        base_model.trainable = False

        self.model_layers = base_model
        logger.info("EfficientNet model layers set")

    def set_output_layers(self):
        """Sets the output layers for the EfficientNet model"""
        pooling_layer = tf.keras.layers.GlobalAveragePooling2D()

        dense_layer = tf.keras.layers.Dense(
            units=EFFICIENTNET_DENSE_UNITS, activation=EFFICIENTNET_DENSE_ACTIVATION
        )

        dropout_layer = tf.keras.layers.Dropout(rate=EFFICIENTNET_DROPOUT_RATE)

        output_layer = tf.keras.layers.Dense(
            1, activation=EFFICIENTNET_OUTPUT_ACTIVATION
        )

        self.output_layers = tf.keras.Sequential(
            [pooling_layer, dense_layer, dropout_layer, output_layer]
        )
        logger.info("EfficientNet output layers set")

    def build(self):
        """Builds the EfficientNet model"""
        model = super().build()
        logger.info("EfficientNet model built")
        return model

    def reset(self):
        """Resets the EfficientNet builder"""
        super().reset()
        logger.info("EfficientNet reset")
