"""
Module with ConvNeXt concrete builder
"""
import logging

import tensorflow as tf

from src.model.builders import ModelBuilder
from src.preprocessing.utils import HEIGHT, WIDTH

CONVNEXT_INPUT_SHAPE = (224, 224, 3)
CONVNEXT_DENSE_UNITS = 128
CONVNEXT_DENSE_ACTIVATION = "relu"
CONVNEXT_DROPOUT_RATE = 0.2
CONVNEXT_OUTPUT_ACTIVATION = "sigmoid"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ConvNeXtBuilder(ModelBuilder):
    """ConvNeXt concrete builder"""
    def __str__(self):
        return "ConvNeXt"

    def set_preprocessing_layers(self):
        """Sets the preprocessing layers for the ConvNeXt model"""
        reshape_layer = tf.keras.layers.Reshape(target_shape=(HEIGHT, WIDTH, 1))

        greyscale_layer = tf.keras.layers.Lambda(tf.image.grayscale_to_rgb)

        resize_layer = tf.keras.layers.Resizing(
            height=CONVNEXT_INPUT_SHAPE[0], width=CONVNEXT_INPUT_SHAPE[1]
        )

        preprocess_input = tf.keras.applications.convnext.preprocess_input

        preprocess_input_layer = tf.keras.layers.Lambda(
            preprocess_input, input_shape=CONVNEXT_INPUT_SHAPE
        )

        self.preprocessing_layers = tf.keras.Sequential(
            [
                reshape_layer,
                greyscale_layer,
                resize_layer,
                preprocess_input_layer,
            ]
        )
        logger.info("ConvNeXt preprocessing layers set")

    def set_model_layers(self):
        """Sets the model layers for the ConvNeXt model"""
        base_model = tf.keras.applications.convnext.ConvNeXtSmall(
            input_shape=CONVNEXT_INPUT_SHAPE, include_top=False, weights="imagenet"
        )

        base_model.trainable = False

        self.model_layers = base_model
        logger.info("ConvNeXt model layers set")

    def set_output_layers(self):
        """Sets the output layers for the ConvNeXt model"""
        pooling_layer = tf.keras.layers.GlobalAveragePooling2D()

        dense_layer = tf.keras.layers.Dense(
            units=CONVNEXT_DENSE_UNITS, activation=CONVNEXT_DENSE_ACTIVATION
        )

        dropout_layer = tf.keras.layers.Dropout(rate=CONVNEXT_DROPOUT_RATE)

        output_layer = tf.keras.layers.Dense(1, activation=CONVNEXT_OUTPUT_ACTIVATION)

        self.output_layers = tf.keras.Sequential(
            [pooling_layer, dense_layer, dropout_layer, output_layer]
        )
        logger.info("ConvNeXt output layers set")

    def build(self):
        """Builds the ConvNeXt model"""
        model = super().build()
        logger.info("ConvNeXt model built")
        return model

    def reset(self):
        """Resets the ConvNeXt builder"""
        super().reset()
        logger.info("ConvNeXt reset")
