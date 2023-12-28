"""
Module with Xception concrete builder
"""
import logging

import tensorflow as tf

from src.model.builders import ModelBuilder
from src.preprocessing.utils import HEIGHT, WIDTH

XCEPTION_INPUT_SHAPE = (224, 224, 3)
XCEPTION_DENSE_UNITS = 128
XCEPTION_DENSE_ACTIVATION = "relu"
XCEPTION_DROPOUT_RATE = 0.2
XCEPTION_OUTPUT_ACTIVATION = "sigmoid"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class XceptionBuilder(ModelBuilder):
    """Xception concrete builder"""

    def set_preprocessing_layers(self):
        """Sets the preprocessing layers for the Xception model"""
        reshape_layer = tf.keras.layers.Reshape(target_shape=(HEIGHT, WIDTH, 1))

        greyscale_layer = tf.keras.layers.Lambda(tf.image.grayscale_to_rgb)

        resize_layer = tf.keras.layers.Resizing(
            height=XCEPTION_INPUT_SHAPE[0], width=XCEPTION_INPUT_SHAPE[1]
        )

        preprocess_input = tf.keras.applications.xception.preprocess_input

        preprocess_input_layer = tf.keras.layers.Lambda(
            preprocess_input, input_shape=XCEPTION_INPUT_SHAPE
        )

        self.preprocessing_layers = tf.keras.Sequential(
            [
                reshape_layer,
                greyscale_layer,
                resize_layer,
                preprocess_input_layer,
            ]
        )
        logger.info("Xception preprocessing layers set")

    def set_model_layers(self):
        """Sets the model layers for the Xception model"""
        base_model = tf.keras.applications.xception.Xception(  
            input_shape=XCEPTION_INPUT_SHAPE, include_top=False, weights="imagenet"
        )

        base_model.trainable = False

        self.model_layers = base_model
        logger.info("Xception model layers set")

    def set_output_layers(self):
        """Sets the output layers for the Xception model"""
        pooling_layer = tf.keras.layers.GlobalAveragePooling2D()

        dense_layer = tf.keras.layers.Dense(
            units=XCEPTION_DENSE_UNITS, activation=XCEPTION_DENSE_ACTIVATION
        )

        dropout_layer = tf.keras.layers.Dropout(rate=XCEPTION_DROPOUT_RATE)

        output_layer = tf.keras.layers.Dense(
            1, activation=XCEPTION_OUTPUT_ACTIVATION
        )

        self.output_layers = tf.keras.Sequential(
            [pooling_layer, dense_layer, dropout_layer, output_layer]
        )
        logger.info("Xception output layers set")

    def build(self):
        """Builds the Xception model"""
        model = super().build()
        logger.info("Xception model built")
        return model

    def reset(self):
        """Resets the Xception builder"""
        super().reset()
        logger.info("Xception reset")
