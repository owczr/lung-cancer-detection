"""
Module with TEMPLATE concrete builder  # TODO: Change the docstring
"""
import logging

import tensorflow as tf

from src.model.builders import ModelBuilder
from src.preprocessing.utils import HEIGHT, WIDTH

# TODO: Change the following constants
TEMPLATE_INPUT_SHAPE = (224, 224, 3)
TEMPLATE_DENSE_UNITS = 128
TEMPLATE_DENSE_ACTIVATION = "relu"
TEMPLATE_DROPOUT_RATE = 0.2
TEMPLATE_OUTPUT_ACTIVATION = "sigmoid"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# TODO: Change the following class name
class TEMPLATEBuilder(ModelBuilder):
    """TEMPLATE concrete builder"""  # TODO: Change the docstring

    def set_preprocessing_layers(self):
        """Sets the preprocessing layers for the TEMPLATE model"""
        reshape_layer = tf.keras.layers.Reshape(target_shape=(HEIGHT, WIDTH, 1))

        greyscale_layer = tf.keras.layers.Lambda(tf.image.grayscale_to_rgb)

        resize_layer = tf.keras.layers.Resizing(
            height=TEMPLATE_INPUT_SHAPE[0], width=TEMPLATE_INPUT_SHAPE[1]
        )

        # TODO: Change the following preprocess_input function
        preprocess_input = tf.keras.applications.TEMPLATE.preprocess_input

        preprocess_input_layer = tf.keras.layers.Lambda(
            preprocess_input, input_shape=TEMPLATE_INPUT_SHAPE
        )

        self.preprocessing_layers = tf.keras.Sequential(
            [
                reshape_layer,
                greyscale_layer,
                resize_layer,
                preprocess_input_layer,
            ]
        )
        # TODO: Change the following log message
        logger.info("TEMPLATE preprocessing layers set")

    def set_model_layers(self):
        """Sets the model layers for the TEMPLATE model"""  # TODO: Change the docstring
        # TODO: Change the following base_model
        base_model = tf.keras.applications.TEMPLATE.TEMPLATE(  
            input_shape=TEMPLATE_INPUT_SHAPE, include_top=False, weights="imagenet"
        )

        base_model.trainable = False

        self.model_layers = base_model
        logger.info("TEMPLATE model layers set")  # TODO: Change the log message

    def set_output_layers(self):
        """Sets the output layers for the TEMPLATE model"""  # TODO: Change the docstring
        pooling_layer = tf.keras.layers.GlobalAveragePooling2D()

        dense_layer = tf.keras.layers.Dense(
            units=TEMPLATE_DENSE_UNITS, activation=TEMPLATE_DENSE_ACTIVATION
        )

        dropout_layer = tf.keras.layers.Dropout(rate=TEMPLATE_DROPOUT_RATE)

        output_layer = tf.keras.layers.Dense(
            1, activation=TEMPLATE_OUTPUT_ACTIVATION
        )

        self.output_layers = tf.keras.Sequential(
            [pooling_layer, dense_layer, dropout_layer, output_layer]
        )
        logger.info("TEMPLATE output layers set")  # TODO: Change the log message

    def build(self):
        """Builds the TEMPLATE model"""  # TODO: Change the docstring
        model = super().build()
        logger.info("TEMPLATE model built")  # TODO: Change the log message
        return model

    def reset(self):
        """Resets the TEMPLATE builder"""  # TODO: Change the docstring
        super().reset()
        logger.info("TEMPLATE reset")  # TODO: Change the log message
