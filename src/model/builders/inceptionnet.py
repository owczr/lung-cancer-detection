"""
Module with InceptionNet concrete builder
"""
import logging

import tensorflow as tf

from src.model.builders import ModelBuilder
from src.preprocessing.utils import HEIGHT, WIDTH

INCEPTIONNET_INPUT_SHAPE = (224, 224, 3)
INCEPTIONNET_DENSE_UNITS = 128
INCEPTIONNET_DENSE_ACTIVATION = "relu"
INCEPTIONNET_DROPOUT_RATE = 0.2
INCEPTIONNET_OUTPUT_ACTIVATION = "sigmoid"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class InceptionNetBuilder(ModelBuilder):
    """InceptionNet concrete builder""" 

    def set_preprocessing_layers(self):
        """Sets the preprocessing layers for the InceptionNet model"""
        reshape_layer = tf.keras.layers.Reshape(target_shape=(HEIGHT, WIDTH, 1))

        greyscale_layer = tf.keras.layers.Lambda(tf.image.grayscale_to_rgb)

        resize_layer = tf.keras.layers.Resizing(
            height=INCEPTIONNET_INPUT_SHAPE[0], width=INCEPTIONNET_INPUT_SHAPE[1]
        )

        preprocess_input = tf.keras.applications.inception_v3.preprocess_input

        preprocess_input_layer = tf.keras.layers.Lambda(
            preprocess_input, input_shape=INCEPTIONNET_INPUT_SHAPE
        )

        self.preprocessing_layers = tf.keras.Sequential(
            [
                reshape_layer,
                greyscale_layer,
                resize_layer,
                preprocess_input_layer,
            ]
        )
        
        logger.info("InceptionNet preprocessing layers set")

    def set_model_layers(self):
        """Sets the model layers for the InceptionNet model"""  
        base_model = tf.keras.applications.inception_v3.InceptionV3(  
            input_shape=INCEPTIONNET_INPUT_SHAPE, include_top=False, weights="imagenet"
        )

        base_model.trainable = False

        self.model_layers = base_model
        logger.info("InceptionNet model layers set")

    def set_output_layers(self):
        """Sets the output layers for the InceptionNet model"""
        pooling_layer = tf.keras.layers.GlobalAveragePooling2D()

        dense_layer = tf.keras.layers.Dense(
            units=INCEPTIONNET_DENSE_UNITS, activation=INCEPTIONNET_DENSE_ACTIVATION
        )

        dropout_layer = tf.keras.layers.Dropout(rate=INCEPTIONNET_DROPOUT_RATE)

        output_layer = tf.keras.layers.Dense(
            1, activation=INCEPTIONNET_OUTPUT_ACTIVATION
        )

        self.output_layers = tf.keras.Sequential(
            [pooling_layer, dense_layer, dropout_layer, output_layer]
        )
        logger.info("InceptionNet output layers set")

    def build(self):
        """Builds the InceptionNet model"""
        model = super().build()
        logger.info("InceptionNet model built")
        return model

    def reset(self):
        """Resets the InceptionNet builder"""
        super().reset()
        logger.info("InceptionNet reset")
